"""Reproduce frozen test-policy evaluation with auditable SQLite queries.

The three SQL tables are normalized *derived views of public experiment rows*,
not production event logs. ``row_id`` is the zero-based source CSV row position,
not a user identifier. A 1:1 join proves row alignment only, not user independence.
The database stays in memory; no raw data or previous experiment is modified.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data import fingerprint, safe_output, write_json
from .evaluation import evaluate_policies

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
CREATE TABLE assignment (
    row_id INTEGER PRIMARY KEY,
    input_position INTEGER NOT NULL UNIQUE,
    treatment INTEGER NOT NULL CHECK (treatment IN (0, 1))
);
CREATE TABLE outcome (
    row_id INTEGER PRIMARY KEY REFERENCES assignment(row_id),
    response INTEGER NOT NULL CHECK (response IN (0, 1))
);
CREATE TABLE policy_scores (
    policy TEXT NOT NULL,
    row_id INTEGER NOT NULL REFERENCES assignment(row_id),
    score REAL NOT NULL,
    tie_priority REAL NOT NULL,
    PRIMARY KEY (policy, row_id)
);
"""

JOIN_AUDIT_SQL = """
SELECT
    (SELECT COUNT(*) FROM assignment) AS assignment_rows,
    (SELECT COUNT(*) FROM outcome) AS outcome_rows,
    (SELECT COUNT(*) FROM assignment a JOIN outcome o USING (row_id)) AS joined_rows,
    (SELECT COUNT(*) FROM assignment a LEFT JOIN outcome o USING (row_id)
        WHERE o.row_id IS NULL) AS missing_outcomes,
    (SELECT COUNT(*) FROM outcome o LEFT JOIN assignment a USING (row_id)
        WHERE a.row_id IS NULL) AS orphan_outcomes;
"""

SCORE_JOIN_AUDIT_SQL = """
SELECT
    COUNT(*) AS score_rows,
    COUNT(DISTINCT s.row_id) AS distinct_score_rows,
    (SELECT COUNT(*) FROM assignment a JOIN outcome o USING (row_id)
        JOIN policy_scores p USING (row_id) WHERE p.policy = :policy) AS joined_rows,
    (SELECT COUNT(*) FROM assignment a LEFT JOIN policy_scores p
        ON a.row_id = p.row_id AND p.policy = :policy
        WHERE p.row_id IS NULL) AS missing_scores
FROM policy_scores s WHERE s.policy = :policy;
"""

CVR_SQL = """
WITH counts AS (
    SELECT COUNT(*) AS n,
        SUM(CASE WHEN a.treatment = 1 THEN 1 ELSE 0 END) AS n_treatment,
        SUM(CASE WHEN a.treatment = 0 THEN 1 ELSE 0 END) AS n_control,
        SUM(CASE WHEN a.treatment = 1 THEN o.response ELSE 0 END) AS responses_treatment,
        SUM(CASE WHEN a.treatment = 0 THEN o.response ELSE 0 END) AS responses_control
    FROM assignment a JOIN outcome o USING (row_id)
)
SELECT *,
    1.0 * responses_treatment / NULLIF(n_treatment, 0) AS cvr_treatment,
    1.0 * responses_control / NULLIF(n_control, 0) AS cvr_control,
    1.0 * responses_treatment / NULLIF(n_treatment, 0)
        - 1.0 * responses_control / NULLIF(n_control, 0) AS difference_in_means,
    (responses_treatment / :e - responses_control / (1.0 - :e)) / n AS ate_ht
FROM counts;
"""

# An input-position final key agrees with NumPy's stable lexsort if the random
# tie priorities ever collide. The priorities themselves never use y or t.
RANKED_SQL = """
WITH ranked AS (
    SELECT a.row_id, a.treatment, o.response, p.score,
        ROW_NUMBER() OVER (
            ORDER BY p.score DESC, p.tie_priority ASC, a.input_position ASC
        ) AS rank,
        NTILE(:n_bins) OVER (
            ORDER BY p.score DESC, p.tie_priority ASC, a.input_position ASC
        ) AS score_bin
    FROM policy_scores p
    JOIN assignment a USING (row_id)
    JOIN outcome o USING (row_id)
    WHERE p.policy = :policy
)
"""

DECILES_SQL = RANKED_SQL + """
SELECT score_bin, COUNT(*) AS n, MIN(rank) AS first_rank, MAX(rank) AS last_rank,
    AVG(score) AS mean_score, MIN(score) AS min_score, MAX(score) AS max_score,
    SUM(treatment) AS n_treatment, SUM(1 - treatment) AS n_control,
    SUM(treatment * response) AS responses_treatment,
    SUM((1 - treatment) * response) AS responses_control,
    1.0 * SUM(treatment * response) / NULLIF(SUM(treatment), 0) AS cvr_treatment,
    1.0 * SUM((1 - treatment) * response) / NULLIF(SUM(1 - treatment), 0) AS cvr_control,
    1.0 * SUM(treatment * response) / NULLIF(SUM(treatment), 0)
        - 1.0 * SUM((1 - treatment) * response) / NULLIF(SUM(1 - treatment), 0)
        AS observed_uplift,
    SUM(treatment * response / :e - (1 - treatment) * response / (1.0 - :e))
        / :n * 100000.0 AS incremental_per_100k_eligible
FROM ranked GROUP BY score_bin ORDER BY score_bin;
"""

POLICY_VALUE_SQL = RANKED_SQL + """
SELECT COUNT(*) AS n_selected,
    COALESCE(SUM(treatment * response / :e
        - (1 - treatment) * response / (1.0 - :e)), 0.0)
        / :n * 100000.0 AS incremental_per_100k
FROM ranked WHERE rank <= :k;
"""


def sql_queries() -> str:
    """Return the exact executed SQL with named bind-parameter documentation."""
    pieces = [
        "-- SQLite evidence: normalized derived tables; no synthetic user IDs.\n"
        "-- Bind parameters: policy (name), e (frozen assignment probability),\n"
        "-- n (all eligible test rows), k=floor(budget*n+1e-12), n_bins.\n"
        "-- Scores are loaded one policy at a time with executemany.\n"
        "-- tie_priority = NumPy default_rng(seed).random(n), aligned to input order.\n"
        "-- NTILE bin 1 contains the highest scores. NULL denotes a missing arm.\n",
    ]
    for name, query in (
        ("schema", SCHEMA_SQL),
        ("assignment/outcome 1:1 join", JOIN_AUDIT_SQL),
        ("policy-score completeness", SCORE_JOIN_AUDIT_SQL),
        ("conditional aggregation", CVR_SQL),
        ("window-function bins", DECILES_SQL),
        ("top-budget HT policy value", POLICY_VALUE_SQL),
    ):
        pieces.append(f"\n-- {name}\n{query.strip()}\n")
    return "".join(pieces)


def audit_arrays(
    row_id,
    y,
    t,
    score_dict: Mapping[str, np.ndarray],
    e: float,
    *,
    budget: float = 0.2,
    seed: int = 42,
    n_bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Validate joins and reconcile SQL with Python for the same frozen policy.

    Only estimates are recomputed: no training, policy reselection, or confidence
    intervals. Independent tie priorities are shared across policies and exactly
    match ``evaluate_policies``. Costs/revenue are not present; the denominator
    is all eligible test rows, not contacted rows. Bins are descriptive only.
    """
    ids = np.asarray(row_id)
    if (
        ids.ndim != 1
        or not np.issubdtype(ids.dtype, np.integer)
        or len(ids) == 0
        or np.any(ids < 0)
        or np.any(ids > np.iinfo(np.int64).max)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError(
            "row_id must contain unique, nonnegative int64 source-row positions"
        )
    if len(ids) != len(y):
        raise ValueError("row_id must match the evaluated rows")
    if not score_dict:
        raise ValueError("at least one frozen score policy is required")
    if (
        not isinstance(n_bins, (int, np.integer))
        or isinstance(n_bins, bool)
        or n_bins < 1
    ):
        raise ValueError("n_bins must be a positive integer")
    # The public evaluator supplies the shared finite/binary/e/budget validation.
    reference = evaluate_policies(
        y, t, score_dict, e, budgets=[budget], bootstrap=0, seed=seed
    )
    y, t = np.asarray(y, dtype=np.int8), np.asarray(t, dtype=np.int8)
    n = len(ids)
    k = int(reference.n_selected.iloc[0])
    priorities = np.random.default_rng(seed).random(n)
    join_records, bins, values = [], [], []
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO assignment VALUES (?, ?, ?)",
            ((int(ids[i]), i, int(t[i])) for i in range(n)),
        )
        conn.executemany(
            "INSERT INTO outcome VALUES (?, ?)",
            ((int(ids[i]), int(y[i])) for i in range(n)),
        )
        join = pd.read_sql_query(JOIN_AUDIT_SQL, conn).iloc[0].to_dict()
        if any(
            join[x] != n for x in ("assignment_rows", "outcome_rows", "joined_rows")
        ):
            raise AssertionError("assignment/outcome 1:1 join changed row count")
        if join["missing_outcomes"] or join["orphan_outcomes"]:
            raise AssertionError("assignment/outcome join lost keys")
        join_records.append(
            {"table_pair": "assignment_outcome", "policy": None, **join}
        )
        cvr = pd.read_sql_query(CVR_SQL, conn, params={"e": float(e)})
        for name, scores in score_dict.items():
            conn.execute("DELETE FROM policy_scores")  # in-memory derived rows only
            conn.executemany(
                "INSERT INTO policy_scores VALUES (?, ?, ?, ?)",
                (
                    (name, int(ids[i]), float(scores[i]), float(priorities[i]))
                    for i in range(n)
                ),
            )
            params = {
                "policy": name,
                "e": float(e),
                "n": n,
                "k": k,
                "n_bins": int(n_bins),
            }
            joined = (
                pd.read_sql_query(SCORE_JOIN_AUDIT_SQL, conn, params=params)
                .iloc[0]
                .to_dict()
            )
            if (
                any(
                    joined[x] != n
                    for x in ("score_rows", "distinct_score_rows", "joined_rows")
                )
                or joined["missing_scores"]
            ):
                raise AssertionError(
                    f"{name}: policy-score join must be complete and 1:1"
                )
            join_records.append(
                {"table_pair": "assignment_outcome_scores", "policy": name, **joined}
            )
            frame = pd.read_sql_query(DECILES_SQL, conn, params=params)
            frame.insert(0, "policy", name)
            bins.append(frame)
            value = pd.read_sql_query(POLICY_VALUE_SQL, conn, params=params).iloc[0]
            expected = reference.loc[reference.policy == name].iloc[0]
            delta = float(value.incremental_per_100k - expected.incremental_per_100k)
            if int(value.n_selected) != k or not np.isclose(
                delta, 0, atol=1e-7, rtol=0
            ):
                raise AssertionError(
                    f"{name}: SQL/Python HT policy discrepancy {delta}"
                )
            values.append(
                {
                    "policy": name,
                    "budget": float(budget),
                    "n_eligible": n,
                    "n_selected": int(value.n_selected),
                    "selected_fraction": k / n,
                    "sql_incremental_per_100k": float(value.incremental_per_100k),
                    "python_incremental_per_100k": float(expected.incremental_per_100k),
                    "sql_minus_python": delta,
                }
            )
        random_sql = k / n * float(cvr.ate_ht.iloc[0]) * 100_000
        random_python = float(
            reference.loc[reference.policy == "random", "incremental_per_100k"].iloc[0]
        )
        if not np.isclose(random_sql, random_python, atol=1e-7, rtol=0):
            raise AssertionError("analytic random-policy reconciliation failed")
        values.insert(
            0,
            {
                "policy": "random",
                "budget": float(budget),
                "n_eligible": n,
                "n_selected": k,
                "selected_fraction": k / n,
                "sql_incremental_per_100k": random_sql,
                "python_incremental_per_100k": random_python,
                "sql_minus_python": random_sql - random_python,
            },
        )
    finally:
        conn.close()
    policies = pd.DataFrame(values)
    policies["vs_random_per_100k"] = policies.sql_incremental_per_100k - random_sql
    return {
        "join_integrity": pd.DataFrame(join_records),
        "cvr": cvr,
        "score_bins": pd.concat(bins, ignore_index=True),
        "policy_values": policies,
    }


def run_sql_audit(source: str | Path, output: str | Path) -> dict:
    """Read a frozen experiment and create a new, non-overwriting audit folder."""
    source = Path(source).resolve()
    output = safe_output(output)
    if output == source or source in output.parents:
        raise ValueError("write the SQL audit outside the frozen source experiment")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    paths = {
        name: source / name
        for name in ("predictions.npz", "manifest.json", "selection.json")
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection.json"].read_text(encoding="utf-8"))
    if selection.get("frozen_before_test") is not True:
        raise ValueError(
            "source must contain a policy selection frozen before test evaluation"
        )
    e, seed = float(manifest["assignment_probability"]), int(manifest["seed"])
    budget = float(selection["primary_budget"])
    with np.load(paths["predictions.npz"], allow_pickle=False) as saved:
        scores = {
            name: saved[name]
            for name in saved.files
            if name not in ("row_id", "y", "t")
        }
        if selection["selected_policy"] not in {"random", *scores}:
            raise ValueError("the frozen selected policy is missing from predictions")
        tables = audit_arrays(
            saved["row_id"], saved["y"], saved["t"], scores, e, budget=budget, seed=seed
        )
    after = {name: fingerprint(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("source artifacts changed during audit")
    tables["policy_values"]["selected_on_validation"] = (
        tables["policy_values"].policy == selection["selected_policy"]
    )
    provenance = {
        "status": "complete",
        "source_directory": str(source),
        "source_files": before,
        "source_unchanged": before == after,
        "sqlite_version": sqlite3.sqlite_version,
        "database": "SQLite :memory:; one score policy loaded at a time",
        "source_row_id": "zero-based source CSV row position; NOT a user identifier",
        "table_origin": "normalized derived tables from public experimental test predictions; NOT real event logs",
        "outcome": manifest["primary_outcome"],
        "assignment_probability": e,
        "assignment_probability_source": manifest.get("probability_source"),
        "seed": seed,
        "n_eligible": int(tables["cvr"].n.iloc[0]),
        "primary_budget": budget,
        "frozen_selected_policy": selection["selected_policy"],
        "selection_unchanged": True,
        "tie_break": "independent default_rng(seed).random(n), score DESC then tie priority ASC then input position ASC",
        "metric": "HT incremental responses per 100k eligible rows; analytic random expectation at k/n",
        "maximum_absolute_sql_python_difference": float(
            tables["policy_values"].sql_minus_python.abs().max()
        ),
        "reconciliation_absolute_tolerance": 1e-7,
        "limitations": [
            "1:1 key integrity establishes row alignment, not user-level independence",
            "descriptive test score bins cannot be used to reselect or recalibrate this frozen policy",
            "no simulated timestamps, user IDs, costs, revenue, or production logs",
            "point estimates only; uncertainty is reported by the original paired-bootstrap policy evaluation",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "queries.sql").write_text(sql_queries(), encoding="utf-8")
    write_json(provenance, output / "provenance.json")
    return provenance

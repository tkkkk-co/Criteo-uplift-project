-- SQLite evidence: normalized derived tables; no synthetic user IDs.
-- Bind parameters: policy (name), e (frozen assignment probability),
-- n (all eligible test rows), k=floor(budget*n+1e-12), n_bins.
-- Scores are loaded one policy at a time with executemany.
-- tie_priority = NumPy default_rng(seed).random(n), aligned to input order.
-- NTILE bin 1 contains the highest scores. NULL denotes a missing arm.

-- schema
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

-- assignment/outcome 1:1 join
SELECT
    (SELECT COUNT(*) FROM assignment) AS assignment_rows,
    (SELECT COUNT(*) FROM outcome) AS outcome_rows,
    (SELECT COUNT(*) FROM assignment a JOIN outcome o USING (row_id)) AS joined_rows,
    (SELECT COUNT(*) FROM assignment a LEFT JOIN outcome o USING (row_id)
        WHERE o.row_id IS NULL) AS missing_outcomes,
    (SELECT COUNT(*) FROM outcome o LEFT JOIN assignment a USING (row_id)
        WHERE a.row_id IS NULL) AS orphan_outcomes;

-- policy-score completeness
SELECT
    COUNT(*) AS score_rows,
    COUNT(DISTINCT s.row_id) AS distinct_score_rows,
    (SELECT COUNT(*) FROM assignment a JOIN outcome o USING (row_id)
        JOIN policy_scores p USING (row_id) WHERE p.policy = :policy) AS joined_rows,
    (SELECT COUNT(*) FROM assignment a LEFT JOIN policy_scores p
        ON a.row_id = p.row_id AND p.policy = :policy
        WHERE p.row_id IS NULL) AS missing_scores
FROM policy_scores s WHERE s.policy = :policy;

-- conditional aggregation
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

-- window-function bins
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

-- top-budget HT policy value
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

SELECT COUNT(*) AS n_selected,
    COALESCE(SUM(treatment * response / :e
        - (1 - treatment) * response / (1.0 - :e)), 0.0)
        / :n * 100000.0 AS incremental_per_100k
FROM ranked WHERE rank <= :k;

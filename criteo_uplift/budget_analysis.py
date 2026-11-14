"""Re-evaluate frozen scores at the budget cutoffs used in the final report."""

import numpy as np
import pandas as pd

from .metrics import evaluate_ranking


BUDGETS = (0., .1, .2, .3, .5, 1.)
BOOTSTRAP_REPEATS = 2_000


def evaluate_budget_margins(path, selection, test_metrics, *, n_bootstrap=BOOTSTRAP_REPEATS):
    selected = selection["selected_policy"]
    uplift = selection["selected_uplift_model"]
    with np.load(path) as saved:
        result = evaluate_ranking(
            saved["y"], saved["t"], {name: saved[name] for name in dict.fromkeys((selected, uplift))},
            selection["assignment_probability"], budgets=BUDGETS, primary_budget=selection["primary_budget"],
            n_bootstrap=n_bootstrap, seed=selection["seed"], include_curves=False, include_random=True,
            return_bootstrap_draws=True,
        )
    point = result["metrics"].set_index(["policy", "budget"])
    columns = ["gain_ht", "gain_plugin"]
    saved_point = test_metrics.set_index(["policy", "budget"])[columns]
    shared = point.index.intersection(saved_point.index)
    np.testing.assert_allclose(point.loc[shared, columns], saved_point.loc[shared], rtol=1e-9, atol=1e-12)

    frozen = point.loc[selected].loc[list(BUDGETS)]
    boot = result["bootstrap"]
    differences = np.diff(boot["draws"][selected].loc[:, list(BUDGETS)].to_numpy(), axis=1)
    low, high = np.quantile(differences, [.025, .975], axis=0)
    margins = pd.DataFrame({
        "start": BUDGETS[:-1], "end": BUDGETS[1:],
        "n_added": np.diff(frozen.n_selected),
        "gain_ht": np.diff(frozen.gain_ht), "gain_plugin": np.diff(frozen.gain_plugin),
        "ci_low": low, "ci_high": high,
    })
    return margins, boot["summary"], boot["pairwise"]

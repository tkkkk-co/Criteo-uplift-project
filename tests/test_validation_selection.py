"""Verify point selection does not request Validation resampling."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from criteo_uplift.config import POLICY_NAMES
from criteo_uplift.metrics import evaluate_ranking
from criteo_uplift.training_validation import fit_and_select_models


class ValidationSelectionTests(TestCase):
    def test_point_selection_preserves_test_bootstrap_setting(self):
        rng = np.random.default_rng(8)
        n = 120
        frame = pd.DataFrame({
            'row_id': np.arange(n), 'treatment': np.tile([0, 1], n // 2),
            'conversion': np.tile([0, 1, 0, 0, 1, 0], n // 6),
        })
        fitted = {}
        for name in POLICY_NAMES:
            entry = {'model_bundle': {'parameters': {}}, 'fit_seconds': 0.0}
            entry['score' if name.startswith('response_') else 'cate'] = rng.random(n)
            fitted[name] = entry
        with TemporaryDirectory() as directory:
            out = Path(directory)
            with patch('criteo_uplift.training_validation._fit_candidates', return_value=fitted),                  patch('criteo_uplift.training_validation.evaluate_ranking', wraps=evaluate_ranking) as evaluate:
                result = fit_and_select_models(
                    frame, frame, outcome='conversion', seed=42, trees=3,
                    primary_budget=.2, assignment_probability=.5, out=out,
                    n_bootstrap=2000, progress=lambda _: None,
                )
            self.assertEqual(evaluate.call_args.kwargs['n_bootstrap'], 0)
            metrics = pd.read_csv(out / 'validation_metrics.csv')
            expected = metrics.loc[metrics.policy.ne('random')].sort_values(
                'gain_ht', ascending=False, kind='stable').policy.iloc[0]
            selection = json.loads((out / 'selection.json').read_text())
            self.assertEqual(selection['selected_policy'], expected)
            self.assertEqual(selection['selection_statistic'], 'point_estimate')
            self.assertEqual(selection['validation_n_bootstrap'], 0)
            self.assertEqual(selection['n_bootstrap'], 2000)
            self.assertEqual(json.loads(json.dumps(result['selection'])), selection)
            self.assertFalse((out / 'validation_bootstrap.csv').exists())
            self.assertFalse((out / 'validation_pairwise.csv').exists())
            self.assertFalse((out / 'validation_selection_frequency.csv').exists())

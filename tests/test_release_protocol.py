"""Release acceptance on synthetic inputs only, never real Test selection."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from criteo_uplift.config import FEATURES, POLICY_NAMES, UPLIFT_NAMES
from criteo_uplift.experiment_protocol import freeze_protocol, load_protocol
from criteo_uplift.preparation import split_sample
from criteo_uplift.models import response_predict, t_learner
from criteo_uplift.training_validation import _fit_candidates, policy_scores
from criteo_uplift.metrics import evaluate_ranking
from criteo_uplift.raw_scan import scan_raw_data


class ReleaseTests(unittest.TestCase):
    def frame(self, n=1000):
        rng = np.random.default_rng(15)
        frame = pd.DataFrame(rng.normal(size=(n, 12)), columns=FEATURES)
        frame['treatment'] = np.tile([0, 1], n // 2)
        frame['conversion'] = rng.binomial(1, .25, n)
        frame['visit'] = frame.conversion
        frame['exposure'] = frame.treatment
        return frame

    def test_frozen_holdout_excludes_history_and_development(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw.csv'
            frame = self.frame()
            frame.to_csv(raw, index=False)
            historical = root / 'history/old/data_preparation'
            historical.mkdir(parents=True)
            pd.DataFrame({'row_id': np.arange(100)}).to_csv(historical / 'test_row_ids.csv', index=False)
            protocol = root / 'protocol'
            freeze_protocol(raw, protocol, history=root / 'history')
            meta, test_ids = load_protocol(protocol)
            other_protocol = root / 'protocol_from_single_history'
            freeze_protocol(raw, other_protocol, history=historical.parent)
            np.testing.assert_array_equal(load_protocol(other_protocol)[1], test_ids)
            from run_pipeline import parse_args
            cli = ['--stage', 'full', '--sample-size', '0', '--protocol', str(protocol)]
            self.assertTrue(parse_args(cli).skip_stability)
            self.assertFalse(parse_args([*cli, '--run-stability']).skip_stability)
            self.assertEqual(len(test_ids), 200)
            self.assertFalse(np.isin(test_ids, np.arange(100)).any())
            sampled = scan_raw_data(raw, sample_size=500, per_arm=10, excluded_row_ids=test_ids, progress=lambda _: None)
            self.assertEqual(len(sampled['sample']), 500)
            self.assertFalse(np.isin(sampled['sample'].row_id, test_ids).any())
            self.assertEqual(sampled['audit']['n_rows'], 1000)
            frame.insert(0, 'row_id', np.arange(len(frame)))
            parts, _, _ = split_sample(frame, frozen_test_ids=test_ids)
            self.assertEqual([len(parts[k]) for k in ('train', 'validation', 'test')], [600, 200, 200])
            np.testing.assert_array_equal(np.sort(parts['test'].row_id), test_ids)
            with self.assertRaises(FileExistsError):
                freeze_protocol(raw, protocol, history=root / 'history')
            np.save(protocol / 'test_row_ids.npy', test_ids[::-1])
            with self.assertRaises(ValueError):
                load_protocol(protocol)

    def test_eight_scores_and_treated_only_response(self):
        frame = self.frame(600)
        train, val = frame.iloc[:400].copy(), frame.iloc[400:].copy()
        results = _fit_candidates(train, val, 'conversion', 42, 3, lambda _: None)
        self.assertEqual(tuple(results), POLICY_NAMES)
        self.assertEqual(len(UPLIFT_NAMES), 6)
        scores = policy_scores(results)
        for model in ('logistic', 'xgboost'):
            response = results[f'response_{model}']
            t_result = results[f't_{model}']
            self.assertIs(response['model_bundle']['mu1_fitted_model'], t_result['model_bundle']['mu1_fitted_model'])
            np.testing.assert_array_equal(response_predict(val[FEATURES], response['model_bundle']), t_result['mu1'])
            self.assertNotIn('mu0_fitted_model', response['model_bundle'])
        # Alter only controls: independently fitted treated predictions must be invariant.
        changed = train.copy()
        changed.loc[changed.treatment == 0, FEATURES] = 999.
        changed.loc[changed.treatment == 0, 'conversion'] = 1
        treated_only = t_learner(val[FEATURES], X_train=changed[FEATURES], t=changed.treatment, y=changed.conversion)
        np.testing.assert_allclose(treated_only['mu1'], results['t_logistic']['mu1'])
        evaluated = evaluate_ranking(val.conversion, val.treatment, scores, .5, n_bootstrap=8, include_random=True)
        self.assertEqual(set(evaluated['metrics'].policy), {*POLICY_NAMES, 'random'})
        random = evaluated['metrics'].query('policy == "random" and budget == 0.2').iloc[0]
        self.assertAlmostEqual(random.gain_ht, .2 * (2 * val.treatment * val.conversion - 2 * (1 - val.treatment) * val.conversion).mean())
        self.assertNotIn('random', evaluated['bootstrap']['selection_frequency'].policy.values)
        for name in POLICY_NAMES:
            self.assertTrue(np.isfinite(scores[name]).all())

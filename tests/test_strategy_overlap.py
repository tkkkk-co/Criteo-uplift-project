import unittest
import numpy as np
from scipy.stats import rankdata
from criteo_uplift.strategy_overlap import score_overlap
from criteo_uplift.metrics import evaluate_ranking


class StrategyOverlapTests(unittest.TestCase):
    def test_known_intersection_and_raw_rank_correlation(self):
        a = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1.])
        b = np.array([9, 8, 10, 7, 6, 5, 4, 3, 2, 1.])
        result = score_overlap(np.arange(100, 110), a, b)
        self.assertEqual(result['k'], 2)
        self.assertEqual(result['common'], 1)
        self.assertEqual(result['only_a'], 1)
        self.assertEqual(result['only_b'], 1)
        self.assertEqual(result['overlap'], .5)
        self.assertAlmostEqual(result['jaccard'], 1 / 3)
        self.assertAlmostEqual(result['spearman'], np.corrcoef(rankdata(a), rankdata(b))[0, 1])

    def test_ties_match_metrics_selection_and_shared_priorities(self):
        a = np.array([2., 2., 2., 2., 2., 2., 1., 1., 1., 1., 1.])
        b = a[::-1].copy()
        ids = np.arange(11)
        seed = 42
        priority = np.random.default_rng(seed).random(len(a))
        chosen = [np.lexsort((priority, -s))[:2] for s in (a, b)]
        r = score_overlap(ids, a, b, seed=seed)
        self.assertEqual(r['common'], len(set(chosen[0]) & set(chosen[1])))
        # Recover each selected position independently from the public HT evaluator.
        t = np.array([0, 1] * 5 + [0])
        for j, score in enumerate((a, b)):
            recovered = []
            for i in ids:
                y = np.eye(len(ids))[i]
                m = evaluate_ranking(y, t, {'model': score}, .5, budgets=[.2], n_bootstrap=0, seed=seed)['metrics']
                if m.gain_ht.iloc[0] != 0:
                    recovered.append(i)
            self.assertEqual(set(recovered), set(chosen[j]))
        expected = np.corrcoef(rankdata(a), rankdata(b))[0, 1]
        self.assertAlmostEqual(r['spearman'], expected)

    def test_identity_disjoint_and_constant_scores(self):
        ids = np.arange(10)
        self.assertEqual(score_overlap(ids, ids, ids)['overlap'], 1)
        self.assertEqual(score_overlap(ids, ids, -ids)['common'], 0)
        self.assertEqual(score_overlap(ids, ids, -ids, budget=1)['jaccard'], 1)
        constant = score_overlap(ids, np.ones(10), np.ones(10))
        self.assertEqual(constant['common'], 2)
        self.assertIsNone(constant['spearman'])

    def test_invalid_rows_scores_and_budget(self):
        with self.assertRaises(ValueError): score_overlap([1, 1], [0, 1], [1, 0], budget=1)
        with self.assertRaises(ValueError): score_overlap([1, 2], [np.nan, 1], [1, 0], budget=1)
        with self.assertRaises(ValueError): score_overlap([1, 2], [0, 1], [1, 0], budget=.2)
        with self.assertRaises(ValueError): score_overlap([1, 2], [0, 1], [1, 0], budget=2)

"""Summarize repeated fits on the same frozen Test, without selecting on Test."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES
from .pipeline_io import write_result


def load_test_features(source_run, row_ids):
    manifest = json.loads((source_run / 'manifest.json').read_text())
    raw_n = json.loads((source_run / 'data_preparation/audit.json').read_text())['n_rows']
    wanted = np.zeros(raw_n, dtype=bool)
    wanted[row_ids] = True
    parts, offset = [], 0
    for chunk in pd.read_csv(manifest['arguments']['data'], usecols=FEATURES, dtype='float32', chunksize=250000):
        ids = np.arange(offset, offset+len(chunk))
        part = chunk.loc[wanted[ids]].copy()
        part.insert(0, 'row_id', ids[wanted[ids]])
        parts.append(part)
        offset += len(chunk)
    if offset != raw_n:
        raise ValueError('Raw row count differs from the saved audit.')
    return pd.concat(parts, ignore_index=True).set_index('row_id').loc[row_ids]


def summarize(source_run, experiment):
    manifest = json.loads((source_run / 'manifest.json').read_text())
    protocol = json.loads((experiment / 'protocol.json').read_text())
    seeds = protocol['seeds']
    labels = [*seeds, manifest['seed']]
    policies = ['response_xgboost', 's_xgboost']
    metrics = pd.read_csv(experiment / 'test_metrics.csv')
    original = pd.read_csv(source_run / 'evaluation/test_metrics.csv').assign(seed=manifest['seed'])
    combined = pd.concat([metrics, original], ignore_index=True)
    primary = combined.loc[combined.budget.eq(.2)].copy()
    best = primary.loc[primary.policy.ne('random')].groupby('seed').gain_ht_per_100k.max()
    primary['gap_to_observed_best'] = primary.seed.map(best) - primary.gain_ht_per_100k
    primary['rank'] = primary.groupby('seed').gain_ht_per_100k.rank(method='min', ascending=False)
    write_result(primary, experiment / 'test_comparison.csv')
    validation = pd.read_csv(experiment / 'longmetrics.csv')
    best_val = validation.loc[validation.policy.ne('random')].groupby('seed').gain_ht_per_100k.max()
    validation['gap_to_observed_best'] = validation.seed.map(best_val) - validation.gain_ht_per_100k
    write_result(validation, experiment / 'validation_comparison.csv')
    selections = pd.read_csv(experiment / 'selections.csv')
    audits = pd.read_csv(experiment / 'split_audit.csv').set_index('seed')
    for row in selections.itertuples():
        write_result({
            'seed': int(row.seed), 'selected_policy': row.winner, 'selected_uplift_model': row.best_uplift,
            'selection_statistic': 'gain_ht_point_estimate', 'primary_budget': .2,
            'validation_bootstrap': 0, 'test_used_for_selection': False,
            'role': 'development sensitivity choice; original formal selection unchanged',
            'assignment_probability': float(audits.loc[row.seed, 'assignment_probability']),
        }, experiment / f'seed{row.seed}/selection.json')
    chosen = primary.merge(selections[['seed', 'winner']], on='seed')
    chosen = chosen.loc[chosen.policy.eq(chosen.winner)]
    write_result(chosen, experiment / 'selected_strategy_test.csv')

    coverage = []
    for (seed, policy), group in combined.loc[combined.policy.isin(policies)].groupby(['seed', 'policy']):
        group = group.set_index('budget')
        for method, column in [('HT', 'gain_ht'), ('Uplift', 'gain_plugin')]:
            g10, g20, g30, full = group.loc[[.1, .2, .3, 1.], column]
            coverage.append({
                'seed': seed, 'policy': policy, 'metric': method,
                'gain10_per100k': g10 * 1e5, 'gain20_per100k': g20 * 1e5,
                'gain30_per100k': g30 * 1e5, 'retention10_vs20': g10 / g20,
                'retention20_vs100': g20 / full,
                'margin10_20_per100k': (g20-g10) * 1e5,
                'margin20_30_per100k': (g30-g20) * 1e5,
                'margin10_20_per1000_slots': (g20-g10) * 10000,
            })
    write_result(pd.DataFrame(coverage), experiment / 'coverage_sensitivity.csv')

    reference = np.load(source_run / 'evaluation/predictions.npz')
    row_ids, t, y = reference['row_id'], reference['t'], reference['y']
    reference.close()
    n = len(row_ids)
    k = int(np.floor(n * .2))
    memberships = {}
    gains = {}
    e_values = pd.read_csv(experiment / 'split_audit.csv').set_index('seed').assignment_probability.to_dict()
    e_values[manifest['seed']] = manifest['assignment_probability']
    for seed in labels:
        path = source_run / 'evaluation/predictions.npz' if seed == manifest['seed'] else experiment / f'seed{seed}/test_predictions.npz'
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['row_id'], row_ids)
            np.testing.assert_array_equal(saved['t'], t)
            np.testing.assert_array_equal(saved['y'], y)
            priority = np.random.default_rng(seed).random(n)
            for policy in policies:
                order = np.lexsort((priority, -saved[policy]))
                mask = np.zeros(n, dtype=bool)
                mask[order[:k]] = True
                memberships[seed, policy] = mask
                e = e_values[seed]
                z = t * y / e - (1-t) * y / (1-e)
                gains[seed, policy] = float(z[mask].sum()/n*1e5)
                expected = primary.loc[primary.seed.eq(seed) & primary.policy.eq(policy), 'gain_ht_per_100k'].iloc[0]
                np.testing.assert_allclose(gains[seed, policy], expected, rtol=1e-8, atol=1e-8)
    overlap_rows, group_rows, profile_rows = [], [], []
    masks_by_policy = {}
    e = manifest['assignment_probability']
    z = t * y / e - (1-t) * y / (1-e)
    for policy in policies:
        for a, b in combinations(labels, 2):
            common = int((memberships[a, policy] & memberships[b, policy]).sum())
            overlap_rows.append({'policy': policy, 'seed_a': a, 'seed_b': b,
                                 'common_n': common, 'overlap': common/k})
        counts = sum((memberships[seed, policy].astype(np.int8) for seed in labels))
        masks = {'all_fits_selected': counts == len(labels), 'some_fits_selected': (counts > 0) & (counts < len(labels)),
                 'never_selected': counts == 0}
        masks_by_policy[policy] = masks
        for name, mask in masks.items():
            size = int(mask.sum())
            group_rows.append({'policy': policy, 'group': name, 'n': size, 'population_share': size/n,
                               'fraction_of_budget': size/k,
                               'treatment_n': int(t[mask].sum()), 'control_n': int((1-t[mask]).sum()),
                               'treatment_events': int((t[mask]*y[mask]).sum()),
                               'control_events': int(((1-t[mask])*y[mask]).sum()),
                               'ht_contribution_per100k_population': float(z[mask].sum()/n*1e5),
                               'ht_mean_per100k_group': float(z[mask].mean()*1e5) if size else None})
    write_result(pd.DataFrame(overlap_rows), experiment / 'selection_overlap.csv')
    write_result(pd.DataFrame(group_rows), experiment / 'stable_population_groups.csv')

    # Descriptive anonymous-feature profiles; ranks are based on feature separation only.
    features = load_test_features(source_run, row_ids)
    for policy, masks in masks_by_policy.items():
        stable = masks['all_fits_selected']
        for name in FEATURES:
            x = features[name].to_numpy(dtype=np.float64)
            mean, sd = float(x.mean()), float(x.std(ddof=1))
            threshold = float(np.quantile(x, .75))
            stable_mean = float(x[stable].mean())
            profile_rows.append({
                'policy': policy, 'feature': name, 'population_mean': mean, 'stable_mean': stable_mean,
                'standardized_mean_shift': (stable_mean-mean)/sd if sd else 0.,
                'population_q75': threshold, 'population_above_q75': float((x>threshold).mean()),
                'stable_above_q75': float((x[stable]>threshold).mean()),
            })
    write_result(pd.DataFrame(profile_rows), experiment / 'anonymous_feature_profiles.csv')
    report = json.loads((experiment / 'report.json').read_text())
    report.update(analysis_complete=True, common_test_n=n, reference_seed=manifest['seed'],
                  profile_interpretation='Exploratory profiles on anonymous features, not demographic attributes or feature causal effects.',
                  comparisons_interpretation='Three overlapping development fits plus original formal fit; common Test revisited for descriptive stability only.',
                  bootstrap_used=False)
    write_result(report, experiment / 'report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source_run, args.experiment)

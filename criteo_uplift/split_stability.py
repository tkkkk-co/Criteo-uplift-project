"""Re-fit current candidates on repeated development splits, excluding frozen Test."""

import argparse
import json
import time
from pathlib import Path

from .config import configure_runtime

configure_runtime()

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, POLICY_NAMES, UPLIFT_NAMES, log
from .metrics import evaluate_ranking
from .pipeline_io import write_result
from .stability import _counts, _validate_seeds
from .training_validation import _fit_candidates, _model_diagnostics, policy_scores
from .models import s_learner, t_learner, x_learner


def load_development(source_run):
    manifest = json.loads((source_run / 'manifest.json').read_text())
    audit = json.loads((source_run / 'data_preparation/audit.json').read_text())
    test_ids = pd.read_csv(source_run / 'data_preparation/test_row_ids.csv').row_id.to_numpy()
    n = audit['n_rows']
    if len(np.unique(test_ids)) != len(test_ids) or np.any((test_ids < 0) | (test_ids >= n)):
        raise ValueError('Frozen Test row IDs must be unique and within the raw data.')
    excluded = np.zeros(n, dtype=bool)
    excluded[test_ids] = True
    columns = [*FEATURES, 'treatment', 'conversion']
    dtypes = {**dict.fromkeys(FEATURES, 'float32'), 'treatment': 'int8', 'conversion': 'int8'}
    source = Path(manifest['arguments']['data'])
    initial_stat = source.stat()
    chunks, test_chunks, offset = [], [], 0
    for chunk in pd.read_csv(source, usecols=columns, dtype=dtypes, chunksize=250_000):
        ids = np.arange(offset, offset + len(chunk), dtype=np.int64)
        if offset + len(chunk) > n:
            raise ValueError('Raw data exceeds the saved audit size.')
        chunk.insert(0, 'row_id', ids)
        chunks.append(chunk.loc[~excluded[ids]])
        test_chunks.append(chunk.loc[excluded[ids]])
        offset += len(chunk)
    final_stat = source.stat()
    if offset != n or (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError('Raw data count or file changed while loading.')
    development = pd.concat(chunks, ignore_index=True)
    if len(development) != n - len(test_ids) or excluded[development.row_id].any():
        raise AssertionError('Development/Test exclusion failed.')
    return development, pd.concat(test_chunks, ignore_index=True), manifest


def run_split_stability(source_run, output, *, seeds=(17, 29, 83)):
    seeds = _validate_seeds(seeds)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log('Loading full development data, excluding frozen Test')
    development, test, manifest = load_development(source_run)
    excluded_n = len(test)
    log(f'Development={len(development):,}; excluded Test={excluded_n:,}')
    trees = manifest['arguments']['trees']
    budget = manifest['selection']['primary_budget']
    protocol = {
        'source_run': source_run.name, 'seeds': list(seeds), 'training_fraction': 1.0,
        'split': '75/25 treatment × conversion stratified development split',
        'development_n': len(development), 'excluded_test_n': excluded_n,
        'candidates': list(POLICY_NAMES), 'trees': trees, 'budget': budget,
        'validation_bootstrap': 0, 'model_seed': 'same as split seed',
        'interpretation': 'Combined split and training randomness; overlapping validation sets.',
        'selection': 'Validation HT Gain point estimate; original deployment selection is unchanged.',
        'test': 'Same frozen Test rows for all seeds; descriptive follow-up, no Test-based selection.',
        'test_bootstrap': 0, 'model_artifacts': 'All eight candidate bundles saved per seed.',
    }
    write_result(protocol, output / 'protocol.json')
    positions = np.arange(len(development))
    strata = 2 * development.treatment.to_numpy() + development.conversion.to_numpy()
    all_metrics, all_test_metrics, all_audits, selections = [], [], [], []
    for seed in seeds:
        seed_out = output / f'seed{seed}'
        seed_out.mkdir()
        train_ids, val_ids = train_test_split(positions, test_size=.25, stratify=strata, random_state=seed)
        if np.intersect1d(train_ids, val_ids).size or len(train_ids) + len(val_ids) != len(development):
            raise AssertionError('Development partition is not disjoint and complete.')
        train, validation = development.iloc[train_ids], development.iloc[val_ids]
        e = float(train.treatment.mean())
        counts = {'seed': seed, **_counts(train, 'conversion', 'train'),
                  **_counts(validation, 'conversion', 'validation'), 'row_overlap': 0,
                  'train_test_overlap': 0, 'validation_test_overlap': 0, 'assignment_probability': e}
        all_audits.append(counts)
        write_result(pd.DataFrame(all_audits), output / 'split_audit.csv')
        log(f'seed={seed}: training {len(train):,}, validation {len(validation):,}')
        fitted = _fit_candidates(train, validation, 'conversion', seed, trees, progress=log)
        scores = policy_scores(fitted)
        warnings = {name: result.get('warnings', []) for name, result in fitted.items()}
        np.savez_compressed(seed_out / 'validation_predictions.npz',
                            row_id=validation.row_id.to_numpy(), t=validation.treatment.to_numpy(),
                            y=validation.conversion.to_numpy(), **scores)
        write_result(warnings, seed_out / 'training_warnings.json')
        np.savez_compressed(seed_out / 'split_row_ids.npz',
                            train=development.iloc[train_ids].row_id.to_numpy(),
                            validation=validation.row_id.to_numpy())
        model_records = {}
        test_scores = {}
        log(f'seed={seed}: saving models and scoring common Test')
        for name, result in fitted.items():
            joblib.dump(result['model_bundle'], seed_out / f'{name}.joblib')
            model_records[name] = {
                'parameters': result['model_bundle']['parameters'],
                'components': _model_diagnostics(result['model_bundle']),
                'fit_seconds': result['fit_seconds'], 'warnings': result.get('warnings', []),
            }
        for learner in ('s', 't', 'x'):
            for model in ('logistic', 'xgboost'):
                name = f'{learner}_{model}'
                predict = {'s': s_learner, 't': t_learner, 'x': x_learner}[learner]
                result = predict(test[FEATURES], model_bundle=fitted[name]['model_bundle'])
                test_scores[name] = result['cate']
                if learner == 't':
                    test_scores[f'response_{model}'] = result['mu1']
        write_result(model_records, seed_out / 'model_records.json')
        np.savez_compressed(seed_out / 'test_predictions.npz',
                            row_id=test.row_id.to_numpy(), t=test.treatment.to_numpy(),
                            y=test.conversion.to_numpy(), **test_scores)
        test_evaluation = evaluate_ranking(
            test.conversion, test.treatment, {name: test_scores[name] for name in POLICY_NAMES},
            e, budgets=[.02, .05, .1, .2, .3, .5, 1.],
            primary_metric='gain_ht', primary_budget=budget, n_bootstrap=0,
            seed=seed, include_curves=False, include_random=True,
        )
        test_metrics = test_evaluation['metrics'].assign(seed=seed)
        all_test_metrics.append(test_metrics)
        write_result(test_metrics, seed_out / 'test_metrics.csv')
        write_result(pd.concat(all_test_metrics, ignore_index=True), output / 'test_metrics.csv')
        del fitted, train, test_scores, test_evaluation, test_metrics
        log(f'seed={seed}: calculating Validation point metrics')
        evaluated = evaluate_ranking(
            validation.conversion, validation.treatment, scores, e, budgets=[budget],
            primary_metric='gain_ht', primary_budget=budget, n_bootstrap=0,
            seed=seed, include_curves=False, include_random=True,
        )
        metrics = evaluated['metrics'].assign(seed=seed)
        all_metrics.append(metrics)
        ordered = metrics.loc[metrics.policy.ne('random')].sort_values('gain_ht', ascending=False, kind='stable')
        winner = ordered.policy.iloc[0]
        uplift = ordered.loc[ordered.policy.isin(UPLIFT_NAMES), 'policy'].iloc[0]
        selections.append({'seed': seed, 'winner': winner, 'best_uplift': uplift})
        write_result(metrics, seed_out / 'metrics.csv')
        write_result(pd.concat(all_metrics, ignore_index=True), output / 'longmetrics.csv')
        write_result(pd.DataFrame(selections), output / 'selections.csv')
        log(f'seed={seed}: winner={winner}, best uplift={uplift}')
        del validation, scores, evaluated, metrics
    combined = pd.concat(all_metrics, ignore_index=True)
    summary = combined.groupby('policy', sort=False)[['gain_ht_per_100k']].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    write_result(summary.reset_index(), output / 'aggregate.csv')
    write_result({'status': 'complete', 'experiments': len(seeds), 'elapsed_seconds': time.perf_counter() - started,
                  'development_n': len(development), 'test_n_excluded': excluded_n,
                  'all_test_row_overlaps_zero': True, 'models_refitted': True,
                  'original_selection_changed': False, 'common_test_scored': True, 'models_saved': True}, output / 'report.json')
    log(f'Completed: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 83])
    args = parser.parse_args()
    run_split_stability(args.source_run, args.output, seeds=args.seeds)

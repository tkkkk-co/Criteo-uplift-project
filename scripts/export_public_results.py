"""Export an allowlisted, aggregate-only snapshot; no raw data/model arrays."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from criteo_uplift.runtime import configure_runtime

configure_runtime()
import argparse
import base64
import hashlib
import json
import shutil
import pandas as pd
from criteo_uplift.data import FEATURES, load_sample, safe_output

BASE = 'conversion_1000000_seed42'
BASE_FILES = (
    'audit.json',
    'manifest.json',
    'selection.json',
    'splits.csv',
    'ate_full.csv',
    'ate_test.json',
    'balance.csv',
    'c2st.csv',
    'c2st_null.csv',
    'c2st_details.json',
    'srm.json',
    'learnability.csv',
    'feature_overlap.json',
    'validation_policies.csv',
    'validation_ranking.csv',
    'test_policies.csv',
    'test_ranking.csv',
    'curves.csv',
    'report.html',
    'calibration_s_logistic.csv',
    'calibration_t_logistic.csv',
    'calibration_t_xgboost.csv',
    'calibration_x_xgboost.csv',
)
EXPORTS = {
    BASE: BASE_FILES,
    'stability_v1': (
        'protocol.json',
        'report.json',
        'aggregate.csv',
        'longmetrics.csv',
        'split_audit.csv',
        'selection_frequency.csv',
    ),
    'metrics_v2': (
        'protocol.json',
        'report.json',
        'ranking.csv',
        'ranking_intervals.csv',
        'curves_display_only.csv',
    ),
    'sql_audit_v1': (
        'queries.sql',
        'provenance.json',
        'cvr.csv',
        'join_integrity.csv',
        'score_bins.csv',
        'policy_values.csv',
    ),
    'planning_v1': (
        'assumptions.json',
        'hypothetical_ab_plans.csv',
        'hypothetical_cost_scenarios.csv',
    ),
}


def sanitize_text(value):
    return value.replace(str(ROOT), '[PROJECT_ROOT]').replace(
        '/opt/miniconda3/envs/causalml/bin/python', 'python'
    )


def export(output):
    output = safe_output(output)
    if output.exists():
        raise FileExistsError(f'Use a new snapshot directory: {output}')
    # Check required sources before creating partial output.
    for directory, names in EXPORTS.items():
        for name in names:
            if not (ROOT / 'artifacts' / directory / name).is_file():
                raise FileNotFoundError(ROOT / 'artifacts' / directory / name)
    output.mkdir(parents=True)
    records = []

    def copy(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original = source.read_bytes()
        if source.suffix in ('.json', '.csv', '.html', '.sql', '.md', '.txt'):
            target.write_text(sanitize_text(original.decode('utf-8')), encoding='utf-8')
        else:
            target.write_bytes(original)
        records.append(
            {
                'source': str(source.relative_to(ROOT)),
                'published': str(relative),
                'source_sha256': hashlib.sha256(original).hexdigest(),
                'published_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )

    for directory, names in EXPORTS.items():
        for name in names:
            copy(ROOT / 'artifacts' / directory / name, Path(directory) / name)
    for source in sorted((ROOT / 'artifacts' / BASE / 'figures').glob('*.png')):
        copy(source, Path(BASE) / 'figures' / source.name)
    for source in sorted((ROOT / 'artifacts/extension_figures').glob('*.png')):
        copy(source, Path('extension_figures') / source.name)
    copy(ROOT / 'artifacts/extension_report.html', Path('extension_report.html'))

    # Only grouped and marginal summaries are exported, never individual rows.
    sample = load_sample(ROOT / 'artifacts/cache/sample_1000000_seed42.npz')
    rows = []
    for treatment, group in sample.groupby('treatment'):
        for label in ['conversion', 'visit', 'exposure']:
            rows.append(
                {
                    'treatment': int(treatment),
                    'label': label,
                    'n': len(group),
                    'positive': int(group[label].sum()),
                    'rate': float(group[label].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(output / BASE / 'sample_group_summary.csv', index=False)
    sample[FEATURES].describe().T.to_csv(
        output / BASE / 'sample_feature_summary.csv', index_label='feature'
    )
    del sample
    # Reuse the actual executed histogram output rather than invent a figure.
    nb = json.loads((ROOT / 'notebooks/01_data_understanding.ipynb').read_text())
    found = False
    for cell in nb['cells']:
        if 'ax.hist(sample[feature]' not in ''.join(cell.get('source', [])):
            continue
        for result in cell.get('outputs', []):
            encoded = result.get('data', {}).get('image/png')
            if encoded:
                if isinstance(encoded, list):
                    encoded = ''.join(encoded)
                (output / BASE / 'figures/sample_features.png').write_bytes(
                    base64.b64decode(encoded)
                )
                found = True
    if not found:
        raise ValueError(
            'Execute notebook 01 before exporting its real feature histogram'
        )
    (output / 'README.md').write_text(
        '# Published aggregate snapshot\n\n'
        'Computed from the historical Criteo experiment. No raw rows, row-ID lists, individual '
        'predictions or fitted models are included. Text paths are redacted. Original source artifacts '
        'stay unchanged. See snapshot_manifest.json for hashes and NOTICE.md for attribution.\n\n'
        'Historical baseline/stability use Qini v1; metrics_v2 uses the explicitly revised '
        'control-only-zero-node convention. Main 20% HT values are unchanged.\n',
        encoding='utf-8',
    )
    generated = [
        'sample_group_summary.csv',
        'sample_feature_summary.csv',
        'figures/sample_features.png',
    ]
    for name in generated:
        p = output / BASE / name
        records.append(
            {
                'source': 'derived from cached sample / executed notebook',
                'published': f'{BASE}/{name}',
                'published_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        'scope': 'aggregate-only public snapshot',
        'selection_changed': False,
        'raw_rows_exported': False,
        'license_for_dataset_derived_outputs': 'CC BY-NC-SA-4.0',
        'files': records,
    }
    (output / 'snapshot_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(f'Exported {len(records)} aggregate evidence files to {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/published')
    export(parser.parse_args().output)

# 从现有实验导出可发布的聚合结果，移除本机路径，不导出逐行数据或模型。

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
from criteo_uplift.data import FEATURES
from criteo_uplift.pipeline_io import load_sample, safe_output

BASE = 'conversion_1000000_seed42'
BASE_FILES = (
    'data_preparation/audit.json',
    'data_preparation/sample_report.json',
    'manifest.json',
    'models/selection.json',
    'data_preparation/split_summary.csv',
    'diagnostics/ate_full.csv',
    'evaluation/ate_test.json',
    'diagnostics/balance.csv',
    'diagnostics/c2st.csv',
    'diagnostics/c2st_null.csv',
    'diagnostics/c2st_details.json',
    'diagnostics/srm.json',
    'diagnostics/learnability.csv',
    'data_preparation/feature_overlap.json',
    'models/validation_policies.csv',
    'models/validation_ranking.csv',
    'evaluation/test_policies.csv',
    'evaluation/test_ranking.csv',
    'evaluation/curves.csv',
    'report.html',
    'evaluation/calibration_s_logistic.csv',
    'evaluation/calibration_t_logistic.csv',
    'evaluation/calibration_t_xgboost.csv',
    'evaluation/calibration_x_xgboost.csv',
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：sanitize_text —— 清理公开文本中的本机路径。
# 输入：
#     value：需要格式化或转换的值。
# 输出：
#     文本：去除本地环境标识后的字符串。
# 说明：清理公开文本中的本机路径。
#-----------------------------------------------------------------------------------------------------------------------
def sanitize_text(value):
    return value.replace(str(ROOT), '[PROJECT_ROOT]').replace(
        '/opt/miniconda3/envs/causalml/bin/python', 'python'
    )


#-----------------------------------------------------------------------------------------------------------------------
# 函数：export —— 导出已有实验的聚合快照。
# 输入：
#     output：本次任务的新输出路径。
# 输出：
#     output：新生成的聚合快照目录。
# 说明：导出已有实验的聚合快照。
#-----------------------------------------------------------------------------------------------------------------------
def export(output):
    output = safe_output(output)
    if output.exists():
        raise FileExistsError(f'Use a new snapshot directory: {output}')
    # 先检查必要的来源文件，再创建输出，避免留下不完整产物。
    for directory, names in EXPORTS.items():
        for name in names:
            base = ROOT / 'artifacts' / directory
            source = base / name
            if not source.is_file():
                raise FileNotFoundError(source)
    output.mkdir(parents=True)
    records = []

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：copy —— 复制并记录一个允许发布的结果文件。
    # 输入：
    #     source：输入来源路径或指纹记录，具体类型由本函数的用途决定。
    #     relative：文件在目标目录中的相对路径。
    # 输出：
    #     返回值：None；保存目标文件并登记来源与指纹。
    # 说明：复制并记录一个允许发布的结果文件。
    #-----------------------------------------------------------------------------------------------------------------------
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
            base = ROOT / 'artifacts' / directory
            source = base / name
            relative = Path(name)
            copy(source, Path(directory) / relative)
    for source in sorted((ROOT / 'artifacts' / BASE / 'figures').glob('*.png')):
        copy(source, Path(BASE) / 'figures' / source.name)
    for source in sorted((ROOT / 'artifacts/extension_figures').glob('*.png')):
        copy(source, Path('extension_figures') / source.name)
    copy(ROOT / 'artifacts/extension_report.html', Path('extension_report.html'))

    # 只导出分组和单变量汇总，不导出逐行记录。
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
    pd.DataFrame(rows).to_csv(output / BASE / 'data_preparation' / 'sample_group_summary.csv', index=False)
    sample[FEATURES].describe().T.to_csv(
        output / BASE / 'data_preparation' / 'sample_feature_summary.csv', index_label='feature'
    )
    del sample
    # 复用 Notebook 实际执行生成的直方图。
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
        'data_preparation/sample_group_summary.csv',
        'data_preparation/sample_feature_summary.csv',
        'figures/sample_features.png',
    ]
    for name in generated:
        p = output / BASE / name
        records.append(
            {
                'source': 'derived from cached sample / executed notebook',
                'published': str(p.relative_to(output)),
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
    parser = argparse.ArgumentParser(description='从现有实验导出可发布的聚合结果，移除本机路径，不导出逐行数据或模型。')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/published')
    export(parser.parse_args().output)

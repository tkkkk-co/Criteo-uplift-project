"""从一个完整实验导出聚合快照；不发布逐行数据、行号、模型或缓存。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import hashlib
import json
from criteo_uplift.pipeline_io import safe_output

BASE_FILES = (
    'manifest.json', 'report.html',
    'data_preparation/audit.json', 'data_preparation/sample_report.json',
    'data_preparation/split_summary.csv', 'data_preparation/feature_overlap.json',
    'diagnostics/balance.csv', 'diagnostics/srm.json', 'diagnostics/c2st.csv',
    'diagnostics/c2st_null.csv', 'diagnostics/c2st_details.json', 'diagnostics/learnability.csv',
    'models/selection.json', 'models/validation_policies.csv', 'models/validation_ranking.csv',
    'evaluation/ate_summary.csv', 'evaluation/test_policies.csv', 'evaluation/test_ranking.csv',
    'evaluation/curves.csv', 'evaluation/ranking_intervals.csv',
    'evaluation/calibration_s_logistic.csv', 'evaluation/calibration_t_logistic.csv',
    'evaluation/calibration_t_xgboost.csv', 'evaluation/calibration_x_xgboost.csv',
    'stability/protocol.json', 'stability/report.json', 'stability/longmetrics.csv',
    'stability/split_audit.csv', 'stability/aggregate.csv', 'stability/selection_frequency.csv',
    'business/assumptions.json', 'business/hypothetical_ab_plans.csv',
    'business/hypothetical_cost_scenarios.csv', 'business/score_policy_status.json',
)


def sanitize_text(value):
    """
    函数名：sanitize_text
    功能：从待发布文本中替换本机项目绝对路径和 Python 解释器路径，避免泄露本地环境信息。
    输入参数：
        - value：需要清理本机路径信息的文本内容。
    输出参数：
        - 完成路径脱敏后的文本字符串。
    """
    return value.replace(str(ROOT), '[PROJECT_ROOT]').replace(sys.executable, 'python')


def export(source, output):
    """
    函数名：export
    功能：从完整实验目录复制明确允许的聚合结果和图像，生成带 SHA256 清单的公开快照。
    输入参数：
        - source：状态为 complete 的实验输出目录。
        - output：新公开快照的目标目录，必须不存在。
    输出参数：
        - 生成的公开快照目录 pathlib.Path。
    """
    source, output = Path(source).resolve(), safe_output(output)
    if output.exists():
        raise FileExistsError(f'Use a new snapshot directory: {output}')
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Only complete experiments can be published')
    paths = [source / name for name in BASE_FILES] + sorted((source / 'figures').glob('*.png'))
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f'Missing or unsafe export source: {path}')
    records = []
    for path in paths:
        relative = Path(source.name) / path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        destination.write_bytes(data if path.suffix == '.png' else sanitize_text(data.decode()).encode())
        records.append({'source': str(path.relative_to(source)), 'published': str(relative),
                        'source_sha256': hashlib.sha256(data).hexdigest(),
                        'published_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
    (output / 'snapshot_manifest.json').write_text(json.dumps({
        'scope': 'aggregate-only public snapshot', 'selection_changed': False,
        'raw_rows_exported': False, 'license_for_dataset_derived_outputs': 'CC BY-NC-SA-4.0',
        'files': records,
    }, ensure_ascii=False, indent=2))
    (output / 'README.md').write_text('# Published aggregate snapshot\n\n'
        f'正式展示来自 `{source.name}`。不含原始行、行号列表、逐行预测或模型。\n'
        '唯一 HTML 为该实验的 report.html；来源及文件哈希见 snapshot_manifest.json。\n')
    print(f'Exported {len(records)} aggregate files to {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/published')
    args = parser.parse_args()
    export(args.source, args.output)

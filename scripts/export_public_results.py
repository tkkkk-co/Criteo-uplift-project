#-------------
# 从一个完整实验导出聚合快照；不发布逐行数据、行号、模型或缓存。
#-------------
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
from criteo_uplift.pipeline_io import safe_output

BASE_FILES = (
    'manifest.json', 'report.html',
    'data_preparation/audit.json', 'data_preparation/sample_report.json',
    'data_preparation/split_summary.csv', 'data_preparation/feature_overlap.json',
    'diagnostics/balance.csv', 'diagnostics/srm.json', 'diagnostics/c2st.csv',
    'diagnostics/overlap_positivity.json', 'diagnostics/overlap_strata.csv',
    'diagnostics/c2st_null.csv', 'diagnostics/c2st_details.json', 'diagnostics/learnability.csv',
    'models/selection.json', 'models/validation_policies.csv', 'models/validation_ranking.csv',
    'evaluation/ate_summary.csv', 'evaluation/test_policies.csv', 'evaluation/test_ranking.csv',
    'evaluation/curves.csv', 'evaluation/ranking_intervals.csv',
    'evaluation/calibration_s_logistic.csv', 'evaluation/calibration_t_logistic.csv',
    'evaluation/calibration_t_xgboost.csv', 'evaluation/calibration_x_xgboost.csv',
    'stability/protocol.json', 'stability/report.json', 'stability/longmetrics.csv',
    'stability/split_audit.csv', 'stability/aggregate.csv', 'stability/selection_frequency.csv',
    'business/assumptions.json', 'business/hypothetical_ab_plans.csv',
    'business/hypothetical_cost_scenarios.csv',
    'figures/sample_features.png', 'figures/c2st_null.png', 'figures/balance.png',
    'figures/ate_full.png', 'figures/policy.png', 'figures/ranking.png',
    'figures/calibration.png', 'figures/stability.png',
)


def export(source, output):
    #-------------
    # 函数名：export
    # 功能：从完整实验目录复制明确允许的聚合结果和图像，生成供 Report 和 Notebook 使用的公开结果。
    # 输入参数：
    #     - source：状态为 complete 的实验输出目录。
    #     - output：新公开快照的目标目录，必须不存在。
    # 输出参数：
    #     - 生成的公开快照目录 pathlib.Path。
    #-------------
    source, output = Path(source).resolve(), safe_output(output)
    if output.exists():
        raise FileExistsError(f'请使用新的公开结果目录：{output}')
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('只能发布已完成的实验')
    paths = [source / name for name in BASE_FILES]
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f'导出来源缺失或路径不安全：{path}')
    for path in paths:
        relative = Path(source.name) / path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        if path.suffix != '.png':
            data = (data.decode().replace(str(ROOT), '[PROJECT_ROOT]')
                    .replace(str(source.parent), '[RESULTS_ROOT]').replace(sys.executable, 'python').encode())
        destination.write_bytes(data)
    (output / 'README.md').write_text('# 公开聚合结果\n\n'
        f'展示来自 `{source.name}`，不含原始行、行号列表、逐行预测、模型或缓存。\n'
        '数据衍生成果遵循 CC BY-NC-SA-4.0；实验参数及来源见 manifest.json。\n')
    print(f'已导出 {len(paths)} 个聚合结果文件：{output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从一个完整实验导出聚合快照；不发布逐行数据、行号、模型或缓存。')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)

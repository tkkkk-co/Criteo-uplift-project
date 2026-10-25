# 使用冻结预测计算扩展排序指标、校准结果与配对区间，保持原模型选择不变。

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from criteo_uplift.runtime import configure_runtime

configure_runtime()
import argparse
import json
import numpy as np
import pandas as pd
from criteo_uplift.pipeline_io import safe_output, write_json, fingerprint
from criteo_uplift.evaluation import (
    uplift_curve,
    ranking_metrics,
    ranking_confidence_intervals,
)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：run —— 对冻结预测计算扩展指标并保存。
# 输入：
#     source：输入来源路径或指纹记录，具体类型由本函数的用途决定。
#     output：本次任务的新输出路径。
#     bootstrap：重抽样次数；支持时可用 0 仅计算点估计。
# 输出：
#     返回值：None；扩展指标和来源记录写入指定目录。
# 说明：对冻结预测计算扩展指标并保存。
#-----------------------------------------------------------------------------------------------------------------------
def run(source, output, bootstrap=200):
    source = Path(source).resolve()
    output = safe_output(output)
    if output == source or source in output.parents:
        raise ValueError(
            'Use an independent output directory outside the frozen source run'
        )
    if output.exists():
        raise FileExistsError(output)
    before = {
        name: fingerprint(source / name)
        for name in ['predictions.npz', 'manifest.json', 'selection.json']
    }
    manifest = json.loads((source / 'manifest.json').read_text())
    selection = json.loads((source / 'selection.json').read_text())
    e, seed = manifest['assignment_probability'], manifest['seed']
    output.mkdir(parents=True)
    protocol = {
        'version': 'metrics_v2',
        'status': 'running',
        'source_files': before,
        'bootstrap': bootstrap,
        'e': e,
        'seed': seed,
        'frozen_selection': selection,
        'purpose': 'Descriptive extension of already-viewed frozen test predictions; no fresh independent confirmation or reselection.',
        'curves': 'Lift=arm rate difference; plug-in gain=q*lift; HT gain=sum(z)/n; Qini=(R1-R0*N1/N0)/n.',
        'integration': 'All defined prefixes plus origin; trapezoids bridge initial undefined region; Qini includes control-only zero nodes.',
        'normalization': 'Population-fraction x axis; y divided by n. No perfect-ranking normalization.',
        'ci': 'Paired within-arm percentile bootstrap; models, e, tie priorities and arm sizes fixed; no multiplicity correction.',
    }
    write_json(protocol, output / 'protocol.json')
    with np.load(source / 'predictions.npz', allow_pickle=False) as predictions:
        y, t = predictions['y'], predictions['t']
        scores = {
            name: predictions[name]
            for name in predictions.files
            if name not in ['y', 't', 'row_id']
        }
    rows, curves = [], []
    for name, score in scores.items():
        rows.append({'policy': name, **ranking_metrics(y, t, score, e, seed)})
        curve = uplift_curve(y, t, score, e, seed)
        # 只压缩可视化数据；前面的面积计算使用每个排序前缀。
        ix = np.unique(
            np.r_[
                np.arange(min(20, len(curve))),
                np.linspace(0, len(curve) - 1, 1001).astype(int),
            ]
        )
        curve = curve.iloc[ix].copy()
        curve.insert(0, 'policy', name)
        curves.append(curve)
    pd.DataFrame(rows).to_csv(output / 'ranking.csv', index=False)
    pd.concat(curves).to_csv(output / 'curves_display_only.csv', index=False)
    ranking_confidence_intervals(y, t, scores, e, bootstrap, seed).to_csv(
        output / 'ranking_intervals.csv', index=False
    )
    after = {name: fingerprint(source / name) for name in before}
    if before != after:
        raise RuntimeError('Source input fingerprint changed during evaluation')
    write_json(
        {**protocol, 'status': 'complete', 'source_unchanged': True, 'n': len(y)},
        output / 'report.json',
    )
    print(f'Completed frozen-score v2 metric extension: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用冻结预测计算扩展排序指标、校准结果与配对区间，保持原模型选择不变。')
    parser.add_argument(
        '--source', type=Path, default=ROOT / 'artifacts/conversion_1000000_seed42'
    )
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/metrics_v2')
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args()
    run(args.source, args.output, args.bootstrap)

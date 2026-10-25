# 为现有冻结实验重新计算扩展评价指标，并写入指定的新目录。
# 保留旧命令的位置参数 source 接口；具体计算仍转交 run_metrics_extension.run。

from run_metrics_extension import run
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='为现有冻结实验重新计算扩展评价指标，并写入指定的新目录。')
    p.add_argument(
        'source',
        nargs='?',
        type=Path,
        default=ROOT / 'artifacts/conversion_1000000_seed42',
    )
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--bootstrap', type=int, default=200)
    a = p.parse_args()
    run(a.source, a.output, a.bootstrap)

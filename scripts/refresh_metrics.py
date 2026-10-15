"""兼容旧入口：指标扩展必须写入独立目录，不覆盖冻结实验。"""

from run_metrics_extension import run
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
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

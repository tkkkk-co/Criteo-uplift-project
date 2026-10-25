# 运行合成数据演示，验证模型训练与策略评价流程。

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from criteo_uplift.runtime import configure_runtime

configure_runtime()
import argparse
from criteo_uplift.demo import run_demo

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='运行合成数据演示，验证模型训练与策略评价流程。')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/demo')
    p.add_argument('--n', type=int, default=6000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--trees', type=int, default=20)
    p.add_argument('--bootstrap', type=int, default=30)
    a = p.parse_args()
    print('SYNTHETIC DEMO — the values below are not Criteo results or online impact.')
    print(run_demo(a.output, a.n, a.seed, a.trees, a.bootstrap).to_string(index=False))

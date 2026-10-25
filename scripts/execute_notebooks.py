# 使用当前 Python 环境逐个执行 Notebook，保存真实输出并在错误时停止。

from pathlib import Path
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='使用当前 Python 环境逐个执行 Notebook，保存真实输出并在错误时停止。')
parser.add_argument(
    '--local-results',
    action='store_true',
    help='Prefer local artifacts; default executes published snapshots',
)
args = parser.parse_args()
if not args.local_results:
    os.environ['CRITEO_RESULTS_MODE'] = 'published'
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts/.mplcache'))
os.environ.setdefault('IPYTHONDIR', str(ROOT / 'artifacts/.ipython'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR', str(ROOT / 'artifacts/.jupyter_runtime'))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import nbformat
from nbclient import NotebookClient

for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}},
    )
    client.km = client.create_kernel_manager()
    # 使用启动当前命令的 Python 解释器执行 Notebook。
    # 只改变内存中的内核启动命令，不修改全局内核配置。
    client.km.kernel_spec.argv = [
        sys.executable,
        '-m',
        'ipykernel_launcher',
        '-f',
        '{connection_file}',
    ]
    client.execute()
    nbformat.write(nb, path)
    n = sum(cell.cell_type == 'code' for cell in nb.cells)
    print(f'{path.name}: {n} code cells executed; no errors', flush=True)

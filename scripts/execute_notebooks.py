"""在现有 Python 环境执行 notebook，保存真实输出并检查错误。"""

from pathlib import Path
import os
import sys
import argparse

parser = argparse.ArgumentParser(description=__doc__)
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
    # Use the interpreter that launched this command, rather than an unrelated
    # global python3 kernel. This only changes the in-memory launch command.
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

# 在导入数值计算库前配置线程和项目运行库，不修改用户的 Conda 环境。

from pathlib import Path
import ctypes
import os
import time

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
DEFAULT_DATA = DATA_ROOT / 'raw/criteo-uplift-v2.1.csv'
FEATURES = [f'f{i}' for i in range(12)]
LABELS = ['treatment', 'conversion', 'visit', 'exposure']
COLUMNS = FEATURES + LABELS
DEFAULT_SAMPLE_SIZE = 1_000_000
DEFAULT_SEED = 42

_OPENMP = None


def configure_runtime():
    global _OPENMP
    root = ROOT
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'artifacts/.mplcache'))
    # 公开项目不随附平台专用的运行库二进制文件。
    # 可通过显式配置指向另行安装且可信的 OpenMP 运行库。
    library = Path(
        os.environ.get(
            'CRITEO_OPENMP_LIBRARY', str(root / '.runtime/openmp/lib/libomp.dylib')
        )
    )
    if library.exists() and _OPENMP is None:
        _OPENMP = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def log(message):
    """输出带时间的进度信息。"""
    print(time.strftime("%H:%M:%S"), message, flush=True)

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
DEFAULT_SAMPLE_SIZE = 5_000_000
DEFAULT_SEED = 42
UPLIFT_NAMES = tuple(f"{learner}_{model}" for learner in ("s", "t", "x") for model in ("logistic", "xgboost"))
RESPONSE_NAMES = ("response_logistic", "response_xgboost")
POLICY_NAMES = (*UPLIFT_NAMES, *RESPONSE_NAMES)

_OPENMP = None


def configure_runtime():
    #-------------
    # 函数名：configure_runtime
    # 功能：设置项目运行所需的线程环境变量，并按需加载 OpenMP 运行库。
    # 输入参数：
    #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
    # 输出参数：
    #     - 无；修改当前 Python 进程的运行环境设置。
    #-------------
    global _OPENMP
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    # 公开项目不随附平台专用的运行库二进制文件。
    # 可通过显式配置指向另行安装且可信的 OpenMP 运行库。
    library = Path(
        os.environ.get(
            'CRITEO_OPENMP_LIBRARY', str(ROOT / '.runtime/openmp/lib/libomp.dylib')
        )
    )
    if library.exists() and _OPENMP is None:
        _OPENMP = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def log(message):
    #-------------
    # 函数名：log
    # 功能：为一条进度消息添加时间戳并立即输出。
    # 输入参数：
    #     - message：需要输出的进度或状态文本。
    # 输出参数：
    #     - 无；将消息写入终端。
    #-------------
    print(time.strftime("%H:%M:%S"), message, flush=True)

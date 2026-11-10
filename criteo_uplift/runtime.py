# 在导入数值计算库前配置线程和项目运行库，不修改用户的 Conda 环境。

from pathlib import Path
import ctypes
import os
import time

_OPENMP = None


#-----------------------------------------------------------------------------------------------------------------------
# 函数：configure_runtime —— 配置数值计算线程和项目运行库。
# 输入：
#     无。
# 输出：
#     返回值：None；更新当前进程环境并按需加载运行库。
# 说明：设置当前进程的线程和绘图缓存；存在明确配置的 OpenMP 库时按需加载。
#-----------------------------------------------------------------------------------------------------------------------
def configure_runtime():
    global _OPENMP
    root = Path(__file__).resolve().parents[1]
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

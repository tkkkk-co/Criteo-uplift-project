"""Project-scoped runtime configuration; no modification to the user's conda env."""

from pathlib import Path
import ctypes
import os

_OPENMP = None


def configure_runtime():
    global _OPENMP
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'artifacts/.mplcache'))
    # Public checkouts do not vendor platform-specific binaries. An explicit
    # override can point to a separately installed trusted OpenMP library.
    library = Path(
        os.environ.get(
            'CRITEO_OPENMP_LIBRARY', str(root / '.runtime/openmp/lib/libomp.dylib')
        )
    )
    if library.exists() and _OPENMP is None:
        _OPENMP = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)

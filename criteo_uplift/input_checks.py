"""公共检查只验证输入，不做转换、插补或删行；调用方决定后续表示。"""
import numpy as np
import pandas as pd


def check_binary_vector(values, name="values", n_rows=None, *, require_both=False):
    """验证非空一维 0/1 标签，可要求长度匹配及两组均存在。"""
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or (n_rows is not None and len(array) != n_rows):
        raise ValueError(f"{name} must be a nonempty one-dimensional vector of matching length")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only finite binary values 0 and 1")
    if require_both and np.unique(array).size != 2:
        raise ValueError(f"{name} requires both treatment arms")


def check_feature_matrix(X, *, allow_nan=False, n_features=None):
    """验证二维数值特征；允许显式开启 NaN，始终拒绝 Inf。"""
    array = np.asarray(X)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("X must be a non-empty two-dimensional feature matrix")
    if array.dtype.kind not in "biuf":
        raise ValueError("X must contain numeric pre-treatment features only")
    if np.isinf(array).any() or (not allow_nan and np.isnan(array).any()):
        raise ValueError("X contains nonfinite features (NaN/Inf，前置特征)")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError(f"X must have {n_features} features")


def check_row_ids(values, name="row_id", n_rows=None, *, require_integer=False):
    """验证来源行号非空、无缺失且唯一；原始位置行号另要求非负整数。"""
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or (n_rows is not None and len(array) != n_rows):
        raise ValueError(f"{name}: row IDs must be nonempty vectors of matching length")
    index = pd.Index(array)
    if index.hasnans or not index.is_unique:
        raise ValueError(f"{name}: row IDs 必须非空且唯一 (non-null and unique)")
    if require_integer and (array.dtype.kind not in "iu" or np.any(array < 0)):
        raise ValueError(f"{name}: row IDs must be nonnegative integer vectors")

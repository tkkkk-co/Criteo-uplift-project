"""公共检查只验证输入，不做转换、插补或删行；调用方决定后续表示。"""
import numpy as np
import pandas as pd


def check_binary_vector(values, name="values", n_rows=None, *, require_both=False):
    """
    函数名：check_binary_vector
    功能：验证输入是一维、非空且只包含 0/1 的二元向量，并按需检查长度和两类是否齐全。
    输入参数：
        - values：待检查的标签或处理分配向量。
        - name：用于错误提示的变量名称。
        - n_rows：可选的预期行数。
        - require_both：是否要求 0 和 1 两类都出现。
    输出参数：
        - 无；校验通过正常返回，失败时抛出 ValueError。
    """
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or (n_rows is not None and len(array) != n_rows):
        raise ValueError(f"{name} must be a nonempty one-dimensional vector of matching length")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only finite binary values 0 and 1")
    if require_both and np.unique(array).size != 2:
        raise ValueError(f"{name} requires both treatment arms")


def check_feature_matrix(X, *, allow_nan=False, n_features=None):
    """
    函数名：check_feature_matrix
    功能：验证特征输入为非空二维数值矩阵，并检查 NaN、Inf 和特征数量约束。
    输入参数：
        - X：待检查的特征矩阵或 DataFrame。
        - allow_nan：是否允许 NaN；Inf 始终不允许。
        - n_features：可选的预期特征列数。
    输出参数：
        - 无；校验通过正常返回，失败时抛出 ValueError。
    """
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
    """
    函数名：check_row_ids
    功能：验证来源行号是一维、非空、无缺失且唯一的向量，并可要求其为非负整数。
    输入参数：
        - values：待检查的来源 row_id 向量。
        - name：用于错误提示的标识名称。
        - n_rows：可选的预期行数。
        - require_integer：是否要求非负整数行号。
    输出参数：
        - 无；校验通过正常返回，失败时抛出 ValueError。
    """
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or (n_rows is not None and len(array) != n_rows):
        raise ValueError(f"{name}: row IDs must be nonempty vectors of matching length")
    index = pd.Index(array)
    if index.hasnans or not index.is_unique:
        raise ValueError(f"{name}: row IDs 必须非空且唯一 (non-null and unique)")
    if require_integer and (array.dtype.kind not in "iu" or np.any(array < 0)):
        raise ValueError(f"{name}: row IDs must be nonnegative integer vectors")

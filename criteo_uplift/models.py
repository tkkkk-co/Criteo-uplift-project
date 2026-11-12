#-------------
# S/T/X 算法与结果分类器、效应回归器分离；所有预处理只在训练数据 fit。
#-------------
from __future__ import annotations
from time import perf_counter

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from .input_checks import check_binary_vector, check_feature_matrix
from .config import configure_runtime

_XGB_DEFAULTS = dict(random_state=42, n_estimators=150, max_depth=4, learning_rate=.05,
                     min_child_weight=20, subsample=.8, colsample_bytree=.9,
                     tree_method="hist", n_jobs=4)


def make_classifier_model(name, **params):
    #-------------
    # 创建结果分类器，Logistic 包含填补和标准化。
    # 输入：name 为分类器名称，params 为构造参数。
    # 输出：未拟合的分类器或 Pipeline。
    #-------------
    if name == "logistic":
        settings = dict(random_state=42, max_iter=500, C=1., solver="lbfgs")
        settings.update(params)
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(**settings))])
    if name == "xgboost":
        configure_runtime()
        from xgboost import XGBClassifier
        settings = {**_XGB_DEFAULTS, "objective": "binary:logistic", "eval_metric": "logloss", **params}
        return XGBClassifier(**settings)
    raise ValueError(f"不支持的结果模型：{name}")


def make_regression_model(name, **params):
    #-------------
    # 创建用于拟合连续伪效应的回归器。
    # 输入：name 为回归器名称，params 为构造参数。
    # 输出：未拟合的回归器或 Pipeline。
    #-------------
    if name == "linear":
        params.pop("random_state", None)
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scaler", StandardScaler()), ("regressor", LinearRegression(**params))])
    if name == "xgboost":
        configure_runtime()
        from xgboost import XGBRegressor
        return XGBRegressor(**{**_XGB_DEFAULTS, "objective": "reg:squarederror", "eval_metric": "rmse", **params})
    raise ValueError(f"不支持的效应模型：{name}")

def _training_inputs_check(X_train, t, y):
    #-------------
    # 检查训练数据并转换成模型输入数组。
    # 输入：训练特征 X_train、处理标签 t、结果 y。
    # 输出：训练数组 X、t、y。
    #-------------
    check_feature_matrix(X_train, allow_nan=True)
    check_binary_vector(t, 't', len(X_train), require_both=True)
    check_binary_vector(y, 'y', len(X_train))

    X_train_array = np.asarray(X_train)
    t, y = np.asarray(t, dtype=np.int8), np.asarray(y, dtype=np.int8)
    return X_train_array, t, y


def _test_inputs_check(X, model_bundle, learner, model_keys):
    #-------------
    # 检查已拟合模型及预测特征，返回 NumPy 矩阵。
    # 输入：预测特征 X、模型字典 model_bundle、预期类型 learner、所需模型键 model_keys。
    # 输出：通过特征数量和列顺序检查的矩阵。
    #-------------
    if (not isinstance(model_bundle, dict) or model_bundle.get('learner') != learner
            or not all(key in model_bundle for key in (*model_keys, 'n_features'))
            or (learner == 'x' and 'treatment_rate' not in model_bundle)):
        raise NotFittedError(f'需要已拟合的 {learner.upper()}-Learner 模型字典')
    check_feature_matrix(X, allow_nan=True, n_features=model_bundle['n_features'])
    if model_bundle.get('feature_names') is not None and hasattr(X, 'columns'):
        if list(X.columns) != model_bundle['feature_names']:
            raise ValueError('预测列及其顺序必须与训练时一致')
    return np.asarray(X)


def s_learner(X, *, X_train=None, t=None, y=None, model_bundle=None,
              outcome_model='logistic', outcome_params=None):
    #-------------
    # 输入：预测特征 X；提供 X_train、t、y 及模型配置来拟合，或提供 model_bundle 只预测。
    # 输出：model_bundle（已拟合模型及训练信息）、mu0、mu1、CATE 和本次拟合耗时。
    #-------------
    fit_seconds = None
    if model_bundle is None:
        started = perf_counter()
        # 训练一个结果模型，将 T 作为最后一列特征。
        features, t, y = _training_inputs_check(X_train, t, y)
        classifier_params = (outcome_params or {}).copy()
        model = make_classifier_model(outcome_model, **classifier_params)
        # 单类别结果使用常数概率，不要求每组同时出现 Y=0 和 Y=1。
        if np.unique(y).size == 1:
            model = DummyClassifier(strategy='constant', constant=int(y[0]))
    
        fitted_model = model.fit(np.column_stack((features, t)), y)
        model_bundle = {
            'learner': 's', 'fitted_model': fitted_model,
            'n_features': features.shape[1],
            'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
            'parameters': {'outcome_model': outcome_model, 'outcome_params': classifier_params,
                           'effect_model': None, 'effect_params': None},
        }
        fit_seconds = perf_counter() - started
    elif any(value is not None for value in (X_train, t, y)):
        raise ValueError('仅预测模式请只提供拟合模型和待预测特征')

    # 固定 X，分别令 T=0 和 T=1，再用预测概率之差估计 CATE。
    features = _test_inputs_check(X, model_bundle, 's', ('fitted_model',))
    fitted_model = model_bundle['fitted_model']
    p0 = fitted_model.predict_proba(np.column_stack((features, np.zeros(len(features)))))
    p1 = fitted_model.predict_proba(np.column_stack((features, np.ones(len(features)))))
    # 按类别标签选取正类列；只有类别 0 时，空列求和得到零概率。
    mu0 = p0[:, fitted_model.classes_ == 1].sum(axis=1, dtype=float)
    mu1 = p1[:, fitted_model.classes_ == 1].sum(axis=1, dtype=float)
    return dict(model_bundle=model_bundle, mu0=mu0, mu1=mu1, cate=mu1 - mu0, fit_seconds=fit_seconds)


def t_learner(X, *, X_train=None, t=None, y=None, model_bundle=None,
              outcome_model='logistic', outcome_params=None):
    #-------------
    # 输入：预测特征 X；提供 X_train、t、y 及模型配置来拟合，或提供 model_bundle 只预测。
    # 输出：model_bundle（已拟合模型及训练信息）、mu0、mu1、CATE 和本次拟合耗时。
    #-------------
    fit_seconds = None
    if model_bundle is None:
        started = perf_counter()
        # 在两臂各用自己的 X 和 Y 训练一个结果模型。
        features, t, y = _training_inputs_check(X_train, t, y)
        classifier_params = (outcome_params or {}).copy()
        X0, y0 = features[t == 0], y[t == 0]
        X1, y1 = features[t == 1], y[t == 1]

        mu0_model = make_classifier_model(outcome_model, **classifier_params)
        if np.unique(y0).size == 1:
            mu0_model = DummyClassifier(strategy='constant', constant=int(y0[0]))
        mu0_fitted_model = mu0_model.fit(X0, y0)

        mu1_model = make_classifier_model(outcome_model, **classifier_params)
        if np.unique(y1).size == 1:
            mu1_model = DummyClassifier(strategy='constant', constant=int(y1[0]))
        mu1_fitted_model = mu1_model.fit(X1, y1)

        model_bundle = {
            'learner': 't', 'mu0_fitted_model': mu0_fitted_model, 'mu1_fitted_model': mu1_fitted_model,
            'n_features': features.shape[1],
            'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
            'parameters': {'outcome_model': outcome_model, 'outcome_params': classifier_params,
                           'effect_model': None, 'effect_params': None},
        }
        fit_seconds = perf_counter() - started
    elif any(value is not None for value in (X_train, t, y)):
        raise ValueError('仅预测模式请只提供拟合模型和待预测特征')

    # 将同一个 X 分别交给两个结果模型，再相减。
    features = _test_inputs_check(X, model_bundle, 't', ('mu0_fitted_model', 'mu1_fitted_model'))
    mu0_fitted_model, mu1_fitted_model = model_bundle['mu0_fitted_model'], model_bundle['mu1_fitted_model']
    # 按类别标签选取正类列；只有类别 0 时，空列求和得到零概率。
    mu0 = mu0_fitted_model.predict_proba(features)[:, mu0_fitted_model.classes_ == 1].sum(axis=1, dtype=float)
    mu1 = mu1_fitted_model.predict_proba(features)[:, mu1_fitted_model.classes_ == 1].sum(axis=1, dtype=float)
    return dict(model_bundle=model_bundle, mu0=mu0, mu1=mu1, cate=mu1 - mu0, fit_seconds=fit_seconds)


def x_learner(X, *, X_train=None, t=None, y=None, model_bundle=None,
              outcome_model='xgboost', effect_model='xgboost', outcome_params=None,
              effect_params=None):
    #-------------
    # 输入：预测特征 X；提供 X_train、t、y 及两阶段模型配置来拟合，或提供 model_bundle 只预测。
    # 输出：model_bundle（已拟合模型及训练信息）、第一阶段 mu0、mu1、第二阶段加权 CATE 和本次拟合耗时。
    #-------------
    fit_seconds = None
    if model_bundle is None:
        started = perf_counter()
        features, t, y = _training_inputs_check(X_train, t, y)
        classifier_params = (outcome_params or {}).copy()
        regressor_params = (effect_params or {}).copy()
        X0, y0 = features[t == 0], y[t == 0]
        X1, y1 = features[t == 1], y[t == 1]

        # 第一阶段：分别训练对照组和处理组的结果模型。
        mu0_model = make_classifier_model(outcome_model, **classifier_params)
        if np.unique(y0).size == 1:
            mu0_model = DummyClassifier(strategy='constant', constant=int(y0[0]))
        mu0_fitted_model = mu0_model.fit(X0, y0)
        
        mu1_model = make_classifier_model(outcome_model, **classifier_params)
        if np.unique(y1).size == 1:
            mu1_model = DummyClassifier(strategy='constant', constant=int(y1[0]))
        mu1_fitted_model = mu1_model.fit(X1, y1)

        # 用另一臂模型填补反事实，两组伪效应均为处理结果减对照结果。
        # 单类别模型同样按类别标签取正类概率，缺少类别 1 时为零。
        mu1_on_control = mu1_fitted_model.predict_proba(X0)[:, mu1_fitted_model.classes_ == 1].sum(axis=1, dtype=float)
        mu0_on_treated = mu0_fitted_model.predict_proba(X1)[:, mu0_fitted_model.classes_ == 1].sum(axis=1, dtype=float)
        d0 = mu1_on_control - y0
        d1 = y1 - mu0_on_treated

        # 第二阶段：分别用对照组和处理组的特征回归伪效应。
        tau0_model = make_regression_model(effect_model, **regressor_params)
        tau0_fitted_model = tau0_model.fit(X0, d0)
        tau1_model = make_regression_model(effect_model, **regressor_params)
        tau1_fitted_model = tau1_model.fit(X1, d1)
        model_bundle = {
            'learner': 'x', 'mu0_fitted_model': mu0_fitted_model, 'mu1_fitted_model': mu1_fitted_model,
            'tau0_fitted_model': tau0_fitted_model, 'tau1_fitted_model': tau1_fitted_model,
            'n_features': features.shape[1],
            'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
            'treatment_rate': float(t.mean()),
            'parameters': {'outcome_model': outcome_model, 'outcome_params': classifier_params,
                           'effect_model': effect_model, 'effect_params': regressor_params},
        }
        fit_seconds = perf_counter() - started
    elif any(value is not None for value in (X_train, t, y)):
        raise ValueError('仅预测模式请只提供拟合模型和待预测特征')

    # 返回第一阶段概率，最终 CATE 用训练集处理比例加权两组效应预测。
    features = _test_inputs_check(X, model_bundle, 'x', ('mu0_fitted_model', 'mu1_fitted_model', 'tau0_fitted_model', 'tau1_fitted_model'))
    mu0_fitted_model, mu1_fitted_model = model_bundle['mu0_fitted_model'], model_bundle['mu1_fitted_model']
    # 按类别标签选取正类列；只有类别 0 时，空列求和得到零概率。
    mu0 = mu0_fitted_model.predict_proba(features)[:, mu0_fitted_model.classes_ == 1].sum(axis=1, dtype=float)
    mu1 = mu1_fitted_model.predict_proba(features)[:, mu1_fitted_model.classes_ == 1].sum(axis=1, dtype=float)
    tau0 = np.asarray(model_bundle['tau0_fitted_model'].predict(features), dtype=float)
    tau1 = np.asarray(model_bundle['tau1_fitted_model'].predict(features), dtype=float)
    e = model_bundle['treatment_rate']
    return dict(model_bundle=model_bundle, mu0=mu0, mu1=mu1, cate=e * tau0 + (1 - e) * tau1,
                fit_seconds=fit_seconds)


LEARNERS = {'s': s_learner, 't': t_learner, 'x': x_learner}

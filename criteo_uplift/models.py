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

def predict_positive_probability(model, X):
    #-------------
    # 按类别标签提取正类概率。
    # 输入：model 为已拟合分类器，X 为预测特征。
    # 输出：每行属于类别 1 的概率向量。
    #-------------
    probabilities = model.predict_proba(X)
    columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    return (np.asarray(probabilities[:, columns[0]], dtype=float) if len(columns)
            else np.zeros(len(X), dtype=float))


def fit_binary_model(X, y, outcome_model, outcome_params=None):
    #-------------
    # 拟合结果分类器，单类别使用常数分类器。
    # 输入：X、y 为训练数据，outcome_model 和 outcome_params 指定分类器。
    # 输出：已拟合的分类器。
    #-------------
    check_feature_matrix(X, allow_nan=True)
    check_binary_vector(y, 'y', len(X))
    model = make_classifier_model(outcome_model, **(outcome_params or {}))
    if np.unique(y).size == 1:
        model = DummyClassifier(strategy='constant', constant=int(np.asarray(y)[0]))
    return model.fit(X, y)


def _training_inputs(X_train, t, y, overlap):
    #-------------
    # 检查训练数据和共享诊断，保留预测校验所需的特征信息。
    # 输入：训练特征 X_train、处理标签 t、结果 y、可选 Overlap 摘要。
    # 输出：训练数组 X、t、y 和训练元数据。
    #-------------
    check_feature_matrix(X_train, allow_nan=True)
    check_binary_vector(t, 't', len(X_train), require_both=True)
    check_binary_vector(y, 'y', len(X_train))
    X_train_array = np.asarray(X_train)
    t, y = np.asarray(t, dtype=np.int8), np.asarray(y, dtype=np.int8)
    if overlap is not None:
        from .diagnostics import overlap_input_signature
        if (overlap.get('input_signature') != overlap_input_signature(X_train, t)
                or overlap.get('status') not in {'checked_no_major_warning', 'checked_with_warning'}):
            raise ValueError('Overlap 诊断与当前训练数据不匹配，必须使用同一份训练数据的诊断')

    outcome_support = []
    for arm in (0, 1):
        labels = y[t == arm]
        positive = int(labels.sum())
        outcome_support.append({
            'treatment': arm, 'n': len(labels), 'positive': positive,
            'negative': len(labels) - positive, 'single_class': positive in (0, len(labels)),
        })
    metadata = {
        'n_features': X_train_array.shape[1],
        'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
        'n_samples': len(X_train),
        'treatment_rate': float(t.mean()),
        'outcome_support': outcome_support,
        'overlap': overlap if overlap is not None else {'status': 'not_checked'},
    }
    return X_train_array, t, y, metadata


def _prediction_features(X, fitted, learner, model_keys):
    #-------------
    # 检查已拟合模型及预测特征，返回 NumPy 矩阵。
    # 输入：预测特征 X、模型字典 fitted、预期类型 learner、所需模型键 model_keys。
    # 输出：通过特征数量和列顺序检查的矩阵。
    #-------------
    if (not isinstance(fitted, dict) or fitted.get('learner') != learner
            or not all(key in fitted for key in (*model_keys, 'n_features', 'treatment_rate'))):
        raise NotFittedError(f'需要已拟合的 {learner.upper()}-Learner 模型字典')
    check_feature_matrix(X, allow_nan=True, n_features=fitted['n_features'])
    if fitted.get('feature_names') is not None and hasattr(X, 'columns'):
        if list(X.columns) != fitted['feature_names']:
            raise ValueError('预测列及其顺序必须与训练时一致')
    return np.asarray(X)


def s_learner(X, *, X_train=None, t=None, y=None, fitted=None,
              outcome_model='logistic', outcome_params=None, overlap=None, **params):
    #-------------
    # 输入：预测特征 X；提供 X_train、t、y 及模型配置来拟合，或提供 fitted 只预测。
    # 输出：模型字典、mu0、mu1、CATE 和本次拟合耗时。
    #-------------
    fit_seconds = None
    if fitted is None:
        started = perf_counter()
        # 训练一个结果模型，将 T 作为最后一列特征。
        features, t, y, fitted = _training_inputs(X_train, t, y, overlap)
        classifier_params = {**params, **(outcome_params or {})}
        model = fit_binary_model(np.column_stack((features, t)), y, outcome_model, classifier_params)
        fitted.update(learner='s', model=model, parameters={
            'outcome_model': outcome_model, 'outcome_params': classifier_params,
            'effect_model': None, 'effect_params': None,
        })
        fit_seconds = perf_counter() - started
    elif any(value is not None for value in (X_train, t, y)):
        raise ValueError('仅预测模式请只提供拟合模型和待预测特征')

    # 固定 X，分别令 T=0 和 T=1，再用预测概率之差估计 CATE。
    features = _prediction_features(X, fitted, 's', ('model',))
    mu0 = predict_positive_probability(fitted['model'], np.column_stack((features, np.zeros(len(features)))))
    mu1 = predict_positive_probability(fitted['model'], np.column_stack((features, np.ones(len(features)))))
    return dict(model=fitted, mu0=mu0, mu1=mu1, cate=mu1 - mu0, fit_seconds=fit_seconds)


def t_learner(X, *, X_train=None, t=None, y=None, fitted=None,
              outcome_model='logistic', outcome_params=None, overlap=None, **params):
    #-------------
    # 输入：预测特征 X；提供 X_train、t、y 及模型配置来拟合，或提供 fitted 只预测。
    # 输出：模型字典、mu0、mu1、CATE 和本次拟合耗时。
    #-------------
    fit_seconds = None
    if fitted is None:
        started = perf_counter()
        # 在两臂各用自己的 X 和 Y 训练一个结果模型。
        features, t, y, fitted = _training_inputs(X_train, t, y, overlap)
        classifier_params = {**params, **(outcome_params or {})}
        control, treated = t == 0, t == 1
        mu0_model = fit_binary_model(features[control], y[control], outcome_model, classifier_params)
        mu1_model = fit_binary_model(features[treated], y[treated], outcome_model, classifier_params)
        fitted.update(learner='t', mu0_model=mu0_model, mu1_model=mu1_model, parameters={
            'outcome_model': outcome_model, 'outcome_params': classifier_params,
            'effect_model': None, 'effect_params': None,
        })
        fit_seconds = perf_counter() - started
    elif any(value is not None for value in (X_train, t, y)):
        raise ValueError('仅预测模式请只提供拟合模型和待预测特征')

    # 将同一个 X 分别交给两个结果模型，再相减。
    features = _prediction_features(X, fitted, 't', ('mu0_model', 'mu1_model'))
    mu0 = predict_positive_probability(fitted['mu0_model'], features)
    mu1 = predict_positive_probability(fitted['mu1_model'], features)
    return dict(model=fitted, mu0=mu0, mu1=mu1, cate=mu1 - mu0, fit_seconds=fit_seconds)


def x_learner(X, *, X_train=None, t=None, y=None, fitted=None,
              outcome_model='xgboost', effect_model='xgboost', outcome_params=None,
              effect_params=None, overlap=None, **params):
    #-------------
    # 输入：预测特征 X；提供 X_train、t、y 及两阶段模型配置来拟合，或提供 fitted 只预测。
    # 输出：模型字典、第一阶段 mu0、mu1、第二阶段加权 CATE 和本次拟合耗时。
    #-------------
    fit_seconds = None
    if fitted is None:
        started = perf_counter()
        features, t, y, fitted = _training_inputs(X_train, t, y, overlap)
        classifier_params = {**params, **(outcome_params or {})}
        regressor_params = {**params, **(effect_params or {})}
        control, treated = t == 0, t == 1
        X0, y0 = features[control], y[control]
        X1, y1 = features[treated], y[treated]

        # 第一阶段：分别训练对照组和处理组的结果模型。
        mu0_model = fit_binary_model(X0, y0, outcome_model, classifier_params)
        mu1_model = fit_binary_model(X1, y1, outcome_model, classifier_params)

        # 用另一臂模型填补反事实，两组伪效应均为处理结果减对照结果。
        d0 = predict_positive_probability(mu1_model, X0) - y0
        d1 = y1 - predict_positive_probability(mu0_model, X1)

        # 第二阶段：分别用对照组和处理组的特征回归伪效应。
        tau0_model = make_regression_model(effect_model, **regressor_params).fit(X0, d0)
        tau1_model = make_regression_model(effect_model, **regressor_params).fit(X1, d1)
        fitted.update(learner='x', mu0_model=mu0_model, mu1_model=mu1_model,
                      tau0_model=tau0_model, tau1_model=tau1_model, parameters={
                          'outcome_model': outcome_model, 'outcome_params': classifier_params,
                          'effect_model': effect_model, 'effect_params': regressor_params,
                      })
        fit_seconds = perf_counter() - started
    elif any(value is not None for value in (X_train, t, y)):
        raise ValueError('仅预测模式请只提供拟合模型和待预测特征')

    # 返回第一阶段概率，最终 CATE 用训练集处理比例加权两组效应预测。
    features = _prediction_features(X, fitted, 'x', ('mu0_model', 'mu1_model', 'tau0_model', 'tau1_model'))
    mu0 = predict_positive_probability(fitted['mu0_model'], features)
    mu1 = predict_positive_probability(fitted['mu1_model'], features)
    tau0 = np.asarray(fitted['tau0_model'].predict(features), dtype=float)
    tau1 = np.asarray(fitted['tau1_model'].predict(features), dtype=float)
    e = fitted['treatment_rate']
    return dict(model=fitted, mu0=mu0, mu1=mu1, cate=e * tau0 + (1 - e) * tau1,
                fit_seconds=fit_seconds)


LEARNERS = {'s': s_learner, 't': t_learner, 'x': x_learner}

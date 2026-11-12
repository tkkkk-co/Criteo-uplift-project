#-------------
# S/T/X 算法与结果分类器、效应回归器分离；所有预处理只在训练数据 fit。
#-------------
from __future__ import annotations
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
    # 函数名：make_classifier_model
    # 功能：创建以 0/1 为训练标签、可输出结果概率的分类模型及其预处理。
    # 输入参数：
    #     - name：分类器名称，当前支持 logistic 和 xgboost。
    #     - **params：传递给对应分类器构造函数的参数。
    # 输出参数：
    #     - 未拟合的结果分类器或 Pipeline。
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
    # 函数名：make_regression_model
    # 功能：创建用于拟合 X-Learner 连续伪效应标签的回归模型及其预处理。
    # 输入参数：
    #     - name：回归器名称，linear 为线性回归，xgboost 为提升树回归。
    #     - **params：传递给对应回归器构造函数的参数。
    # 输出参数：
    #     - 未拟合的效应回归器或 Pipeline。
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
    # 功能：按类别标签提取正类概率，兼容只有单个类别的分类器。
    # 输入参数：
    #     - model：已拟合的结果分类器。
    #     - X：预测特征矩阵。
    # 输出参数：
    #     - 每行属于类别 1 的概率向量。
    #-------------
    probabilities = model.predict_proba(X)
    columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    return (np.asarray(probabilities[:, columns[0]], dtype=float) if len(columns)
            else np.zeros(len(X), dtype=float))


def fit_binary_model(X, y, outcome_model, outcome_params=None):
    #-------------
    # 功能：拟合二元结果模型；单类别标签使用标准库常数分类器保持原有行为。
    # 输入参数：
    #     - X：当前训练范围的特征矩阵。
    #     - y：当前训练范围的二元结果标签。
    #     - outcome_model：分类器名称。
    #     - outcome_params：分类器构造参数。
    # 输出参数：
    #     - 已拟合分类器；单类别不意味着因果效应不可识别。
    #-------------
    check_feature_matrix(X, allow_nan=True)
    check_binary_vector(y, 'y', len(X))
    model = make_classifier_model(outcome_model, **(outcome_params or {}))
    if np.unique(y).size == 1:
        model = DummyClassifier(strategy='constant', constant=int(np.asarray(y)[0]))
    return model.fit(X, y)


def validate_model_inputs(X, t, y):
    #-------------
    # 功能：校验训练数据，要求两臂非空；这不等同于完整的条件正值性诊断。
    # 输入参数：
    #     - X：训练特征矩阵，允许 NaN，拒绝无穷值。
    #     - t：二元处理标签。
    #     - y：二元结果标签。
    # 输出参数：
    #     - 特征、处理标签、结果标签三个 NumPy 数组。
    #-------------
    check_feature_matrix(X, allow_nan=True)
    check_binary_vector(t, 't', len(X), require_both=True)
    check_binary_vector(y, 'y', len(X))
    return np.asarray(X), np.asarray(t, dtype=np.int8), np.asarray(y, dtype=np.int8)


def uplift_model_parameters(learner, outcome_model, effect_model=None, *,
                            outcome_params=None, effect_params=None, **params):
    #-------------
    # 功能：校验模型组合，合并公共参数和阶段专属参数；不创建或训练模型。
    # 输入参数：
    #     - learner：s、t 或 x。
    #     - outcome_model：logistic 或 xgboost。
    #     - effect_model：X 阶段回归器，linear 或 xgboost。
    #     - outcome_params：结果分类器专属参数。
    #     - effect_params：效应回归器专属参数。
    #     - **params：公共参数，专属参数优先。
    # 输出参数：
    #     - 可审计的模型参数字典。
    #-------------
    if learner not in {'s', 't', 'x'}:
        raise ValueError(f'不支持的学习器：{learner}')
    if outcome_model not in {'logistic', 'xgboost'}:
        raise ValueError(f'不支持的结果模型：{outcome_model}')
    if learner == 'x' and effect_model not in {'linear', 'xgboost'}:
        raise ValueError('X-Learner 需要指定效应模型：linear 或 xgboost')
    if learner != 'x' and (effect_model is not None or effect_params):
        raise ValueError('S/T 学习器不使用效应模型')
    return dict(outcome_model=outcome_model, effect_model=effect_model,
                outcome_params={**params, **(outcome_params or {})},
                effect_params={**params, **(effect_params or {})} if learner == 'x' else None)


def _training_result(X, t, y, learner, parameters, overlap):
    #-------------
    # 功能：记录训练范围、参数、结果标签支持情况及共用诊断的来源。
    # 输入参数：
    #     - X：已校验的训练特征，保留 DataFrame 列信息。
    #     - t：已校验的处理标签。
    #     - y：已校验的结果标签。
    #     - learner：学习器类型。
    #     - parameters：合并后的模型参数。
    #     - overlap：同一训练数据的诊断摘要；未提供时标记未检查。
    # 输出参数：
    #     - 普通训练结果字典，随后由训练函数填入已拟合模型。
    #-------------
    names = list(X.columns) if hasattr(X, 'columns') else None
    if overlap is not None:
        from .diagnostics import overlap_input_signature
        if (overlap.get('input_signature') != overlap_input_signature(X, t)
                or overlap.get('status') not in {'checked_no_major_warning', 'checked_with_warning'}):
            raise ValueError('Overlap 诊断与当前训练数据不匹配，必须使用同一份训练数据的诊断')
    support = []
    for arm in (0, 1):
        labels = y[t == arm]
        support.append(dict(treatment=arm, n=len(labels), positive=int(labels.sum()),
                            negative=int(len(labels) - labels.sum()),
                            single_class=bool(np.unique(labels).size == 1)))
    return dict(learner=learner, parameters=parameters, n_features=np.asarray(X).shape[1],
                feature_names=names, n_samples=len(X), treatment_rate=float(t.mean()),
                outcome_support=support,
                overlap=overlap if overlap is not None else {'status': 'not_checked'})


def fit_s_learner(X, t, y, outcome_model='logistic', *, outcome_params=None, overlap=None, **params):
    #-------------
    # 功能：在全部训练数据上，将 T 追加为特征，只拟合一个结果分类器。
    # 输入参数：
    #     - X：处理前训练特征。
    #     - t：处理分配标签。
    #     - y：二元结果标签。
    #     - outcome_model：结果分类器名称。
    #     - outcome_params：分类器专属参数。
    #     - overlap：可复用的同一 Train 诊断摘要。
    #     - **params：分类器公共参数，包括随机种子。
    # 输出参数：
    #     - 训练字典，含 model、配置、训练范围及诊断摘要。
    #-------------
    features, treatment, outcome = validate_model_inputs(X, t, y)
    config = uplift_model_parameters('s', outcome_model, outcome_params=outcome_params, **params)
    result = _training_result(X, treatment, outcome, 's', config, overlap)
    result['model'] = fit_binary_model(np.column_stack((features, treatment)), outcome,
                                       outcome_model, config['outcome_params'])
    return result


def fit_t_learner(X, t, y, outcome_model='logistic', *, outcome_params=None, overlap=None, **params):
    #-------------
    # 功能：分别在两臂上训练独立结果分类器，两个模型的输入均只有 X。
    # 输入参数：
    #     - X：处理前训练特征。
    #     - t：处理分配标签。
    #     - y：二元结果标签。
    #     - outcome_model：结果分类器名称。
    #     - outcome_params：分类器专属参数。
    #     - overlap：可复用的同一 Train 诊断摘要。
    #     - **params：分类器公共参数，包括随机种子。
    # 输出参数：
    #     - 训练字典，含 mu0_model、mu1_model 和训练元数据。
    #-------------
    features, treatment, outcome = validate_model_inputs(X, t, y)
    config = uplift_model_parameters('t', outcome_model, outcome_params=outcome_params, **params)
    result = _training_result(X, treatment, outcome, 't', config, overlap)
    control, treated = treatment == 0, treatment == 1
    result['mu0_model'] = fit_binary_model(features[control], outcome[control], outcome_model, config['outcome_params'])
    result['mu1_model'] = fit_binary_model(features[treated], outcome[treated], outcome_model, config['outcome_params'])
    return result


def fit_x_learner(X, t, y, outcome_model='xgboost', effect_model='xgboost', *,
                  outcome_params=None, effect_params=None, overlap=None, **params):
    #-------------
    # 功能：按结果模型、跨臂伪效应、效应回归三个步骤训练 X-Learner。
    # 输入参数：
    #     - X：处理前训练特征。
    #     - t：处理分配标签。
    #     - y：二元结果标签。
    #     - outcome_model：第一阶段分类器名称。
    #     - effect_model：第二阶段回归器名称。
    #     - outcome_params：分类器专属参数。
    #     - effect_params：回归器专属参数。
    #     - overlap：可复用的同一 Train 诊断摘要。
    #     - **params：两个阶段的公共参数，包括随机种子。
    # 输出参数：
    #     - 训练字典，含两个结果模型、两个效应模型和固定训练处理比例。
    #-------------
    features, treatment, outcome = validate_model_inputs(X, t, y)
    config = uplift_model_parameters('x', outcome_model, effect_model,
                                     outcome_params=outcome_params, effect_params=effect_params, **params)
    result = _training_result(X, treatment, outcome, 'x', config, overlap)
    control, treated = treatment == 0, treatment == 1
    # 第一步：两臂各自训练结果模型，与 T-Learner 的第一阶段相同。
    mu0 = fit_binary_model(features[control], outcome[control], outcome_model, config['outcome_params'])
    mu1 = fit_binary_model(features[treated], outcome[treated], outcome_model, config['outcome_params'])
    # 第二步：用另一臂模型填补反事实，保留原有伪效应方向。
    d0 = predict_positive_probability(mu1, features[control]) - outcome[control]
    d1 = outcome[treated] - predict_positive_probability(mu0, features[treated])
    # 第三步：连续伪效应必须交给回归器，不能交给二元分类器。
    tau0 = make_regression_model(effect_model, **config['effect_params']).fit(features[control], d0)
    tau1 = make_regression_model(effect_model, **config['effect_params']).fit(features[treated], d1)
    result.update(mu0_model=mu0, mu1_model=mu1, tau0_model=tau0, tau1_model=tau1)
    return result


def fit_uplift_model(X, t, y, learner, outcome_model, effect_model=None, *,
                     outcome_params=None, effect_params=None, overlap=None, **params):
    #-------------
    # 功能：按已固定的候选配置分发训练，算法步骤保留在三个独立训练函数中。
    # 输入参数：
    #     - X：训练特征。
    #     - t：处理标签。
    #     - y：结果标签。
    #     - learner：s、t 或 x。
    #     - outcome_model：结果分类器名称。
    #     - effect_model：X 第二阶段回归器名称。
    #     - outcome_params：分类器专属参数。
    #     - effect_params：回归器专属参数。
    #     - overlap：可复用的同一训练数据诊断摘要。
    #     - **params：公共模型参数。
    # 输出参数：
    #     - 已拟合学习器的普通字典，可使用 joblib 保存。
    #-------------
    config = uplift_model_parameters(learner, outcome_model, effect_model,
                                     outcome_params=outcome_params, effect_params=effect_params, **params)
    if learner == 'x':
        return fit_x_learner(X, t, y, overlap=overlap, **config)
    return {'s': fit_s_learner, 't': fit_t_learner}[learner](
        X, t, y, outcome_model, outcome_params=config['outcome_params'], overlap=overlap)


def _prediction_features(fitted, X):
    #-------------
    # 功能：拒绝未拟合字典和错误特征结构，预测阶段不运行 Overlap 诊断。
    # 输入参数：
    #     - fitted：训练函数返回的模型字典。
    #     - X：待预测特征矩阵。
    # 输出参数：
    #     - 校验通过的 NumPy 特征矩阵。
    #-------------
    if not isinstance(fitted, dict) or fitted.get('learner') not in {'s', 't', 'x'}:
        raise NotFittedError('请先使用学习器训练函数得到模型字典；旧类模型需在新目录重新训练')
    keys = ['model'] if fitted['learner'] == 's' else ['mu0_model', 'mu1_model']
    if fitted['learner'] == 'x':
        keys += ['tau0_model', 'tau1_model']
    if not all(key in fitted for key in [*keys, 'n_features', 'treatment_rate']):
        raise NotFittedError('模型字典缺少已拟合模型或训练元数据')
    check_feature_matrix(X, allow_nan=True, n_features=fitted['n_features'])
    if fitted.get('feature_names') is not None and hasattr(X, 'columns'):
        if list(X.columns) != fitted['feature_names']:
            raise ValueError('预测列及其顺序必须与训练时一致')
    return np.asarray(X)


def predict_outcomes(fitted, X):
    #-------------
    # 功能：返回第一阶段的两个潜在结果概率；X 的效应另由第二阶段预测。
    # 输入参数：
    #     - fitted：已拟合的 S/T/X 模型字典。
    #     - X：需要预测的处理前特征。
    # 输出参数：
    #     - 二元组 mu0、mu1，每个元素是概率向量。
    #-------------
    features = _prediction_features(fitted, X)
    if fitted['learner'] == 's':
        return tuple(predict_positive_probability(fitted['model'], np.column_stack((features, np.full(len(features), arm, dtype=float))))
                     for arm in (0, 1))
    return (predict_positive_probability(fitted['mu0_model'], features),
            predict_positive_probability(fitted['mu1_model'], features))


def predict_s_learner(fitted, X):
    #-------------
    # 功能：计算 S-Learner 的两种处理状态预测概率之差。
    # 输入参数：
    #     - fitted：已拟合 S-Learner 字典。
    #     - X：预测特征。
    # 输出参数：
    #     - 每行的 CATE 向量 mu1 - mu0。
    #-------------
    if not isinstance(fitted, dict) or fitted.get('learner') != 's':
        raise ValueError('需要 S-Learner 模型字典')
    mu0, mu1 = predict_outcomes(fitted, X)
    return mu1 - mu0


def predict_t_learner(fitted, X):
    #-------------
    # 功能：计算 T-Learner 两个独立结果模型的预测概率之差。
    # 输入参数：
    #     - fitted：已拟合 T-Learner 字典。
    #     - X：预测特征。
    # 输出参数：
    #     - 每行的 CATE 向量 mu1 - mu0。
    #-------------
    if not isinstance(fitted, dict) or fitted.get('learner') != 't':
        raise ValueError('需要 T-Learner 模型字典')
    mu0, mu1 = predict_outcomes(fitted, X)
    return mu1 - mu0


def predict_x_learner(fitted, X):
    #-------------
    # 功能：第四步，按训练集处理比例加权两组效应回归预测，不作截断。
    # 输入参数：
    #     - fitted：已拟合 X-Learner 字典。
    #     - X：预测特征。
    # 输出参数：
    #     - 每行的 CATE 向量 e × tau0 + (1-e) × tau1。
    #-------------
    features = _prediction_features(fitted, X)
    if fitted['learner'] != 'x':
        raise ValueError('需要 X-Learner 模型字典')
    tau0 = np.asarray(fitted['tau0_model'].predict(features), dtype=float)
    tau1 = np.asarray(fitted['tau1_model'].predict(features), dtype=float)
    e = fitted['treatment_rate']
    return e * tau0 + (1 - e) * tau1


def predict_uplift(fitted, X):
    #-------------
    # 功能：按学习器类型调用相应的效应预测函数。
    # 输入参数：
    #     - fitted：已拟合模型字典。
    #     - X：预测特征。
    # 输出参数：
    #     - 每行 CATE 向量。
    #-------------
    if not isinstance(fitted, dict) or fitted.get('learner') not in {'s', 't', 'x'}:
        raise NotFittedError('需要训练函数返回的已拟合模型字典')
    return {'s': predict_s_learner, 't': predict_t_learner, 'x': predict_x_learner}[fitted['learner']](fitted, X)


def predict_policy_scores(models, X):
    #-------------
    # 功能：生成四个候选的 CATE 分数和处理状态下的响应概率基线。
    # 输入参数：
    #     - models：候选名称到已拟合字典的映射。
    #     - X：待评分特征矩阵。
    # 输出参数：
    #     - 分数字典；propensity 仍表示响应概率，不是处理分配概率。
    #-------------
    scores = {name: predict_uplift(fitted, X) for name, fitted in models.items()}
    scores['propensity'] = predict_outcomes(models['t_xgboost'], X)[1]
    return scores

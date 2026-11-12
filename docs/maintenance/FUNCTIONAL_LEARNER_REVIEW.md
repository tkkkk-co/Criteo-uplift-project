# 函数式 Learner、Overlap 诊断与 C2ST 报告验收

本次从 `f3e8f8a` 的干净工作区开始，只调整模型接口及其调用方、增加经验覆盖诊断和报告展示。没有修改 ATE、指标公式、样本划分、四个候选、模型参数或 C2ST 算法，没有新增 Python 文件。

## 修改文件与职责

| 文件 | 本次修改 |
|---|---|
| `criteo_uplift/models.py` | 删除所有自定义 Learner 类与常数分类器类；独立函数训练、预测，普通字典保存训练结果 |
| `criteo_uplift/diagnostics.py` | 新增 `diagnose_overlap()` 和训练数据签名函数，原 SRM/SMD/KS/C2ST 不变 |
| `criteo_uplift/training_validation.py` | Train 只诊断一次，四个候选共用；保存模型字典、诊断文件和模型摘要 |
| `criteo_uplift/stability.py` | 每个实际训练子集诊断一次，四个候选共用；使用函数式训练和参数记录 |
| `criteo_uplift/reporting.py` | 嵌入已有置换 AUC 直方图，显示原始 p、Holm p、观测 AUC 和 0.5 参考线；增加 Overlap 和标签支持说明 |
| `score_policy.py` | 加载普通模型字典，使用函数预测，维持原 X-only 分块和预算选择规则 |
| `tests/test_models.py` | 原算法与边界测试迁移到函数接口；新增共享诊断、参数覆盖、所有候选 joblib 往返、全缺失特征测试 |
| `tests/test_diagnostics.py` | 新增覆盖充分、缺臂、少数臂过少、常数/重复/缺失/空箱、只读和参考概率边界测试 |
| `tests/test_end_to_end.py` | 验证一次 Train 诊断、诊断来源和实际 Train 行一致、模型摘要一致及 HTML 确实嵌入图像 |
| `tests/test_scoring.py` | 将测试模型容器改为字典，保留原分块、冻结、同分、预算、只读等测试 |
| `tests/test_stability.py` | 函数接口迁移，验证 12 次拟合共用 3 次实际训练子集诊断 |
| `notebooks/criteo_uplift_project.ipynb` | 更新函数式 API 示例，增加只读 Overlap 展示并执行保存全部单元格 |
| `docs/PROJECT_PIPELINE.md` | 更新接口、诊断规则和序列化边界 |
| 本文 | 记录验收配置、数值对照与限制 |

`run_pipeline.py` 仍调用原来的 `fit_and_select_models()`，`evaluation.py` 仍调用 `predict_policy_scores()`，不需要修改它们。`preparation.py` 保持原样。

## 模型函数与返回值

| 函数 | 数学步骤与返回内容 |
|---|---|
| `make_classifier_model()` | 创建原 Logistic 或 XGBoost 二元结果模型，未拟合 |
| `make_regression_model()` | 创建原线性或 XGBoost 连续效应回归器，未拟合 |
| `fit_s_learner(X,t,y,...)` | 训练一个 `[X,T] → Y` 模型；字典含 `model` |
| `fit_t_learner(X,t,y,...)` | 分别用两臂的 `X → Y` 训练两个独立模型；字典含 `mu0_model`、`mu1_model` |
| `fit_x_learner(X,t,y,...)` | 训练两组结果模型，构造 `D0=mu1(X0)-Y0`、`D1=Y1-mu0(X1)`，拟合两组连续效应；字典另外含 `tau0_model`、`tau1_model` |
| `predict_s_learner(fitted,X)` | 返回 `mu1-mu0` |
| `predict_t_learner(fitted,X)` | 返回 `mu1-mu0` |
| `predict_x_learner(fitted,X)` | 返回 `e*tau0+(1-e)*tau1`，其中 `e` 是原训练处理比例；不截断 |
| `predict_outcomes(fitted,X)` | 返回第一阶段的 `(mu0,mu1)`，不能将其差当成 X 第二阶段结果 |
| `fit_uplift_model()` / `predict_uplift()` | 只按配置分发到独立算法函数，供 Pipeline 和打分入口使用 |
| `validate_model_inputs()` / `predict_positive_probability()` / `fit_binary_model()` | 真正重复的输入校验、按类别 1 提取概率、二元结果拟合与单类别安全处理 |

每个训练字典另外保存 `learner`、`parameters`、`n_features`、`feature_names`、`n_samples`、`treatment_rate`、`outcome_support` 和 `overlap`。仅接受模型名称及参数，不接收外部 Estimator，因此每次工厂创建独立底层模型，不存在需要 clone 外部实例的情况。原有额外模型组合能力保留，但正式候选仍仅 S-Logistic、T-Logistic、T-XGBoost、X-XGBoost。

单类别标签改用 sklearn 的 `DummyClassifier`；正类概率函数根据 `classes_` 定位，兼容只有 0 或只有 1 的模型。这个实现保持原常数概率预测，不将 Outcome 标签单一误报为因果不可识别。

## Overlap 的输入、输出与含义

`diagnose_overlap(X,t,features=None,expected_treatment_rate=None,probability_source=None,bins=10,min_arm_count=20,scope='train')` 只使用当前训练数据的处理前特征与处理标签。整体缺臂或非法输入报错；逐列分位数箱内缺臂或任一臂少于设定人数时仅警告。重复分位点合并，常数单箱，NaN 单箱，保留空区间但不将其误报为观测缺臂。

返回 `summary` 字典与 `strata` DataFrame，分别保存到实验目录的 `diagnostics/overlap_positivity.json` 和 `diagnostics/overlap_strata.csv`。摘要含人数、比例、参考值及来源、与参考值之差、分箱边界、经验阈值、风险状态、来源签名及解释限制。各行记录箱界、人数、两臂人数、处理比例、较少臂人数、空箱与警告。

Pipeline 在任何候选拟合前检查一次，四个候选引用同一个摘要，并核验其与当前有序 X/T 的内容签名一致。独立训练未提供完整诊断时记为 `not_checked`；预测时不执行诊断。Stability 按每个实际训练子集检查一次，不能沿用原 Train 的覆盖结论。

目前未提供核实过的设计概率，所以正式调用不填预期值，只报告 Train 的处理比例；显式传参考值时必须同时给来源。`checked_no_major_warning` 表示单变量经验规则未预警，`checked_with_warning` 表示局部覆盖薄弱，都不等同于证明识别假设。真实设计若保证各目标区域的分配概率严格位于 0 与 1 之间，才从设计上支持正值性；一致性、无干扰和可交换性需处理定义、执行机制和设计论证，不能由分箱表证明。

原百万建模样本的固定 Train 在事后展示检查中为 600,000 行：Treatment 510,246，Control 89,754，比例 0.850410。重复分位点合并后 12 个特征共 36 箱，缺臂箱和不足 20 人箱均为 0，非空箱最少臂人数为 6,782。该结果不能排除高维联合区域的覆盖问题，也没有用于重选原模型。

## 数值对照和验收结果

在修改代码前保存完整合成实验输出，修改后使用相同输入、划分、参数重新执行。合成数据 4,000 行、seed=814，Train/Validation/Test=2,400/800/800；XGBoost 5 棵树；C2ST 1,000 行、2 次置换；Bootstrap 5 次；Stability seeds=[17,29,83]、训练比例=[0.25,0.5,1.0]，共 12 个配置和 48 次模型拟合。这是工程回归 Smoke Test，不能替代大样本统计推断。

| Test 政策 | 修改前每 10 万人增量，20% 预算 | 修改后 | 逐行分数最大绝对差 |
|---|---:|---:|---:|
| S-Logistic | 4580.896686 | 4580.896686 | 0 |
| T-Logistic | 3963.612736 | 3963.612736 | 0 |
| T-XGBoost | 3460.038986 | 3460.038986 | 0 |
| X-XGBoost | 2964.587394 | 2964.587394 | 0 |
| 响应概率基线 | 3460.038986 | 3460.038986 | 0 |

两次 Validation 均冻结 S-Logistic。逐行预测数组、划分、验证政策/排序指标、Test 政策/排序指标及 Bootstrap 区间、全部 CATE 校准表、ATE、C2ST 观测与置换值、SMD/KS、Stability 选择频率全部精确相等。Stability 长表和聚合表除实际计时字段外也精确相等；未放宽统计容差。

- `python -m unittest discover -s tests -v`：113 项测试通过。没有删除任何原测试覆盖；原 sklearn Learner clone 接口断言按新字典契约改为独立拟合及保存重载断言。
- `python scripts/execute_notebooks.py`：全部公开 Notebook（1 本、30 个代码单元）执行完成，无错误。冻结快照没有 Overlap 文件时显示 `not_checked`，不伪造新结果。
- `python scripts/check_publication.py`：42 个冻结快照文件校验通过，原快照内容未修改。
- 完整合成 Pipeline、函数式模型 joblib 往返、新人群 CSV 分块打分通过。
- 图像已直接查看；端到端测试核验 HTML 包含实际 PNG 的完整内嵌内容和诊断文字。浏览器工具安全策略不允许打开本地 file URL，因此未声称完成浏览器整页视觉验收。
- 没有重新扫描真实原始 CSV。原百万样本仅从现有 NPZ 读取，用原种子划分并与已保存三份 row_id 逐行核对。

## 报告与兼容性边界

`artifacts/report_functional_overlap_review/report.html` 是原冻结聚合结果的独立展示副本，已加入 C2ST 置换图及原 Train 的事后覆盖描述。它没有重新训练百万样本模型、没有重跑 499 次 C2ST，也没有修改或替换 `reports/published/`；报告开头明确标注上述范围。原始 Logistic 的 AUC=0.5177、p=0.006、Holm p=0.012 保持不变，XGBoost 同样保持原结果。

不再保留旧类包装或别名，因此旧类 joblib 无法直接在新接口加载，需要保留旧环境或在新实验目录重新训练。新的函数式模型字典已验证保存、加载及预测。历史 CSV、PNG、HTML 和快照仍可只读展示。

本次不自动发布新的百万样本模型，不用 Smoke Test 冒充正式实证结果。公开历史报告保持冻结；要形成新的正式实验记录，应运行当前 Pipeline 并指定新的输出目录。

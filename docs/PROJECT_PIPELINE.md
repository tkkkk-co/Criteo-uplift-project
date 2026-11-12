# 项目流程与阅读顺序

学习目标是看清每一步的输入、作用与输出。先读 run_pipeline.py 的 run，再进入对应模块。语法只在妨碍理解流程时展开。

| 阶段 | 调用 | 输入与输出 | 核心边界 |
|---|---|---|---|
| 01 配置 | config、safe_output | 参数 → 全部输出目录 | 拒绝写入原始 data，拒绝覆盖已有实验 |
| 02 准备 | prepare_sample、split_sample | 原始 CSV/缓存 → audit、样本、60/20/20 划分 | 最多一次 CSV 分块扫描；三任务独立缓存 |
| 03 诊断 | run_diagnostics | 扫描统计、C2ST 样本、Train/Validation → SRM、SMD/KS、C2ST、Learnability、Train Overlap | diagnostics 不读原始 CSV、不重新审计 |
| 04 全量效果 | ate_summary_from_audit | 两组人数及正例数 → 全量 ATE/ITT | 全量结果只提供背景，不选模型 |
| 05 训练/验证 | fit_and_select_models | Train 拟合，Validation 比较 → 模型与 selection.json | 固定四候选，按主预算 HT 增量选策略 |
| 06 Test | evaluate_test_set | 冻结模型与 Test → CATE 排序、策略值、Bootstrap、校准、Test ATE | 不按 Test 改选；统一 ATE 表 |
| 07 稳定性 | run_stability | 缓存减去原 Test → 12 个开发配置 | 原种子、嵌套训练比例、特征分组方案不变 |
| 08 规划 | run_planning | 冻结策略效果与明确假设 → A/B 样本量、经济情景 | 假设全部保存，不称为真实收入 |
| 09 展示 | finalize_run、build_report | 已存结果 → manifest 与唯一 report.html | 不重新训练、不重新计算统计量 |


## 输入检查三层

1. Preparation 同一次扫描检查列顺序、数值、二元标签、缺失/Inf 与生成的来源行号。结构错误停止；逻辑异常和重复内容按原规则记录，不能偷偷删行。
2. 缓存恢复检查矩阵维度、标签、来源行号。划分检查行号非空唯一、互斥、覆盖与比例，保存匿名特征重叠诊断。
3. 模型与指标在训练、评估入口完成必要检查，重点验证长度、列顺序及拟合状态；已检查的数据不在内部重复校验。

`input_checks.py` 只验证，不改变 dtype 或修改数组。模型接收 NaN 时由训练内插补处理，原始 CSV 入口保留有限值检查。

## 一次扫描与独立缓存

`preparation.prepare_sample()` 固定准备建模样本、全量审计和诊断输入。外部没有任务开关或两种参数传入方式；只需传入主流程的参数对象。各自缓存缺失时，由底层 `scan_raw_data()` 的内部任务标记决定计算范围，多个任务共用一次 CSV 扫描。

| 函数或类 | 职责 |
|---|---|
| `scan_raw_data()` | 建立读取器和临时资源，执行一次读取循环，分发当前数据块 |
| `validate_chunk()` | 逐块输入校验；不删除非法行 |
| `ModelSampler` | 建模随机优先级蓄水池，最终特征为 float32 |
| `BalancedSampler` | 两臂独立的等量 C2ST 抽样，特征保持 float64 |
| `AuditAccumulator` | 全量重复指纹、标签计数、异常计数及稳定的分组矩统计 |
| `DiagnosticsAccumulator` | 全量 SMD/KS、SRM 和 C2ST 样本准备 |
| `finalize_scan()`、`scan_metadata()`、`sample_report()` | 扫描结束校验、结果整理和来源说明 |

每个任务只在自身缓存未命中时进入扫描。Sample 命中、Audit 失效时，只更新 Audit。所有启用任务均命中时不调用扫描函数，也不读取 CSV 表头。所有未命中任务共享同一个 float64 数据块；不会逐任务重读 CSV。原始位置 `row_id`、抽样种子和两组均匀无放回抽样规则不变。建模随机流与两臂 C2ST 随机流互相独立。

全量 KS 需要完整分布。扫描把必要列写入临时文件，之后逐列用内存映射调用诊断计算函数；不会为了计算 KS 再解析原 CSV。诊断和审计同时重算时共享全量特征统计；仅审计重算时使用稳定的流式均值和中心平方和，允许原有近常量特征浮点容差。文件指纹始终按块计算，用于来源与扫描前后完整性核验；指纹读取不等于再次解析 CSV。

```text
artifacts/cache/
  sample_1000000_seed42.npz
  sample_1000000_seed42.json
  full_audit_<source_sha256>.json
  c2st_sample_n20000_seed42.npz
  c2st_sample_n20000_seed42.json
```

建模缓存依赖来源、规模、种子及抽样规则，沿用原文件名和报告结构。说明与 NPZ 内容都校验通过才算命中；文件损坏、缺字段、形状或数据异常会在扫描计划形成前判为未命中，与其他待计算任务共用一次扫描重建样本和说明。Audit 依赖来源和统计字段完整性；Diagnostics 依赖来源、C2ST 规模、种子、统计字段和必要数组校验。缓存没有规则版本字段。诊断 NPZ 保存 float64 样本及平衡统计，避免 JSON 丢失无限 SMD 或特征精度。置换次数不影响诊断输入缓存；真实 C2ST 拟合仍由 `run_diagnostics()` 执行。

`prepare_sample()` 返回建模样本、全量审计、来源指纹和诊断字典。`task_status` 仅记录各任务是重新计算还是读取缓存。`split_sample()` 保留 60/20/20 分层划分和来源行互斥检查。

## 模型与评价

`make_classifier_model()` 创建使用二元结果标签的分类器，`make_regression_model()` 创建拟合连续伪效应标签的回归器，`s_learner()`、`t_learner()`、`x_learner()` 各自实现完整的拟合和预测。`linear` 是 X-Learner 第二阶段的可选线性回归器，当前正式候选 `x_xgboost` 使用 XGBoost 回归。四个固定候选在 `training_validation.py` 中逐一调用。分类器参数统一通过 `outcome_params` 传入，第二阶段通过 `effect_params` 传入，不再接受公共参数覆盖层。

X-Learner 保留原式 D0 = mu1(X0) − Y0、D1 = Y1 − mu0(X1)，再按 e × tau0 + (1−e) × tau1 加权。以 X 为例，`x_learner(X_val, X_train=X_train, t=t_train, y=y_train)` 返回拟合模型和验证预测。`x_learner(X_test, model_bundle=result["model_bundle"])` 复用模型进行预测。三个函数均返回 `model_bundle`、`mu0`、`mu1`、`cate` 及 `fit_seconds`。X 的 `mu0`、`mu1` 来自第一阶段，`cate` 来自两组效应回归的加权预测。

`diagnostics.run_diagnostics()` 在诊断阶段调用一次 `diagnose_overlap()`，仅对 Train 的处理前特征做逐列分位数分箱。默认最多 10 箱，重复分位点合并、缺失值单列；每箱任一臂少于 20 人时警告，这只是经验阈值。结果写入 `diagnostics/overlap_positivity.json` 和 `diagnostics/overlap_strata.csv`，与检查划分重复特征的 `data_preparation/feature_overlap.json` 不同。诊断结果独立保存，不传入模型、不计算训练数据签名；模型入口仍检查整体两臂非空。Stability 对每个实际训练子集检查一次，不能复用不同样本的结论。

当前流程没有核实原始实验每行的设计分配概率，诊断默认不填预期值，只报告 Train 观察比例；单独调用可传参考比例及其来源。`checked_no_major_warning` 只代表预设单变量分箱未发现主要警告，不能证明高维联合正值性、无混杂、一致性或无干扰。局部缺臂也可能是有限样本波动。Outcome 单类别使用 sklearn `DummyClassifier` 保持原有常数概率行为，并在模型元数据中记录两臂正负例数，这不是因果不可识别的判定。

模型字典用于本地保存和冻结 Test 预测，直接传给 Learner 的 `model_bundle` 参数。不提供外部批量打分接口。

HTML Diagnostics 会嵌入已保存的 C2ST 置换 AUC 直方图，展示真实标签的 AUC、原始 p、Holm p 和 0.5 参考线，不重新执行置换。旧聚合快照若没有 Overlap 文件，展示 `not_checked`，不能伪造通过状态。

metrics.py 保留完整排序前缀、原点、传统 Qini 的仅对照组零值节点、种子控制的平分处理与配对臂分层 Bootstrap。HT 与 plug-in 口径分开标明，显示曲线抽点不能用来重新积分。

ATE 的逐行和 Audit 两条路径共用 `_ate_summary_from_counts()`。经用户确认，区间统一为原逐行样本方差；p 值仍为 pooled-null 双侧检验，并保留极小概率的对数与文本表示。全量 SE/CI 相对历史 Audit 口径略变，不更改历史文件。当前估计按分配，不能称为 ATT。

Stability 直接读取当前建模样本并排除原 Test 行号，不再比较缓存说明、阶段清单或文件哈希。`protocol.json` 保存实验的种子、训练比例和统计口径；不再进行协议指纹自校验。行号和特征分组的实际交集检查保留，以防直接数据泄漏。

## 运行与展示

完整运行命令与环境说明见根 README。Notebook 固定读取 `reports/published/conversion_1000000_seed42_refactored` 的公开聚合结果，不再在本地缓存和公开结果间自动切换。

```bash
python scripts/execute_notebooks.py  # 仅用公开聚合快照验收
```

项目不再保留 diagnostic refresh、Demo、SQL Audit、独立 metrics extension 或独立 Stability 执行入口。历史结果原样保留，清理只针对已迁移的源码与正式展示入口。

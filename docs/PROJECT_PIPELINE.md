# 项目流程与阅读顺序

学习目标是看清每一步的输入、作用与输出。先读 run_pipeline.py 的 run，再进入对应模块。语法只在妨碍理解流程时展开。

| 阶段 | 调用 | 输入与输出 | 核心边界 |
|---|---|---|---|
| 01 配置 | config、safe_output | 参数 → 全部输出目录 | 拒绝写入原始 data，拒绝覆盖已有实验 |
| 02 准备 | prepare_sample、split_sample | 原始 CSV/缓存 → audit、样本、60/20/20 划分 | 最多一次 CSV 分块扫描；三任务独立缓存 |
| 03 诊断 | run_diagnostics | 扫描统计、C2ST 样本、Train/Validation → SRM、SMD/KS、C2ST、Learnability | diagnostics 不读原始 CSV、不重新审计 |
| 04 全量效果 | ate_summary_from_audit | 两组人数及正例数 → 全量 ATE/ITT | 全量结果只提供背景，不选模型 |
| 05 训练/验证 | fit_and_select_models | Train 拟合，Validation 比较 → 模型与 selection.json | 固定四候选，按主预算 HT 增量选策略 |
| 06 Test | evaluate_test_set | 冻结模型与 Test → CATE 排序、策略值、Bootstrap、校准、Test ATE | 不按 Test 改选；统一 ATE 表 |
| 07 稳定性 | run_stability | 缓存减去原 Test → 12 个开发配置 | 原种子、嵌套训练比例、特征分组方案不变 |
| 08 规划 | run_planning | 冻结策略效果与明确假设 → A/B 样本量、经济情景 | 假设全部保存，不称为真实收入 |
| 09 展示 | finalize_run、build_report | 已存结果 → manifest 与唯一 report.html | 不重新训练、不重新计算统计量 |

Score Policy 独立运行。它面对未来仅有 X 的数据，因此不属于本轮离线 Test 指标计算。未提供外部数据时，报告明确显示未执行，不生成虚构的打分结果。

## 输入检查三层

1. Preparation 同一次扫描检查列顺序、数值、二元标签、缺失/Inf 与生成的来源行号。结构错误停止；逻辑异常和重复内容按原规则记录，不能偷偷删行。
2. 缓存恢复检查矩阵维度、标签、来源行号。划分检查行号非空唯一、互斥、覆盖与比例，保存匿名特征重叠诊断。
3. 模型、指标和打分接口使用公共轻量检查，并保留各自的长度、列顺序、拟合状态、有限分数约束。不重复做全量审计。

`input_checks.py` 只验证，不改变 dtype 或修改数组。模型接收 NaN 时由训练内插补处理，原始 CSV 入口和新人群打分沿用原先的有限值要求。

## 一次扫描与独立缓存

`preparation.prepare_sample()` 负责三个独立开关、缓存检查、读取和写入，以及返回值组装。`sample_enabled`、`audit_enabled`、`diagnostics_enabled` 可以任意组合，默认全部开启；全部关闭会明确报错。开关可作为关键字参数传入，也可放在 `args` 的同名属性中，显式参数优先。关闭的任务不读取缓存、不计算，不要求提供其专用参数。

```python
sample, audit, source, prepared = prepare_sample(
    args, sample_enabled=True, audit_enabled=True,
    diagnostics_enabled=False, chunk_size=250_000,
)
```

只新增 `raw_scan.py`，所有扫描任务都集中于该文件。`scan_raw_data()` 与调度入口一样，使用显式的 `sample_enabled`、`audit_enabled`、`diagnostics_enabled`；`sample_size=None` 不再关闭 Sample。Diagnostics 同时负责全量统计及 C2ST，`per_arm` 仅指定每臂规模（默认 10,000），不作为任务开关：

| 函数或类 | 职责 |
|---|---|
| `scan_raw_data()` | 建立读取器和临时资源，执行一次读取循环，分发当前数据块 |
| `validate_scan_options()`、`validate_chunk()` | 参数及逐块输入校验；不删除非法行 |
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

建模缓存依赖来源、规模、种子及抽样规则，沿用原文件名和报告结构。说明与 NPZ 内容都校验通过才算命中；文件损坏、缺字段、形状或数据异常会在扫描计划形成前判为未命中，与其他待计算任务共用一次扫描重建样本和说明。Audit 依赖来源和统计字段完整性；Diagnostics 依赖来源、C2ST 规模、种子、统计字段与载荷指纹。审计和诊断均没有规则版本字段。诊断 NPZ 保存 float64 样本及平衡统计，避免 JSON 丢失无限 SMD 或特征精度。置换次数不影响诊断输入缓存；真实 C2ST 拟合仍由 `run_diagnostics()` 执行。

`prepare_sample()` 的四元组接口和实验结果目录不变，第四项不重复持有建模大表。未启用任务返回 `None`。返回的 Metadata 描述本次请求的三个开关，`task_status` 分别记录 `computed`、`cache` 或 `disabled`，不会沿用诊断缓存生成时的开关；仅使用建模缓存时，来源说明中未知的全量两臂人数为 `None`，不以样本人数冒充全量人数。`split_sample()` 保留原有分层划分及互斥检查。不存在旧扫描入口或缓存迁移分支。

## 模型与评价

`make_classifier_model()` 创建使用二元结果标签的分类器，`make_regression_model()` 创建拟合连续伪效应标签的回归器，`make_uplift_model()` 组合 S/T/X。`linear` 是 X-Learner 第二阶段的可选线性回归器，当前正式候选 `x_xgboost` 使用 XGBoost 回归。候选名单定义在 `training_validation.py` 中。

X-Learner 保留原式 D0 = mu1(X0) − Y0、D1 = Y1 − mu0(X1)，再按 e × tau0 + (1−e) × tau1 加权。predict_outcomes 返回第一阶段概率；predict 返回加权 CATE。

metrics.py 保留完整排序前缀、原点、传统 Qini 的仅对照组零值节点、种子控制的平分处理与配对臂分层 Bootstrap。HT 与 plug-in 口径分开标明，显示曲线抽点不能用来重新积分。

ATE 的逐行和 Audit 两条路径共用 `_ate_summary_from_counts()`。经用户确认，区间统一为原逐行样本方差；p 值仍为 pooled-null 双侧检验，并保留极小概率的对数与文本表示。全量 SE/CI 相对历史 Audit 口径略变，不更改历史文件。当前估计按分配，不能称为 ATT。

Stability 运行期间核验输入指纹与冻结 protocol。嵌入主流程时读取 status=models_frozen 的阶段 manifest；最终报告完成后主 manifest 才标记 complete。因此 protocol 中 base_manifest 指纹记录的是稳定性启动时的阶段快照，不能误读为最终 manifest 的指纹。

## 运行与展示

完整运行命令与环境说明见根 README。Notebook 默认优先读取 `artifacts/conversion_1000000_seed42_refactored`，没有本地结果时读取同名公开快照。可以设置 `CRITEO_RUN` 指定其他实验名，或 `CRITEO_RESULTS_MODE=published` 强制只读快照。

```bash
python scripts/execute_notebooks.py                 # 仅用公开快照验收
python scripts/execute_notebooks.py --local-results # 本地优先
```

项目不再保留 diagnostic refresh、Demo、SQL Audit、独立 metrics extension 或独立 Stability 执行入口。历史结果原样保留，清理只针对已迁移的源码与正式展示入口。

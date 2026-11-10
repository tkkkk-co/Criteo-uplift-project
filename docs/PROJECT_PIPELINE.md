# 项目流程与阅读顺序

学习目标是看清每一步的输入、作用与输出。先读 run_pipeline.py 的 run，再进入对应模块。语法只在妨碍理解流程时展开。

| 阶段 | 调用 | 输入与输出 | 核心边界 |
|---|---|---|---|
| 01 配置 | config、safe_output | 参数 → 全部输出目录 | 拒绝写入原始 data，拒绝覆盖已有实验 |
| 02 准备 | prepare_sample、split_sample | 原始 CSV/缓存 → audit、样本、60/20/20 划分 | 一次 CSV 分块扫描；审计与样本独立缓存 |
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

`preparation._scan_raw_data()` 保留两个独立随机流。建模样本是均匀无放回的随机优先级蓄水池；C2ST 是两组分别均匀抽样，每组默认 10,000 行。建模抽样大小不会改变诊断抽样。

全量 KS 需要完整分布。扫描把必要列写入临时文件，之后用内存映射调用 diagnostics 的纯计算函数；不会为了计算 KS 再解析原 CSV。全量原始统计保持 float64，建模缓存使用 float32。文件内容指纹按块计算，用于来源与扫描前后完整性核验；指纹读取不等于再次解析 CSV。

```text
artifacts/cache/
  sample_1000000_seed42.npz
  sample_1000000_seed42.json
  full_audit_<source_sha256>.json
```

样本 metadata 记录来源、原始/抽样规模、种子、方法、表示和版本，不内嵌全量审计。Sample 与 Audit 版本在 preparation.py 中独立维护。一方失效不自动使另一方失效；为本轮全量诊断仍执行一次统一扫描。

## 模型与评价

`make_outcome_model()` 创建分类器，`make_effect_model()` 创建连续伪效应回归器，`make_uplift_model()` 组合 S/T/X。模型算法不依赖正式候选名单；名单只在 training_validation.py 中定义。

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

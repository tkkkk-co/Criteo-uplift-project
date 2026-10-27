# Notebook 与结果文件导航

> 本文只解释本项目结果的来源、如何阅读，以及哪些结论可以/不可以从现有证据得出。C2ST、ATE、AUUC/Qini、置信区间等通用理论请查 DSDAHANDBOOK；本仓库只保留实现映射、当前项目口径和结果解释。

## 1. 先区分三种东西

1. **计算代码**：在 criteo_uplift/ 中计算指标、训练模型或执行诊断。
2. **结果产物**：保存为 CSV / JSON / NPZ / 模型文件 / 图片。它们是计算的持久化结果。
3. **Notebook**：读取已保存的结果，组织表格、图和项目结论。大多数 Notebook 单元不会重新计算正式模型或指标。

因此，看到 Notebook 里的 pd.read_csv(RUN / "diagnostics" / "c2st.csv")，它的作用是读取已生成的结果，不是计算 C2ST。应先查本表的“计算函数 / 写入函数”，再回到 Notebook 看结果解释。

## 2. Notebook 01–06 导航

| Notebook | 业务问题 | 主要调用链 / 计算位置 | 关键结果文件 | 阅读时重点 |
|---|---|---|---|---|
| 01_data_understanding.ipynb | 数据来源、质量、抽样和 split 是否可复核？ | run_pipeline.py::run → preparation.py::prepare_sample → diagnostics.py::scan_raw_data; data.py::split_sample | audit.json、ate_full.csv、sample_group_summary.csv、sample_feature_summary.csv、split_summary.csv、feature_overlap.json、manifest.json | 区分全量审计与 100 万行建模样本；特征重叠只代表匿名特征向量重叠，不是识别用户 |
| 02_sanity_and_ate.ipynb | 公开数据计数、单变量平衡、联合可区分性与结果可学习性如何？ | diagnostics.py::scan_raw_data, srm_summary, covariate_balance, c2st, update_experiment_diagnostics; evaluation.py::aggregate_ate_from_audit; training.py::run_learnability_diagnostics | ate_full.csv、srm.json、balance.csv、c2st.csv、c2st_details.json、c2st_null.csv、learnability.csv | C2ST 用两组等量抽样；不显著不等于联合分布相同，也不能证明随机化 |
| 03_uplift_models.ipynb | 候选模型怎样产生 CATE 分数，validation 如何冻结策略？ | models.py::make_model; training.py::fit_and_select_models; evaluation.py::ranking_metrics | selection.json、validation_policies.csv、validation_ranking.csv、模型 .joblib、calibration_*.csv | 选型只看 validation；不要因为 test 上另一个点估计更高就重新选模型 |
| 04_policy_evaluation.ipynb | 冻结策略在 test 上的预算价值和不确定性如何？ | holdout_evaluation.py::evaluate_test_set; evaluation.py::evaluate_policies, ranking_confidence_intervals, uplift_curve | test_policies.csv、test_ranking.csv、curves.csv、ranking_intervals.csv、ate_test.json、predictions.npz、figures/policy.png | 主预算为 20%；每十万人增量以全部待选用户为分母；区间有明确条件限制 |
| 05_metrics_worked_example.ipynb | 指标实现能否在一个小型手算例子中逐项复核？ | evaluation.py::uplift_curve, ranking_metrics；Notebook 内用 Fraction 独立复算梯形面积 | 八行手算数据、精确分数表、artifacts/extension_figures/worked_metrics.png、ht_vs_plugin.png | 这是人为构造的教学/单元核验例子，不是 Criteo 真实结果 |
| 06_robustness_sql_and_decisions.ipynb | 开发期稳定性、SQL 一致性和业务假设情景说明什么？ | stability.py::run_stability; sql_audit.py::run_sql_audit; planning.py::plan_policy_ab, cost_value_scenario | stability_v1/、metrics_v2/、sql_audit_v1/ 下的 report.json、指标表、审计表和 queries.sql | 这些是扩展分析；不能包装成新的独立测试集，也不能把假设成本/价值当成真实财务数据 |

## 3. 主实验结果文件速查

下表用文件名定位产物；新实验按下述子目录存放。当前实验目录例如 artifacts/conversion_1000000_seed42/；Notebook 的 RUN 变量会按 CRITEO_RESULTS_MODE 选择本地结果或公开快照。

当前主实验目录按阶段组织：

- `data_preparation/`：audit.json、sample_report.json、split_summary.csv、三份 row_ids.csv、feature_overlap.json，以及公开样本摘要。
- `diagnostics/`：ate_full.csv、srm.json、balance.csv、c2st*.csv/json、learnability.csv。
- `models/`：模型 .joblib、selection.json、validation_policies.csv、validation_ranking.csv。
- `evaluation/`：test_policies.csv、test_ranking.csv、predictions.npz、curves.csv、校准表、置信区间和 ate_test.json。
- 顶层保留 manifest.json、report.html 和 figures/，方便打开总报告。

读写均显式指定对应子目录，例如 `RUN / "data_preparation" / "split_summary.csv"`。主流程先统一创建子目录，再把具体目录传给各保存步骤；项目不再按文件名推断目录或兼容旧名称。汇总表 `split_summary.csv` 记录人数与正例数，不包含划分后的完整样本。

### 数据与来源

| 文件 | 由谁计算/写入 | 用来回答什么 |
|---|---|---|
| manifest.json | pipeline_io.py::write_run_manifest，最后由 run_pipeline.py::finalize_run 更新状态 | 参数、seed、来源指纹、依赖版本、选型记录和运行状态 |
| audit.json | diagnostics.py::scan_raw_data 产生；pipeline_io.py::write_audit_artifacts 保存 | 仅全量数据计数、质量审计和特征统计，不含抽样信息 |
| sample_report.json | diagnostics.py::scan_raw_data 生成，prepare_sample 管理缓存；run_pipeline.py 保存 | 当前抽样请求规模、实际行数、seed、方法、精度及样本组别人数；直接读取独立说明文件 |
| ate_full.csv | evaluation.py::aggregate_ate_from_audit 计算；pipeline_io.py::write_audit_artifacts 保存 | 全量公开数据上的描述性差均值估计 |
| sample_group_summary.csv、sample_feature_summary.csv | 公开快照导出流程 | 仅在没有逐行数据时展示历史聚合样本摘要 |
| split_summary.csv、train_row_ids.csv、validation_row_ids.csv、test_row_ids.csv | pipeline_io.py::write_split_artifacts | split 规模、分组计数和原始行号 |
| feature_overlap.json | pipeline_io.py::write_feature_overlap | validation/test 特征向量与 train 的精确重合比例；不是用户身份识别 |

### 实验可信度与可学习性

| 文件 | 由谁计算/写入 | 用来回答什么 |
|---|---|---|
| srm.json | diagnostics.py::srm_summary / update_experiment_diagnostics | 公开数据计数与 85% 外部参考比例的一致性检查 |
| balance.csv、figures/balance.png | diagnostics.py::covariate_balance / update_experiment_diagnostics | 各特征的均值差、SMD 和 KS 统计量 |
| c2st.csv | diagnostics.py::c2st / update_experiment_diagnostics | Logistic 与 HGB 的留出 ROC AUC、原始和 Holm 校正 p 值 |
| c2st_null.csv | diagnostics.py::c2st / update_experiment_diagnostics | 每个模型每次置换的 AUC 零分布 |
| c2st_details.json | diagnostics.py::update_experiment_diagnostics | 抽样规模、内部划分、seed、置换次数、p 值分辨率、来源指纹和解释边界 |
| learnability.csv | training.py::run_learnability_diagnostics 调用 diagnostics.py::learnability | 处理前特征对 visit/conversion 的结果预测能力；不是 CATE 可预测性的证明 |

### 模型、选型与 test 评价

| 文件 | 由谁计算/写入 | 用来回答什么 |
|---|---|---|
| *.joblib | training.py::fit_and_select_models | 已拟合的候选模型 |
| validation_policies.csv | training.py::fit_and_select_models | validation 主预算下各策略的 HT 增量指标 |
| validation_ranking.csv | training.py::fit_and_select_models | validation 上的排序指标 |
| selection.json | training.py::fit_and_select_models | 最终策略和 uplift 候选的冻结选择、选择依据 |
| test_policies.csv | holdout_evaluation.py::evaluate_test_set | 各预算下的策略估计值、区间及相对基线差异 |
| test_ranking.csv、curves.csv | holdout_evaluation.py::evaluate_test_set | 排序指标和绘图用曲线；曲线文件为绘图而压缩，指标用完整曲线计算 |
| ranking_intervals.csv | evaluation.py::ranking_confidence_intervals | AUUC/Qini 等排序指标及策略差值的逐项区间 |
| calibration_*.csv | holdout_evaluation.py::evaluate_test_set | 预测分数分箱后的校准诊断 |
| ate_test.json | evaluation.py::ate_summary | test 集上的描述性 ATE 结果 |
| predictions.npz | holdout_evaluation.py::evaluate_test_set | 冻结模型的逐行分数和行标识；公开快照不包含该文件 |
| report.html、figures/ | reporting.py::build_report | 将已经保存的产物渲染为可浏览报告和图表 |

## 4. 当前快照的结果应怎样解释

以下数字是仓库当前公开快照中已经保存的历史实验结果，不是本次重新运行得到的结果，也不是线上广告实验收益。

- **主预算**：20% 触达比例；“每十万人增量”以全部待选人群为分母。这里是“恰好选择 20%”的离线预算约束，不是人民币预算、竞价优化或收入最大化问题；当前假设每次触达成本与每次转化价值相同。
- **全量审计与抽样**：当前记录的公开来源为 13,979,592 行、16 个原始字段；全字段重复行指纹的多余行数为 1,259,545（约 9.01%），全量有限数值检查未发现非有限值。指纹重复不等于重复用户，也不应据此自动删除。首轮建模使用 seed=42 的 1,000,000 行等概率无放回样本，按 treatment × conversion 分层为 60/20/20；这与全量审计不是同一个样本范围。
- **冻结策略**：当前首轮 validation 选出的最终策略是 response propensity。公开快照中其 20% 预算增量点估计约为每 10 万待选人 117.23；相对解析随机策略的差值约为 92.61，配对 95% CI 约为 [50.02, 132.15]。
- **模型比较与事件稀疏**：S/T/X learner 均已实现，但当前实验没有证明复杂 X-Learner 优于强 response 基线。首轮 100 万样本的 600k 训练集里，control 约 89,754 行、conversion 事件仅 164 个；test control 的 conversion 事件为 55 个。开发期 3 个 seed 的完整训练规模平均主指标约为 response 127.51、S-Logistic 126.67、X-Learner 99.03（每十万待选人）；对应跨 seed 样本标准差约 4.54、5.09、5.93，标准差不是置信区间。事件稀疏提示估计可能不稳定，但不能据此断言扩样一定有效或无效。
- **ATE**：ate_full.csv 来自公开数据的全量汇总，是描述性差均值，不等于真实线上广告的增量收益。
- **SRM**：85% 是数据集公开资料提供的外部参考比例。本检查比较发布数据计数与参考比例，不能证明原始线上实验通过 SRM。
- **SMD / KS**：单变量统计不能证明联合分布一致；大样本下 p 值可能非常小，应同时看差异量级和数据生成背景。
- **C2ST**：每组从原始来源等概率无放回抽样，默认 10,000 control + 10,000 treatment；内部按 treatment 分层做 70/30 train/holdout；Logistic 和 HGB 的标签在两个集合分别置换并完整重拟合，p 值再做 Holm 校正。AUC 高于 0.5 表示分类器在该设计下有区分能力；不显著不能证明联合分布完全相同或随机化成立。
- **置信区间**：当前配对 bootstrap 区间条件于已拟合模型、冻结分数、固定的分配概率估计和观测臂样本量；不包含重训、模型选型、分配概率估计误差或未知用户依赖。
- **稳定性与记录敏感性**：扩展分析排除旧测试行，但分析者此前已看过旧测试结果，因此它是后续开发诊断，不是全新独立确认。另有一份 2026-10-20 的本机记录核验与特征组隔离归档，位于被 Git 忽略的 artifacts/audits/record_duplicates_20261020/，不包含在公开仓库；没有用户 ID，不能把记录/特征匹配解释为真实用户身份。
- **SQL**：用于核对 join 完整性、分组汇总、预算排名和 Python 评价口径；row_id 是原始行号，不是真实用户 ID。
- **成本/价值规划**：planning.py 的价值和成本是显式假设情景，不能写成 Criteo 的真实收入、利润或广告成本。

## 5. 如何刷新 Notebook 输出

Notebook 的结果展示单元应该只负责读结果、格式化和解释；重训或重新诊断必须通过明确的 pipeline 命令启动。

~~~bash
# 默认强制使用公开快照，适合只看项目结果
python scripts/execute_notebooks.py

# 优先使用本地 artifacts（若存在）
python scripts/execute_notebooks.py --local-results

# 只更新现有 run 的 SRM / SMD-KS / C2ST 诊断，不重训模型
python run_pipeline.py --diagnostics-only --output artifacts/conversion_1000000_seed42
~~~

如果编辑了 Notebook 中的结果展示代码，需要重新执行 Notebook 以更新缓存输出；不要把旧的 HTML/图片输出当作新代码已成功运行的证据。

## 6. 项目结果的对外表达边界

可以说：在当前公开基准、既定 20% 预算和现有评估口径下，response propensity 是首轮 validation 冻结选择的最终策略；已经完成了候选 uplift 模型、离线预算评价、配对区间、开发期稳定性与 SQL 核对。

不应说：已经证明线上收入提升、识别了真实用户级重复、证明随机化完全成立、证明扩样一定改善 X-Learner，或用假设成本情景声称真实 ROI。真实部署前仍需要预先冻结策略、候选池、转化/归因窗口、主指标、成本护栏、频控/负反馈指标和停止规则，并进行独立线上随机实验。随机单元应是可信的用户或实体；各策略臂使用相同预算约束，比较全部入组人群的结果，而不是只比较各自被选中的人群。

# 结果阅读指南

正式结果目录是 `artifacts/conversion_1000000_seed42_refactored/`；发布副本在 `reports/published/conversion_1000000_seed42_refactored/`。Notebook 直接读取公开目录中的这次实验。

| 文件/目录 | 内容 | 阅读要点 |
|---|---|---|
| manifest.json | 来源指纹、参数、软件版本、冻结选择、完成状态 | complete 表示主流程与报告均完成 |
| data_preparation/audit.json | 全量分组计数、逻辑异常、重复指纹、特征统计 | 不把内容重复直接当重复用户，不擅自删行 |
| data_preparation/sample_report.json | 百万样本的来源、抽样方法、随机种子、构成 | 与全量审计分开；缓存 metadata 不承担模型配置 |
| data_preparation/split_summary.csv | 三集合分组人数与事件数 | Train/Validation/Test = 60/20/20 |
| data_preparation/*_row_ids.csv | 原始 CSV 位置行号 | 非用户 ID，不发布逐行列表 |
| data_preparation/feature_overlap.json | 行号互斥/覆盖及匿名特征重叠 | 特征相同不代表同一用户 |
| diagnostics/srm.json | 相对外部 85% 参考分流比例的计数检验 | 不证明原始线上实验随机化 |
| diagnostics/balance.csv | 全量 float64 的 SMD、KS | 同时看效应量与统计显著性 |
| diagnostics/c2st*.csv/json | 两组各 1 万行的 C2ST、置换零分布、来源 | 两个模型完整重拟合置换，Holm 校正；不能证明分布相同 |
| diagnostics/learnability.csv | Train 拟合、Validation 评价的结果可预测性 | 不选择正式 uplift 策略 |
| diagnostics/overlap_positivity.json、overlap_strata.csv | Train 的逐特征分箱与两臂支持 | 新 Pipeline 在 Diagnostics 阶段保存；历史快照如缺失则按原状态展示 |
| models/*.joblib | 四种候选的拟合模型 | 只在本地用于冻结模型的 Test 预测 |
| models/validation_*.csv | Validation 排序与主预算策略值 | 选择依据仅来自 Validation |
| models/selection.json | 主预算、候选、冻结策略与最佳 uplift 候选 | 最佳 uplift 不一定是最终策略 |
| evaluation/ate_summary.csv | full_dataset/test × conversion/visit 的统一 ATE | 同一套统计字段；ITT 分配口径，不是 ATT |
| evaluation/test_policies.csv | 各预算 HT 增量与配对区间 | 每十万人以全部可选对象为分母 |
| evaluation/test_ranking.csv | AUUC、Qini、评分分布摘要 | 区分 HT 与 plug-in，不混用面积标度 |
| evaluation/ranking_intervals.csv | 固定评分的配对 Bootstrap 区间 | 不包括重新训练或选型的不确定性 |
| evaluation/curves.csv | 绘图用抽点曲线 | 指标计算使用完整前缀，不从抽点图重新积分 |
| evaluation/calibration_*.csv | CATE 分箱预测与观察差值 | 缺组时保留未定义统计，不捏造结果 |
| evaluation/predictions.npz | Test 原始行号、标签和冻结分数 | 本地保留，不发布 |
| stability/ | protocol、逐配置指标、隔离审计、聚合和选择频率 | 排除原 Test；重复开发结果不是独立 Test 结论 |
| business/assumptions.json | 全部业务假设与冻结策略来源 | 不把假设成本、价值写成观察事实 |
| business/hypothetical_*.csv | A/B 所需样本量与经济敏感性情景 | 价值区间条件于固定经济假设与可迁移效果 |
| figures/、report.html | 结果图与完整展示报告 | HTML 图片内嵌，可离线打开 |

固定评分 Bootstrap 在两臂内分层，同一次重抽样用于各策略，每次重新选 Top-k。相对 response 的配对区间包含零，应说“证据不足”，不能据此宣布没有差异，也不能按点估计重新选择策略。

公开样本上的分配效果不等于真实广告曝光效果。匿名特征、公开数据筛选机制、稀疏转化、控制组规模、未知用户依赖与效果可迁移性仍限制外推。线上验证需要新的预先冻结 A/B。

正式实验的统计表保留原数值。轻量化只清理代码接口和展示中的废弃功能；逐批验收见 [重构记录](maintenance/FINAL_REFACTOR.md)。

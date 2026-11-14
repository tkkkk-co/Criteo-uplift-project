# V1.0 交付说明

- 正式策略为验证集冻结的 Response-XGBoost，最佳 Uplift 候选为 S-XGBoost。500 万开发阶段冻结 T-Logistic。
- 两份离线 HTML 均有七章研究正文与十个主题附录。正文提供主指标与业务解读，附录集中呈现九策略指标、全部预算、配对区间、实验诊断、冻结记录和经济情景。
- 新增已保存评分的重叠分析。全量正式测试集 Overlap@20% 为 96.80%，Jaccard 为 0.9379，Spearman 为 0.9373。开发阶段 T-Logistic 与 Response-XGBoost 的对应结果为 85.37%、0.7447、0.8070。
- 每份报告旁有 `strategy_overlap.json` 和 `report_provenance.json`，记录总数、估计规则及来源指纹。正式结论仍以既定的验证选型与留出评价为准。
- 项目进入 V1.0 冻结状态。后续业务验证方向为目标人群中的用户级随机策略实验。

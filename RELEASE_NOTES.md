# 交付说明

- 正式策略为验证集冻结的 Response-XGBoost，最佳 Uplift 候选为 S-XGBoost。500 万开发阶段冻结 T-Logistic。
- 两份离线 HTML 均有七章研究正文与十个主题附录。正文提供主指标与业务解读，附录集中呈现九策略指标、全部预算、配对区间、实验诊断、冻结记录和经济情景。
- 全量正式测试集 Overlap@20% 为 96.80%，Jaccard 为 0.9379，Spearman 为 0.9373。开发阶段 T-Logistic 与 Response-XGBoost 的对应结果为 85.37%、0.7447、0.8070。
- 正式报告展示 Top 20% 的九策略 HT Gain 与 Uplift Gain 排名，以及冻结策略五档扩量的边际收益和配对区间。扩量区间与关键策略比较均使用 2,000 次分层配对 Bootstrap。
- 每份报告旁有 `strategy_overlap.json` 和 `report_provenance.json`，记录总数、估计规则及来源指纹。正式结论仍以既定的验证选型与留出评价为准。
- 线上验证方案明确了切换策略所需的转化改善、成本和护栏条件。

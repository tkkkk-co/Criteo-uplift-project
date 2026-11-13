# 结果阅读指南

本次新结果写入 `--output` 指定的独立目录。已发布的百万行实验是此前选型规则下的历史快照。新的代码和 Notebook 读取本轮评价生成的表。

| 文件 | 内容 |
|---|---|
| `manifest.json` | 数据来源、运行参数、软件版本和完成状态 |
| `models/selection.json` | 唯一主指标、预设覆盖率、Bootstrap 次数和冻结选择 |
| `models/validation_metrics.csv` | 五个策略在预设覆盖率的三种 Gain 与完整曲线的两项面积 |
| `models/validation_bootstrap.csv` | 主指标点估计、均值、点态区间和有效重复数 |
| `models/validation_pairwise.csv` | 主指标逐轮配对差值、区间和有效配对数 |
| `models/validation_selection_frequency.csv` | 每个候选在全部策略均有定义的重复中获胜的频率 |
| `evaluation/test_metrics.csv` | 五个策略在各预算的三种 Gain 与两项面积 |
| `evaluation/test_bootstrap.csv` | Test 主指标点估计、均值及区间，Gain 按预算组织，面积指标的预算列为空 |
| `evaluation/test_pairwise.csv` | Test 主指标的配对差值，包含冻结策略、Propensity 和 Random 参考 |
| `evaluation/curves.csv` | 三种 Gain 的绘图节点，每策略最多 501 行 |
| `evaluation/ate_summary.csv` | 全量及 Test 上 conversion、visit 的独立 ATE |
| `evaluation/calibration_*.csv` | CATE 分箱预测与观察差值 |
| `evaluation/predictions.npz` | 本地保存的 Test 行号、真实标签和固定评分 |
| `stability/` | 排除原 Test 后的开发期实验，按相同主指标 Bootstrap 均值选型 |
| `business/` | 冻结策略的 A/B 样本量规划和假设经济情景 |
| `report.html` | 可离线阅读的完整报告 |

`evaluate_ranking(y, t, score_dict, e, *, budgets, primary_metric, primary_budget, n_bootstrap, seed, include_curves)` 一次接收五种评分并返回 `metrics`、`curves`、`bootstrap`。最后一项包含 `summary`、`pairwise`、`selection_frequency`。关闭 Bootstrap 时返回 `None`。

默认主指标为 20% HT Gain。通过 `--primary-metric` 可选择 `auuc`、`qini_score`、`gain_plugin`、`qini_traditional` 或 `gain_ht`。Validation 用主指标 Bootstrap 均值冻结策略。面积选型使用完整排序前缀，`--primary-budget` 仍可指定投放情景。Test Gain 主指标区间按覆盖率逐点计算。

各策略复用同一轮两臂有放回抽样，评分与分配概率固定。单策略区间使用自身有效重复，配对区间使用共同有效重复。缺组前缀保留未定义值。获胜频率仅描述固定评分下的重抽样选择稳定性。Random 是解析参考，Propensity 复用 T-XGBoost 的处理下转化概率。

Gain 按总体人数归一化。HT Gain 乘 100,000 是每十万总体可触达用户的估计增量转化。AUUC 和传统 Qini Score 的表格显示缩放为面积 × 10⁵。边际增量使用同一冻结排序的相邻预算 HT 点估计差值。Planning 仅在主指标是 HT Gain 时展示相应价值区间。覆盖率表示触达人群比例，价值与成本均为情景假设。

使用新运行目录或其聚合导出运行 Notebook：

```bash
CRITEO_RESULTS_DIR=artifacts/my_new_run python scripts/execute_notebooks.py
```

聚合导出使用 `scripts/export_public_results.py`，包含指标和报告，逐行预测与模型仅保留在本地。

# 结果阅读指南

正式结果目录是 `artifacts/conversion_1000000_seed42_refactored/`；发布副本在 `reports/published/conversion_1000000_seed42_refactored/`。Notebook 直接读取公开目录中的这次实验。

本次评价与报告升级单独保存在 `artifacts/conversion_1000000_seed42_evaluation_upgrade/`，打开其中的 `report.html` 查看。它复用原实验的 200,000 条冻结 Test 评分、分配概率、seed=42 和 200 次配对 Bootstrap，仅扩展预算评价与报告，不重新扫描原始 CSV、训练模型或选型。原正式结果、公开快照及 Notebook 指向保持原样。

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
| evaluation/test_policies.csv | 各预算 HT 增量与配对区间 | 升级版含 2%、5%、10%、20%、30%、50%、100% 及配置的主预算；每十万人以全部可选对象为分母 |
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

## 升级报告的评价章节

- **Model Selection**：仍仅按 Validation 的 20% HT Gain 冻结策略；Test 展示点估计、区间和相对基线的配对差值，不重新选择。
- **Ranking Performance**：从 `test_ranking.csv` 读取全部六项统计，从 `ranking_intervals.csv` 读取五项面积指标的区间及对 Propensity 的配对差值。展示 HT Gain、Uplift Gain、Traditional Qini 三类曲线，各参考线连接自身终点。绘图使用 `curves.csv`，不从抽点曲线重新积分。
- **Budget Sensitivity Analysis**：从 `test_policies.csv` 读取冻结策略、Random、Propensity 在 10%、20%、30%、100% 的结果。冻结策略与基线重合时不重复列出；可展开查看全部策略和预算。
- **Marginal Incremental Gain**：`build_report()` 对冻结策略计算 `V(b) - V(a)`，其中 `V(0)=0`，展示 0%–10%、10%–20%、20%–30%、30%–100% 的新增人数与增量。仅为点估计，不相减累计 CI 端点，不推断显著性。
- **Full Coverage Scenario**：展示所有策略共享的 `100000 × ATE_HT` 终点，以及冻结策略在较低覆盖率下与全量的差值。

表头中的面积 `× 10⁵` 仅为显示缩放，不能解释成固定覆盖率下的新增转化次数，不能跨不同面积定义直接比较大小。Contact Fraction 是可触达人群比例，不是货币预算；没有实际成本与收入时不能推断最优商业预算、利润或 ROI。

### 仅从冻结评分重新生成评价报告

在仓库根目录运行下列代码，指定一个**尚不存在**的新目录。已有结果缺少 30% 时必须用 `predictions.npz` 调用 `evaluate_policies()` 补算；报告不会插值伪造这个预算点。该 NPZ 含逐行标签和分数，仅保留在本地。

```python
from pathlib import Path
import json
import shutil
import numpy as np
from criteo_uplift.evaluation import DEFAULT_BUDGETS, evaluate_policies
from criteo_uplift.pipeline_io import write_result
from criteo_uplift.reporting import build_report

source = Path("artifacts/conversion_1000000_seed42_refactored")
out = Path("artifacts/conversion_1000000_seed42_evaluation_review")
shutil.copytree(source, out)  # 目录已存在时停止，避免覆盖结果。
manifest = json.loads((source / "manifest.json").read_text())
selection = json.loads((source / "models/selection.json").read_text())
with np.load(source / "evaluation/predictions.npz", allow_pickle=False) as data:
    scores = {name: data[name] for name in data.files if name not in {"row_id", "y", "t"}}
    policies = evaluate_policies(
        data["y"], data["t"], scores, manifest["assignment_probability"],
        budgets=sorted({*DEFAULT_BUDGETS, selection["primary_budget"]}),
        bootstrap=manifest["arguments"]["bootstrap"], seed=manifest["seed"],
    )
write_result(policies, out / "evaluation/test_policies.csv")
manifest["report_revision"] = f"复用 {source} 的冻结 Test 评分补算预算与报告；未训练或重新选型。"
write_result(manifest, out / "manifest.json")
build_report(out / "manifest.json", out / "report.html")
```

这条路径保留原排序统计、模型、Validation 表与 `selection.json`；只在新目录更新预算表、报告图和 HTML。生成时间及环境不是原模型训练时间及环境，原 manifest 的实验配置继续描述来源实验。

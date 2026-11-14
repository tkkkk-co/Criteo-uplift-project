# Criteo Uplift Project V1.0

固定覆盖 20% 的合资格用户时，怎样分配广告资格才能带来更多增量转化？本项目用 Criteo Uplift v2.1 的 13,979,592 条公开记录，比较六种 CATE 模型、两种转化倾向模型和同覆盖率的随机策略。

- [全量正式实验报告](reports/final/report.html)
- [500 万开发实验报告](reports/development/report.html)
- [V1.0 交付说明](RELEASE_NOTES.md)

下载 HTML 即可离线阅读。正文沿业务问题、因果识别、策略评价和投放决策展开，技术附录提供完整的统计证据和复现信息。

## 冻结结论

HT Gain 与配对差值均以每 10 万名全部合资格用户的预期增量转化表示。

| 阶段 | 验证集冻结策略 | HT Gain @20%（95% CI） | 相对 Random（配对 95% CI） |
| --- | --- | ---: | ---: |
| 500 万开发 | T-Logistic | 113.66 [92.83, 136.05] | +89.82 [73.28, 107.64] |
| 全量正式 | Response-XGBoost | 109.46 [96.04, 121.57] | +86.56 [75.51, 96.01] |

全量验证集最佳 Uplift 候选 S-XGBoost 在正式测试集上的点估计高出 **0.04**，配对区间 **[−1.70, 1.98]** 包含零。正式推荐沿用验证集冻结的 Response-XGBoost。两策略各选 559,183 行，共同选中 541,274 行，Overlap@20% 为 **96.80%**。重叠指标描述排序行为，策略收益差异以配对评价为准。

## 数据与研究设计

- [Criteo Uplift v2.1](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) 记录广告资格分配。主结果是转化，辅助结果是访问。模型输入限定在 12 个匿名处理前特征。
- 正式测试集的 2,795,919 行在新增开发前冻结。开发样本来自正式测试集之外，训练集用于拟合，验证集按固定 20% HT Gain 的分层配对 Bootstrap 均值选型。
- 因果目标是广告资格分配的 ITT。HT 权重以训练集处理比例代入，推断依赖公开分配设计与近似恒定处理概率等假设。匿名行号不能验证用户级隔离。条件区间未纳入全部重训、选型及概率估计的不确定性。
- 线上建议在用户级随机实验中比较同覆盖率策略，以全部随机入组用户的转化率为主指标，同时核验成本和护栏。[数据与报告许可](NOTICE.md)

## 复现

使用 Python 3.11，测试依赖见 [requirements-tested.txt](requirements-tested.txt)。macOS 的 XGBoost 所需 OpenMP 可由 `brew install libomp` 提供。生成中文图表时使用本机可用的中文字体。

```bash
python -m pip install -e . -c requirements-tested.txt
python -m unittest discover -s tests -v
```

用对应保存的运行目录重建发布报告。命令同时生成评分重叠摘要与来源指纹。

```bash
python - <<'PY'
from pathlib import Path
from criteo_uplift.reporting import build_report

for stage, run in [
    ('development', 'development_conversion_5000000_seed42'),
    ('final', 'final_conversion_full_seed42'),
]:
    build_report(Path('artifacts') / run / 'manifest.json',
                 Path('reports') / stage / 'report.html')
PY
```

报告附录 J 列出了原始运行配置。Pipeline 入口为 `run_pipeline.py --help`，正式实验复现需要本地原始数据与 `artifacts/release_protocol_seed42` 中的冻结行号及指纹。新运行请使用空目录。原始数据和行级工件保存在本地 `data/` 与 `artifacts/`。

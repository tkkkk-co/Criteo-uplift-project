# Criteo Uplift｜基于随机实验的广告增量评估与投放策略优化

**Causal Inference · Uplift Modeling · Policy Evaluation · A/B Testing · Python / XGBoost**

在广告资格仅覆盖 **20% 合资格用户**的约束下，如何分配名额，才能提高实际增量转化？本项目基于 **13,979,592 条 Criteo Uplift v2.1 公开随机实验记录**，搭建从数据审计、因果诊断、增量策略比较到选人分析与预算决策的离线研究流程，评估 Response Targeting、Uplift Targeting 与同覆盖率 Random 的策略价值。

**[在线阅读完整研究报告（交互式网页）](https://tkkkk-co.github.io/Criteo-uplift-project/)** · [一页研究综述（PDF）](https://tkkkk-co.github.io/Criteo-uplift-project/reports/final/executive_summary.pdf) · [报告来源与冻结记录](reports/final/report_provenance.json)

## 核心发现

在独立冻结的 **2,795,919 条 Test 记录**上，按照验证集预设的 HT Gain @20% 指标选出 **Response-XGBoost** 作为正式策略。同等 20% 资格名额下，模型定向相对随机分配取得明确的离线增量收益；显式 Uplift 学习带来的额外收益尚不足以支持替换正式策略。

| 核心结果 | 点估计 | 95% 置信区间 |
| --- | ---: | ---: |
| 全量广告资格分配 ITT（百分点） | **+0.1152** | [0.1085, 0.1219] |
| Response-XGBoost：HT Gain @20% | **109.46** | [94.20, 123.39] |
| Response-XGBoost − Random：配对收益差 | **+86.56** | [74.37, 97.72] |
| S-XGBoost − Response-XGBoost：配对收益差 | +0.04 | [-1.64, 1.88] |

*HT Gain 及配对差值的单位为每 10 万名全部合资格用户的预期增量转化次数；以全员无资格为收益基准。区间基于固定策略下的 2,000 次处理臂内分层 Bootstrap。*

策略选择进一步得到三类决策证据：

- **模型替换**：Response-XGBoost 与 S-XGBoost 的 Top 20% 名单重叠 **96.80%**；两者的配对收益差区间跨零。保留验证集冻结的 Response-XGBoost，将 S-XGBoost 作为线上挑战策略。
- **目标人群**：两策略入选记录均表现出 f8、f3 较低、f9 较高的分布特征。Response-XGBoost 最高分十分层的观测 ITT 为 **0.8323 个百分点**，高于同一测试集整体的 **0.1146 个百分点**，为高分人群优先试点提供描述性证据。
- **预算覆盖**：Response-XGBoost 前 10% 名额已贡献前 20% 约 **95.7% 的 HT Gain**（按 Uplift Gain 为 **84.6%**）。从 10% 扩至 20%，每千新增名额约增加 **0.474 次转化**；结合真实获客成本与转化价值，建议将 10% 覆盖率列为独立试点方案。

补充三轮完整重划分与重训后，共四轮验证选型的首位模型存在变化，但选型流程的 Top 20% 名单共同入选比例达到 **93.49%**；Response-XGBoost 距各轮测试观测最高点最多 **1.61 次／10 万人**。这些结果支持保留当前基线并开展后续线上挑战实验。

## 研究设计与技术实现

| 环节 | 实施内容 |
| --- | --- |
| 数据质量与实验诊断 | 对 1,398 万条记录进行流式字段、标签、重复与特征分布审计；结合 SRM、SMD、KS、C2ST、单变量 Overlap 和结果可学习性检查实验数据基础 |
| 估计目标 | 将 `treatment` 定义为广告资格分配，以 `conversion` 为主结果估计 ITT；使用 12 个处理前匿名特征（f0–f11）进行策略评分 |
| 模型比较 | S / T / X-Learner × Logistic Regression / XGBoost，共 **6 个 Uplift 候选**；另设 **2 个 Response 候选**与 Random 基线 |
| 训练与评价 | 按 `treatment × conversion` 分层划分 Train / Validation / Test（约 6:2:2）；预处理仅在 Train 拟合，Validation 选型并冻结，Test 用于正式评价和配对区间 |
| 决策分析 | 比较 Top-K 收益与名单重叠、入选特征分布、十分层效应、逐档扩量的边际增益；以三轮补充重训检查选型及人群稳定性 |
| 工程交付 | 提供数据处理、诊断、训练、策略评价与报告生成模块；保留实验协议和来源指纹，使用自动化测试验证核心实现与合成数据端到端流程 |

完整的实验配置、统计口径、置信区间、分析图表和业务决策依据均在 **[正式研究报告（八章，在线网页）](https://tkkkk-co.github.io/Criteo-uplift-project/)** 中展开。GitHub 的 HTML 文件页显示源代码，在线网页链接可直接阅读排版后的报告。

## 项目代码与验证

| 路径 | 职责 |
| --- | --- |
| [`run_pipeline.py`](run_pipeline.py) | 研究流程入口与阶段组织 |
| [`criteo_uplift/`](criteo_uplift/) | 流式审计、实验诊断、Meta-Learner、选型、收益评价、画像和稳定性分析 |
| `reports/final/report.html` | 唯一正式研究报告（Report v2）；[在网页中阅读](https://tkkkk-co.github.io/Criteo-uplift-project/) |
| [`reports/final/report_provenance.json`](reports/final/report_provenance.json) | 冻结报告指纹与历史实验来源 |
| [`tests/`](tests/) · [GitHub Actions](.github/workflows/ci.yml) | 单元测试与合成数据端到端验收 |

基于 Python 3.11 的安装与代码测试：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e . -c requirements-tested.txt
python -m unittest discover -s tests -v
```

仓库公开源代码、测试及完整研究报告；原始 Criteo 数据、冻结行号、已训练模型和行级评分工件保存在本地，**精确复现报告数值需要相应原始数据及冻结实验工件**。数据来源：[Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)。

## 业务建议与研究边界

离线结论支持在固定预算下采用模型定向：**Response-XGBoost 作为正式基线，S-XGBoost 进入同覆盖率的用户级 A/B 挑战实验**；覆盖率 10% 与 20% 的成本收益比较独立验证。线上实验计划以全部随机入组用户的固定窗口转化率为主指标，并预设样本量、业务阈值与执行护栏。

研究结论对应 Criteo 公开样本及其随机分配设计；匿名记录缺少可靠用户标识，四轮重训共用正式 Test Set，当前置信区间反映固定模型条件下的评价不确定性。项目尚未实施线上实验，真实 ROI、最优覆盖率及跨流量迁移效果需要进一步验证。

**参考与许可**：Diemert et al., *A Large Scale Benchmark for Uplift Modeling* (AdKDD, 2018)；Künzel et al., [*Metalearners for Estimating Heterogeneous Treatment Effects*](https://arxiv.org/abs/1706.03461)。项目代码采用 [MIT License](LICENSE)，数据和衍生研究材料的来源及使用条件见 [NOTICE.md](NOTICE.md)。

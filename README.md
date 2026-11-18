# Criteo Uplift：广告增量评估与投放策略优化

**固定 20% 广告资格名额时，如何选择用户，才能获得更多增量转化？**

本项目基于 Criteo Uplift v2.1 的 **13,979,592 条公开随机增量实验记录**，比较增量定向、转化倾向定向和同覆盖率随机分配。研究从数据审计与因果诊断出发，通过冻结测试评价、人群画像、分层效应和覆盖率分析，形成模型选择、目标人群与资源配置的离线决策依据。

> **核心结论：** 同等名额下，Response-XGBoost 相对 Random 每 10 万名合资格用户多产生约 **86.56 次预期增量转化**；S-XGBoost 尚未建立足以替换正式基线的额外收益证据。增量集中于高分人群，较低覆盖率值得结合实际成本进行线上试点。

## 阅读入口

| 内容 | 入口 | 用途 |
| --- | --- | --- |
| **正式展示报告** | [完整研究报告](reports/final/report.html) | 首页一页综述，随后为八章研究正文与可展开的技术证据 |
| **一页项目综述** | [研究结论与业务决策 · PDF](reports/final/executive_summary.pdf) | 面试快速阅读与附件展示 |
| 发布与复现边界 | [交付说明](RELEASE_NOTES.md) | 冻结版本、来源记录及生成报告的注意事项 |
| 数据与报告使用说明 | [NOTICE.md](NOTICE.md) | 数据来源、论文引用与许可范围 |

**正式展示版已于 2026-11-18 冻结**，由校订通过的 v2 原样发布至 `reports/final/report.html`。仓库仅保留这一份完整报告及其一页 PDF 综述，所有展示链接统一指向正式入口。

GitHub 文件页不直接渲染交互 HTML：可下载正式报告后用浏览器打开，或按本文末尾配置 GitHub Pages。图表已嵌入 HTML；公式排版依赖 MathJax CDN，完整公式渲染需要联网。阅读报告和 PDF 无需安装 Python 或运行模型。

## 业务问题与分析路径

广告资格有限时，高转化概率用户未必具有高广告增量。本项目将两类排序目标放在同一名额约束下评价：

- **Response Targeting**：按获得资格后的预测转化概率排序，学习目标为 `P(Y=1 | X=x, T=1)`。
- **Uplift Targeting**：按预测 CATE 排序，估计获得与未获得资格两种状态的条件平均转化差。
- **Random**：相同覆盖率下的随机分配期望收益，衡量定向本身的价值。

研究依次回答：资格分配平均是否有价值；定向是否提高固定名额的增量产出；显式增量学习是否优于响应定向；两套规则如何影响实际选人；继续扩量的边际回报是否值得投入。

```text
全量审计与特征诊断
        ↓
因果识别诊断与平均资格效应（ITT）
        ↓
六个 Uplift 候选 + 两个 Response 候选 + Random
        ↓
训练拟合 → 验证集选型并冻结 → 正式测试与配对区间
        ↓
名单重叠与画像 → 十分层效应 → 覆盖率边际收益
        ↓
重划分与重训敏感性 → 交付规则与用户级线上实验方案
```

## 主要研究结果

### 1. 同等名额下，模型定向提高增量转化

**HT Gain @20%** 表示：向排序前 20% 分配资格，相对全员不分配资格，**每 10 万名全部合资格用户的预期增量转化次数**。分母包含全部候选用户。

| 正式测试策略 | HT Gain @20% | 95% Bootstrap CI |
| --- | ---: | ---: |
| **Response-XGBoost · 验证集冻结策略** | **109.46** | **[94.20, 123.39]** |
| S-XGBoost · 验证集首位 Uplift 候选 | 109.50 | [94.20, 123.43] |
| Random · 同覆盖率期望收益 | 22.90 | [19.75, 25.79] |

| 配对比较 | 差值／10 万人 | 配对 95% CI | 决策含义 |
| --- | ---: | ---: | --- |
| Response-XGBoost − Random | **+86.56** | **[74.37, 97.72]** | 支持模型定向的离线增量价值 |
| S-XGBoost − Response-XGBoost | +0.04 | [−1.64, 1.88] | 额外收益优势尚未确立，保留冻结基线 |

全量公开样本的转化 ITT 为 **+0.1152 个百分点**，95% CI 为 **[0.1085, 0.1219] 个百分点**。这是总体平均资格效应基准，与冻结测试集上固定覆盖率的策略收益分别报告。

### 2. 人群分析解释模型决策，并支持规则监测

- **名单重叠**：Response-XGBoost 与 S-XGBoost 各选 559,183 条记录，共同选中 541,274 条，Overlap@20% 为 **96.80%**。模型替换的价值主要取决于差异名额的净转化贡献。
- **入选画像**：两策略均偏向 f8、f3 较低、f9 较高的记录。以 Response-XGBoost 为例，f9 严格高于全体第 75% 分位点者，全体占比为 **20.51%**，入选者占比为 **77.48%**。匿名特征的集中取值使严格高于分位点的整体比例未必为 25%。
- **分层效应**：按各自冻结分数十分层，Response-XGBoost、S-XGBoost 首层观测 ITT 为 **0.8323、0.8225 个百分点**，高于同一测试总体的 **0.1146 个百分点**。分层点估计支持高分人群优先试点，画像则提供新流量和名单分布的监测基准。

画像描述人群结构，分层描述组内平均效应；它们未建立匿名特征的因果作用或精确 CATE 校准结论。

### 3. 收益集中与边际回报支持预算试点

Response-XGBoost 前 10% 的 HT Gain 为 **104.72 次／10 万人**，保留前 20% 收益的约 **95.7%**；按 Uplift Gain，保留比例约为 **84.6%**。两种估计口径均显示增量集中于首批名额，但具体集中程度存在差异。

从 10% 扩至 20%：

- 新增 HT Gain 为 **4.74 次／10 万人**，配对 95% CI **[0.45, 8.64]**。
- 每千新增名额约增加 **0.474 次转化**，低于前一档的 **10.472 次**。
- 若单次增量转化价值为 `V`，则每千新增名额的估计毛增量价值为 `0.474 × V`，可与同口径新增成本比较。

因此，10% 覆盖率是后续试点候选，20% 保留为预设模型比较口径。公开数据缺少真实成本与转化金额，当前结果尚未确定实际 ROI 或利润最优覆盖率。

### 4. 重训验证实际收益与选人的稳定性

固定同一正式测试集，在其余数据上补充 Seed 17、29、83 三轮完整划分、拟合与验证选型，连同正式 Seed 42 共四轮：

- 验证首位为 Response-XGBoost 三次、S-XGBoost 一次。
- Response-XGBoost 距各轮测试观测最高点最多 **1.61 次／10 万人**。
- 四轮选型流程共同入选 **522,797** 条记录，占单轮名额 **93.49%**。

当前扰动下，排名存在波动，候选收益与主要分配区域较稳定。四轮测试点估计的小幅领先使 S-XGBoost 成为挑战实验候选；各轮共用测试集，属于相关的重训敏感性分析。

## 数据与研究设计

| 项目 | 设置 |
| --- | --- |
| 数据 | Criteo Uplift v2.1，13,979,592 条记录，16 个字段 |
| 处理前特征 | f0–f11，共 12 个匿名数值特征 |
| 处理 | `treatment`，广告资格分配；估计资格分配 ITT |
| 主结果／辅助结果 | `conversion`／`visit` |
| 排除评分字段 | `exposure`、`visit`、来源行号及结果标签 |
| 分组与事件 | 约 85%／15%；全量转化率 0.2917%，对照转化事件 4,063 个 |
| 正式划分 | Train 8,387,755；Validation 2,795,918；Test 2,795,919 |
| 划分口径 | 按 `treatment × conversion` 分层；来源行号隔离；预处理仅在 Train 拟合 |
| 候选策略 | S／T／X-Learner × Logistic Regression／XGBoost，两个 Response，Random |
| 选型 | **验证集 HT Gain @20% 点估计**；Test 不参与选型 |
| 正式推断 | 2,000 次处理臂内分层配对 Bootstrap |
| 补充验证 | 固定正式 Test，三轮重划分与完整重训；覆盖率敏感性分析 |

诊断覆盖全量字段与标签审计、重复及 IQR 分布、SRM、SMD、KS、C2ST、单变量覆盖和结果可学习性。正式报告在大样本下以 KS 统计量衡量分布差异幅度，显著性作为补充依据；C2ST 同时报告 Logistic 与 XGBoost 的检验结果。

## 工程实现与项目结构

工程工作服务于数据与评价过程的可追溯性：按批扫描原始 CSV；保存划分行号、审计与诊断；训练内拟合预处理；验证选型与冻结测试分离；留存模型、评分、选型、人群及图表输入。

```text
Criteo-uplift-project/
├── README.md
├── index.html                         # GitHub Pages 入口，跳转正式报告
├── run_pipeline.py                    # 研究流程入口
├── pyproject.toml                     # 包与依赖配置
├── requirements-tested.txt            # Python 3.11 已用直接依赖版本快照
├── criteo_uplift/
│   ├── raw_scan.py / preparation.py    # 全量审计、抽样与划分
│   ├── diagnostics.py                 # 分配、平衡、覆盖与可学习性诊断
│   ├── models.py                      # S/T/X-Learner 与 Response
│   ├── training_validation.py         # 拟合、验证选型与冻结
│   ├── metrics.py / evaluation.py     # 排序指标、测试与配对推断
│   ├── split_stability.py             # 补充完整划分与重训
│   ├── stability_analysis.py          # 收益、覆盖率与名单汇总
│   ├── profile_effect_analysis.py     # 画像、分层及成员分析
│   ├── planning.py                    # 条件经济情景与实验规划
│   └── reporting.py                   # 实验报告生成
├── reports/
│   ├── final/report.html              # 冻结的正式展示报告
│   ├── final/executive_summary.pdf     # 一页项目综述
│   └── final/report_provenance.json    # 正式发布与实验来源指纹
├── tests/                             # 单元测试与合成数据端到端测试
├── .github/workflows/ci.yml           # 安装与测试工作流
├── data/                              # 本地原始数据，不纳入版本控制
└── artifacts/                         # 本地协议、模型与实验工件，不纳入版本控制
```

## 安装、测试与复现

### 环境准备

建议使用 Python 3.11 和独立虚拟环境。在仓库根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -c requirements-tested.txt
```

Windows 激活命令为 `.venv\Scripts\activate`。`requirements-tested.txt` 是直接依赖的已用版本快照，并非跨平台或传递依赖锁文件。macOS 的 XGBoost 如需 OpenMP，可安装 `libomp`；绘制中文图表还需环境中可用的中文字体。

### 公开仓库可执行的检查

```bash
python -m unittest discover -s tests -v
python run_pipeline.py --help
```

测试包含指标、数据审计、模型、冻结选型及合成数据端到端流程等。测试用于程序行为校验，不等同于对全量公开样本研究结论的重新验证；当前 CI 运行状态请查看仓库 Actions。

### 原始数据与严格复现条件

原始数据请从 [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) 获取，并参考 [数据卡](https://huggingface.co/datasets/criteo/criteo-uplift)。本地默认路径为：

```text
data/raw/criteo-uplift-v2.1.csv
```

**严格复现冻结结果还需要作者本地的协议与实验工件。** 当前仓库未分发原始数据、`artifacts/release_protocol_seed42` 中的冻结行号与协议、已拟合模型及行级评分。仅克隆仓库即可阅读报告并运行合成数据测试，不能据此承诺直接重建同一批正式结果。

在具备原始数据及冻结协议的本地环境，可按协议配置运行全量实验到新目录：

```python
from pathlib import Path
import subprocess
import sys
from criteo_uplift.experiment_protocol import load_protocol

protocol_path = Path("artifacts/release_protocol_seed42")
metadata, _ = load_protocol(protocol_path)

command = [
    sys.executable, "run_pipeline.py",
    "--data", "data/raw/criteo-uplift-v2.1.csv",
    "--stage", "full", "--sample-size", "0",
    "--protocol", str(protocol_path),
    "--output", "artifacts/reproduction_full_seed42",
]
for option in ("seed", "bootstrap", "trees", "c2st_sample", "permutations"):
    command += ["--" + option.replace("_", "-"), str(metadata[option])]
subprocess.run(command, check=True)
```

输出目录须为空。该入口的 `--run-stability` 控制其额外重训模块；正式报告第 07 章的三轮完整补充实验另保存在本地 `full_split_stability_test_20261117` 运行目录，由相应稳定性与画像分析模块汇总。标准入口并不自动恢复这批补充工件。

### 冻结展示版与重新生成的实验报告

`reports/final/report.html` 是人工校订后冻结的 v2 展示版，也是仓库唯一完整报告。发布 SHA-256、原始实验输入与生成器指纹统一保存在同目录 `report_provenance.json`；原始实验记录作为历史来源嵌入，不代表生成器可逐字重建最终编辑内容。

`reporting.build_report()` 或重新运行 Pipeline 生成的是实验报告，不保证恢复最终编辑的版式与叙述。后续生成请写入新的 `artifacts/` 目录，**不要把冻结展示报告作为生成输出路径**。测试和常规阅读无需重新生成展示版。

## 业务建议与适用边界

正式基线采用 Response-XGBoost，S-XGBoost 为挑战策略；冻结模型、训练预处理、同分规则与 Top-K 覆盖率共同构成资格分配规则，画像用于解释和监测名单结构。

线上规划先在相同 20% 覆盖率下做用户级策略随机实验，以全部随机入组用户的固定窗口转化率为主指标，预设最小业务提升、样本量与观察期，核验实际成本与执行护栏；随后固定策略，单独比较 10%／20% 覆盖率的边际净价值。

研究对应非均匀抽样后的公开样本，因果解释依赖原实验设计、近似恒定分配概率及相关识别条件；数据缺少可靠用户标识，单变量覆盖尚不等于已验证高维联合支持。Bootstrap 区间为固定模型等条件下的评价不确定性，四轮重训共用正式测试集。项目尚未上线，真实 ROI、最优预算和迁移效果待新流量与成本数据验证。

## 在线展示与引用

仓库提供根目录 `index.html`，可在 GitHub Pages 选择发布分支的 `/(root)`，访问时自动进入正式报告；配置步骤见 [GitHub 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。Pages 尚未启用时，请使用上方本地报告或 PDF 入口。

原始代码使用 [MIT License](LICENSE)。数据及报告衍生结果的来源与使用范围见 [NOTICE.md](NOTICE.md)，原始数据不随仓库分发。本项目为独立学习研究项目。

数据论文：Eustache Diemert, Artem Betlei, Christophe Renaudin, Massih-Reza Amini. *A Large Scale Benchmark for Uplift Modeling*. AdKDD, 2018。

Meta-Learner 方法参考：[Künzel et al., Metalearners for estimating heterogeneous treatment effects using machine learning](https://arxiv.org/abs/1706.03461)。

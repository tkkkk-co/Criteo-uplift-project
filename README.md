# Criteo Uplift｜随机实验下的广告增量评估与定向策略优化

**Causal Inference · Uplift Modeling · A/B Testing · Python · XGBoost**

> **业务问题：** 当广告资格只覆盖 20% 的合资格用户时，怎样选人才能创造更多**真实增量转化**？  
> 本项目使用 **13,979,592 条 Criteo Uplift v2.1 公开随机实验记录**，建立从实验可信度诊断、ITT/CATE 估计，到离线策略比较、预算敏感性及线上 A/B 测试规划的完整分析流程。

### 📖 从这里开始

**[完整正式研究报告｜Final Report（HTML，七章正文＋附录 A–J）](reports/final/report.html)** · [500 万样本开发报告](reports/development/report.html) · [交付说明](RELEASE_NOTES.md)

正式报告汇总研究方法、全量结果、策略收益曲线、置信区间、选人重叠与成本敏感性。报告为自包含 HTML，可下载后直接在浏览器离线阅读；仓库公开后也可通过 GitHub Pages 提供在线阅读链接。

## 一眼看懂最终结论

**验证集冻结的正式策略：Response-XGBoost。** 在独立留出的 **2,795,919 条 Test 记录**上，同样只分配 **Top 20% 广告资格**：

| 策略 / 比较 | HT Gain @20%（每 10 万名合资格用户） | 95% CI |
| --- | ---: | ---: |
| **Response-XGBoost（验证集预先冻结）** | **109.46** | **[94.20, 123.39]** |
| S-XGBoost（最佳 Uplift 候选） | 109.50 | [94.20, 123.43] |
| Random（相同覆盖率） | 22.90 | [19.75, 25.79] |
| **Response-XGBoost − Random（配对差值）** | **+86.56** | **[74.37, 97.72]** |
| S-XGBoost − Response-XGBoost（配对差值） | +0.04 | [−1.64, 1.88] |

**解读：** 基于预测评分的定向策略显著优于同覆盖率的随机分配；在该公开样本及指标下，额外引入 Uplift 模型并未得到可确认的策略收益优势。按预设的验证集规则继续采用 Response-XGBoost，将 S-XGBoost 保留为后续线上验证候选。

两种策略在 Test 的 Top 20% 目标集合有 **96.80% 重叠**（各入选 559,183 条、共同 541,274 条；Jaccard 0.9379，原始评分 Spearman 0.9373）。重叠度刻画了评分策略的选人行为，供解释和后续检验使用。

> 以上 Gain 和配对差值均表示：**对每 100,000 名全部合资格用户，按给定资格分配策略预计产生的增量转化次数**。HT Gain 以全部合资格用户为分母；不是每 10 万实际投放人数的转化，也不是广告收入。

## 研究设计：从因果识别到投放决策

1. **实验与数据诊断。** 检查标签有效性、SRM、SMD、KS、C2ST 及分箱的处理/对照支持。公开数据中处理组占 85%，全量转化率 ITT 估计为 **+0.1152 个百分点**（95% CI [0.1085, 0.1219]）。Logistic C2ST AUC=0.5177、Holm p=0.012，显示轻微可观察分组可分性；因此结合效应量与检验结果综合评估。
2. **严格划分与预先选型。** 使用 12 个处理前匿名特征（f0–f11），正式阶段 Train / Validation / Test 分别为 **8,387,755 / 2,795,918 / 2,795,919** 条。新增模型开发前通过冻结协议保留正式 Test 行号并排除可追溯历史建模行。Train 拟合、Validation 按预设 HT Gain@20% 的 Bootstrap 均值选型、Test 仅用于最终留出评价。
3. **统一比较九种投放策略。** S/T/X-Learner 分别搭配 Logistic Regression 与 XGBoost，产生 **六种 CATE 评分**；另评估 **两种 Response Targeting 评分**（复用 T-Learner 的处理组响应模型），以及同覆盖率 Random 参考策略。所有策略共享数据划分、特征和评价尺度。
4. **增量收益与区间推断。** 以固定 20% 覆盖率下的 **Horvitz–Thompson Gain（HT Gain）** 为唯一选型主指标，同时报告 Uplift Gain、Traditional Qini Gain、AUUC 与 Qini Score。验证集和常规九策略 Test 评价使用 200 次处理臂内分层配对 Bootstrap；正式报告对关键策略比较、冻结策略的预算边际收益进一步计算 **2,000 次配对 Bootstrap**。
5. **业务决策与线上实验规划。** 固定评分下，覆盖率从 **10% 增至 20%** 的 HT 边际收益为 **4.74 次增量转化／每 10 万名合资格用户**（95% CI [0.45, 8.64]）。在报告明确设定的转化价值和单位资格成本下，这一档扩量的**假设边际净价值为 −76.30**（条件 95% CI [−97.76, −56.80]）。进一步给出按用户随机、两臂均覆盖 Top 20%、以全体入组用户转化率为主指标的线上 A/B 方案、样本量与护栏设计；线上实验尚未执行。

### HT Gain 的业务口径

令 (T_i) 为随机分配的广告资格，(Y_i) 为转化，(e) 为训练集的处理分配比例代入值，(pi_i) 为模型在指定覆盖率下的选人决策：

[
\widehat{G}_{HT}(\pi)=\frac{100,000}{N}\sum_{i=1}^{N}\pi_i
\left(\frac{T_iY_i}{e}-\frac{(1-T_i)Y_i}{1-e}\right).
]

每个策略按照预测评分选出 (K=\lfloor0.2N\rfloor) 条记录；同分使用与结果无关的共同随机优先级。这里的处理定义为**广告资格分配（ITT）**，实际曝光 `exposure` 不作为建模特征或样本筛选条件。

## 项目结构

```text
.
├── README.md                         # 项目入口与核心结论
├── reports/
│   ├── final/report.html             # ★ 冻结的全量正式研究报告
│   ├── final/report_provenance.json  # 来源指纹、生成与计算记录
│   ├── final/strategy_overlap.json   # 冻结评分的选人重叠审计
│   └── development/report.html       # 500 万开发实验报告
├── criteo_uplift/
│   ├── experiment_protocol.py        # 正式 Test 冻结与历史行排除
│   ├── preparation.py                # 数据准备及划分
│   ├── diagnostics.py                # SRM、SMD/KS、C2ST 等
│   ├── models.py                     # S/T/X-Learner、Response
│   ├── training_validation.py        # Validation 选型与冻结记录
│   ├── metrics.py                    # HT/Uplift/Qini/AUUC、Bootstrap
│   ├── evaluation.py                 # 独立 Test 评估
│   ├── budget_analysis.py            # 2000 次关键比较和扩量区间
│   ├── planning.py                   # 假设经济情景、线上 A/B 规划
│   └── reporting.py                  # HTML 报告生成
├── run_pipeline.py                   # 端到端流程入口
├── tests/                            # 单元测试与合成数据端到端验收
├── pyproject.toml
├── requirements-tested.txt
└── NOTICE.md
```

## 安装、测试与复现

推荐 Python 3.11。安装当前项目并运行仓库自带的单元与合成数据端到端测试：

```bash
python -m pip install -e . -c requirements-tested.txt
python -m unittest discover -s tests -v
python run_pipeline.py --help
```

`requirements-tested.txt` 是 macOS arm64 / Python 3.11 下测试过的直接依赖版本快照，并非全平台锁文件。macOS 使用 XGBoost 时，如需 OpenMP，可通过 `brew install libomp` 安装。

若本地保留了原始数据、`artifacts/release_protocol_seed42` 的**预先冻结行号与指纹**，可在新的空输出目录启动正式阶段：

```bash
python run_pipeline.py \
  --stage full \
  --protocol artifacts/release_protocol_seed42 \
  --data data/raw/criteo-uplift-v2.1.csv \
  --sample-size 0 \
  --output artifacts/final_conversion_full_seed42_recheck
```

已有完整本地运行工件时，可以从冻结记录重建报告：

```bash
python - <<'PY'
from pathlib import Path
from criteo_uplift.reporting import build_report

root = Path("artifacts/final_conversion_full_seed42")
build_report(root / "manifest.json", Path("reports/final/report.html"))
PY
```

**复现范围：** 仓库提供源码、测试、完整正式报告与报告来源指纹。原始 Criteo 数据、行级预测、模型文件和冻结 Test 行号工件存放在作者本地的 `data/`、`artifacts/`，未随仓库发布。重新生成冻结版数值需要对应的原始源文件及冻结工件；重新创建协议会得到另一份实验划分。报告中的关键结果与来源可在 [正式报告附录](reports/final/report.html) 和 [provenance](reports/final/report_provenance.json) 中核对。

## 适用范围与研究限制

- **识别对象：** Criteo v2.1 公开、经非均匀抽样的数据中的广告**资格分配**效果。公开数据不支持将估计值直接外推至原始完整广告总体；固定处理概率是估计假设。
- **依赖和区间：** 匿名记录缺少可核验的用户 ID，来源行隔离不等于用户级隔离；本文 Bootstrap 区间条件于固定模型、评分及处理概率，未纳入完整模型重训和重新选型的不确定性。
- **经济结论：** 价值、资格成本与 A/B 测试基线转化率均为显式**假设**，报告未声称观察到真实线上收入或 ROI。跨种子重训稳定性分析在正式阶段未执行。

## 数据来源与许可

Criteo Uplift Prediction Dataset v2.1： [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) / [数据卡](https://huggingface.co/datasets/criteo/criteo-uplift)。

原始项目代码按 [MIT](LICENSE) 许可；公开数据及相应派生报告材料遵循来源规定的 **CC BY-NC-SA 4.0** 条款，详见 [NOTICE.md](NOTICE.md)。本项目为独立教育研究项目，未获 Criteo 官方背书。

# 固定投放预算下的增量转化优化

基于 Criteo Uplift v2.1 的端到端数据科学项目。在单位触达成本与转化价值相同、满足无干扰等识别假设的条件下，比较固定覆盖 20% 合资格用户时的预期增量转化。

## 报告

- [500 万开发实验报告](reports/development/report.html)
- [全量正式实验报告](reports/final/report.html)

两份 HTML 均包含内嵌图表与公式，可离线打开。正文采用中文，保留通用统计术语、模型名称和必要的代码标识。所有结果来自对应真实运行，报告不代表已经实现的线上收益。

## 本轮真实运行结果

两阶段均完成，使用相同固定模型参数，没有依据正式测试结果调参或重新选型。

| 阶段 | 验证集冻结策略 | HT Gain @20% / 10 万人（95% CI） | 相对 Random 配对差值（95% CI） |
| --- | --- | --- | --- |
| 500 万开发实验 | T-Logistic | 113.66 [92.83, 136.05] | +89.82 [73.28, 107.64] |
| 全量正式实验 | Response-XGBoost | 109.46 [96.04, 121.57] | +86.56 [75.51, 96.01] |

全量验证集最佳 Uplift 候选是 S-XGBoost。其正式测试点估计比 Response-XGBoost 高 **0.04 / 10 万人**，配对区间 **[−1.70, 1.98]** 包含零。最终策略保留验证集预先冻结的 Response-XGBoost。该策略相对 Response-Logistic 的配对差值为 **+3.58 [1.15, 5.64] / 10 万人**。

两阶段实际流程分别约 **162 秒**和 **374 秒**，进程峰值常驻内存约 **2.33 GiB**和 **3.25 GiB**。这不包括冻结协议、后续对账与视觉验收时间，也不代表其他硬件上的耗时。84 项自动化测试通过，全部九策略点估计已从保存评分独立重算核对。

## 数据与独立测试设计

[Criteo 公开数据](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)包含 **13,979,592** 行。本项目只用 `f0`–`f11` 排序，以 `conversion` 为主要结果、`visit` 为辅助结果。`treatment` 是广告资格分配，`exposure` 是处理后的实际曝光，不作为预测特征。主要因果目标为分配的 ITT。数据及派生成果许可见 [NOTICE.md](NOTICE.md)。

新增开发工作前，按 treatment × conversion 分层固定 **2,795,919** 行正式测试集。排除有保存行号的历史建模记录 **1,000,000** 行，500 万开发样本再从正式测试集以外无放回抽取。

| 阶段 | 训练集 | 验证集 | 测试集 |
| --- | ---: | ---: | ---: |
| 开发实验 | 3,000,000 | 1,000,000 | 1,000,000 |
| 全量正式实验 | 8,387,755 | 2,795,918 | 2,795,919 |

开发阶段的测试集是内部留出评价。全量正式实验使用预冻结行号，其余数据按预定比例分成训练与验证集。来源行两两隔离，但匿名特征不能验证用户级独立性。历史全量审计、ATE 与特征诊断已经查看过，无法确认未保存的历史使用情况，因此不宣称这是完全未接触过的新实验。

## 九种策略与统计口径

| 策略 | Logistic | XGBoost |
| --- | --- | --- |
| S-Learner | `s_logistic` | `s_xgboost` |
| T-Learner | `t_logistic` | `t_xgboost` |
| X-Learner | `x_logistic` | `x_xgboost` |
| Response | `response_logistic` | `response_xgboost` |

另有 `random`，为同覆盖率随机分配的解析期望，不训练、不参加选型。Response 复用对应 T-Learner 仅用训练集处理组拟合的结果模型，输出处理下转化概率，不是 CATE，也不是处理分配倾向分数。X-Logistic 第二阶段使用线性回归拟合连续伪效应；X-XGBoost 使用 XGBRegressor。

- 唯一主选型指标为 **HT Gain @20%**，按验证集 **200 次处理臂内分层配对 Bootstrap 的均值**最大化选择，并在正式测试前冻结模型和策略。
- HT 处理概率使用训练集观察比例代入，并非已知的原始实验设计概率。分母为全部合资格评价用户，选中人数为 `floor(0.20 × N)`，不足一人的差异来自整数预算约束。
- 测试集提供九策略点估计、单策略区间、相对两个 Response 基线和 Random 的配对区间。自比较为零。区间条件于冻结模型、评分和分配概率，不包含全部训练、选型或用户依赖的不确定性。
- Uplift Gain、Traditional Qini Gain、AUUC、Qini Score 为辅助指标。AUUC 对 Uplift Gain 积分；Qini Score 为传统 Qini 曲线面积减去终点参考线面积，不做完美排序归一化。Qini Gain 与 HT 不同尺度。
- 全量审计及 SMD/KS 使用全部原始记录；C2ST 从两臂各抽取 10,000 行，使用固定 499 次完整重拟合置换。SRM 的 85% 是外部参考，只用于公开样本分配比例诊断。
- 转化与访问 ATE/ITT 均由全量审计计数计算，使用未合并标准误与 95% CI，双侧两比例检验使用零假设下合并比例。
- 主覆盖率固定 20%，保留 10%、20%、30%、50%、100% 等预设预算的描述性分析。两阶段默认关闭额外跨种子和训练规模重训，需使用 `--run-stability` 才会执行；报告明确标注，不借用旧结果冒充本轮稳定性证据。

## 复现

使用 Python 3.11，依赖版本见 [requirements-tested.txt](requirements-tested.txt)。macOS 运行 XGBoost 需要 OpenMP，可使用 `brew install libomp`。生成中文图表的机器需要可用中文字体，已发布 HTML 已将图片和公式内嵌，不需要安装运行时或联网。

```bash
python -m pip install -e . -c requirements-tested.txt
python -m unittest discover -s tests -v
```

原始文件放在 `data/raw/criteo-uplift-v2.1.csv`。现有实验直接复用 `artifacts/release_protocol_seed42`。首次冻结时指定**本轮之前的历史运行目录**，不要混入新增实验目录：

```bash
python -m criteo_uplift.experiment_protocol \
  --data data/raw/criteo-uplift-v2.1.csv \
  --history artifacts/final_conversion_1000000_seed42 \
  --output artifacts/release_protocol_seed42 --seed 42
```

历史目录在此仅用来核验已接触过的行号，不再提供旧模型运行接口。协议目录拒绝覆盖；缺少历史行号或冻结协议时，无法声称复现了完全相同的正式测试集。

```bash
python run_pipeline.py \
  --stage development --protocol artifacts/release_protocol_seed42 \
  --sample-size 5000000 --outcome conversion --seed 42 \
  --primary-metric gain_ht --primary-budget 0.20 --bootstrap 200 \
  --permutations 499 --c2st-sample 20000 --trees 150 --skip-stability \
  --output artifacts/development_conversion_5000000_seed42

python run_pipeline.py \
  --stage full --protocol artifacts/release_protocol_seed42 \
  --sample-size 0 --outcome conversion --seed 42 \
  --primary-metric gain_ht --primary-budget 0.20 --bootstrap 200 \
  --permutations 499 --c2st-sample 20000 --trees 150 --skip-stability \
  --output artifacts/final_conversion_full_seed42
```

输出目录必须为空。再次运行请选择新目录，保留同一冻结协议。每次运行保存全量审计、划分行号、训练诊断、八份策略模型、验证与测试评分、指标、区间、运行源码指纹和 `manifest.json`。

只重新生成报告时无需训练：

```bash
python -c "from pathlib import Path; from criteo_uplift.reporting import build_report; p=Path('artifacts/development_conversion_5000000_seed42'); build_report(p/'manifest.json', Path('reports/development/report.html'))"
python -c "from pathlib import Path; from criteo_uplift.reporting import build_report; p=Path('artifacts/final_conversion_full_seed42'); build_report(p/'manifest.json', Path('reports/final/report.html'))"
```

## 仓库结构

| 路径 | 职责 |
| --- | --- |
| `run_pipeline.py` | 两阶段运行入口与固定协议校验 |
| `criteo_uplift/experiment_protocol.py` | 预冻结测试行号、历史排除与指纹验证 |
| `criteo_uplift/` | 审计、诊断、建模、评价、可选稳定性分析、业务规划与报告 |
| `tests/` | 正式自动化回归测试，全部使用合成数据 |
| `reports/development/report.html` | 500 万开发报告 |
| `reports/final/report.html` | 全量正式报告 |
| `data/`、`artifacts/` | 本地原始数据与可复现运行工件，不纳入 Git |

仓库不保留演示入口、临时调试脚本或旧版发布报告。测试集证据只用于评价预冻结策略；任何线上结论仍需目标业务人群中的随机策略对照实验。

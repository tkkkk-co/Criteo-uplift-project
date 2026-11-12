# Criteo Uplift Project

在固定触达预算下，比较响应概率策略与增量模型能否带来更多增量转化。项目围绕公开随机实验数据，完成数据审计、实验诊断、ATE/ITT、S/T/X Learner、冻结策略评价、稳定性分析与业务实验规划。

正式展示只有两个入口：

- [完整学习 Notebook](notebooks/criteo_uplift_project.ipynb)：项目逻辑、理论解释、结果阅读和八行指标教学核验。
- [完整 HTML Report](reports/published/conversion_1000000_seed42_refactored/report.html)：业务背景、模型比较、Test、Stability、打分接口和 Planning。

## 当前完成的实验

原始数据全量审计，建模使用均匀无放回抽取的 1,000,000 行。按 treatment × conversion 联合分层划分 Train / Validation / Test = 60% / 20% / 20%。正式模型和预处理只在 Train 拟合；Validation 在 20% 主预算下选择并冻结策略，Test 只用于评价。

四个候选保持为 S＋Logistic、T＋Logistic、T＋XGBoost、X＋XGBoost。XGBoost 默认 150 轮、depth=4、learning_rate=0.05、min_child_weight=20、subsample=0.8、colsample_bytree=0.9、hist；本项目没有做超参数搜索或 early stopping。诊断中的 XGBoost 有独立配置，不能与正式基模型混淆。

Validation 冻结的最终策略是 **response propensity**，即 P(Y=1 | X,T=1)，不是处理分配倾向分数。Test 上 20% 预算的增量估计为每 10 万名符合条件者约 **117.23** 次转化，95% 条件性 Bootstrap 区间约 **[64.09, 166.21]**。这是假设成立时的离线估计，不是实际广告投放或收入增长。最佳 uplift 候选为 S＋Logistic；不能按 Test 上更高的点估计回头改选。

Stability 包含种子 17/29/83、训练比例 25%/50%/100% 和特征分组划分，共 12 个开发配置、48 次模型拟合，排除原 Test 的 200,000 行。重复开发结果的标准差不是独立 Test 置信区间。

## 安装与验证

```bash
python -m pip install -e '.[notebooks]'
python -m unittest discover -s tests
python scripts/execute_notebooks.py
python scripts/check_publication.py
```

不下载原始数据也可以执行测试、查看聚合快照和运行 Notebook。测试中的人工数据仅用于核验实现，不作为 Criteo 实验结果。macOS 的 XGBoost 需要可用的 OpenMP 运行库；已有可信库可通过 `CRITEO_OPENMP_LIBRARY` 指定。项目不发布本机运行库二进制。

## 完整 Pipeline

自行获取 Criteo Uplift v2.1 原始 CSV，默认放在 `data/raw/criteo-uplift-v2.1.csv`。data 目录只读。

```bash
python run_pipeline.py --output artifacts/conversion_1000000_seed42_repeat
```

默认完整执行 C2ST 499 次完整重拟合置换、150 轮 XGBoost、200 次 Bootstrap，以及全部稳定性配置。已有非空输出目录会被拒绝。`--sample-size 0` 表示全量建模，不改变默认的百万行实验。

```bash
python run_pipeline.py --help
python run_pipeline.py --outcome visit --sample-size 1000000 --seed 42 \
  --output artifacts/visit_1000000_seed42
```

`--stability-seeds`、`--stability-fractions` 可显式配置敏感性分析；正式默认设计不变。`--planning-assumptions assumptions.json` 接受完整业务假设对象，字段见 `planning.DEFAULT_PLANNING_ASSUMPTIONS`。未指定时沿用原项目明确标记的教学情景，所有参数都会保存到 `business/assumptions.json`。

## 新用户打分

```bash
python score_policy.py --run artifacts/conversion_1000000_seed42_refactored \
  --input /path/to/new_users.csv \
  --output artifacts/conversion_1000000_seed42_refactored/business/score_policy_scores.csv \
  --budget 0.2
```

输入只要求 f0–f11，按块读取。输出 `source_row_id`、`score`、`selected`，不进行训练或真实广告投放。当前没有提供真实新人群数据，报告明确标注未执行。没有 T、Y 就不能计算观测 ATE、ATT 或 ITT。

## 代码与结果结构

```text
criteo_uplift/
  config.py                 路径、列名、默认参数、运行环境与日志
  input_checks.py           二元标签、特征矩阵、行号的公共检查
  pipeline_io.py            指纹、路径保护、结果与缓存读写
  preparation.py            三任务开关、独立缓存、统一调度、样本划分
  raw_scan.py               单次流式读取、校验、抽样、审计及诊断任务
  diagnostics.py            SRM、SMD/KS、C2ST、Learnability
  models.py                 结果/效应模型工厂与通用 S/T/X
  training_validation.py    正式候选、Train 拟合、Validation 冻结
  metrics.py                排序指标、HT、Qini、Bootstrap 公式
  evaluation.py             ATE、策略值、校准与 Test 评价
  stability.py              排除 Test 的开发敏感性分析
  planning.py               假设业务参数、A/B 规划、经济情景
  reporting.py              读取结果、生成图表与唯一正式报告
run_pipeline.py             主流程入口
score_policy.py             X-only 新人群打分入口
notebooks/criteo_uplift_project.ipynb
tests/
docs/
reports/published/          可发布聚合快照
artifacts/                  本地缓存与实验结果，不发布
```

每个实验目录下统一为 `data_preparation/`、`diagnostics/`、`models/`、`evaluation/`、`stability/`、`business/`、`figures/`、`manifest.json`、`report.html`。全量和 Test 的 ATE 都保存在 `evaluation/ate_summary.csv`，用 `scope` 与 `outcome` 区分。

Sample、Audit、Diagnostics 各自缓存在 `artifacts/cache/`，独立检查来源和所需参数。审计与诊断规则固定，不使用版本号；建模样本保留抽样规则校验。全部命中时不解析原始 CSV，任一任务未命中时最多执行一次共用的分块扫描。模型及实验评价不会写入全局缓存。

详细说明见 [项目流程](docs/PROJECT_PIPELINE.md)、[结果阅读指南](docs/RESULTS_GUIDE.md)、[函数迁移与验收清单](docs/maintenance/FINAL_REFACTOR.md)。

## 当前结果与缓存

本地 `artifacts` 中的旧缓存和旧实验结果已清理，当前内容由现有代码完整重跑生成；公开聚合快照及 Notebook 输出也已同步刷新。原始 CSV 保持不变，不保留旧缓存迁移或旧模型兼容文件。

C2ST 的 20,000 行 float64 样本保存在 `artifacts/cache/c2st_sample_n20000_seed42.npz`，同名 JSON 记录来源及校验信息。实验目录的 `diagnostics/` 保存检验结果，而非样本。独立的 [C2ST 敏感性分析](reports/analysis/c2st_sensitivity.md) 复用该缓存；[英文 Report 小节草稿](reports/analysis/c2st_sensitivity_report_section.md) 单独交付，没有替换正式主分析。

## 发布聚合快照

```bash
python scripts/export_public_results.py \
  --source artifacts/conversion_1000000_seed42_refactored \
  --output reports/new_snapshot
python scripts/check_publication.py
```

导出拒绝覆盖已有快照，只包含明确列出的聚合统计和展示图。原始数据、样本、行号列表、逐行预测、模型和本机绝对路径不进入正式快照。`scripts/build_release.py` 使用发布清单生成 ZIP。

## 解释与许可

`treatment` 是分配，`exposure` 是处理后信息，仅作审计，不进入特征或人群筛选。匿名特征不能确认用户身份；保留重复内容候选，不自动去重。公开样本的事后诊断不能证明原始线上随机化，也不能消除未知实体依赖。

代码许可见 [LICENSE](LICENSE)，数据与衍生成果使用边界见 [NOTICE.md](NOTICE.md)。数据来源为 [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)，Meta Learner 方法参考 [Künzel et al.](https://arxiv.org/abs/1706.03461)。

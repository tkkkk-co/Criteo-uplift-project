# 项目 Pipeline 导航

> 本文只解释这个仓库的执行路径、模块职责、输入输出和复现方式。通用统计、因果推断、C2ST、AUUC/Qini 等理论推导统一放在 DSDAHANDBOOK，不在仓库重复维护。

## 从哪里开始读代码

**先读 §1 的流程图，再打开 run_pipeline.py 的 run()。不要一开始就逐行阅读所有 criteo_uplift/*.py。**

主流程入口是仓库根目录的 run_pipeline.py。它负责串联阶段；每个阶段的实现留在对应模块。run_stability.py 与 score_policy.py 是两个独立工作流，不是第二套主 Pipeline。

## 1. 主实验执行顺序

~~~text
run_pipeline.py::main / parse_args
  └─ run_pipeline.py::run
      1. 创建新的实验输出目录，并检查 data/ 只读规则
      2. preparation.py::prepare_sample
          ├─ data.py::fingerprint 核验原始 CSV
          ├─ 分别检查/复用建模样本缓存与全量审计缓存
          └─ diagnostics.py::scan_raw_data
              ├─ 全量数据审计
              ├─ SRM 与协变量平衡所需统计
              ├─ 建模样本抽样（需要时）
              └─ C2ST 两臂等量抽样
      3. data.py::split_sample → train / validation / test
      4. pipeline_io.py::write_split_artifacts / write_feature_overlap
      5. pipeline_io.py::write_audit_artifacts → audit.json / ate_full.csv；另存 sample_report.json
      6. diagnostics.py::update_experiment_diagnostics
          └─ diagnostics.py::c2st → c2st.csv / c2st_null.csv / c2st_details.json
          └─ 保存 srm.json、balance.csv
      7. training.py::run_learnability_diagnostics → learnability.csv
      8. training.py::fit_and_select_models
          ├─ models.py::make_model
          ├─ 仅用 train 拟合候选模型
          └─ 仅用 validation 冻结策略选择
      9. holdout_evaluation.py::evaluate_test_set
          └─ 对冻结模型做一次 test 评价并保存结果
     10. pipeline_io.py::write_run_manifest
     11. run_pipeline.py::finalize_run → reporting.py::build_report
~~~

注意：当前代码在 prepare_sample() 内通过一次分块 CSV 扫描复用多个任务的结果。不要为了让函数看起来更短，把这次扫描拆成多次独立扫描；那可能改变计算成本、抽样流和诊断结果。

## 2. 主流程各阶段的职责

| 阶段 | 入口/核心函数 | 输入 | 关键输出 |
|---|---|---|---|
| 配置和编排 | run_pipeline.py::parse_args, run() | CLI 参数 | 决定本次运行的参数与阶段顺序 |
| 来源核验、缓存和扫描 | preparation.py::prepare_sample | 原始 CSV、抽样规模、seed、缓存 | 样本、审计字典、来源指纹、可复用扫描结果 |
| 全量扫描 | diagnostics.py::scan_raw_data | 原始 CSV 分块 | 审计、SRM/SMD/KS 所需统计、建模样本、C2ST 样本 |
| 数据划分 | data.py::split_sample | 建模样本 | 60% train / 20% validation / 20% test |
| 产物写入 | pipeline_io.py::write_* | 已算好的统计和元数据 | CSV、JSON、缓存、manifest |
| 诊断 | diagnostics.py::srm_summary, covariate_balance, c2st, update_experiment_diagnostics | 原始来源统计和 C2ST 样本 | SRM、SMD/KS、C2ST 结果 |
| 结果可学习性 | training.py::run_learnability_diagnostics → diagnostics.py::learnability | train、validation | learnability.csv |
| 模型定义 | models.py::make_model 和模型类 | 特征、T、Y | S/T/X learner 对象 |
| 训练与选型 | training.py::fit_and_select_models | train、validation | 模型文件、验证指标、selection.json |
| 测试集评价 | holdout_evaluation.py::evaluate_test_set → evaluation.py | test、已冻结模型 | 策略值、排序指标、区间、校准、预测 |
| 报告 | reporting.py::build_report | 已保存产物 | report.html 和图表 |

## 3. Python 文件应该怎样读

| 文件 | 业务边界 | 什么时候打开 |
|---|---|---|
| run_pipeline.py | CLI、主流程编排、收尾 | 想知道“整体什么时候调用什么”时首先打开 |
| criteo_uplift/preparation.py | 样本缓存校验和准备流程 | 研究原始 CSV 如何变成可用样本时 |
| criteo_uplift/data.py | 数据列/路径常量、文件指纹、train/validation/test 划分 | 研究来源指纹或 split 规则时 |
| criteo_uplift/diagnostics.py | 全量扫描、SRM、SMD/KS、C2ST、结果可学习性 | 学 Notebook 01–02 时 |
| criteo_uplift/pipeline_io.py | 路径保护、JSON、样本缓存、实验产物写入、结果目录解析 | 需要知道某个文件怎么保存或从哪里读取时 |
| criteo_uplift/models.py | S/T/X learner 与模型工厂 | 学模型具体实现时 |
| criteo_uplift/training.py | 训练候选模型、validation 选型、冻结选择 | 学模型如何被比较和选出时 |
| criteo_uplift/evaluation.py | 纯指标和统计计算 | 追踪 ATE、HT、AUUC/Qini、policy、CI 等计算时 |
| criteo_uplift/holdout_evaluation.py | 在 test 上执行固定模型评价并写结果 | 学 Notebook 04 时 |
| criteo_uplift/reporting.py | 读取已有结果、渲染图表和 HTML | 想知道报告里的图表从哪里来时 |
| criteo_uplift/planning.py | 样本量、预算和成本/价值假设情景 | 学 Notebook 06 的决策情景时 |
| criteo_uplift/sql_audit.py | SQL 结果与 Python 结果核对 | 学 Notebook 06 的 SQL 部分时 |
| criteo_uplift/stability.py | 开发样本上的训练规模和切分稳定性分析 | 学 Notebook 06 的稳定性部分时 |
| criteo_uplift/runtime.py | 数值库线程等运行时设置 | 环境启动问题时；通常不需要先学 |
| criteo_uplift/demo.py | 合成数据演示 | 验证接口/环境，不用于解释真实 Criteo 结果 |

## 4. 三个入口不是同一件事

### 主 Pipeline：run_pipeline.py

负责新建一轮实验，按顺序准备数据、诊断、训练、验证集选型、测试集评价并生成报告。默认输出目录按结果、样本规模和 seed 命名，例如 artifacts/conversion_1000000_seed42/。若该目录已有正式结果，完整运行会拒绝覆盖。

### 稳定性工作流：run_stability.py

调用 criteo_uplift/stability.py，分析不同训练规模/划分设置下的开发期稳定性。它是扩展分析，不会自动成为主 Pipeline 的下一阶段，也不应据此回头更换已经冻结的测试集结果。

### 冻结策略打分：score_policy.py

读取已经保存并冻结的策略/模型，对新特征文件进行离线评分和预算筛选。它不负责重新训练或重新选型；输入应只读，输出不得写入 data/。

SQL 审计和指标扩展则分别通过 scripts/run_sql_audit.py 与 scripts/run_metrics_extension.py 等脚本启动。具体命令参数以各入口的 --help 为准。

## 5. 结果目录、缓存与公开快照

- data/ 是原始数据目录，只读。
- artifacts/ 存放本地正式运行、缓存和扩展分析产物。缓存通常位于 artifacts/cache/。
- 抽样结果 `sample_<size>_seed<seed>.npz` 与配套 `.json` 只描述样本；全量审计按原始来源 SHA256 独立保存为 `cache/full_audit_<sha256>.json`，内部记录审计版本。更换抽样规模或 seed 不会使同一来源的全量审计失效。
- 当前抽样说明统一保存于 `data_preparation/sample_report.json`，全量审计保存于 `data_preparation/audit.json`。现有实验的历史指标和 HTML 内容保留。
- 每次主实验的文件分为 `data_preparation/`、`diagnostics/`、`models/`、`evaluation/`；顶层保留 `manifest.json`、`report.html`、`figures/`。划分汇总使用 `data_preparation/split_summary.csv`。
- `run()` 在开头一次性创建固定子目录，并把具体输出目录传给对应步骤。保存函数直接写文件，读取也使用同一固定路径；不保留文件名分派、旧名映射或旧目录回退。
- reports/published/ 存放可以随仓库分发的聚合结果快照，不包含完整逐行预测、训练样本或 fitted models。
- criteo_uplift/pipeline_io.py::result_directory(name, root) 默认先找本地 artifacts/name/，再找 reports/published/name/。
- 环境变量 CRITEO_RESULTS_MODE=published 强制使用公开快照；auto（默认）优先使用本地 artifacts/。
- Notebook 默认展示已保存的结果，不会因为展示结果而自动重训正式模型。要重跑主实验，必须显式运行 Pipeline。
- 输出路径必须经过 safe_output() 保护；不要将结果写入 data/，也不要覆盖已有正式实验目录。

## 6. 环境、数据与常用命令

### 首次建立本地环境

在项目根目录执行；项目要求 Python 3.11 或兼容版本：

~~~bash
python -m venv .venv
python -m pip install -e '.[notebooks]'
python -m unittest discover -s tests -v
~~~

然后可以先执行公开快照模式的 Notebook 检查：

~~~bash
python scripts/execute_notebooks.py
~~~

这一步默认不需要原始 Criteo CSV。若要重训、运行稳定性扩展或生成逐行预测，需要自行从 [Criteo 官方下载页](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)获取符合项目版本要求的数据，放入 data/raw/criteo-uplift-v2.1.csv，并遵守数据集许可证与 NOTICE.md。原始数据、逐行预测、缓存模型和本机审计备份不随公开仓库分发。

### 常用命令

在项目根目录、已安装项目依赖的 Python 环境中执行：

~~~bash
# 查看主实验参数，不启动训练
python run_pipeline.py --help

# 运行一轮新的 100 万行开发实验（使用一个新的输出目录）
python run_pipeline.py --sample-size 1000000 --outcome conversion --output artifacts/conversion_repeat

# 只更新已有实验的诊断；不重新训练正式模型
python run_pipeline.py --diagnostics-only --output artifacts/conversion_1000000_seed42

# 查看独立稳定性和策略打分入口
python run_stability.py --help
python score_policy.py --help

# 使用当前 Python 环境刷新 Notebook 输出（默认读取公开快照，不需要原始 CSV）
python scripts/execute_notebooks.py
# 如需优先读取本机 artifacts，明确指定
python scripts/execute_notebooks.py --local-results
~~~

首次配置环境、数据来源和发布快照的细节可从 README 与 NOTICE.md 开始；本文件不重复记录通用 Python 或统计学习知识。

## 7. 学习代码的推荐顺序

1. 只读 run_pipeline.py::run()，把每个编号阶段和上表对应起来。
2. 学 Notebook 01 时沿着 prepare_sample → scan_raw_data → split_sample 阅读。
3. 学 Notebook 02 时沿着 update_experiment_diagnostics → c2st 阅读，再回到 Notebook 查看结果。
4. 学 Notebook 03 时读 models.py → training.py::fit_and_select_models。
5. 学 Notebook 04 时读 holdout_evaluation.py::evaluate_test_set → evaluation.py。
6. 学 Notebook 05 时把八行手算输入与 evaluation.py 输出逐项核对。
7. 学 Notebook 06 时分别读 stability.py、sql_audit.py 和 planning.py。

每次只沿当前业务问题追踪相关函数。Notebook 是结果解释入口，run_pipeline.py 是调用地图，模块函数才是具体实现。

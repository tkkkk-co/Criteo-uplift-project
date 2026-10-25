# Python 职责重构与验收报告

## 1. 基线和执行范围

- 基线提交：`cbb59d4677379094bbc9cd0ba0992e933136f3d9`。
- 工作区原本包含上一轮注释、数值边界修复与用户 Notebook 编辑。本次以这些当前文件为基线，先保存副本和指纹，没有回退、提交或覆盖原有编辑。
- 按方案完成 Phase 0–5；临时工作目录为 `/tmp/criteo-refactor-20261025`。
- 生产代码只调整职责和导入；现有数据、随机规则、拆分、训练、选型、指标、SQL 和结果字段保持原定义。测试与新图全部写入临时目录。

## 2. 新的 Python 文件树

共 41 个 Python 文件：核心包 15 个、顶层入口 3 个、脚本 12 个、测试 11 个。

```text
Criteo-uplift-project/
├── run_pipeline.py
├── run_stability.py
├── score_policy.py
├── criteo_uplift/
│   ├── __init__.py
│   ├── data.py
│   ├── demo.py
│   ├── diagnostics.py
│   ├── evaluation.py
│   ├── holdout_evaluation.py
│   ├── models.py
│   ├── pipeline_io.py
│   ├── planning.py
│   ├── preparation.py
│   ├── reporting.py
│   ├── runtime.py
│   ├── sql_audit.py
│   ├── stability.py
│   └── training.py
├── scripts/
│   ├── build_extension_report.py
│   ├── build_release.py
│   ├── check_publication.py
│   ├── create_extension_notebooks.py
│   ├── create_notebooks.py
│   ├── execute_notebooks.py
│   ├── export_public_results.py
│   ├── refresh_metrics.py
│   ├── run_demo.py
│   ├── run_metrics_extension.py
│   ├── run_sql_audit.py
│   └── verify_integrity.py
├── tests/
│   ├── test_data.py
│   ├── test_diagnostics.py
│   ├── test_end_to_end.py
│   ├── test_evaluation.py
│   ├── test_models.py
│   ├── test_planning.py
│   ├── test_publication.py
│   ├── test_raw_scan.py
│   ├── test_scoring.py
│   ├── test_sql_audit.py
│   └── test_stability.py
```

## 3. 函数迁移清单

| 函数 | 原文件 | 目标文件 | 状态 |
|---|---|---|---|
| `safe_output` | `criteo_uplift/data.py` | `criteo_uplift/pipeline_io.py` | 迁移，保留原计算 |
| `write_json` | `criteo_uplift/data.py` | `criteo_uplift/pipeline_io.py` | 迁移，保留原计算 |
| `write_json.convert` | `criteo_uplift/data.py` | `criteo_uplift/pipeline_io.py` | 迁移，保留原计算 |
| `fingerprint` | `criteo_uplift/data.py` | `criteo_uplift/pipeline_io.py` | 迁移，保留原计算 |
| `save_sample` | `criteo_uplift/data.py` | `criteo_uplift/pipeline_io.py` | 迁移，保留原计算 |
| `load_sample` | `criteo_uplift/data.py` | `criteo_uplift/pipeline_io.py` | 迁移，保留原计算 |
| `finalize_run` | `criteo_uplift/pipeline_io.py` | `run_pipeline.py` | 迁移，保留原计算 |
| `update_diagnostics_only` | `criteo_uplift/pipeline_io.py` | `run_pipeline.py` | 迁移，保留原计算 |
| `result_directory` | `criteo_uplift/results.py` | `criteo_uplift/pipeline_io.py` | 迁移，保留原计算 |

- `write_json.convert` 继续作为内部函数，随 `write_json` 迁移，未提升为公共 API。
- `run_pipeline.prepare_sample()` 保留：`tests/test_raw_scan.py` 仍导入此入口，继续转发到 `preparation.prepare_sample()`。
- `scripts/refresh_metrics.py` 保留：它提供旧的、带位置参数 `source` 的 CLI，帮助检查和兼容调用继续可用。外部旧脚本无法通过仓库搜索排除，因此保留薄入口。
- `runtime.py`、模型、指标、规划、SQL、训练与测试评价等业务边界继续独立。
- 按方案第 4.2 节补齐 `write_split_artifacts()` 的入口保护：原先依赖上层 `run()` 检查输出路径，现直接使用原有 `safe_output(out)`。合法路径下的文件内容和字段不变；单独调用该函数写入 `data/` 或其符号链接时会提前拒绝。没有改变 `safe_output` 本身的保护规则。

## 4. 删除、引用与文件变动

- 删除 `criteo_uplift/results.py`：函数已迁移，删除前检查了受版本管理的 Python、Notebook、Markdown、HTML、配置和 CI 文本，旧导入及旧文件链接均已处理。最终源码中的旧结果模块导入残留为 0。
- 原 11 个测试文件保持分组，没有合并测试文件。
- 未修改发布 allowlist 的规则、数据格式或公开快照内容。

导入变动：

| 原导入归属 | 新导入归属 | 同步范围 |
|---|---|---|
| `data` 中五个 IO 函数 | `pipeline_io` | 核心包、入口、脚本、测试、Notebook 1 与生成器、学习文档 |
| `results.result_directory` | `pipeline_io.result_directory` | 6 个 Notebook、2 个生成器、发布测试、学习文档链接 |
| `pipeline_io.finalize_run`、`pipeline_io.update_diagnostics_only` | `run_pipeline` 内直接定义 | 主入口调用与依赖关系 |

以下变动以本次开始时的工作区为参照，不把之前已存在的编辑重复计为本次修改：

| 文件 | 本次状态 |
|---|---|
| `README.md` | 修改 |
| `criteo_uplift/data.py` | 修改 |
| `criteo_uplift/demo.py` | 修改 |
| `criteo_uplift/diagnostics.py` | 修改 |
| `criteo_uplift/holdout_evaluation.py` | 修改 |
| `criteo_uplift/pipeline_io.py` | 修改 |
| `criteo_uplift/preparation.py` | 修改 |
| `criteo_uplift/reporting.py` | 修改 |
| `criteo_uplift/results.py` | 删除 |
| `criteo_uplift/sql_audit.py` | 修改 |
| `criteo_uplift/stability.py` | 修改 |
| `criteo_uplift/training.py` | 修改 |
| `docs/01_数据审计与EDA学习笔记.html` | 修改 |
| `docs/01_数据审计与EDA学习笔记.md` | 修改 |
| `notebooks/01_data_understanding.ipynb` | 修改 |
| `notebooks/02_sanity_and_ate.ipynb` | 修改 |
| `notebooks/03_uplift_models.ipynb` | 修改 |
| `notebooks/04_policy_evaluation.ipynb` | 修改 |
| `notebooks/05_metrics_worked_example.ipynb` | 修改 |
| `notebooks/06_robustness_sql_and_decisions.ipynb` | 修改 |
| `run_pipeline.py` | 修改 |
| `score_policy.py` | 修改 |
| `scripts/build_extension_report.py` | 修改 |
| `scripts/build_release.py` | 修改 |
| `scripts/create_extension_notebooks.py` | 修改 |
| `scripts/create_notebooks.py` | 修改 |
| `scripts/export_public_results.py` | 修改 |
| `scripts/refresh_metrics.py` | 修改 |
| `scripts/run_metrics_extension.py` | 修改 |
| `tests/test_data.py` | 修改 |
| `tests/test_diagnostics.py` | 修改 |
| `tests/test_end_to_end.py` | 修改 |
| `tests/test_publication.py` | 修改 |
| `tests/test_raw_scan.py` | 修改 |
| `tests/test_scoring.py` | 修改 |
| `tests/test_sql_audit.py` | 修改 |
| `tests/test_stability.py` | 修改 |
| `docs/PYTHON_REFACTOR_REPORT.md` | 新增 |
| `docs/PYTHON_REFACTOR_VALIDATION.json` | 新增 |

## 5. 分阶段测试结果

| 阶段 | 测试范围 | 通过数 | 失败 / 错误 / 跳过 |
|---|---|---:|---|
| phase0 | 完整测试基线 | 85 | 0 / 0 / 0 |
| phase1 | test_data、test_raw_scan、test_scoring | 23 | 0 / 0 / 0 |
| phase2 | test_end_to_end、test_diagnostics、test_publication | 14 | 0 / 0 / 0 |
| phase3 | test_publication | 3 | 0 / 0 / 0 |
| phase4 | test_raw_scan + 10 个 CLI 帮助入口 | 7 | 0 / 0 / 0 |
| phase5 | 完整测试验收 | 85 | 0 / 0 / 0 |

测试日志与复现命令：

```bash
/opt/miniconda3/envs/causalml/bin/python -B /tmp/criteo-refactor-20261025/check_stage.py phase0
/opt/miniconda3/envs/causalml/bin/python -B /tmp/criteo-refactor-20261025/check_stage.py phase1 test_data test_raw_scan test_scoring
/opt/miniconda3/envs/causalml/bin/python -B /tmp/criteo-refactor-20261025/check_stage.py phase2 test_end_to_end test_diagnostics test_publication
/opt/miniconda3/envs/causalml/bin/python -B /tmp/criteo-refactor-20261025/check_stage.py phase3 test_publication
/opt/miniconda3/envs/causalml/bin/python -B /tmp/criteo-refactor-20261025/check_stage.py phase4 test_raw_scan
/opt/miniconda3/envs/causalml/bin/python -B /tmp/criteo-refactor-20261025/check_stage.py phase5
```

运行器调用 unittest discovery，并在导入数值库前使用项目已有 runtime 配置；每阶段日志位于上述临时目录的 `phaseN-tests.log`。日常完整测试也可使用 `python -B -m unittest discover -s tests -v`。

- Phase 1 首次运行出现 1 个测试错误：`test_raw_scan.py` 已更新为调用 `pipeline_io.fingerprint`，但遗漏模块导入；补上导入后 23 项全部通过，初次日志保留为 `phase1-first-attempt-tests.log`。
- Notebook 校验工具最初把 JSON 单元格源码的分行表示差异误报为内容差异；统一为源码字符串后，确认仅导入改变，原输出与执行记录相同。Jupyter 绑定本地端口被沙箱阻止后，已通过授权在沙箱外执行临时副本。
- 模型对比工具最初未调用 runtime，导致本机 XGBoost 缺少已配置的 OpenMP 加载步骤；使用项目现有配置后 8 份模型的参数和预测对比均通过。上述问题没有引起算法或依赖配置修改。

## 6. 最终验收与等价性证据

- 10 个 CLI 的 `--help` 均成功，与基线输出逐字一致：主实验、稳定性、打分、SQL、扩展指标、旧指标入口、演示、Notebook 执行、发布打包、公开结果导出。
- 重构前后均执行 7 项合成数据流程：首次完整运行、缓存复用运行、仅更新诊断、离线打分、SQL 核对、扩展指标、稳定性分析；模型文件、冻结选择与预测文件在仅更新诊断前后指纹一致。
- 对比 94 份临时产物：50 个 CSV、20 个 JSON、3 个 NPZ、8 个模型、10 张 PNG、2 个 HTML、1 份 SQL。CSV 非耗时字段逐值相同，NPZ 键、dtype、shape 和数组完全相同，模型参数与预测相同，JSON schema 与非运行元数据相同，图片与 SQL 字节一致。HTML 仅允许显示耗时不同。
- 等价性对比明确排除重新运行必然变化的耗时、生成文件修改时间，以及包含耗时记录的 manifest/protocol 指纹；未忽略数据、抽样行号、统计量、p 值、选择结果、预算或置信区间。
- 113 个生产顶层函数/类的 AST 对比通过：除导入迁移与上文明确记录的一处 IO 入口保护外，其余执行语句不变。15 个现存核心模块中的常量 AST 相同；`evaluation.py`、`models.py`、`planning.py`、`runtime.py` 的文件字节与工作区基线一致。
- 依赖检查通过：`data` 不依赖 IO；IO 不直接或间接加载 diagnostics、reporting、主入口；业务模块不反向导入主入口。
- 两个 Notebook 生成器各生成前后对照副本，6 个生成 Notebook 的源码仅有导入迁移差异；原有 6 个 Notebook 的 Markdown、输出、执行计数和元数据保持不变。
- 6 个 Notebook 共 32 个代码单元以公开快照模式在独立 Jupyter 内核执行通过；执行副本和新图片保存在临时目录。
- 发布检查核验 manifest 中的 56 份公开快照文件通过；完整文件保护进一步核验了 `reports/published/` 下的全部 58 个文件。发布打包命令也在临时目录成功生成 ZIP，未包含原始数据、缓存或逐行模型数组。
- 对 `data/` 2 个文件、`artifacts/` 168 个文件、`reports/published/` 58 个文件，共 228 个文件比较相对路径、大小、mtime_ns 和 SHA256：新增 0、删除 0、修改 0。

Notebook 明细：

| Notebook | 已执行代码单元 | 结果 |
|---|---:|---|
| `01_data_understanding.ipynb` | 6 | 通过 |
| `02_sanity_and_ate.ipynb` | 7 | 通过 |
| `03_uplift_models.ipynb` | 4 | 通过 |
| `04_policy_evaluation.ipynb` | 4 | 通过 |
| `05_metrics_worked_example.ipynb` | 5 | 通过 |
| `06_robustness_sql_and_decisions.ipynb` | 6 | 通过 |

## 7. 兼容性与使用边界

- 现有命令入口、参数默认值、数据格式、模型类路径、结果文件名与字段 schema 保持可用；未重训或覆盖正式实验。
- 按方案删除旧结果模块且不让 data 反向依赖 IO，因此仓库外仍使用旧 Python 导入路径的脚本，需要按迁移表更新；仓库内 Python、Notebook、生成器和文档引用均已同步。
- 临时合成数据前后对比用于验证重构等价性，不构成新的真实 Criteo 实验结论。

## 8. 完整函数与类归属清单

以下逐项覆盖基线中的真实 Python 定义。生成器代码字符串中的教学函数另通过生成 Notebook 的源码对比验证。

| 原文件 | 定义 | 新文件 | 状态 |
|---|---|---|---|
| `criteo_uplift/data.py` | `safe_output` | `criteo_uplift/pipeline_io.py` | 迁移 |
| `criteo_uplift/data.py` | `write_json` | `criteo_uplift/pipeline_io.py` | 迁移 |
| `criteo_uplift/data.py` | `write_json.convert` | `criteo_uplift/pipeline_io.py` | 迁移 |
| `criteo_uplift/data.py` | `fingerprint` | `criteo_uplift/pipeline_io.py` | 迁移 |
| `criteo_uplift/data.py` | `save_sample` | `criteo_uplift/pipeline_io.py` | 迁移 |
| `criteo_uplift/data.py` | `load_sample` | `criteo_uplift/pipeline_io.py` | 迁移 |
| `criteo_uplift/data.py` | `split_sample` | `criteo_uplift/data.py` | 保留 |
| `criteo_uplift/demo.py` | `run_demo` | `criteo_uplift/demo.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `scan_raw_data` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `_audit_feature_stats` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `_features` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `_binary` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `srm_summary` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `covariate_balance` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `_model` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `_probability_metrics` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `c2st` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `_holm` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `learnability` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/diagnostics.py` | `update_experiment_diagnostics` | `criteo_uplift/diagnostics.py` | 保留 |
| `criteo_uplift/evaluation.py` | `_binary_inputs` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `_scores` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `_ht_outcome` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `_rank` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `_mean_difference` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `aggregate_ate_from_audit` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `ate_summary` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `uplift_curve` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `_area` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `_ranking_from_ordered` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `ranking_metrics` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `ranking_confidence_intervals` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `ranking_confidence_intervals.values` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `_prefix_values` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `evaluate_policies` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/evaluation.py` | `calibration_table` | `criteo_uplift/evaluation.py` | 保留 |
| `criteo_uplift/holdout_evaluation.py` | `evaluate_test_set` | `criteo_uplift/holdout_evaluation.py` | 保留 |
| `criteo_uplift/models.py` | `_feature_array` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_binary_vector` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_ConstantProbabilityClassifier` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_ConstantProbabilityClassifier.fit` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_ConstantProbabilityClassifier.predict_proba` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_positive_probability` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_UpliftEstimator` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_UpliftEstimator._prepare_fit` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_UpliftEstimator._prepare_predict` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_UpliftEstimator._fit_binary` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_UpliftEstimator.predict` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_LogisticConfiguration` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_LogisticConfiguration.__init__` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_LogisticConfiguration._new_classifier` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `SLearnerLogistic` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `SLearnerLogistic.fit` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `SLearnerLogistic.predict_outcomes` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_TwoOutcomeModels` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_TwoOutcomeModels._fit_outcomes` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_TwoOutcomeModels.fit` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_TwoOutcomeModels.predict_outcomes` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `TLearnerLogistic` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_XGBoostConfiguration` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_XGBoostConfiguration.__init__` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_XGBoostConfiguration._tree_parameters` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_XGBoostConfiguration._xgboost_classes` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_XGBoostConfiguration._new_classifier` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `_XGBoostConfiguration._new_regressor` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `TLearnerXGBoost` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `XLearnerXGBoost` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `XLearnerXGBoost.fit` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `XLearnerXGBoost.predict` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/models.py` | `make_model` | `criteo_uplift/models.py` | 保留 |
| `criteo_uplift/pipeline_io.py` | `write_split_artifacts` | `criteo_uplift/pipeline_io.py` | 保留，补齐现有路径保护 |
| `criteo_uplift/pipeline_io.py` | `write_feature_overlap` | `criteo_uplift/pipeline_io.py` | 保留 |
| `criteo_uplift/pipeline_io.py` | `write_audit_artifacts` | `criteo_uplift/pipeline_io.py` | 保留 |
| `criteo_uplift/pipeline_io.py` | `write_run_manifest` | `criteo_uplift/pipeline_io.py` | 保留 |
| `criteo_uplift/pipeline_io.py` | `finalize_run` | `run_pipeline.py` | 迁移 |
| `criteo_uplift/pipeline_io.py` | `update_diagnostics_only` | `run_pipeline.py` | 迁移 |
| `criteo_uplift/planning.py` | `plan_policy_ab` | `criteo_uplift/planning.py` | 保留 |
| `criteo_uplift/planning.py` | `cost_value_scenario` | `criteo_uplift/planning.py` | 保留 |
| `criteo_uplift/preparation.py` | `prepare_sample` | `criteo_uplift/preparation.py` | 保留 |
| `criteo_uplift/reporting.py` | `_json` | `criteo_uplift/reporting.py` | 保留 |
| `criteo_uplift/reporting.py` | `_num` | `criteo_uplift/reporting.py` | 保留 |
| `criteo_uplift/reporting.py` | `_p` | `criteo_uplift/reporting.py` | 保留 |
| `criteo_uplift/reporting.py` | `_interval` | `criteo_uplift/reporting.py` | 保留 |
| `criteo_uplift/reporting.py` | `_table` | `criteo_uplift/reporting.py` | 保留 |
| `criteo_uplift/reporting.py` | `_image` | `criteo_uplift/reporting.py` | 保留 |
| `criteo_uplift/reporting.py` | `_errorbar` | `criteo_uplift/reporting.py` | 保留 |
| `criteo_uplift/reporting.py` | `build_report` | `criteo_uplift/reporting.py` | 保留 |
| `criteo_uplift/reporting.py` | `build_report.save` | `criteo_uplift/reporting.py` | 保留 |
| `criteo_uplift/results.py` | `result_directory` | `criteo_uplift/pipeline_io.py` | 迁移 |
| `criteo_uplift/runtime.py` | `configure_runtime` | `criteo_uplift/runtime.py` | 保留 |
| `criteo_uplift/sql_audit.py` | `sql_queries` | `criteo_uplift/sql_audit.py` | 保留 |
| `criteo_uplift/sql_audit.py` | `audit_arrays` | `criteo_uplift/sql_audit.py` | 保留 |
| `criteo_uplift/sql_audit.py` | `run_sql_audit` | `criteo_uplift/sql_audit.py` | 保留 |
| `criteo_uplift/stability.py` | `row_ids_sha256` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `_checked_ids` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `_binary_vectors` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `_validate_fractions` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `_validate_seeds` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `feature_groups` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `load_development_sample` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `nested_stratified_indices` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `development_splits` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `_counts` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `_summarize` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/stability.py` | `run_stability` | `criteo_uplift/stability.py` | 保留 |
| `criteo_uplift/training.py` | `run_learnability_diagnostics` | `criteo_uplift/training.py` | 保留 |
| `criteo_uplift/training.py` | `fit_and_select_models` | `criteo_uplift/training.py` | 保留 |
| `scripts/build_extension_report.py` | `table` | `scripts/build_extension_report.py` | 保留 |
| `scripts/build_extension_report.py` | `figure` | `scripts/build_extension_report.py` | 保留 |
| `scripts/build_extension_report.py` | `build` | `scripts/build_extension_report.py` | 保留 |
| `scripts/build_release.py` | `build` | `scripts/build_release.py` | 保留 |
| `scripts/check_publication.py` | `publication_files` | `scripts/check_publication.py` | 保留 |
| `scripts/check_publication.py` | `check` | `scripts/check_publication.py` | 保留 |
| `scripts/create_extension_notebooks.py` | `md` | `scripts/create_extension_notebooks.py` | 保留 |
| `scripts/create_extension_notebooks.py` | `code` | `scripts/create_extension_notebooks.py` | 保留 |
| `scripts/create_extension_notebooks.py` | `save` | `scripts/create_extension_notebooks.py` | 保留 |
| `scripts/create_notebooks.py` | `md` | `scripts/create_notebooks.py` | 保留 |
| `scripts/create_notebooks.py` | `code` | `scripts/create_notebooks.py` | 保留 |
| `scripts/create_notebooks.py` | `save` | `scripts/create_notebooks.py` | 保留 |
| `scripts/export_public_results.py` | `sanitize_text` | `scripts/export_public_results.py` | 保留 |
| `scripts/export_public_results.py` | `export` | `scripts/export_public_results.py` | 保留 |
| `scripts/export_public_results.py` | `export.copy` | `scripts/export_public_results.py` | 保留 |
| `scripts/run_metrics_extension.py` | `run` | `scripts/run_metrics_extension.py` | 保留 |
| `scripts/run_sql_audit.py` | `main` | `scripts/run_sql_audit.py` | 保留 |
| `scripts/verify_integrity.py` | `sha` | `scripts/verify_integrity.py` | 保留 |
| `scripts/verify_integrity.py` | `verify` | `scripts/verify_integrity.py` | 保留 |
| `tests/test_data.py` | `_synthetic_frame` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.setUp` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.tearDown` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.test_sampling_reads_entire_file_is_reproducible_and_chunk_independent` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.test_full_data_audit_counts_duplicates_without_removing_rows` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.test_sample_larger_than_input_retains_every_original_row` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.test_readonly_guard_rejects_direct_and_symlinked_data_destinations` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.test_sample_round_trip_uses_an_output_directory` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.test_split_is_disjoint_reproducible_and_preserves_joint_strata` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.test_invalid_schema_nonfinite_and_nonbinary_labels_fail` | `tests/test_data.py` | 保留 |
| `tests/test_data.py` | `DataTests.test_split_rejects_unsupported_outcome_and_too_small_strata` | `tests/test_data.py` | 保留 |
| `tests/test_diagnostics.py` | `DiagnosticsTests` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `DiagnosticsTests.test_smd_matches_pooled_standard_deviation` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `DiagnosticsTests.test_post_treatment_features_are_rejected` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `DiagnosticsTests.test_srm_defaults_to_published_reference_and_accepts_other_ratios` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `DiagnosticsTests.test_srm_rejects_missing_or_invalid_reference_ratios` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `DiagnosticsTests.test_c2st_complete_refits_and_permutation_pvalues` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `DiagnosticsTests.test_learnability_dummy_uses_training_rate` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `DiagnosticsTests.test_single_class_outcomes_are_explicit` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `PublicDiagnosticsTests` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `PublicDiagnosticsTests.frame` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `PublicDiagnosticsTests.test_full_stats_balanced_sampling_precision_and_chunk_reproducibility` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `PublicDiagnosticsTests.test_invalid_data_fails_without_silent_row_deletion` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_diagnostics.py` | `PublicDiagnosticsTests.test_update_replaces_only_existing_diagnostic_files` | `tests/test_diagnostics.py` | 保留 |
| `tests/test_end_to_end.py` | `EndToEndTests` | `tests/test_end_to_end.py` | 保留 |
| `tests/test_end_to_end.py` | `EndToEndTests.test_complete_pipeline_outputs_and_frozen_selection` | `tests/test_end_to_end.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.setUp` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_difference_in_means_and_pooled_null` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_aggregate_ate_matches_row_level_with_singleton_arm` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_aggregate_ate_rejects_invalid_counts` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_constant_outcomes_and_missing_control_response` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_ht_curve_preserves_fixed_assignment_probability` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_area_uses_full_curve_and_centering` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_traditional_qini_bridges_initial_undefined_prefix` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_traditional_qini_centers_its_own_full_coverage_endpoint` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_traditional_score_is_not_changed_by_ht_assignment_probability` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_tie_break_is_seeded_and_does_not_use_row_order` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_weighted_prefix_reselects_boundary` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_same_score_paired_difference_is_exactly_zero` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_bootstrap_is_reproducible_and_point_estimates_unchanged` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_budget_rounding_and_analytic_random` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_paired_difference_matches_direct_resampling` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_calibration_bins_cover_data_and_missing_arms_are_nan` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `EvaluationTests.test_invalid_inputs_fail_explicitly` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `ExtendedMetricTests` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `ExtendedMetricTests.setUp` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `ExtendedMetricTests.test_full_eight_row_hand_example` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `ExtendedMetricTests.test_qini_keeps_defined_control_only_zero_nodes` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `ExtendedMetricTests.test_relative_lift_undefined_for_nonpositive_total_effect` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `ExtendedMetricTests.test_paired_ranking_intervals_same_score_zero_difference` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `ExtendedMetricTests.test_ranking_bootstrap_matches_independent_expanded_sample` | `tests/test_evaluation.py` | 保留 |
| `tests/test_evaluation.py` | `ExtendedMetricTests.test_no_draws_and_invalid_bootstrap` | `tests/test_evaluation.py` | 保留 |
| `tests/test_models.py` | `_FixedProbabilityModel` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `_FixedProbabilityModel.__init__` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `_FixedProbabilityModel.fit` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `_FixedProbabilityModel.predict_proba` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `_RecordingRegressor` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `_RecordingRegressor.__init__` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `_RecordingRegressor.fit` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `_RecordingRegressor.predict` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.setUp` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_logistic_contracts_pickle_and_clone` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_s_learner_has_only_additive_treatment_feature` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_preprocessing_is_fit_per_training_arm_only` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_constant_outcome_arms_have_defined_positive_probabilities` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_probability_selection_uses_class_label` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_probability_selection_uses_class_label.ReversedClasses` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_probability_selection_uses_class_label.ReversedClasses.predict_proba` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_x_learner_cross_arm_targets_and_weight_direction` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_validation_and_unfitted_behavior` | `tests/test_models.py` | 保留 |
| `tests/test_models.py` | `ModelTests.test_factory_and_tree_configuration_need_no_xgboost_import` | `tests/test_models.py` | 保留 |
| `tests/test_planning.py` | `PlanningTests` | `tests/test_planning.py` | 保留 |
| `tests/test_planning.py` | `PlanningTests.test_equal_arm_textbook_formula_and_power` | `tests/test_planning.py` | 保留 |
| `tests/test_planning.py` | `PlanningTests.test_mde_and_allocation_and_budget_semantics` | `tests/test_planning.py` | 保留 |
| `tests/test_planning.py` | `PlanningTests.test_economics_hand_example_and_ci` | `tests/test_planning.py` | 保留 |
| `tests/test_planning.py` | `PlanningTests.test_reject_invalid_assumptions` | `tests/test_planning.py` | 保留 |
| `tests/test_publication.py` | `PublicationTests` | `tests/test_publication.py` | 保留 |
| `tests/test_publication.py` | `PublicationTests.test_local_priority_and_explicit_snapshot_mode` | `tests/test_publication.py` | 保留 |
| `tests/test_publication.py` | `PublicationTests.test_invalid_paths_and_modes` | `tests/test_publication.py` | 保留 |
| `tests/test_publication.py` | `PublicationTests.test_synthetic_demo_real_learners_freeze_selection_and_no_overwrite` | `tests/test_publication.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests.setUp` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests.scan` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests.test_one_csv_pass_preserves_both_sampling_streams_and_full_statistics` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests.test_audit_only_variance_matches_full_diagnostics_for_near_constant_features` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests.test_diagnostic_sample_independent_of_model_size_and_chunk_size` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests.test_audit_diagnostics_and_two_samples_are_independent` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests.test_source_changes_and_insufficient_arms_fail` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests.test_update_reuses_scan_and_rejects_mismatched_settings` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_raw_scan.py` | `UnifiedScanTests.test_prepare_sample_fresh_and_cached_paths_each_parse_csv_once` | `tests/test_raw_scan.py` | 保留 |
| `tests/test_scoring.py` | `_FeatureOnlyModel` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `_FeatureOnlyModel.__init__` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `_FeatureOnlyModel._check_features` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `_FeatureOnlyModel.predict` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `_FeatureOnlyModel.predict_outcomes` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.setUp` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.tearDown` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests._save_run` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests._score` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.test_feature_only_chunks_exact_budget_and_input_unchanged` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.test_propensity_uses_treated_response_probability` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.test_random_scores_and_tie_breaking_are_chunk_independent` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.test_budget_boundaries_and_decimal_floor` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.test_tied_score_selection_matches_evaluation_for_same_row_order` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.test_readonly_input_data_symlink_and_existing_output_guards` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.test_missing_run_model_features_and_unfrozen_selection_have_clear_errors` | `tests/test_scoring.py` | 保留 |
| `tests/test_scoring.py` | `ScoringTests.test_invalid_features_and_excessive_chunk_size_fail_before_output` | `tests/test_scoring.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests.setUp` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests.audit` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests.test_hand_calculated_cvr_bins_and_policy_value` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests.test_assignment_probability_is_fixed_and_not_test_treatment_fraction` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests.test_tied_scores_use_independent_priorities_shared_with_python` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests.test_zero_and_full_budget_have_correct_denominators` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests.test_schema_rejects_duplicate_and_orphan_keys` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests.test_invalid_inputs_fail_before_sql` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_sql_audit.py` | `SqlAuditTests.test_runner_preserves_source_and_rejects_overwrites_and_data` | `tests/test_sql_audit.py` | 保留 |
| `tests/test_stability.py` | `synthetic_inputs` | `tests/test_stability.py` | 保留 |
| `tests/test_stability.py` | `StabilityTests` | `tests/test_stability.py` | 保留 |
| `tests/test_stability.py` | `StabilityTests.test_nested_proportional_sampling` | `tests/test_stability.py` | 保留 |
| `tests/test_stability.py` | `StabilityTests.test_invalid_sampling_inputs` | `tests/test_stability.py` | 保留 |
| `tests/test_stability.py` | `StabilityTests.test_old_test_excluded_and_group_split_has_zero_overlap` | `tests/test_stability.py` | 保留 |
| `tests/test_stability.py` | `StabilityTests.test_requested_sample_larger_than_source_uses_audited_row_count` | `tests/test_stability.py` | 保留 |
| `tests/test_stability.py` | `StabilityTests.test_bad_provenance_and_unknown_test_id_rejected` | `tests/test_stability.py` | 保留 |
| `tests/test_stability.py` | `StabilityTests.test_small_integration_freezes_protocol_and_preserves_inputs` | `tests/test_stability.py` | 保留 |
| `tests/test_stability.py` | `StabilityTests.test_small_integration_freezes_protocol_and_preserves_inputs.checked_make_model` | `tests/test_stability.py` | 保留 |
| `tests/test_stability.py` | `StabilityTests.test_small_integration_freezes_protocol_and_preserves_inputs.checked_make_model.fit` | `tests/test_stability.py` | 保留 |
| `run_pipeline.py` | `log` | `run_pipeline.py` | 保留 |
| `run_pipeline.py` | `prepare_sample` | `run_pipeline.py` | 保留兼容包装 |
| `run_pipeline.py` | `run` | `run_pipeline.py` | 保留 |
| `run_pipeline.py` | `parse_args` | `run_pipeline.py` | 保留 |
| `run_pipeline.py` | `main` | `run_pipeline.py` | 保留 |
| `run_stability.py` | `parse_args` | `run_stability.py` | 保留 |
| `score_policy.py` | `_read_json` | `score_policy.py` | 保留 |
| `score_policy.py` | `_load_frozen_policy` | `score_policy.py` | 保留 |
| `score_policy.py` | `_check_input_header` | `score_policy.py` | 保留 |
| `score_policy.py` | `score_policy` | `score_policy.py` | 保留 |
| `score_policy.py` | `main` | `score_policy.py` | 保留 |

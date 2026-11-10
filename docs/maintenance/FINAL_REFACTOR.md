# 最终重构迁移与验收记录

## Phase 1 依赖检查

基线 101 项测试通过（8.902 秒）。工作区仅 preparation.py 的两条缓存中文注释及空行未提交，保留注释含义。六本 Notebook 已逐页检查；保留数据准备、诊断、模型、策略、八行指标核验和稳定性/规划内容，SQL 章节按要求退出正式展示。

依赖方向为 config/input_checks → pipeline_io/metrics/models → diagnostics/preparation/evaluation → training_validation/stability/planning → reporting/主入口。preparation 可以调用 diagnostics 的纯诊断函数；diagnostics 不再导入 preparation 或读取原始 CSV，消除当前延迟导入隐藏的循环依赖。

缓存按最新指令第七节保留全局目录。模型类名改变后历史 joblib 不再直接兼容；不保留组合类别名，历史结果完整保留，新目录重新训练。X-Learner 保留原有伪效应加权公式，其 predict 不等于 predict_outcomes 的差。

ATE 冲突已由用户确认统一为原逐行口径：两条路径均使用 n−1 修正。原全量 Audit 使用 n，故新全量 SE/CI 会微调。其余统计公式、默认参数、候选顺序、置换检验和抽样随机流不变。

## 函数迁移清单

| 原文件 | 原函数/类 | 新文件 | 新函数/类 | 处理方式 |
|---|---|---|---|---|
| run_stability.py | `parse_args` | run_pipeline.py | `parse_args / run` | 合并执行入口 |
| score_policy.py | `_read_json` | score_policy.py | `_read_json` | 保留 |
| score_policy.py | `_load_frozen_policy` | score_policy.py | `_load_frozen_policy` | 保留 |
| score_policy.py | `_check_input_header` | score_policy.py | `_check_input_header` | 保留 |
| score_policy.py | `score_policy` | score_policy.py | `score_policy` | 保留 |
| score_policy.py | `main` | score_policy.py | `main` | 保留 |
| run_pipeline.py | `run` | run_pipeline.py | `run` | 保留 |
| run_pipeline.py | `parse_args` | run_pipeline.py | `parse_args` | 保留 |
| run_pipeline.py | `main` | run_pipeline.py | `main` | 保留 |
| scripts/export_public_results.py | `sanitize_text` | scripts/export_public_results.py | `sanitize_text` | 保留 |
| scripts/export_public_results.py | `export` | scripts/export_public_results.py | `export` | 保留 |
| scripts/build_extension_report.py | `table` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| scripts/build_extension_report.py | `figure` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| scripts/build_extension_report.py | `build` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| scripts/run_metrics_extension.py | `run` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| scripts/check_publication.py | `publication_files` | scripts/check_publication.py | `publication_files` | 保留 |
| scripts/check_publication.py | `check` | scripts/check_publication.py | `check` | 保留 |
| scripts/run_sql_audit.py | `main` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| scripts/build_release.py | `build` | scripts/build_release.py | `build` | 保留 |
| scripts/verify_integrity.py | `sha` | scripts/verify_integrity.py | `sha` | 保留 |
| scripts/verify_integrity.py | `verify` | scripts/verify_integrity.py | `verify` | 保留 |
| criteo_uplift/stability.py | `row_ids_sha256` | criteo_uplift/stability.py | `row_ids_sha256` | 保留 |
| criteo_uplift/stability.py | `_checked_ids` | criteo_uplift/stability.py | `_checked_ids` | 保留 |
| criteo_uplift/stability.py | `_binary_vectors` | criteo_uplift/stability.py | `_binary_vectors` | 保留 |
| criteo_uplift/stability.py | `_validate_fractions` | criteo_uplift/stability.py | `_validate_fractions` | 保留 |
| criteo_uplift/stability.py | `_validate_seeds` | criteo_uplift/stability.py | `_validate_seeds` | 保留 |
| criteo_uplift/stability.py | `feature_groups` | criteo_uplift/stability.py | `feature_groups` | 保留 |
| criteo_uplift/stability.py | `load_development_sample` | criteo_uplift/stability.py | `load_development_sample` | 保留 |
| criteo_uplift/stability.py | `nested_stratified_indices` | criteo_uplift/stability.py | `nested_stratified_indices` | 保留 |
| criteo_uplift/stability.py | `development_splits` | criteo_uplift/stability.py | `development_splits` | 保留 |
| criteo_uplift/stability.py | `_counts` | criteo_uplift/stability.py | `_counts` | 保留 |
| criteo_uplift/stability.py | `_summarize` | criteo_uplift/stability.py | `_summarize` | 保留 |
| criteo_uplift/stability.py | `run_stability` | criteo_uplift/stability.py | `run_stability` | 保留 |
| criteo_uplift/models.py | `_feature_array` | criteo_uplift/input_checks.py | `check_feature_matrix` | 迁移/合并 |
| criteo_uplift/models.py | `_binary_vector` | criteo_uplift/input_checks.py | `check_binary_vector` | 迁移/合并 |
| criteo_uplift/models.py | `_ConstantProbabilityClassifier` | criteo_uplift/models.py | `_ConstantProbabilityClassifier` | 保留 |
| criteo_uplift/models.py | `_positive_probability` | criteo_uplift/models.py | `_positive_probability` | 保留 |
| criteo_uplift/models.py | `_UpliftEstimator` | criteo_uplift/models.py | `_BaseUpliftLearner` | 迁移/合并 |
| criteo_uplift/models.py | `_LogisticConfiguration` | criteo_uplift/models.py | `make_outcome_model` | 迁移/合并 |
| criteo_uplift/models.py | `SLearnerLogistic` | criteo_uplift/models.py | `SLearner` | 迁移/合并 |
| criteo_uplift/models.py | `_TwoOutcomeModels` | criteo_uplift/models.py | `TLearner` | 迁移/合并 |
| criteo_uplift/models.py | `TLearnerLogistic` | criteo_uplift/models.py | `TLearner` | 迁移/合并 |
| criteo_uplift/models.py | `_XGBoostConfiguration` | criteo_uplift/models.py | `make_outcome_model / make_effect_model` | 迁移/合并 |
| criteo_uplift/models.py | `TLearnerXGBoost` | criteo_uplift/models.py | `TLearner` | 迁移/合并 |
| criteo_uplift/models.py | `XLearnerXGBoost` | criteo_uplift/models.py | `XLearner` | 迁移/合并 |
| criteo_uplift/models.py | `make_model` | criteo_uplift/models.py | `make_uplift_model` | 迁移/合并 |
| criteo_uplift/models.py | `predict_policy_scores` | criteo_uplift/models.py | `predict_policy_scores` | 保留 |
| criteo_uplift/pipeline_io.py | `safe_output` | criteo_uplift/pipeline_io.py | `safe_output` | 保留 |
| criteo_uplift/pipeline_io.py | `write_result` | criteo_uplift/pipeline_io.py | `write_result` | 保留 |
| criteo_uplift/pipeline_io.py | `save_sample` | criteo_uplift/pipeline_io.py | `save_sample` | 保留 |
| criteo_uplift/pipeline_io.py | `load_sample` | criteo_uplift/pipeline_io.py | `load_sample` | 保留 |
| criteo_uplift/pipeline_io.py | `result_directory` | criteo_uplift/pipeline_io.py | `result_directory` | 保留 |
| criteo_uplift/diagnostics.py | `scan_raw_data` | criteo_uplift/preparation.py | `_scan_raw_data` | 迁移/合并 |
| criteo_uplift/diagnostics.py | `_features` | criteo_uplift/diagnostics.py | `_features（保留诊断列约束，基础检查委托公共模块）` | 迁移/合并 |
| criteo_uplift/diagnostics.py | `_binary` | criteo_uplift/input_checks.py | `check_binary_vector` | 合并 |
| criteo_uplift/diagnostics.py | `srm_summary` | criteo_uplift/diagnostics.py | `srm_summary` | 保留 |
| criteo_uplift/diagnostics.py | `covariate_balance` | criteo_uplift/diagnostics.py | `covariate_balance` | 保留 |
| criteo_uplift/diagnostics.py | `_model` | criteo_uplift/diagnostics.py | `_make_c2st_model` | 迁移/合并 |
| criteo_uplift/diagnostics.py | `_probability_metrics` | criteo_uplift/diagnostics.py | `_probability_metrics` | 保留 |
| criteo_uplift/diagnostics.py | `c2st` | criteo_uplift/diagnostics.py | `c2st` | 保留 |
| criteo_uplift/diagnostics.py | `learnability` | criteo_uplift/diagnostics.py | `learnability` | 保留 |
| criteo_uplift/diagnostics.py | `run_learnability_diagnostics` | criteo_uplift/diagnostics.py | `run_diagnostics` | 迁移/合并 |
| criteo_uplift/diagnostics.py | `update_experiment_diagnostics` | criteo_uplift/diagnostics.py | `run_diagnostics` | 迁移/合并 |
| criteo_uplift/evaluation.py | `_binary_inputs` | criteo_uplift/metrics.py | `_binary_inputs` | 迁移/合并 |
| criteo_uplift/evaluation.py | `_scores` | criteo_uplift/metrics.py | `_scores` | 迁移/合并 |
| criteo_uplift/evaluation.py | `_ht_outcome` | criteo_uplift/metrics.py | `_ht_outcome` | 迁移/合并 |
| criteo_uplift/evaluation.py | `_rank` | criteo_uplift/metrics.py | `_rank` | 迁移/合并 |
| criteo_uplift/evaluation.py | `_normal_pvalue` | criteo_uplift/evaluation.py | `_normal_pvalue` | 保留 |
| criteo_uplift/evaluation.py | `_mean_difference` | criteo_uplift/evaluation.py | `_ate_summary_from_counts` | 迁移/合并 |
| criteo_uplift/evaluation.py | `aggregate_ate_from_audit` | criteo_uplift/evaluation.py | `ate_summary_from_audit` | 迁移/合并 |
| criteo_uplift/evaluation.py | `ate_summary` | criteo_uplift/evaluation.py | `ate_summary` | 保留 |
| criteo_uplift/evaluation.py | `uplift_curve` | criteo_uplift/metrics.py | `uplift_curve` | 迁移/合并 |
| criteo_uplift/evaluation.py | `_prefix_statistics` | criteo_uplift/metrics.py | `_prefix_statistics` | 迁移/合并 |
| criteo_uplift/evaluation.py | `_area` | criteo_uplift/metrics.py | `_area` | 迁移/合并 |
| criteo_uplift/evaluation.py | `_ranking_from_ordered` | criteo_uplift/metrics.py | `_ranking_from_ordered` | 迁移/合并 |
| criteo_uplift/evaluation.py | `ranking_metrics` | criteo_uplift/metrics.py | `ranking_metrics` | 迁移/合并 |
| criteo_uplift/evaluation.py | `ranking_confidence_intervals` | criteo_uplift/metrics.py | `ranking_confidence_intervals` | 迁移/合并 |
| criteo_uplift/evaluation.py | `_prefix_values` | criteo_uplift/metrics.py | `_prefix_values` | 迁移/合并 |
| criteo_uplift/evaluation.py | `evaluate_policies` | criteo_uplift/evaluation.py | `evaluate_policies` | 保留 |
| criteo_uplift/evaluation.py | `calibration_table` | criteo_uplift/evaluation.py | `calibration_table` | 保留 |
| criteo_uplift/runtime.py | `configure_runtime` | criteo_uplift/config.py | `configure_runtime` | 迁移/合并 |
| criteo_uplift/runtime.py | `log` | criteo_uplift/config.py | `log` | 迁移/合并 |
| criteo_uplift/sql_audit.py | `sql_queries` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| criteo_uplift/sql_audit.py | `audit_arrays` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| criteo_uplift/sql_audit.py | `run_sql_audit` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| criteo_uplift/holdout_evaluation.py | `evaluate_test_set` | criteo_uplift/evaluation.py | `evaluate_test_set` | 迁移/合并 |
| criteo_uplift/planning.py | `plan_policy_ab` | criteo_uplift/planning.py | `plan_policy_ab` | 保留 |
| criteo_uplift/planning.py | `cost_value_scenario` | criteo_uplift/planning.py | `cost_value_scenario` | 保留 |
| criteo_uplift/demo.py | `run_demo` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| criteo_uplift/diagnostic_refresh.py | `update_diagnostics_only` | — | `—` | 删除入口；相关保留功能已映射至主流程/评价/测试 |
| criteo_uplift/training.py | `fit_and_select_models` | criteo_uplift/training_validation.py | `fit_and_select_models` | 迁移/合并 |
| criteo_uplift/preparation.py | `fingerprint` | criteo_uplift/pipeline_io.py | `fingerprint` | 迁移/合并 |
| criteo_uplift/preparation.py | `prepare_sample` | criteo_uplift/preparation.py | `prepare_sample` | 保留 |
| criteo_uplift/preparation.py | `split_sample` | criteo_uplift/preparation.py | `split_sample` | 保留 |
| criteo_uplift/reporting.py | `save_sample_feature_plot` | criteo_uplift/reporting.py | `save_sample_feature_plot` | 保留 |
| criteo_uplift/reporting.py | `save_c2st_plot` | criteo_uplift/reporting.py | `save_c2st_plot` | 保留 |
| criteo_uplift/reporting.py | `_json` | criteo_uplift/reporting.py | `_json` | 保留 |
| criteo_uplift/reporting.py | `_num` | criteo_uplift/reporting.py | `_num` | 保留 |
| criteo_uplift/reporting.py | `_p` | criteo_uplift/reporting.py | `_p` | 保留 |
| criteo_uplift/reporting.py | `_interval` | criteo_uplift/reporting.py | `_interval` | 保留 |
| criteo_uplift/reporting.py | `_table` | criteo_uplift/reporting.py | `_table` | 保留 |
| criteo_uplift/reporting.py | `_image` | criteo_uplift/reporting.py | `_image` | 保留 |
| criteo_uplift/reporting.py | `_errorbar` | criteo_uplift/reporting.py | `_errorbar` | 保留 |
| criteo_uplift/reporting.py | `build_report` | criteo_uplift/reporting.py | `build_report` | 保留 |
| criteo_uplift/reporting.py | `finalize_run` | criteo_uplift/reporting.py | `finalize_run` | 保留 |

## 清理文件

- `criteo_uplift/runtime.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `criteo_uplift/training.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `criteo_uplift/holdout_evaluation.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `criteo_uplift/diagnostic_refresh.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `criteo_uplift/demo.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `criteo_uplift/sql_audit.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `scripts/run_demo.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `scripts/run_sql_audit.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `tests/test_sql_audit.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `scripts/build_extension_report.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `scripts/run_metrics_extension.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `run_stability.py`：职责已迁移至目标模块或按指令移除独立 Demo、SQL、扩展报告入口。
- `notebooks/05_metrics_worked_example.ipynb`：已整合到唯一学习 Notebook，29 个代码单元格执行成功后清理。
- `notebooks/01_data_understanding.ipynb`：已整合到唯一学习 Notebook，29 个代码单元格执行成功后清理。
- `notebooks/04_policy_evaluation.ipynb`：已整合到唯一学习 Notebook，29 个代码单元格执行成功后清理。
- `notebooks/02_sanity_and_ate.ipynb`：已整合到唯一学习 Notebook，29 个代码单元格执行成功后清理。
- `notebooks/06_robustness_sql_and_decisions.ipynb`：已整合到唯一学习 Notebook，29 个代码单元格执行成功后清理。
- `notebooks/03_uplift_models.ipynb`：已整合到唯一学习 Notebook，29 个代码单元格执行成功后清理。

## 最终目录树

```text
criteo_uplift/
  __init__.py
  config.py
  input_checks.py
  pipeline_io.py
  preparation.py
  diagnostics.py
  models.py
  training_validation.py
  metrics.py
  evaluation.py
  stability.py
  planning.py
  reporting.py
run_pipeline.py
score_policy.py
notebooks/criteo_uplift_project.ipynb
scripts/
  execute_notebooks.py
  export_public_results.py
  check_publication.py
  build_release.py
  verify_integrity.py
tests/                      原保留功能测试 + 公共检查、缓存损坏、通用组合、统一 ATE、完整流程
reports/published/
  snapshot_manifest.json
  conversion_1000000_seed42_refactored/
    data_preparation/ diagnostics/ models/ evaluation/ stability/ business/ figures/
    manifest.json
    report.html
artifacts/
  cache/                    样本与全量审计，独立版本
  conversion_1000000_seed42_refactored/
    data_preparation/ diagnostics/ models/ evaluation/ stability/ business/ figures/
    manifest.json
    report.html
  conversion_1000000_seed42/ 原有实验原样保留
  archive/published_before_final_refactor/ 原公开快照原样归档
```

## 每个 Python 文件的最终函数/类清单

以下由最终源码 AST 生成，包含类方法；测试文件同样列出。参数与返回语义见函数实现和项目流程表。常量集中在 config、preparation、training_validation、stability、planning 对应职责内。`__init__.py` 无函数。

### criteo_uplift/__init__.py

无顶层函数；仅配置或脚本执行逻辑。

### criteo_uplift/config.py

- `configure_runtime()`
- `log(message)`

### criteo_uplift/diagnostics.py

- `_features(df: pd.DataFrame, features: Sequence[str])`
- `srm_summary(n1: int, n0: int, expected_ratio: float=0.85)`
- `covariate_balance(df: pd.DataFrame, features: Sequence[str])`
- `_make_c2st_model(name: str, seed: int)`
- `_probability_metrics(y: np.ndarray, prediction: np.ndarray)`
- `c2st(df: pd.DataFrame, features: Sequence[str], sample_size: int=20000, permutations: int=99, seed: int=42)`
- `learnability(train: pd.DataFrame, val: pd.DataFrame, features: Sequence[str], outcome: str, by_arm: bool=True, seed: int=42)`
- `run_diagnostics(out, *, audit, source, scanned, train, validation, sample_size=20000, permutations=499, seed=42, progress=print)`

### criteo_uplift/evaluation.py

- `_normal_pvalue(z)`
- `_ate_summary_from_counts(n0, n1, r0, r1)`
- `ate_summary(y, t)`
- `ate_summary_from_audit(audit, outcome)`
- `evaluate_policies(y, t, score_dict: Mapping[str, Sequence[float]], e: float, budgets: Sequence[float]=DEFAULT_BUDGETS, bootstrap: int=200, seed: int=42)`
- `calibration_table(y, t, cate, n_bins: int=10)`
- `evaluate_test_set(test, models, *, outcome, assignment_probability, seed, bootstrap, primary_budget, out, progress=print, full_ate=None)`

### criteo_uplift/input_checks.py

- `check_binary_vector(values, name='values', n_rows=None, *, require_both=False)`
- `check_feature_matrix(X, *, allow_nan=False, n_features=None)`
- `check_row_ids(values, name='row_id', n_rows=None, *, require_integer=False)`

### criteo_uplift/metrics.py

- `_binary_inputs(y, t)`
- `_scores(scores, n)`
- `_ht_outcome(y, t, e)`
- `_rank(scores, seed)`
- `uplift_curve(y, t, scores, e: float, seed: int=42)`
- `_prefix_statistics(y, t, e)`
- `_area(x, values)`
- `_ranking_from_ordered(y, t, e)`
- `ranking_metrics(y, t, scores, e: float, seed: int=42)`
- `ranking_confidence_intervals(y, t, score_dict: Mapping[str, Sequence[float]], e: float, bootstrap: int=200, seed: int=42)`
- `_prefix_values(z_sorted, counts_sorted, k)`

### criteo_uplift/models.py

- `make_outcome_model(name, **params)`
- `make_effect_model(name, **params)`
- `class _ConstantProbabilityClassifier`
- `_ConstantProbabilityClassifier.fit(self, X, y)`
- `_ConstantProbabilityClassifier.predict_proba(self, X)`
- `_positive_probability(model, X)`
- `class _BaseUpliftLearner`
- `_BaseUpliftLearner.__init__(self, outcome_model='logistic', effect_model=None, outcome_params=None, effect_params=None)`
- `_BaseUpliftLearner._new_classifier(self)`
- `_BaseUpliftLearner._new_regressor(self)`
- `_BaseUpliftLearner._prepare_fit(self, X, t, y)`
- `_BaseUpliftLearner._prepare_predict(self, X)`
- `_BaseUpliftLearner._fit_binary(self, X, y)`
- `_BaseUpliftLearner.predict(self, X)`
- `class SLearner`
- `SLearner.fit(self, X, t, y)`
- `SLearner.predict_outcomes(self, X)`
- `class TLearner`
- `TLearner._fit_outcomes(self, features, treatment, outcome)`
- `TLearner.fit(self, X, t, y)`
- `TLearner.predict_outcomes(self, X)`
- `class XLearner`
- `XLearner.fit(self, X, t, y)`
- `XLearner.predict(self, X)`
- `make_uplift_model(learner, outcome_model, effect_model=None, *, outcome_params=None, effect_params=None, **params)`
- `predict_policy_scores(models, X)`

### criteo_uplift/pipeline_io.py

- `fingerprint(path)`
- `safe_output(path)`
- `write_result(value, path)`
- `save_sample(sample, path)`
- `load_sample(path)`
- `result_directory(name, root=ROOT)`

### criteo_uplift/planning.py

- `plan_policy_ab(p0, absolute_mde, allocation_ratio=1.0, budget=0.2, alpha=0.05, power=0.8)`
- `cost_value_scenario(incremental_per_100k, *, budget, value_per_conversion, contact_cost, eligible_n=100000, ci_low=None, ci_high=None)`
- `run_planning(policies, selection, out, *, assumptions=None)`

### criteo_uplift/preparation.py

- `_scan_raw_data(path=DEFAULT_DATA, sample_size=1000000, seed=42, chunk_size=250000, *, per_arm=None, audit_enabled=True, diagnostics_enabled=True, progress=print, source=None)`
- `prepare_sample(args, *, root: Path=ROOT, progress=None)`
- `split_sample(df, outcome='conversion', seed=42)`

### criteo_uplift/reporting.py

- `save_sample_feature_plot(sample, path)`
- `save_c2st_plot(out)`
- `_json(out: Path, name: str)`
- `_num(value, digits: int=2)`
- `_p(value)`
- `_interval(row, low='ci_low', high='ci_high', scale=1, digits=2)`
- `_table(frame: pd.DataFrame)`
- `_image(path: Path, alt: str, caption: str)`
- `_errorbar(ax, point, low, high, positions, *, horizontal=False, color='#2171ad')`
- `build_report(manifest_path: Path, report_path: Path)`
- `finalize_run(*, manifest_path: Path, report_path: Path, args, source, sample_n, assignment_probability, fitted, elapsed_seconds)`

### criteo_uplift/stability.py

- `row_ids_sha256(ids)`
- `_checked_ids(values, label)`
- `_binary_vectors(t, y)`
- `_validate_fractions(fractions)`
- `_validate_seeds(seeds)`
- `feature_groups(frame)`
- `load_development_sample(cache, base_run, outcome='conversion')`
- `nested_stratified_indices(t, y, fractions=DEFAULT_FRACTIONS, seed=17)`
- `development_splits(frame, seed=17, outcome='conversion')`
- `_counts(frame, outcome, prefix)`
- `_summarize(longmetrics)`
- `run_stability(cache, base_run, output, *, seeds=DEFAULT_SEEDS, fractions=DEFAULT_FRACTIONS, trees=150, primary_budget=0.2, outcome='conversion')`

### criteo_uplift/training_validation.py

- `fit_and_select_models(train, validation, *, outcome, seed, trees, primary_budget, assignment_probability, out, progress=print)`

### run_pipeline.py

- `run(args)`
- `parse_args(argv=None)`
- `main(argv=None)`

### score_policy.py

- `_read_json(path)`
- `_load_frozen_policy(run)`
- `_check_input_header(path)`
- `score_policy(run, input_path, output_path, budget=0.2, chunk_size=MAX_CHUNK_SIZE)`
- `main(argv=None)`

### scripts/build_release.py

- `build(output)`

### scripts/check_publication.py

- `publication_files(root=ROOT)`
- `check(root=ROOT)`

### scripts/execute_notebooks.py

无顶层函数；仅配置或脚本执行逻辑。

### scripts/export_public_results.py

- `sanitize_text(value)`
- `export(source, output)`

### scripts/verify_integrity.py

- `sha(path)`
- `verify()`

### tests/test_data.py

- `_synthetic_frame(n=400)`
- `class DataTests`
- `DataTests.setUp(self)`
- `DataTests.tearDown(self)`
- `DataTests.test_sampling_reads_entire_file_is_reproducible_and_chunk_independent(self)`
- `DataTests.test_full_data_audit_counts_duplicates_without_removing_rows(self)`
- `DataTests.test_sample_larger_than_input_retains_every_original_row(self)`
- `DataTests.test_readonly_guard_rejects_direct_and_symlinked_data_destinations(self)`
- `DataTests.test_sample_round_trip_uses_an_output_directory(self)`
- `DataTests.test_split_is_disjoint_reproducible_and_preserves_joint_strata(self)`
- `DataTests.test_split_rejects_missing_null_or_duplicate_source_ids(self)`
- `DataTests.test_split_detects_corrupted_partition_membership(self)`
- `DataTests.test_split_reports_all_feature_pairs_without_treating_matches_as_row_duplicates(self)`
- `DataTests.test_invalid_schema_nonfinite_and_nonbinary_labels_fail(self)`
- `DataTests.test_split_rejects_unsupported_outcome_and_too_small_strata(self)`
- `DataTests.test_write_result_preserves_json_values_and_csv_columns(self)`
- `DataTests.test_write_result_rejects_invalid_format_before_writing(self)`
- `class InputChecksTests`
- `InputChecksTests.test_checks_do_not_transform_inputs(self)`
- `InputChecksTests.test_corrupt_cached_arrays_are_rejected(self)`

### tests/test_diagnostics.py

- `class DiagnosticsTests`
- `DiagnosticsTests.test_smd_matches_pooled_standard_deviation(self)`
- `DiagnosticsTests.test_post_treatment_features_are_rejected(self)`
- `DiagnosticsTests.test_srm_defaults_to_published_reference_and_accepts_other_ratios(self)`
- `DiagnosticsTests.test_srm_rejects_missing_or_invalid_reference_ratios(self)`
- `DiagnosticsTests.test_c2st_complete_refits_and_permutation_pvalues(self)`
- `DiagnosticsTests.test_learnability_dummy_uses_training_rate(self)`
- `DiagnosticsTests.test_single_class_outcomes_are_explicit(self)`
- `class PublicDiagnosticsTests`
- `PublicDiagnosticsTests.frame(self)`
- `PublicDiagnosticsTests.test_full_stats_balanced_sampling_precision_and_chunk_reproducibility(self)`
- `PublicDiagnosticsTests.test_invalid_data_fails_without_silent_row_deletion(self)`
- `PublicDiagnosticsTests.test_update_replaces_only_existing_diagnostic_files(self)`

### tests/test_end_to_end.py

- `class EndToEndTests`
- `EndToEndTests.test_complete_pipeline_outputs_and_frozen_selection(self)`

### tests/test_evaluation.py

- `class EvaluationTests`
- `EvaluationTests.setUp(self)`
- `EvaluationTests.test_difference_in_means_and_pooled_null(self)`
- `EvaluationTests.test_ate_row_and_audit_paths_agree_for_every_statistic(self)`
- `EvaluationTests.test_normal_tail_retains_subnormal_probability_and_log_beyond_float_range(self)`
- `EvaluationTests.test_ate_extreme_pvalues_preserve_test_and_estimates(self)`
- `EvaluationTests.test_aggregate_ate_rejects_invalid_counts(self)`
- `EvaluationTests.test_constant_outcomes_and_missing_control_response(self)`
- `EvaluationTests.test_ht_curve_preserves_fixed_assignment_probability(self)`
- `EvaluationTests.test_area_uses_full_curve_and_centering(self)`
- `EvaluationTests.test_traditional_qini_bridges_initial_undefined_prefix(self)`
- `EvaluationTests.test_traditional_qini_centers_its_own_full_coverage_endpoint(self)`
- `EvaluationTests.test_traditional_score_is_not_changed_by_ht_assignment_probability(self)`
- `EvaluationTests.test_tie_break_is_seeded_and_does_not_use_row_order(self)`
- `EvaluationTests.test_weighted_prefix_reselects_boundary(self)`
- `EvaluationTests.test_same_score_paired_difference_is_exactly_zero(self)`
- `EvaluationTests.test_bootstrap_is_reproducible_and_point_estimates_unchanged(self)`
- `EvaluationTests.test_budget_rounding_and_analytic_random(self)`
- `EvaluationTests.test_paired_difference_matches_direct_resampling(self)`
- `EvaluationTests.test_calibration_bins_cover_data_and_missing_arms_are_nan(self)`
- `EvaluationTests.test_invalid_inputs_fail_explicitly(self)`
- `class ExtendedMetricTests`
- `ExtendedMetricTests.setUp(self)`
- `ExtendedMetricTests.test_full_eight_row_hand_example(self)`
- `ExtendedMetricTests.test_qini_keeps_defined_control_only_zero_nodes(self)`
- `ExtendedMetricTests.test_relative_lift_undefined_for_nonpositive_total_effect(self)`
- `ExtendedMetricTests.test_paired_ranking_intervals_same_score_zero_difference(self)`
- `ExtendedMetricTests.test_ranking_bootstrap_matches_independent_expanded_sample(self)`
- `ExtendedMetricTests.test_no_draws_and_invalid_bootstrap(self)`

### tests/test_models.py

- `class _FixedProbabilityModel`
- `_FixedProbabilityModel.__init__(self, probability)`
- `_FixedProbabilityModel.fit(self, X, y)`
- `_FixedProbabilityModel.predict_proba(self, X)`
- `class _RecordingRegressor`
- `_RecordingRegressor.__init__(self, prediction)`
- `_RecordingRegressor.fit(self, X, y)`
- `_RecordingRegressor.predict(self, X)`
- `class ModelTests`
- `ModelTests.setUp(self)`
- `ModelTests.test_logistic_contracts_pickle_and_clone(self)`
- `ModelTests.test_s_learner_has_only_additive_treatment_feature(self)`
- `ModelTests.test_preprocessing_is_fit_per_training_arm_only(self)`
- `ModelTests.test_constant_outcome_arms_have_defined_positive_probabilities(self)`
- `ModelTests.test_probability_selection_uses_class_label(self)`
- `ModelTests.test_x_learner_cross_arm_targets_and_weight_direction(self)`
- `ModelTests.test_validation_and_unfitted_behavior(self)`
- `ModelTests.test_factory_and_extra_combinations(self)`

### tests/test_planning.py

- `class PlanningTests`
- `PlanningTests.test_equal_arm_textbook_formula_and_power(self)`
- `PlanningTests.test_mde_and_allocation_and_budget_semantics(self)`
- `PlanningTests.test_economics_hand_example_and_ci(self)`
- `PlanningTests.test_reject_invalid_assumptions(self)`

### tests/test_publication.py

- `class PublicationTests`
- `PublicationTests.test_learning_notebook_uses_project_functions_and_does_not_train(self)`
- `PublicationTests.test_local_priority_and_explicit_snapshot_mode(self)`
- `PublicationTests.test_invalid_paths_and_modes(self)`

### tests/test_raw_scan.py

- `class UnifiedScanTests`
- `UnifiedScanTests.setUp(self)`
- `UnifiedScanTests.scan(self, **kwargs)`
- `UnifiedScanTests.test_one_csv_pass_preserves_both_sampling_streams_and_full_statistics(self)`
- `UnifiedScanTests.test_audit_only_variance_matches_full_diagnostics_for_near_constant_features(self)`
- `UnifiedScanTests.test_diagnostic_sample_independent_of_model_size_and_chunk_size(self)`
- `UnifiedScanTests.test_audit_diagnostics_and_two_samples_are_independent(self)`
- `UnifiedScanTests.test_source_changes_and_insufficient_arms_fail(self)`
- `UnifiedScanTests.test_update_reuses_scan_and_rejects_mismatched_settings(self)`
- `UnifiedScanTests.test_prepare_sample_fresh_and_cached_paths_each_parse_csv_once(self)`
- `UnifiedScanTests.test_prepare_sample_rebuilds_without_metadata_even_with_legacy_files(self)`
- `UnifiedScanTests.test_sample_and_audit_caches_are_reused_independently(self)`
- `UnifiedScanTests.test_incomplete_sample_metadata_is_regenerated_by_scan(self)`
- `UnifiedScanTests.test_audit_cache_survives_size_seed_changes_and_invalidates_by_source_and_version(self)`
- `UnifiedScanTests.test_duplicate_excess_counts_are_not_unique_duplicate_patterns(self)`
- `UnifiedScanTests.test_missing_duplicate_fields_rebuild_audit_without_resampling(self)`
- `UnifiedScanTests.test_source_change_invalidates_both_caches(self)`

### tests/test_scoring.py

- `class _FeatureOnlyModel`
- `_FeatureOnlyModel.__init__(self, constant=False, max_rows=7)`
- `_FeatureOnlyModel._check_features(self, X)`
- `_FeatureOnlyModel.predict(self, X)`
- `_FeatureOnlyModel.predict_outcomes(self, X)`
- `class ScoringTests`
- `ScoringTests.setUp(self)`
- `ScoringTests.tearDown(self)`
- `ScoringTests._save_run(self, policy, model=None)`
- `ScoringTests._score(self, **kwargs)`
- `ScoringTests.test_feature_only_chunks_exact_budget_and_input_unchanged(self)`
- `ScoringTests.test_propensity_uses_treated_response_probability(self)`
- `ScoringTests.test_random_scores_and_tie_breaking_are_chunk_independent(self)`
- `ScoringTests.test_budget_boundaries_and_decimal_floor(self)`
- `ScoringTests.test_tied_score_selection_matches_evaluation_for_same_row_order(self)`
- `ScoringTests.test_readonly_input_data_symlink_and_existing_output_guards(self)`
- `ScoringTests.test_missing_run_model_features_and_unfrozen_selection_have_clear_errors(self)`
- `ScoringTests.test_invalid_features_and_excessive_chunk_size_fail_before_output(self)`

### tests/test_stability.py

- `synthetic_inputs(root, n=2000)`
- `class StabilityTests`
- `StabilityTests.test_nested_proportional_sampling(self)`
- `StabilityTests.test_invalid_sampling_inputs(self)`
- `StabilityTests.test_old_test_excluded_and_group_split_has_zero_overlap(self)`
- `StabilityTests.test_requested_sample_larger_than_source_uses_source_row_count(self)`
- `StabilityTests.test_bad_provenance_and_unknown_test_id_rejected(self)`
- `StabilityTests.test_small_integration_freezes_protocol_and_preserves_inputs(self)`

## 验收证据

- 基线 101 项测试通过。最终 94 项通过。数量变化来自按指令删除 SQL/Demo/诊断刷新专属测试，并补入公共检查、缓存损坏检查；所有保留功能测试已迁移。通用组合、ATE 同口径和完整 Stability 等断言已扩展到现有测试中。
- 四个原有组合在固定教学回归样本上的 CATE 与潜在结果预测，逐元素完全一致。新增 S＋XGBoost、X＋Logistic＋Linear 可拟合与预测，clone/pickle、单类概率、X 伪效应及加权方向通过。
- 真实完整 Pipeline 使用原默认 1,000,000 样本、seed=42、150 trees、499 次 C2ST 置换、200 次 Bootstrap，全部执行成功，manifest.status=complete，耗时约 96.81 秒。
- 与原百万行实验的逐行划分、全部 Test 预测数组、Validation 排序/策略值/冻结选择、Test 策略值/排序/配对区间、SMD/KS、C2ST 与零分布逐项完全一致。ATE 只按用户确认统一全量 SE/CI 的口径，其余统计定义不变。
- Stability 默认 12 配置、48 模型，原 Test 的 200,000 行全部排除，来源行交集与特征分组交集符合原设计。冻结协议及输入在 Stability 运行期间未改变。
- 统一端到端测试实际加载新 joblib，使用仅包含 X 的人工测试文件分块打分并核对预算。该测试不伪装为真实部署，也不写入正式业务产物。
- 本地结果模式与仅公开快照模式，唯一 Notebook 均执行全部 29 个代码单元格，无错误。原六本在新本执行成功后才清理。
- HTML 包含全部 11 个必需主题、8 张结果图；不训练或重算指标。只生成完整报告，没有新增 Demo/SQL/Extension Report。稳定性图已检查，展示已保存的结果。
- 13 个核心模块全部可导入，静态依赖图无环。源码编译检查和 Git diff 空白检查通过。
- 2 个原始数据文件 SHA256、大小、mtime 均与原完整性记录一致。112 个既有结果/缓存文件内容哈希不变，旧公开快照的文件在归档路径逐一验证。
- 当前发布快照 42 个聚合文件通过哈希检查，不含 NPZ、joblib、逐行预测或行号列表。

### 必测项目映射

| 要求 | 已执行验证 |
|---|---|
| 1 原始分块；2 全量审计 | test_raw_scan、test_data：一次 CSV 解析、精度、重复内容统计、异常停止 |
| 3 Sample；4 Metadata；5 Audit；6 独立失效 | test_raw_scan：首次/命中、缺 metadata、版本变化、来源变化、规模/种子变化；test_data：缓存损坏拒绝 |
| 7 三集合互斥 | test_data、test_end_to_end：互斥、覆盖、比例、分层和全部来源行 |
| 8 公共检查 | test_data.InputChecksTests：只检查不修改、NaN/Inf、形状、二元标签与行号 |
| 9 S/T/X；10 基模型组合；11 X 权重 | test_models：原组合与新组合、伪效应符号、权重方向、训练内预处理、拟合状态、序列化 |
| 12 通用 ATE；13 Audit ATE；14 两路一致 | test_evaluation：同一份数据的全部统计字段完全一致，单例/常量/极小 p 值/非法计数 |
| 15 AUUC/Qini | test_evaluation：八行精确手算、仅对照零节点、固定 e、完整积分、独立重抽样核验 |
| 16 Test Evaluation | test_end_to_end：冻结选择、完整结果结构、独立评价、统一 ATE |
| 17 Stability 三维度 | test_stability、test_end_to_end：种子/规模/分组、嵌套抽样、协议冻结、排除 Test |
| 18 X-only Score Policy | test_scoring、test_end_to_end：真实新模型载入、分块、精确预算、输入保护 |
| 19 Planning | test_planning、test_end_to_end：样本量公式、功效、价值与成本、条件性区间及假设记录 |
| 20 HTML | test_end_to_end：报告成功生成、全部必需主题、图表文件 |

## 运行与交付位置

```bash
python run_pipeline.py --output artifacts/conversion_1000000_seed42_repeat
python -m unittest discover -s tests
python scripts/execute_notebooks.py
python scripts/check_publication.py
```

- 学习 Notebook：`notebooks/criteo_uplift_project.ipynb`。
- 本地正式报告：`artifacts/conversion_1000000_seed42_refactored/report.html`。
- 发布副本：`reports/published/conversion_1000000_seed42_refactored/report.html`。
- 完整入口/参数说明与独立打分命令见根 README。

## 已知边界与未完成的外部工作

本次重构功能均已执行验收。真实新人群打分未执行，因为没有提供外部 X-only 数据；报告明确标注，不生成虚构结果。Planning 沿用原教学假设，不声明线上收入。

旧组合类按指令移除，历史 joblib 不能直接由新类加载；历史文件未改动，需切回原提交读取。新实验已经重新训练并核验，当前打分使用新模型。没有保留类名别名或额外兼容框架。

原始 CSV 入口仍沿用原规则拒绝非有限数据和非法标签；逻辑异常/重复候选仅记录，不删除。KS、C2ST 的置换可交换性、行独立、未知用户依赖与效果迁移等既有统计限制原样保留。

源码与本地结果尚未提交或推送 GitHub，本轮未执行任何远程发布。

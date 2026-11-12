# Criteo Uplift 项目轻量化重构验收记录

## 项目定位

本仓库服务于固定 Criteo Uplift 数据集的端到端 Data Science 研究和求职展示。主运行入口为 `run_pipeline.py`，交付结果为 `report.html`、聚合统计表和 `notebooks/criteo_uplift_project.ipynb`。公开结果由 `scripts/export_public_results.py` 导出。

## 分批清理

| 批次 | 已完成的调整 | 代表提交 |
|---|---|---|
| 第一批 | 模型 S/T/X 主线可读，Learner 不再接收 Overlap 或核对训练数据签名 | `3539b74` |
| 第二批 | Preparation 固定任务、简化多缓存检查；Stability 删除历史来源哈希链和冻结协议自检 | `3d6c773` |
| 第三批 | 删除 Scoring、ZIP 发布、历史完整性校验入口；Notebook 固定读聚合结果；清理发布与文档残留 | `ebfbb2d` 及后续收尾 |

Overlap 统一在 `run_diagnostics()` 中使用 Train 的处理前特征分箱，并保存为 `diagnostics/overlap_positivity.json` 和 `diagnostics/overlap_strata.csv`。`training_validation.py` 只负责四个候选的训练、Validation 评价和策略冻结。

## 保留的统计计算

- 大文件分块扫描、全量审计、可复现抽样和 60/20/20 Train/Validation/Test 划分。
- SRM、Balance、C2ST 置换检验、Learnability、经验 Overlap。
- ATE/ITT、S/T/X-Learner、AUUC/Qini、HT 策略价值、Bootstrap 与 CATE 校准。
- 排除原 Test 的 Stability、基于明确假设的 A/B 规划与经济情景。
- 现有中文注释风格，模型数值、数据隔离和合成端到端测试。

不再提供独立的新人群 Scoring、部署、ZIP 打包、历史完整性校验或探索性敏感性分析执行接口。`scripts/execute_notebooks.py` 只用于 Notebook 验收，不参与模型拟合。

## 运行和验收

```bash
python -m unittest discover -s tests -v
python scripts/execute_notebooks.py
```

获取原始 CSV 后，可以使用新输出目录执行主流程：

```bash
python run_pipeline.py --output artifacts/conversion_1000000_seed42_repeat
```

从已完成的本地结果导出聚合展示文件：

```bash
python scripts/export_public_results.py \
  --source artifacts/conversion_1000000_seed42_repeat \
  --output reports/new_snapshot
```

公开导出清单不含原始行、逐行预测、来源行号列表、模型或缓存。已有的百万行发布快照是历史实验产物，本次重构没有重新训练这批正式模型。

2026-11-12 的 [GitHub Actions 验收](https://github.com/tkkkk-co/Criteo-uplift-project/actions/runs/43189487562) 已通过。本文件保留阶段性改动和复核入口，不再维护过时的函数迁移与历史文件清单。

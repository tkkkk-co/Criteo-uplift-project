# 正式展示版交付说明

2026-11-18：用户校订通过的 report v2 冻结为项目正式展示版，原样替换 `reports/final/report.html`。本轮为展示报告与文档发布整理，不重新训练模型或计算实验结果。

## 正式入口

- 完整报告：`reports/final/report.html`。
- 一页综述：`reports/final/executive_summary.pdf`。
- 发布与来源记录：`reports/final/report_provenance.json`。
- 本地或 GitHub Pages 根入口：`index.html`，跳转正式报告。

正式报告包含「研究结论与业务决策」一页综述和八章正文。关键数据分析与业务解释直接展示，参数、公式推导、完整比较结果及补充图按需展开。HTML 内嵌图表，公式渲染依赖 MathJax CDN。

## 冻结内容与唯一入口

- 正式策略：验证集 HT Gain @20% 点估计冻结的 Response-XGBoost；验证集首位 Uplift 候选为 S-XGBoost。
- Test Set 用于冻结后的收益评价，正式关键比较使用 2,000 次处理臂内分层配对 Bootstrap。
- 仓库只保留 `reports/final/report.html` 这一份完整报告，其内容为校订通过的 v2。
- 同目录保留 `executive_summary.pdf` 与 `report_provenance.json`，分别为一页综述和统一来源记录。
- 开发报告、旧版报告、v2 重复副本及校订过程文档已从展示目录移除；原实验来源指纹嵌入统一发布记录。

## 复现与后续生成

原始数据、冻结协议、模型及行级工件保留本地，未随代码仓库分发。公开阅读与合成数据测试、严格重跑正式实验具有不同输入要求，详见 README。

正式展示版经过人工编辑，生成器不保证逐字重建最终版式与叙述。后续实验报告应输出到新的 `artifacts/` 运行目录，避免覆盖 `reports/final/report.html`。发布记录中的 HTML SHA-256 对应最终文件；历史生成器与实验输入指纹嵌入同一发布记录。

本轮未提交、推送或更改 GitHub 仓库可见性；GitHub Pages 配置仍需在远端完成。

"""Create a self-contained Chinese report from a completed pipeline run.

The module reads saved result tables only; it neither refits models nor changes
model selection. Figures use Matplotlib's bundled font and English labels.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import safe_output

_COLORS = {
    "random": "#7b8793",
    "propensity": "#e38636",
    "s_logistic": "#8dbce1",
    "t_logistic": "#5594c5",
    "t_xgboost": "#2171ad",
    "x_xgboost": "#123f69",
}
_LABELS = {
    "random": "Random",
    "propensity": "Response propensity",
    "s_logistic": "S / Logistic",
    "t_logistic": "T / Logistic",
    "t_xgboost": "T / XGBoost",
    "x_xgboost": "X / XGBoost",
}


def _json(out: Path, name: str) -> dict:
    return json.loads((out / name).read_text(encoding="utf-8"))


def _num(value, digits: int = 2) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.{digits}f}"


def _p(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    if value == 0:
        return "≈0（数值下溢）"
    return f"{value:.3g}"


def _interval(row, low="ci_low", high="ci_high", scale=1, digits=2) -> str:
    return f"[{_num(row[low] * scale, digits)}, {_num(row[high] * scale, digits)}]"


def _table(frame: pd.DataFrame) -> str:
    return (
        '<div class="table-wrap">'
        + frame.to_html(index=False, border=0, escape=True, na_rep="—")
        + "</div>"
    )


def _image(path: Path, alt: str, caption: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'
        f'<figcaption>{caption}</figcaption></figure>'
    )


def _errorbar(ax, point, low, high, positions, *, horizontal=False, color="#2171ad"):
    """Draw endpoints directly, including rare percentile intervals excluding the point."""
    valid = np.isfinite(point) & np.isfinite(low) & np.isfinite(high)
    if horizontal:
        ax.hlines(positions[valid], low[valid], high[valid], color=color, lw=2)
        ax.scatter(point, positions, c=color, s=35, zorder=3)
    else:
        ax.vlines(positions[valid], low[valid], high[valid], color=color, lw=2)
        ax.scatter(positions, point, c=color, s=35, zorder=3)


def build_report(out: Path) -> Path:
    """Read the pipeline's frozen outputs and write report.html plus five PNGs."""
    out = safe_output(out)
    manifest, audit = _json(out, "manifest.json"), _json(out, "audit.json")
    selection, srm = _json(out, "selection.json"), _json(out, "srm.json")
    overlap = _json(out, "feature_overlap.json")
    c2st_details, ate_test = _json(out, "c2st_details.json"), _json(
        out, "ate_test.json"
    )
    frames = {
        name: pd.read_csv(out / f"{name}.csv")
        for name in (
            "ate_full",
            "balance",
            "c2st",
            "validation_policies",
            "test_policies",
            "test_ranking",
            "learnability",
            "splits",
            "curves",
        )
    }
    selected = selection["selected_policy"]
    selected_uplift = selection["selected_uplift_model"]
    budget = float(selection["primary_budget"])
    outcome = manifest["primary_outcome"]
    e = float(manifest["assignment_probability"])
    policies = frames["test_policies"]
    primary = policies[np.isclose(policies["budget"], budget)].copy()
    primary = primary.sort_values(
        "incremental_per_100k", ascending=False, kind="stable"
    )
    selected_row = primary.loc[primary["policy"] == selected].iloc[0]
    calibration_model = (
        selected if selected not in {"random", "propensity"} else selected_uplift
    )
    calibration = pd.read_csv(out / f"calibration_{calibration_model}.csv")
    figures = safe_output(out / "figures")
    figures.mkdir(exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    paths = {
        name: safe_output(figures / f"{name}.png")
        for name in ("ate_full", "balance", "ranking", "policy", "calibration")
    }
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelcolor": "#344454",
            "text.color": "#23313f",
            "axes.edgecolor": "#c4ced7",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#e6ebef",
            "axes.axisbelow": True,
        }
    ):

        def save(fig, name):
            fig.savefig(paths[name], dpi=160, bbox_inches="tight")
            plt.close(fig)

        ate = frames["ate_full"]
        fig, axes = plt.subplots(
            1, len(ate), figsize=(10.8, 3.4), squeeze=False, constrained_layout=True
        )
        for ax, row in zip(axes[0], ate.itertuples()):
            ax.axhline(0, color="#a7b3be", lw=1)
            _errorbar(
                ax,
                np.array([row.ate * 100]),
                np.array([row.ci_low * 100]),
                np.array([row.ci_high * 100]),
                np.array([0.0]),
                color="#2171ad",
            )
            ax.set_xlim(-0.8, 0.8)
            ax.set_xticks([0], [row.outcome.capitalize()])
            ax.set_ylabel("Absolute effect (percentage points)")
            ax.set_title(f"{row.outcome.capitalize()} | full public sample")
            ax.grid(axis="y")
            ax.text(
                0.03,
                0.96,
                f"ATE {row.ate * 100:.3f} pp\n95% CI [{row.ci_low * 100:.3f}, {row.ci_high * 100:.3f}]",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )
            ax.margins(y=0.35)
        save(fig, "ate_full")

        balance = frames["balance"].sort_values("abs_smd", ascending=True)
        fig, ax = plt.subplots(figsize=(10.8, 4.7), constrained_layout=True)
        ax.barh(balance["feature"], balance["abs_smd"], color="#2171ad", height=0.65)
        ax.axvline(
            0.1,
            color="#e38636",
            linestyle="--",
            lw=1.5,
            label="0.1 descriptive reference",
        )
        ax.set_xlabel("Absolute standardized mean difference (training split)")
        ax.set_title("Observed covariate balance")
        ax.set_xlim(
            0, max(0.12, float(balance["abs_smd"].replace(np.inf, np.nan).max()) * 1.2)
        )
        ax.grid(axis="x")
        ax.legend(frameon=False, loc="lower right")
        save(fig, "balance")

        curves = frames["curves"]
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), constrained_layout=True)
        first = curves[curves["policy"] == curves["policy"].iloc[0]].sort_values(
            "fraction"
        )
        for name, group in curves.groupby("policy", sort=False):
            group = group.sort_values("fraction")
            color = _COLORS.get(name, "#2171ad")
            weight = 2.4 if name == selected else 1.4
            for ax, column in zip(axes, ("gain_ht", "qini_traditional")):
                values = group[column].copy()
                values.loc[group["fraction"] == 0] = 0
                ax.plot(
                    group["fraction"],
                    values * 100_000,
                    color=color,
                    lw=weight,
                    label=_LABELS.get(name, name) + (" *" if name == selected else ""),
                )
        axes[0].plot(
            first["fraction"],
            first["random_gain_ht"] * 100_000,
            color=_COLORS["random"],
            ls="--",
            lw=1.6,
            label="Random expected gain",
        )
        q_endpoint = float(first["qini_traditional"].iloc[-1])
        axes[1].plot(
            [0, 1],
            [0, q_endpoint * 100_000],
            color=_COLORS["random"],
            ls="--",
            lw=1.6,
            label="Endpoint reference line",
        )
        for ax in axes:
            ax.axhline(0, color="#c4ced7", lw=0.8)
            ax.set_xlabel("Fraction of eligible population contacted")
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=0.7)
            ax.legend(frameon=False, fontsize=8, loc="best")
        axes[0].set_title("HT incremental gain | held-out test")
        axes[0].set_ylabel("Incremental responses / 100,000 eligible people")
        axes[1].set_title("Traditional Qini curve | auxiliary diagnostic")
        axes[1].set_ylabel("Traditional Qini ordinate x 100,000")
        save(fig, "ranking")

        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
        for name, group in policies.groupby("policy", sort=False):
            group = group.sort_values("budget")
            axes[0].plot(
                group["budget"],
                group["incremental_per_100k"],
                color=_COLORS.get(name, "#2171ad"),
                marker="o",
                markersize=3,
                lw=2.4 if name == selected else 1.3,
                linestyle="--" if name == "random" else "-",
                label=_LABELS.get(name, name) + (" *" if name == selected else ""),
            )
        axes[0].axvline(budget, color="#a7b3be", lw=1, ls=":")
        axes[0].set_title("Policy value by contact budget")
        axes[0].set_xlabel("Contact fraction (not currency)")
        axes[0].set_ylabel("Incremental responses / 100,000 eligible people")
        axes[0].xaxis.set_major_formatter(PercentFormatter(1))
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].grid(alpha=0.7)
        shown = primary.iloc[::-1]
        for j, row in enumerate(shown.itertuples()):
            color = _COLORS.get(row.policy, "#2171ad")
            _errorbar(
                axes[1],
                np.array([row.incremental_per_100k]),
                np.array([row.ci_low]),
                np.array([row.ci_high]),
                np.array([j]),
                horizontal=True,
                color=color,
            )
        axes[1].set_yticks(
            np.arange(len(shown)),
            [
                _LABELS.get(name, name) + (" *" if name == selected else "")
                for name in shown["policy"]
            ],
        )
        axes[1].axvline(0, color="#a7b3be", lw=1)
        axes[1].set_xlabel("Incremental responses / 100,000 eligible people")
        axes[1].set_title(f"Primary budget: {budget:.0%} | pointwise 95% CI")
        axes[1].grid(axis="x")
        save(fig, "policy")

        fig, ax = plt.subplots(figsize=(10.8, 4.2), constrained_layout=True)
        positions = calibration["bin"].to_numpy()
        _errorbar(
            ax,
            calibration["observed_ate"].to_numpy() * 100,
            calibration["ci_low"].to_numpy() * 100,
            calibration["ci_high"].to_numpy() * 100,
            positions,
            color="#2171ad",
        )
        ax.plot(
            positions,
            calibration["predicted_cate"] * 100,
            color="#e38636",
            marker="o",
            lw=1.6,
            markersize=4,
            label="Mean predicted CATE",
        )
        ax.plot(
            [],
            [],
            color="#2171ad",
            marker="o",
            linestyle="none",
            label="Observed arm difference + 95% CI",
        )
        ax.axhline(0, color="#a7b3be", lw=1)
        ax.set_xticks(positions)
        ax.set_xlabel("Equal-count bins: low to high predicted CATE")
        ax.set_ylabel("Effect (percentage points)")
        ax.set_title(
            f"CATE calibration | {_LABELS.get(calibration_model, calibration_model)}"
        )
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.grid(axis="y")
        save(fig, "calibration")

    esc = html.escape
    test_split = frames["splits"].query("split == 'test'")
    n_test = int(test_split["n"].sum())
    event_column = f"{outcome}_positive"
    n_events = int(test_split[event_column].sum())
    n_control_events = int(
        test_split.loc[test_split["treatment"] == 0, event_column].sum()
    )
    primary_pct = f"{budget:.0%}"
    source = manifest["source"]
    source_file = esc(Path(source["path"]).name)
    delta_lower, delta_upper = (
        selected_row["vs_random_ci_low"],
        selected_row["vs_random_ci_high"],
    )
    if selected == "random":
        conclusion = "验证集在主预算下选择了随机投放；当前这次模型比较没有选出优于该基线的定向政策。"
    elif pd.notna(delta_lower) and delta_lower > 0:
        conclusion = "冻结政策在测试集上相对随机投放的配对区间完全高于 0，在本次条件推断下支持增量改善。"
    elif pd.notna(delta_upper) and delta_upper < 0:
        conclusion = "冻结政策在测试集上相对随机投放的配对区间完全低于 0，本次测试不支持上线该定向政策。"
    else:
        conclusion = "冻结政策相对随机投放的配对区间跨越或接触 0，现有测试结果不足以确认定向带来额外收益。"
    if selected == "propensity":
        conclusion += "本次获选的是转化倾向基线，不能据此宣称 uplift 模型胜出。"

    ate_display = pd.DataFrame(
        {
            "指标": ate["outcome"],
            "对照率": ate["response_control"].map(lambda x: f"{100*x:.4f}%"),
            "处理率": ate["response_treatment"].map(lambda x: f"{100*x:.4f}%"),
            "ATE（百分点）": ate["ate"].map(lambda x: _num(100 * x, 4)),
            "95% CI（百分点）": ate.apply(
                lambda row: _interval(row, scale=100, digits=4), axis=1
            ),
            "相对增幅": ate["relative_uplift"].map(
                lambda x: "—" if pd.isna(x) else f"{100*x:.2f}%"
            ),
            "双侧 p": ate["p_value"].map(_p),
        }
    )
    c2 = frames["c2st"]
    c2_display = pd.DataFrame(
        {
            "模型": c2["model"],
            "ROC AUC": c2["roc_auc"].map(lambda x: _num(x, 4)),
            "LogLoss": c2["log_loss"].map(lambda x: _num(x, 5)),
            "Dummy LogLoss": c2["dummy_log_loss"].map(lambda x: _num(x, 5)),
            "置换 p": c2["pvalue"].map(_p),
            "Holm p（两模型）": c2["pvalue_holm"].map(_p),
        }
    )
    val_primary = frames["validation_policies"]
    val_primary = val_primary[np.isclose(val_primary["budget"], budget)].sort_values(
        "incremental_per_100k", ascending=False
    )
    val_display = pd.DataFrame(
        {
            "政策": val_primary["policy"],
            "验证集增量 / 10 万人": val_primary["incremental_per_100k"].map(_num),
            "相对随机": val_primary["vs_random_per_100k"].map(_num),
            "已冻结选择": val_primary["policy"].map(
                lambda x: "是" if x == selected else ""
            ),
        }
    )
    policy_display = pd.DataFrame(
        {
            "政策": primary["policy"].map(
                lambda x: x + ("（验证集获选）" if x == selected else "")
            ),
            "增量 / 10 万人": primary["incremental_per_100k"].map(_num),
            "95% CI": primary.apply(_interval, axis=1),
            "对随机差值": primary["vs_random_per_100k"].map(_num),
            "配对 95% CI（对随机）": primary.apply(
                lambda r: _interval(r, "vs_random_ci_low", "vs_random_ci_high"), axis=1
            ),
            "对倾向差值": primary["vs_propensity_per_100k"].map(_num),
            "配对 95% CI（对倾向）": primary.apply(
                lambda r: _interval(r, "vs_propensity_ci_low", "vs_propensity_ci_high"),
                axis=1,
            ),
        }
    )
    ranking = frames["test_ranking"]
    ranking_display = pd.DataFrame({"政策": ranking["policy"]})
    for column, label in (
        ("auuc_ht", "HT AUUC × 10⁵"),
        ("qini_score_traditional", "传统 Qini score × 10⁵"),
        ("qini_ht", "HT 中心面积 × 10⁵（补充）"),
    ):
        ranking_display[label] = (
            ranking[column].map(lambda x: _num(x * 100_000))
            if column in ranking
            else "未生成"
        )
    learning = frames["learnability"].query("scope == 'whole'")
    learning_display = pd.DataFrame(
        {
            "指标": learning["outcome"],
            "模型": learning["model"],
            "LogLoss": learning["log_loss"].map(lambda x: _num(x, 5)),
            "AP": learning["average_precision"].map(lambda x: _num(x, 5)),
            "Brier": learning["brier"].map(lambda x: _num(x, 5)),
            "ROC AUC": learning["roc_auc"].map(lambda x: _num(x, 4)),
        }
    )
    calibration_display = pd.DataFrame(
        {
            "分箱": calibration["bin"],
            "样本量": calibration["n"],
            "对照人数": calibration["n_control"],
            "处理人数": calibration["n_treatment"],
            "预测 CATE（百分点）": calibration["predicted_cate"].map(
                lambda x: _num(x * 100, 3)
            ),
            "观察差值（百分点）": calibration["observed_ate"].map(
                lambda x: _num(x * 100, 3)
            ),
            "95% CI（百分点）": calibration.apply(
                lambda r: _interval(r, scale=100, digits=3), axis=1
            ),
        }
    )
    overlap_data = overlap.get("fraction_matching_training_features", {})
    overlap_text = "、".join(f"{esc(str(k))} {v:.2%}" for k, v in overlap_data.items())
    if srm["status"] == "skipped":
        srm_text = "设计分流比例未知，SRM 检验已跳过。观测处理比例不能作为同一份数据的零假设比例。"
    else:
        srm_text = f"以提供的设计处理比例 {srm['expected_ratio']:.2%} 计算 SRM：p = {_p(srm['pvalue'])}。"
    calibration_note = (
        f"展示冻结获选政策 {esc(calibration_model)} 的 CATE 校准。"
        if selected == calibration_model
        else f"获选政策 {esc(selected)} 不输出 CATE，因此不能做 CATE 校准；这里展示验证集内最优 uplift 模型 {esc(calibration_model)} 的辅助诊断，不能把它视作获选政策。"
    )
    b = int(c2["permutations"].iloc[0])
    boot = int(selected_row["bootstrap_replicates"])
    significant_c2st = c2.loc[c2["pvalue_holm"] < 0.05, "model"].tolist()
    c2st_conclusion = (
        "以 5% 作为探索性参考，Holm 校正后存在可观测分配差异信号的模型为 "
        + "、".join(esc(name) for name in significant_c2st)
        + "。"
        if significant_c2st
        else "以 5% 作为探索性参考，两模型经 Holm 校正后都未检出可观测分配差异。"
    )
    split_display = frames["splits"].rename(
        columns={
            "split": "划分",
            "treatment": "处理组标识",
            "n": "样本量",
            "conversion_positive": "转化数",
            "visit_positive": "访问数",
        }
    )
    footer_versions = ", ".join(
        f"{esc(k)} {esc(str(v))}" for k, v in manifest.get("versions", {}).items()
    )
    sections = [
        f'<header><p class="eyebrow">CRITEO UPLIFT · OFFLINE EXPERIMENT</p><h1>从广告实验到增量定向</h1>'
        f'<p class="subtitle">主结果：{esc(outcome)} · 主预算：{primary_pct} · 验证集冻结选择，测试集独立评估</p></header>',
        '<section class="lead"><h2>本次结果</h2>'
        f'<p class="key">验证集选中的政策是 <strong>{esc(selected)}</strong>。</p><p>{conclusion}</p>'
        f'<div class="cards"><div><span>测试集规模</span><strong>{n_test:,}</strong></div>'
        f'<div><span>{primary_pct} 预算下增量 / 10 万人</span><strong>{_num(selected_row["incremental_per_100k"])}</strong></div>'
        f'<div><span>该增量的 95% CI</span><strong>{_interval(selected_row)}</strong></div></div>'
        '<p>单位均为每 10 万名可触达对象的预期增量结果，比较假设每次触达成本及每个结果价值相同。预算指触达人数比例，未包含真实广告成本、收入或利润，不能解释成线上 ROI。</p></section>',
        '<section><h2>数据、任务和评估边界</h2>'
        f'<p>源文件 <code>{source_file}</code> 共 {int(audit["n_rows"]):,} 行；本次建模样本 {int(manifest["sample_n"]):,} 行，'
        f'抽样方法：{esc(audit["sampling"])}，随机种子 {int(manifest["seed"])}。固定采用 train / validation / test = 60% / 20% / 20%，'
        '按 treatment × 主结果联合分层；模型均只在 train 拟合，没有 train + validation 重拟合。</p>'
        '<p>研究对象是公开样本中 treatment <strong>分配</strong>对结果的 ITT 风格效应。模型仅用 f0–f11 处理前特征；'
        'exposure、visit、conversion 不作为特征，也不按曝光或访问行为筛选人群。propensity 在本报告中表示 '
        '<code>P(Y=1 | X,T=1)</code> 的结果倾向排序，不表示处理分配概率。</p>'
        f'<p>全量扫描发现：缺失/非有限数 {int(audit["missing_or_nonfinite"]):,}，非二元标签 {int(audit["invalid_binary_labels"]):,}，'
        f'对照组曝光 {int(audit["control_exposed"]):,}，转化但未访问 {int(audit["conversion_without_visit"]):,}；'
        f'全字段重复指纹 {int(audit["duplicate_row_fingerprints"]):,}。重复指纹只是记录特征诊断，没有 user ID，因此保留原行，不能直接推断重复用户。</p>'
        f'<p>与训练集匿名特征向量相同的比例：{overlap_text}。该检查基于建模时 float32 表示；相同向量不等于同一用户，也不能排除未知用户依赖。</p>'
        + _table(split_display)
        + '</section>',
        '<section><h2>总体实验效果</h2><p>以下为源文件全量聚合 ATE，作为实验背景，不参与模型或政策选择。'
        'ATE 是处理组结果率减去对照组结果率；绝对效应用百分点表示，相对增幅以对照率为分母。</p>'
        + _table(ate_display)
        + _image(
            paths["ate_full"],
            "全量平均处理效应",
            "点为差均值，线为 95% Wald 区间；两个指标采用各自纵轴尺度。",
        )
        + f'<p>独立测试集 {esc(outcome)} 的差均值 ATE 为 {_num(ate_test["ate"] * 100, 4)} 个百分点，'
        f'95% CI {_interval(ate_test, scale=100, digits=4)}。差均值 ATE 与下文固定 e 的 HT 估计是不同有限样本计算，数值不必完全一致。</p></section>',
        '<section><h2>随机化与可学习性诊断</h2>'
        f'<p>{srm_text} 当前训练集处理比例 e = {e:.6f} 只作为 HT 评估的 plug-in 分配概率，并非已知实验设计参数。</p>'
        + _image(
            paths["balance"],
            "处理前协变量平衡",
            "SMD 在训练集计算；0.1 仅是描述性参考，不是随机化成立的检验阈值。KS p 值保存在 balance.csv，未经多重校正。",
        )
        + f'<p>C2ST 对训练集作与 T/Y 无关的等概率抽样（{int(c2st_details["sample_n"]):,} 行），固定 70/30 训练/留出样本位置；'
        f'在两部分内分别打乱 treatment，完整重拟合两类固定模型，共 {b} 次。诊断结果如下：</p>'
        + _table(c2_display)
        + f'<p>{c2st_conclusion}训练集最大 |SMD| 为 {frames["balance"]["abs_smd"].max():.4f}。</p>'
        + '<p>高于置换基线的 AUC 提示可观测分配差异，需要检查实验设计、抽样和特征时序；不显著不能证明随机化。'
        '该置换方案假设可交换分配；如存在未知实验分层或多个广告实验混合，需使用匹配实际设计的置换方案。</p>'
        '<h3>结果可预测性：固定模型对照 Dummy</h3>'
        + _table(learning_display)
        + '<p>上表使用 train 拟合、validation 评估，仅显示 whole 范围；按处理组分别训练的结果保存在 learnability.csv。'
        'Dummy 预测训练集正例率。LogLoss/Brier 越低越好，AP/ROC AUC 越高越好；AP 的基线取决于正例率。'
        '这些指标诊断结果模型，不评估个体因果效应；普通结果预测较弱不能推出不存在效应异质性。</p></section>',
        '<section><h2>验证集选择与测试集政策价值</h2>'
        f'<p>事先设定主预算 {primary_pct}，用验证集“每 10 万名可触达对象的 HT 增量”在 random、propensity 及四个 uplift 模型间选择。'
        f'选择已在测试评估前冻结：<strong>{esc(selected)}</strong>。表格中的测试排名不会重新改变这一选择。</p>'
        + _table(val_display)
        + _image(
            paths["policy"],
            "预算与政策增量及主预算置信区间",
            f"星号标识验证集冻结政策；左图为预算敏感性描述，右图为 {primary_pct} 主预算的点估计和点对点 95% 区间。",
        )
        + _table(policy_display)
        + f'<p>测试集共 {n_test:,} 行，其中 {esc(outcome)} 正例 {n_events:,} 个、对照组正例 {n_control_events:,} 个。'
        '低预算区域和小分箱中的事件更少，估计容易波动。辅助预算不用于测试后选择“最好”的投放比例。</p>'
        f'<p>CI 由 {boot} 次组内配对百分位 bootstrap 得到：各处理组人数、训练好的模型、分数与 e 保持固定，'
        '每次重抽样重新选取 top-k；同一重抽样用于全部政策与基线，因此对随机/倾向的差值区间是配对区间。'
        '这些区间不包含重新训练、验证选择、估计 e 或未知用户依赖的额外不确定性；多个政策/预算区间是点对点区间，未作同时覆盖校正。</p></section>',
        '<section><h2>增量排序与辅助 Qini</h2>'
        + _image(
            paths["ranking"],
            "HT 增量曲线与传统 Qini 曲线",
            "横轴是触达比例；HT 图虚线是随机触达的预期增量，传统 Qini 图虚线连接原点和全样本终点。图只抽取约 501 个点，面积由完整分辨率计算。",
        )
        + _table(ranking_display)
        + '<p>主比较仍是固定主预算的政策增量；HT AUUC 汇总所有预算，传统 Qini score 用作辅助。'
        '二者尺度与归一化不同，不能与其他库的同名分数直接混用；HT 中心面积也不等于传统 Qini score。</p>'
        '<div class="formula"><strong>本报告的计算定义</strong>'
        '<p><code>zᵢ = TᵢYᵢ/e − (1−Tᵢ)Yᵢ/(1−e)</code></p>'
        '<p><code>k = floor(qn), G_HT(q) = (1/n) Σᵢ∈top-k zᵢ</code>；每 10 万人增量 = <code>100000 × G_HT(q)</code>。</p>'
        '<p>随机政策期望 = <code>(k/n) × mean(z)</code>；HT AUUC = <code>∫₀¹ G_HT(q) dq</code>；'
        'HT 中心面积 = <code>HT AUUC − G_HT(1)/2</code>。</p>'
        '<p>传统曲线 <code>Q_trad(q) = [R₁(q) − R₀(q) × N₁(q)/N₀(q)] / n</code>；'
        '<code>Nₐ</code> 为已选前缀内各组人数，<code>Rₐ</code> 为对应正例数。传统 Qini score = '
        '<code>∫₀¹ [Q_trad(q) − q × Q_trad(1)] dq</code>。面积计算将原点 (0,0) 与所有同时包含两组的前缀做线性插值，'
        '桥接最初未定义的区间；原始曲线仍保留相应缺失值。未进行“完美排序面积”归一化，因此不等于某些库的 normalized Qini score。</p>'
        '<p>相同分数用独立于 outcome 的固定随机优先级破同分；模型必须在被评估样本之外训练。'
        '</p></div></section>',
        '<section><h2>CATE 分箱校准</h2>'
        f'<p>{calibration_note}</p>'
        + _image(
            paths["calibration"],
            "CATE 分箱校准",
            "只按预测 CATE 从低到高分为等人数箱。橙线为平均预测，蓝点为组间观察差值，蓝线为点对点 Wald 区间。",
        )
        + _table(calibration_display)
        + '<p>没有个体真实反事实标签，不能在该数据上直接计算真实 ITE/CATE 的 PEHE。分箱观察差值只是组水平诊断；'
        '该测试结果没有用于重新拟合或校准模型，多个箱的区间也不是同时置信带。</p></section>',
        '<section><h2>推断口径与下一步</h2>'
        '<p>全量/分箱 ATE 区间使用未合并方差的正态近似；ATE 双侧 p 值使用零假设下合并比例的 z 检验，'
        '因此两者并非严格有限样本对偶，稀疏分箱尤其需要谨慎解释。p 值不表示零假设为真的概率。</p>'
        '<p>C2ST 的 p 值是 AUC 上尾检验：<code>p = (1 + #{AUC_perm ≥ AUC_obs}) / (B + 1)</code>，'
        f'本次 B = {b}，未校正 p 的最小可能值为 {1/(b+1):.4g}，最大为 1；两模型使用 Holm 校正。'
        'ATE 的渐近 p 值不是该离散网格上的值。政策表只报告点估计和区间，没有把 bootstrap 区间换算为未经定义的 p 值。</p>'
        '<p>公开匿名样本的离线 ITT 结论不能直接外推到当前线上人群、曝光效果或盈利效果。实际收益还需要成本、转化价值、'
        '可触达条件及线上随机验证。</p>'
        '<ol><li>在资源允许时扩到 500 万样本或全量，保留独立留出集；本报告没有宣称已完成该规模训练。</li>'
        '<li>预先定义多随机种子或重复拆分方案，评估训练与选择稳定性；不能反复依据当前测试结果调参。</li>'
        '<li>对相同匿名特征向量按组拆分作敏感性分析，并比较结果；这不是用户级去重，本次尚未实施。</li>'
        '<li>补充成本/价值场景与上线门槛，冻结候选政策后开展新的线上随机实验。</li></ol></section>',
        '<section class="provenance"><h2>复现记录</h2>'
        f'<p>来源 SHA256：<code>{esc(source["sha256"])}</code><br>'
        f'源文件大小：{int(source["size"]):,} bytes · seed：{int(manifest["seed"])} · '
        f'模型阶段累计耗时：{float(manifest["elapsed_seconds"]):.1f} s</p>'
        f'<p>{footer_versions}</p><p>详细参数、数据指纹与不确定性边界见 manifest.json；划分行号、CSV 表格、预测和模型保存在同一输出目录。'
        '此 HTML 的图片全部内嵌，无网络依赖；PNG 也单独保存在 figures/。</p>'
        '<p>方法假设：公开样本中近似恒定的处理分配概率，以及行独立、一致性和无干扰。C2ST 不能证明这些假设。</p>'
        '<p>参考：<a href="https://ailab.criteo.com/criteo-uplift-prediction-dataset/">Criteo AI Lab 数据集与版本说明</a>；'
        '<a href="https://arxiv.org/abs/1706.03461">Künzel et al. (2019), Meta-learners</a>。</p></section>',
    ]
    css = """
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#263746;font:15px/1.8 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    main{max-width:1200px;margin:auto;padding:44px 28px 60px}header{margin:0 0 30px;padding:0 8px}.eyebrow{font-size:12px;letter-spacing:.16em;color:#2171ad;font-weight:700}h1{font-size:36px;line-height:1.25;margin:12px 0}.subtitle{color:#5e7180}
    section{background:#fff;border:1px solid #e2e9ef;border-radius:8px;padding:28px;margin:20px 0}h2{font-size:23px;margin:0 0 16px}h3{font-size:18px;margin-top:24px}p{margin:12px 0}.lead{border-top:4px solid #2171ad}.key{font-size:21px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}.cards div{background:#f2f7fb;padding:16px;border-radius:5px}.cards span{display:block;color:#667887;font-size:13px}.cards strong{font-size:25px;font-weight:650}
    figure{margin:24px 0}img{width:100%;height:auto;display:block}figcaption{font-size:13px;color:#647887;margin-top:10px}.table-wrap{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th{background:#eff5f9;text-align:left;padding:11px 12px;border-bottom:2px solid #d5e3ee}td{padding:9px 12px;border-bottom:1px solid #e8eef3}tr:nth-child(even) td{background:#fafcfe}.formula{background:#f3f7fb;border-left:3px solid #2171ad;padding:16px 20px;margin:22px 0}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}.provenance{font-size:13px;color:#667887}.provenance h2{font-size:19px}li{margin:8px 0}
    @media(max-width:700px){main{padding:24px 12px}section{padding:18px}h1{font-size:28px}.cards{grid-template-columns:1fr}.cards strong{font-size:23px}}@media print{body{background:white}main{padding:0;max-width:none}section{border:0;padding:12px 0;break-inside:auto}figure,.cards{break-inside:avoid}.table-wrap{overflow:visible}table{white-space:normal;font-size:10px}h2,h3{break-after:avoid}}
    """
    document = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Criteo 增量定向实验报告</title><style>'
        + css
        + '</style></head><body><main>'
        + "\n".join(sections)
        + '</main></body></html>'
    )
    destination = safe_output(out / "report.html")
    destination.write_text(document, encoding="utf-8")
    return destination

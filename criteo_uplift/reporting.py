# 保存实验运行记录，读取已有结果生成图表和 HTML 报告。
# finalize_run 记录复现信息与完成状态；build_report 渲染结果。来源见 docs/RESULTS_GUIDE.md。

from __future__ import annotations

import base64
import html
import importlib.metadata
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES
from .metrics import RANKING_METRIC_DESCRIPTION
from .pipeline_io import safe_output, write_result

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


def save_sample_feature_plot(sample, path):
    #-------------
    # 函数名：save_sample_feature_plot
    # 功能：绘制建模样本中 f0–f11 的特征分布直方图并保存为 PNG。
    # 输入参数：
    #     - sample：包含 f0–f11 的建模样本 DataFrame。
    #     - path：PNG 图像输出路径。
    # 输出参数：
    #     - 已保存图像的 pathlib.Path。
    #-------------
    from matplotlib.figure import Figure

    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(13, 8), constrained_layout=True)
    axes = fig.subplots(3, 4)
    for feature, ax in zip(FEATURES, axes.flat):
        ax.hist(sample[feature], bins=40, color="#2476a8", log=True)
        ax.set_title(feature)
        ax.set_ylabel("Count (log)")
    fig.suptitle(f"Feature distributions | modeling sample n = {len(sample):,}")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def save_c2st_plot(out):
    #-------------
    # 函数名：save_c2st_plot
    # 功能：读取 C2ST 观测 AUC 和置换零分布，绘制各诊断模型的 AUC 直方图并保存。
    # 输入参数：
    #     - out：本次实验输出目录，需包含 diagnostics 结果。
    # 输出参数：
    #     - 无显式返回值；在实验 figures 目录保存 c2st_null.png。
    #-------------
    from matplotlib.figure import Figure

    out = Path(out)
    scores = pd.read_csv(out / "diagnostics/c2st.csv")
    null = pd.read_csv(out / "diagnostics/c2st_null.csv")
    details = json.loads((out / "diagnostics/c2st_details.json").read_text())
    path = safe_output(out / "figures/c2st_null.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(11, 4), constrained_layout=True)
    axes = fig.subplots(1, len(scores), squeeze=False)[0]
    for ax, row in zip(axes, scores.itertuples()):
        ax.hist(null.loc[null["model"] == row.model, "roc_auc"],
                bins=24, color="#2171ad", alpha=.8)
        ax.axvline(row.roc_auc, color="#ba6b24", label=f"Observed {row.roc_auc:.4f}")
        ax.axvline(.5, color="#718096", linestyle="--", linewidth=1, label="Chance AUC = 0.5")
        ax.set_title(f"{row.model} | p = {row.pvalue:.3f} | Holm p = {row.pvalue_holm:.3f}")
        ax.set_xlabel("Diagnostic holdout ROC AUC")
        ax.set_ylabel("Permutation count")
        ax.legend(frameon=False)
    fig.suptitle(
        f"Full-source balanced C2ST | {details['n_control']:,} control + "
        f"{details['n_treatment']:,} treatment\n"
        f"{details['permutations']} complete-refit permutations; internal 70/30 split"
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def _num(value, digits: int = 2) -> str:
    #-------------
    # 函数名：_num
    # 功能：将数值格式化为带千位分隔符的字符串，并对缺失值显示占位符。
    # 输入参数：
    #     - value：待格式化的数值。
    #     - digits：保留的小数位数。
    # 输出参数：
    #     - 格式化后的字符串。
    #-------------
    return "—" if value is None or pd.isna(value) else f"{value:,.{digits}f}"


def _p(value) -> str:
    #-------------
    # 函数名：_p
    # 功能：将 p 值格式化为报告展示文本，并明确处理缺失值和数值下溢。
    # 输入参数：
    #     - value：待展示的 p 值。
    # 输出参数：
    #     - 用于 HTML 报告的 p 值字符串。
    #-------------
    if value is None or pd.isna(value):
        return "—"
    if value == 0:
        return "≈0（数值下溢）"
    return f"{value:.3g}"


def _interval(row, low="ci_low", high="ci_high", scale=1, digits=2) -> str:
    #-------------
    # 函数名：_interval
    # 功能：读取一行结果中的置信区间上下界，按指定倍率和精度格式化为区间字符串。
    # 输入参数：
    #     - row：包含区间上下界字段的 Series 或映射对象。
    #     - low：下界字段名。
    #     - high：上界字段名。
    #     - scale：展示时乘以的数值倍率。
    #     - digits：区间端点保留的小数位数。
    # 输出参数：
    #     - 形如 [下界, 上界] 的字符串。
    #-------------
    return f"[{_num(row[low] * scale, digits)}, {_num(row[high] * scale, digits)}]"


def _table(frame: pd.DataFrame) -> str:
    #-------------
    # 函数名：_table
    # 功能：将 DataFrame 转换为带横向滚动容器的 HTML 表格。
    # 输入参数：
    #     - frame：需要嵌入报告的表格数据。
    # 输出参数：
    #     - HTML 表格字符串。
    #-------------
    return (
        '<div class="table-wrap">'
        + frame.to_html(index=False, border=0, escape=True, na_rep="—")
        + "</div>"
    )


def _image(path: Path, alt: str, caption: str) -> str:
    #-------------
    # 函数名：_image
    # 功能：将 PNG 文件编码为 Base64，并生成带替代文本和图注的 HTML figure。
    # 输入参数：
    #     - path：PNG 图像文件路径。
    #     - alt：图像的替代文本。
    #     - caption：显示在图像下方的说明文字。
    # 输出参数：
    #     - 包含内嵌图片和图注的 HTML 字符串。
    #-------------
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'
        f'<figcaption>{caption}</figcaption></figure>'
    )


def _equation(expression: str, label: str) -> str:
    #-------------
    # 将数学表达式排版为内嵌矢量图，使报告离线打开时也能显示分式、积分和上下标。
    # 输入参数：
    #     - expression：Matplotlib MathText 支持的数学表达式，不包含美元分隔符。
    #     - label：公式的中文标题，同时用于辅助阅读。
    # 输出参数：
    #     - 带标题、替代文本和横向滚动容器的 HTML 公式块。
    #-------------
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser

    formula = f'${expression}$'
    font = FontProperties(size=17, math_fontfamily='stix')
    width, height, depth, _, _ = MathTextParser('path').parse(formula, dpi=72, prop=font)
    with rc_context({'svg.hashsalt': 'criteo-report-math', 'svg.fonttype': 'path'}):
        fig = Figure(figsize=((width + 12) / 72, (height + 16) / 72))
        fig.text(6 / (width + 12), (depth + 8) / (height + 16), formula,
                 fontproperties=font, color='#183c56')
        buffer = io.BytesIO()
        fig.savefig(buffer, format='svg', transparent=True, metadata={'Date': None})
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return (
        f'<div class="equation"><div class="equation-label">{html.escape(label)}</div>'
        f'<div class="equation-scroll" tabindex="0" role="group" aria-label="{html.escape(label)}">'
        f'<img class="math-equation" src="data:image/svg+xml;base64,{encoded}" '
        f'alt="{html.escape(expression, quote=True)}"></div></div>'
    )


def _errorbar(ax, point, low, high, positions, *, horizontal=False, color="#2171ad"):
    #-------------
    # 函数名：_errorbar
    # 功能：在给定 Matplotlib 坐标轴上绘制点估计及其上下置信区间。
    # 输入参数：
    #     - ax：Matplotlib 坐标轴对象。
    #     - point：点估计数组。
    #     - low：置信区间下界数组。
    #     - high：置信区间上界数组。
    #     - positions：各点在另一坐标轴上的位置。
    #     - horizontal：是否绘制水平误差线。
    #     - color：误差线和点的颜色。
    # 输出参数：
    #     - 无；直接在 ax 上添加误差线和点。
    #-------------
    valid = np.isfinite(point) & np.isfinite(low) & np.isfinite(high)
    if horizontal:
        ax.hlines(positions[valid], low[valid], high[valid], color=color, lw=2)
        ax.scatter(point, positions, c=color, s=35, zorder=3)
    else:
        ax.vlines(positions[valid], low[valid], high[valid], color=color, lw=2)
        ax.scatter(positions, point, c=color, s=35, zorder=3)


def _overlap_section(out, manifest):
    #-------------
    # 功能：展示 Train 的经验覆盖结果，并与设计正值性及结果标签支持区别说明。
    # 输入参数：
    #     - out：当前报告的结果目录。
    #     - manifest：当前实验复现信息，含模型训练摘要。
    # 输出参数：
    #     - 可嵌入诊断章节的 HTML。
    #-------------
    path = out / 'diagnostics/overlap_positivity.json'
    heading = '<h3>Overlap / Positivity · 训练集经验覆盖</h3>'
    summary = json.loads(path.read_text())
    strata = pd.read_csv(out / 'diagnostics/overlap_strata.csv')
    counts = strata.assign(nonempty_min=strata.min_arm_n.where(~strata.empty_bin))
    by_feature = counts.groupby('feature', sort=False).agg(
        分箱数=('bin', 'size'), 缺臂箱数=('missing_arm', 'sum'),
        人数不足箱数=('insufficient_arm_count', 'sum'), 非空箱最少臂人数=('nonempty_min', 'min'))
    display = by_feature.reset_index().rename(columns={'feature': '特征'})
    reference = summary['expected_treatment_rate']
    reference_text = '未提供' if reference is None else f"{reference:.6f}；观察减参考 = {summary['observed_minus_expected']:.6f}"
    content = (heading + f"<p>范围：{html.escape(summary['scope'])}；经验状态：<strong>{html.escape(summary['status'])}</strong>。"
               f"共 {summary['n']:,} 人，Treatment {summary['n_treatment']:,} 人，Control {summary['n_control']:,} 人；"
               f"观察处理比例 {summary['observed_treatment_rate']:.6f}，对照比例 {summary['observed_control_rate']:.6f}。</p>"
               f"<p>参考分配概率：{reference_text}。来源：{html.escape(summary['probability_source'])}。</p>"
               f"<p>每个特征最多 {summary['rules']['requested_bins']} 个分位数箱，重复边界合并、缺失单独成箱。"
               f"每箱任一臂少于 {summary['rules']['min_arm_count']} 人时提示警告；这是经验阈值，不是正值性的数学判定。"
               f"缺臂 {summary['missing_arm_bins']} 箱，人数不足 {summary['weak_bins']} 箱。分箱边界仅从 Train 得到，未使用 Validation/Test。</p>"
               + ('<p>' + html.escape(summary['execution_context']) + '</p>' if summary.get('execution_context') else '')
               + _table(display)
               + '<p>' + html.escape(summary['interpretation']) + '</p>')
    support = next((info['outcome_support'] for info in manifest.get('models', {}).values()
                    if 'outcome_support' in info), None)
    if support is not None:
        content += ('<p>以下为训练结果标签支持情况，服务于分类器拟合，不是因果识别判据。'
                    '单类别范围使用常数结果概率；两臂标签常数不同也可以产生非零效应估计。'
                    '即使两类都存在，正例很少仍可能导致估计不稳定。</p>'
                    + _table(pd.DataFrame(support).rename(columns={'treatment': '处理臂', 'n': '人数',
                        'positive': '正例', 'negative': '负例', 'single_class': '单类别'})))
    return content


def build_report(manifest_path: Path, report_path: Path) -> Path:
    #-------------
    # 函数名：build_report
    # 功能：读取实验 manifest 和各模块结果，生成包含数据诊断、ATE、Uplift、Stability 与 Planning 的完整 HTML 报告。
    # 输入参数：
    #     - manifest_path：本次实验 manifest.json 路径。
    #     - report_path：最终 HTML 报告输出路径。
    # 输出参数：
    #     - 生成的 HTML 报告 pathlib.Path。
    #-------------
    manifest_path = Path(manifest_path)
    out = safe_output(manifest_path.parent)
    report_path = safe_output(report_path)
    manifest = json.loads((out / manifest_path.name).read_text(encoding="utf-8"))
    audit = json.loads((out / "data_preparation" / "audit.json").read_text(encoding="utf-8"))
    selection, srm = json.loads((out / "models" / "selection.json").read_text(encoding="utf-8")), json.loads((out / "diagnostics" / "srm.json").read_text(encoding="utf-8"))
    overlap = json.loads((out / "data_preparation" / "feature_overlap.json").read_text(encoding="utf-8"))
    c2st_details = json.loads((out / "diagnostics" / "c2st_details.json").read_text(encoding="utf-8"))
    frames = {
        name: pd.read_csv(out / folder / f"{name}.csv", dtype={"p_value_text": "string"} if name == "ate_summary" else None)
        for folder, names in (
            ("diagnostics", ("balance", "c2st", "learnability")),
            ("models", ("validation_bootstrap",)),
            ("evaluation", ("ate_summary", "test_metrics", "test_bootstrap", "test_pairwise", "curves")),
            ("data_preparation", ("split_summary",)),
        )
        for name in names
    }
    frames["ate_full"] = frames["ate_summary"].query("scope == 'full_dataset'")
    ate_test = frames["ate_summary"].loc[
        (frames["ate_summary"].scope == "test") &
        (frames["ate_summary"].outcome == manifest["primary_outcome"])
    ].iloc[0].to_dict()
    sample_report = json.loads((out / "data_preparation" / "sample_report.json").read_text(encoding="utf-8"))
    selected = selection["selected_policy"]
    selected_uplift = selection["selected_uplift_model"]
    budget = float(selection["primary_budget"])
    outcome = manifest["primary_outcome"]
    e = float(manifest["assignment_probability"])
    primary_metric = selection["primary_metric"]
    metric_labels = {
        "auuc": "AUUC", "qini_score": "Traditional Qini Score",
        "gain_ht": "HT Gain", "gain_plugin": "Uplift Gain", "qini_traditional": "Traditional Qini Gain",
    }
    area_primary = primary_metric in ("auuc", "qini_score")
    primary_label = metric_labels[primary_metric]
    criterion = f"完整曲线的 {primary_label}" if area_primary else f"{budget:.0%} 覆盖率的 {primary_label}"
    scale_label = "面积 × 10⁵" if area_primary else "每 10 万总体用户标度"
    test_bootstrap = frames["test_bootstrap"]
    test_pairs = frames["test_pairwise"]
    policies = frames["test_metrics"].copy()
    ranking = policies.drop_duplicates("policy")
    full = policies[policies.budget.eq(1.)].iloc[0]
    random = policies[policies.policy.eq(selected)].copy()
    random["policy"] = "random"
    random[["auuc", "qini_score"]] = [full.gain_plugin / 2, 0.]
    for metric in ("gain_ht", "gain_plugin", "qini_traditional"):
        random[metric] = random.n_selected / full.n_selected * full[metric]
    random["gain_ht_per_100k"] = random.gain_ht * 100_000
    policies = pd.concat([policies, random], ignore_index=True)
    policies["incremental_per_100k"] = policies["gain_ht_per_100k"]
    random_gain = random.set_index("budget").gain_ht_per_100k
    propensity_gain = policies[policies.policy.eq("propensity")].set_index("budget").gain_ht_per_100k
    policies["vs_random_per_100k"] = policies.incremental_per_100k - policies.budget.map(random_gain)
    policies["vs_propensity_per_100k"] = policies.incremental_per_100k - policies.budget.map(propensity_gain)
    if primary_metric == "gain_ht":
        intervals = test_bootstrap[["policy", "budget", "ci_low", "ci_high"]].copy()
        intervals[["ci_low", "ci_high"]] *= 100_000
        policies = policies.merge(intervals, on=["policy", "budget"], validate="one_to_one")
    else:
        policies[["ci_low", "ci_high"]] = np.nan
    primary = policies[policies.budget.eq(budget)].sort_values("incremental_per_100k", ascending=False)
    selected_row = primary.loc[primary.policy.eq(selected)].iloc[0]
    calibration_model = selected_uplift if selected == "propensity" else selected
    calibration = pd.read_csv(out / "evaluation" / f"calibration_{calibration_model}.csv")
    figures = safe_output(out / "figures")
    figures.mkdir(exist_ok=True)
    save_c2st_plot(out)

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
            #-------------
            # 函数名：build_report.save
            # 功能：将当前 Matplotlib Figure 保存到预先登记的报告图像路径并关闭图形。
            # 输入参数：
            #     - fig：待保存的 Matplotlib Figure。
            #     - name：图像名称，用于查找对应输出路径。
            # 输出参数：
            #     - 无；保存 PNG 并关闭 Figure。
            #-------------
            fig.savefig(paths[name], dpi=160, bbox_inches="tight")
            plt.close(fig)

        ate = frames["ate_full"]
        # 从当前表格现场绘图；原始列保持比例单位，绘图副本转换为百分点（pp）。
        from matplotlib.ticker import MaxNLocator

        ate_plot = (
            ate.set_index('outcome')
            .loc[['conversion', 'visit'], ['ate', 'ci_low', 'ci_high']]
            .mul(100)  # 转换为百分点。
        )

        with plt.rc_context(  # 临时使用 Notebook 的图片样式。
            {
                'font.family': 'DejaVu Sans',
                'font.size': 11,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.left': False,
                'axes.edgecolor': '#C8D0D8',
                'axes.labelcolor': '#344454',
                'text.color': '#23313F',
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
            }
        ):
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(12, 6.2),
                sharex=False,
                dpi=130,
                constrained_layout=True,
            )
            fig.suptitle(
                'Absolute treatment effects in the full public sample\n'
                '95% confidence intervals | Separate x-axis scales | Dashed line: zero effect',
                fontsize=13,
            )

            for ax, (ate_outcome, row), color in zip(
                axes,
                ate_plot.iterrows(),
                ['#2166AC', '#087F8C'],
            ):
                ax.axvline(
                    0,
                    color='#536170',
                    linestyle='--',
                    linewidth=1.3,
                    zorder=2,
                )
                ax.errorbar(
                    x=[row['ate']],
                    y=[0],
                    xerr=[
                        [row['ate'] - row['ci_low']],
                        [row['ci_high'] - row['ate']],
                    ],
                    fmt='none',
                    color=color,
                    elinewidth=3.5,
                    capsize=10,
                    capthick=2.2,
                    zorder=3,
                )

                # 单独绘制小型空心点，避免较窄的 Visit 区间被实心标记遮挡。
                ax.plot(row['ate'], 0, marker='o', markersize=4.5,
                        markerfacecolor='white', markeredgecolor=color,
                        markeredgewidth=1.6, linestyle='none', zorder=4)

                # 每个子图独立缩放，但始终包含零点、点估计和完整 CI；保持线性坐标。
                left = min(0.0, row['ci_low'], row['ate'])
                right = max(0.0, row['ci_high'], row['ate'])
                span = right - left
                padding = 0.08 * span if span > 0 else 0.01
                ax.set_xlim(
                    left - padding,
                    right + padding,
                )

                ax.set_ylim(-0.45, 0.85)
                ax.set_yticks(
                    [0],
                    [ate_outcome.capitalize()],
                )
                ax.tick_params(axis='y', length=0, pad=12)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

                ax.set_xlabel('Absolute treatment effect (percentage points, pp)')
                ax.set_axisbelow(True)
                ax.grid(axis='x', color='#E6EBEF', linewidth=0.8)

                # 标注使用绘图区相对坐标，放在点和 CI 上方，避免遮挡。
                ax.text(
                    0.98,
                    0.94,
                    f"ATE = {row['ate']:+.4f} pp\n"
                    f"95% CI: [{row['ci_low']:.4f}, {row['ci_high']:.4f}] pp",
                    transform=ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=12,
                    fontweight='bold',
                    color=color,
                    linespacing=1.5,
                )

            save(fig, "ate_full")

        balance = frames["balance"].sort_values("abs_smd", ascending=True)
        full_balance = balance.get("scope", pd.Series(dtype=str)).eq("full_published_dataset").all() and "scope" in balance
        balance_scope = "Full public data | raw float64" if full_balance else "Training split"
        n_balance = int(balance.iloc[0]["n_treatment"] + balance.iloc[0]["n_control"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        axes[0].barh(balance["feature"], balance["abs_smd"], color="#2171ad", height=.65)
        axes[0].axvline(.1, color="#e38636", linestyle="--", lw=1.5, label="0.1 descriptive reference")
        axes[0].set_xlabel("Absolute standardized mean difference")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].barh(balance["feature"], balance["ks_statistic"], color="#24858b", height=.65)
        axes[1].set_xlabel("Two-sided KS statistic")
        for ax in axes:
            ax.grid(axis="x")
        fig.suptitle(f"{balance_scope} | n = {n_balance:,}\nMarginal T/C balance; no randomization pass/fail threshold")
        save(fig, "balance")


        curves = frames["curves"]
        fig, axes = plt.subplots(3, 1, figsize=(12.6, 10.5), constrained_layout=True)
        first = curves[curves["policy"] == curves["policy"].iloc[0]].sort_values(
            "fraction"
        )
        for name, group in curves.groupby("policy", sort=False):
            group = group.sort_values("fraction")
            color = _COLORS.get(name, "#2171ad")
            weight = 2.4 if name == selected else 1.4
            for ax, column in zip(axes, ("gain_ht", "gain_plugin", "qini_traditional")):
                values = group[column].copy()
                values.loc[group["fraction"] == 0] = 0
                ax.plot(
                    group["fraction"],
                    values * 100_000,
                    color=color,
                    lw=weight,
                    label=_LABELS.get(name, name) + (" *" if name == selected else ""),
                )
        # 三种曲线分别连接自己的全量终点；抽点只用于绘图，不重新积分。
        for ax, column in zip(axes, ("gain_ht", "gain_plugin", "qini_traditional")):
            ax.plot([0, 1], [0, float(first[column].iloc[-1]) * 100_000],
                    color=_COLORS["random"], ls="--", lw=1.6,
                    label="Reference: q x own endpoint")
        for ax in axes:
            ax.axhline(0, color="#c4ced7", lw=0.8)
            ax.set_xlabel("Fraction of eligible population contacted")
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=0.7)
            ax.legend(frameon=False, fontsize=8, loc="best", ncol=3)
        axes[0].set_title("HT incremental gain | held-out test")
        axes[0].set_ylabel("Incremental responses / 100,000 eligible people")
        axes[1].set_title("Uplift Gain curve | plug-in estimate")
        axes[1].set_ylabel("Uplift Gain x 100,000")
        axes[2].set_title("Traditional Qini curve | auxiliary diagnostic")
        axes[2].set_ylabel("Traditional Qini ordinate x 100,000")
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
        axes[0].axvline(budget, color="#a7b3be", lw=1, ls=":", label=f"Scenario: {budget:.0%}")
        axes[0].set_xticks(sorted({0., .1, .2, .3, .5, 1., budget}))
        axes[0].set_title("Contact fraction | full coverage 100%")
        axes[0].set_xlabel("Contact fraction (not currency)")
        axes[0].set_ylabel("HT incremental responses / 100,000 eligible users")
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
        axes[1].set_title(f"HT Gain at {budget:.0%}" + (" | pointwise 95% CI" if primary_metric == "gain_ht" else " | point estimates"))
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
    test_split = frames["split_summary"].query("split == 'test'")
    n_test = int(test_split["n"].sum())
    event_column = f"{outcome}_positive"
    n_events = int(test_split[event_column].sum())
    n_control_events = int(
        test_split.loc[test_split["treatment"] == 0, event_column].sum()
    )
    primary_pct = f"{budget:.0%}"
    source = manifest["source"]
    source_file = esc(Path(source["path"]).name)
    frozen_random = test_pairs[test_pairs.policy.eq(selected) & test_pairs.reference.eq("random")]
    if not area_primary:
        frozen_random = frozen_random[frozen_random.budget.eq(budget)]
    delta = frozen_random.iloc[0]
    if pd.isna(delta.ci_low):
        conclusion = f"冻结策略对 Random 的 {criterion} 配对区间缺少有效重复。"
    elif delta.ci_low > 0:
        conclusion = f"冻结策略对 Random 的 {criterion} 配对区间完全高于 0。"
    elif delta.ci_high < 0:
        conclusion = f"冻结策略对 Random 的 {criterion} 配对区间完全低于 0。"
    else:
        conclusion = f"冻结策略对 Random 的 {criterion} 配对区间包含 0。"
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
            "双侧 p": ate["p_value_text"] if "p_value_text" in ate else ate["p_value"].map(_p),
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
    # 只展示主指标的不确定性。面积按 10⁵ 缩放，Gain 使用每十万总体用户标度。
    val = frames["validation_bootstrap"].query("policy != 'random'").sort_values(
        "bootstrap_mean", ascending=False, kind="stable",
    )
    val_display = pd.DataFrame({
        "政策": val.policy, "点估计": val.point_estimate.map(lambda x: _num(x * 100_000)),
        "Bootstrap 均值": val.bootstrap_mean.map(lambda x: _num(x * 100_000)),
        "95% CI": val.apply(lambda r: _interval(r, scale=100_000), axis=1),
        "获胜频率": val.win_frequency.map(lambda x: _num(x * 100, 1) + "%" if pd.notna(x) else "—"),
        "有效重复": val.valid_replicates,
        "冻结": val.policy.map(lambda name: "是" if name == selected else ""),
    })
    test_display = pd.DataFrame({
        "政策": test_bootstrap.policy,
        "覆盖率": test_bootstrap.budget.map(lambda q: f"{q:.1%}" if pd.notna(q) else "完整曲线"),
        "点估计": test_bootstrap.point_estimate.map(lambda x: _num(x * 100_000)),
        "Bootstrap 均值": test_bootstrap.bootstrap_mean.map(lambda x: _num(x * 100_000)),
        "95% CI": test_bootstrap.apply(lambda r: _interval(r, scale=100_000), axis=1),
        "有效重复": test_bootstrap.valid_replicates,
    })
    comparisons = test_pairs[
        test_pairs.reference.isin({selected, "propensity", "random"})
        & (test_pairs.policy.eq(selected) | test_pairs.reference.ne("random"))
    ]
    comparison_display = pd.DataFrame({
        "政策": comparisons.policy, "参考策略": comparisons.reference,
        "覆盖率": comparisons.budget.map(lambda q: f"{q:.1%}" if pd.notna(q) else "完整曲线"),
        "点估计差值": comparisons.difference.map(lambda x: _num(x * 100_000)),
        "配对 95% CI": comparisons.apply(lambda r: _interval(r, scale=100_000), axis=1),
        "有效配对": comparisons.valid_replicates,
    })
    ranking_display = pd.DataFrame({
        "政策": ranking.policy,
        "AUUC × 10⁵": ranking.auuc.map(lambda x: _num(x * 100_000)),
        "Traditional Qini Score × 10⁵": ranking.qini_score.map(lambda x: _num(x * 100_000)),
    })
    budget_display = pd.DataFrame({
        "政策": policies.policy.map(lambda name: name + ("（冻结）" if name == selected else "")),
        "投放覆盖率": policies.budget.map(lambda q: f"{q:.0%}"),
        "选中人数": policies.n_selected,
        "Uplift Gain × 10⁵": policies.gain_plugin.map(lambda x: _num(x * 100_000)),
        "Traditional Qini Gain × 10⁵": policies.qini_traditional.map(lambda x: _num(x * 100_000)),
        "HT 增量 / 10 万总体用户": policies.incremental_per_100k.map(_num),
        "HT 对 Random 差值": policies.vs_random_per_100k.map(_num),
        "HT 对 Propensity 差值": policies.vs_propensity_per_100k.map(_num),
    })
    if primary_metric == "gain_ht":
        budget_display["HT 点态 95% CI"] = policies.apply(_interval, axis=1)
    key_budgets = [.1, .2, .3, 1.]
    key_rows = []
    for name in dict.fromkeys([selected, "random", "propensity"]):
        for q in key_budgets:
            key_rows.extend(policies.index[(policies["policy"] == name) & np.isclose(policies["budget"], q)])
    budget_key_display = budget_display.loc[key_rows]
    frozen = policies.loc[policies["policy"] == selected].set_index("budget").loc[key_budgets]

    # 固定排序的新增区间贡献等于两个累计值之差；不相减置信区间端点。
    values = np.r_[0., frozen["incremental_per_100k"].to_numpy()]
    counts = np.r_[0, frozen["n_selected"].to_numpy()]
    marginal = np.diff(values)
    added_counts = np.diff(counts)
    boundaries = [0., *key_budgets]
    marginal_display = pd.DataFrame({
        "新增覆盖区间": [f"{a:.0%}–{b:.0%}" for a, b in zip(boundaries[:-1], boundaries[1:])],
        "新增覆盖比例（实际）": [f"{n / n_test:.2%}" for n in added_counts],
        "新增人数": added_counts,
        "新增估计转化 / 10 万总体用户（点估计）": [_num(v) for v in marginal],
    })
    marginal_findings = "；".join(
        f"{label} 为{'正' if value > 0 else '负' if value < 0 else '零'}增量 {_num(value)}"
        for label, value in zip(marginal_display["新增覆盖区间"], marginal)
    )
    marginal_trend = (
        "20%–30% 的新增贡献低于 10%–20%，这两个等宽区间的点估计呈递减。"
        if marginal[2] < marginal[1] else
        "20%–30% 的新增贡献不低于 10%–20%，这两个等宽区间的点估计未呈递减。"
    )
    full_value = float(values[-1])
    full_comparisons = "；".join(
        f"{q:.0%} 覆盖率下为 {_num(row.incremental_per_100k)}，比全量{'多' if row.incremental_per_100k >= full_value else '少'} "
        f"{_num(abs(row.incremental_per_100k - full_value))}"
        for q, row in frozen.iloc[:3].iterrows()
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
        srm_text = f"全量公开数据计数与外部参考比例 {srm['expected_ratio']:.2%} 的一致性检查：p = {_p(srm['pvalue'])}；不能据此验证原始线上实验SRM或证明随机化。"
    calibration_note = (
        f"展示冻结获选政策 {esc(calibration_model)} 的 CATE 校准。"
        if selected == calibration_model
        else f"获选政策 {esc(selected)} 不输出 CATE，因此不能做 CATE 校准；这里展示验证集内最优 uplift 模型 {esc(calibration_model)} 的辅助诊断，不能把它视作获选政策。"
    )
    full_diagnostic = c2st_details.get("scope") == "full_source_arm_balanced_sample"
    balance_caption = (
        "原始全量float64特征SMD/KS；0.1仅是经验参考。KS p值未经多重校正，不能仅凭显著性判断实际意义。"
        if full_balance else "训练集SMD/KS；0.1仅是经验参考，不能证明随机化。"
    )
    diagnostic_population = "全量公开数据" if full_balance else "训练集"
    c2st_sampling_text = (
        f"C2ST从全量原始数据分别等概率无放回抽取两组各 {int(c2st_details['n_control']):,} 行"
        if full_diagnostic else f"C2ST使用开发数据的随机样本，共 {int(c2st_details['sample_n']):,} 行"
    )
    max_ks_row = frames["balance"].loc[frames["balance"]["ks_statistic"].idxmax()]
    b = int(c2["permutations"].iloc[0])
    boot = int(selection["n_bootstrap"])
    full_interval = f"，点态 95% CI {_interval(frozen.iloc[-1])}" if primary_metric == "gain_ht" else ""
    significant_c2st = c2.loc[c2["pvalue_holm"] < 0.05, "model"].tolist()
    c2st_conclusion = (
        "以 5% 作为探索性参考，Holm 校正后存在可观测分配差异信号的模型为 "
        + "、".join(esc(name) for name in significant_c2st)
        + "。"
        if significant_c2st
        else "以 5% 作为探索性参考，两模型经 Holm 校正后都未检出可观测分配差异。"
    )
    split_display = frames["split_summary"].rename(
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
        f'<p class="subtitle">主结果：{esc(outcome)} · 选型指标：{esc(criterion)} · 验证集冻结选择，测试集独立评估</p></header>',
        '<section class="lead"><h2>Business Background · 业务问题与本次结果</h2>'
        f'<p class="key">验证集选中的政策是 <strong>{esc(selected)}</strong>。</p><p>{conclusion}</p>'
        f'<div class="cards"><div><span>测试集规模</span><strong>{n_test:,}</strong></div>'
        f'<div><span>{primary_pct} 预算下增量 / 10 万人</span><strong>{_num(selected_row["incremental_per_100k"])}</strong></div>'
        + (f'<div><span>该增量的点态 95% CI</span><strong>{_interval(selected_row)}</strong></div>' if primary_metric == "gain_ht" else "") + '</div>'
        '<p>单位均为每 10 万名可触达对象的预期增量结果，比较假设每次触达成本及每个结果价值相同。预算指触达人数比例，未包含真实广告成本、收入或利润，不能解释成线上 ROI。</p></section>',
        '<section><h2>Data Preparation · 数据与分析边界</h2>'
        f'<p>源文件 <code>{source_file}</code> 共 {int(audit["n_rows"]):,} 行；本次建模样本 {int(manifest["sample_n"]):,} 行，'
        f'抽样方法：{esc(sample_report["sampling"])}，随机种子 {int(manifest["seed"])}。固定采用 train / validation / test = 60% / 20% / 20%，'
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
        '<section><h2>ATE / ITT · 总体分配效果</h2><p>以下为源文件全量聚合 ATE，作为实验背景，不参与模型或政策选择。'
        'ATE 是处理组结果率减去对照组结果率；绝对效应用百分点表示，相对增幅以对照率为分母。</p>'
        + _table(ate_display)
        + _image(
            paths["ate_full"],
            "全量平均处理效应",
            "空心点为差均值，横线为 95% Wald 区间；两个指标采用各自横轴尺度，虚线表示零效应。",
        )
        + f'<p>独立测试集 {esc(outcome)} 的差均值 ATE 为 {_num(ate_test["ate"] * 100, 4)} 个百分点，'
        f'95% CI {_interval(ate_test, scale=100, digits=4)}。差均值 ATE 与下文固定 e 的 HT 估计是不同有限样本计算，数值不必完全一致。</p></section>',
        '<section><h2>Experiment Diagnostics · 分组与可学习性诊断</h2>'
        f'<p>{srm_text} 当前训练集处理比例 e = {e:.6f} 只作为 HT 评估的 plug-in 分配概率，并非已知实验设计参数。</p>'
        + _image(
            paths["balance"],
            "处理前协变量平衡",
            balance_caption,
        )
        + f'<p>{c2st_sampling_text}，固定70/30内部训练/留出划分；'
        f'在两部分内分别打乱 treatment，完整重拟合两类固定模型，共 {b} 次。诊断结果如下：</p>'
        + _table(c2_display)
        + _image(figures / "c2st_null.png", "C2ST 置换 AUC 分布",
                 f"柱高表示 {b} 次完整重拟合置换中落入各 AUC 区间的次数；橙线为真实标签的观测 AUC，灰色虚线为 0.5。"
                 "原始 p 按逐次 AUC 计算，未用直方图分箱近似；Holm 校正仅包含原定的两个分类器。")
        + f'<p>{c2st_conclusion}{diagnostic_population}最大 |SMD| 为 {frames["balance"]["abs_smd"].max():.6f}；最大KS为 {max_ks_row["ks_statistic"]:.6f}（{esc(max_ks_row["feature"])}）。置换p值分辨率为 {1 / (b + 1):.3f}。</p>'
        + '<p>高于置换基线的 AUC 提示可观测分配差异，需要检查实验设计、抽样和特征时序；不显著不能证明随机化。'
        '该置换方案假设可交换分配；如存在未知实验分层或多个广告实验混合，需使用匹配实际设计的置换方案。</p>'
        + _overlap_section(out, manifest)
        + '<h3>结果可预测性：固定模型对照 Dummy</h3>'
        + _table(learning_display)
        + '<p>上表使用 train 拟合、validation 评估，仅显示 whole 范围；按处理组分别训练的结果保存在 learnability.csv。'
        'Dummy 预测训练集正例率。LogLoss/Brier 越低越好，AP/ROC AUC 越高越好；AP 的基线取决于正例率。'
        '这些指标诊断结果模型，不评估个体因果效应；普通结果预测较弱不能推出不存在效应异质性。</p></section>',
        '<section><h2>Model Selection · Training &amp; Validation 冻结选择</h2>'
        f'<p>Validation 以预设的 {esc(criterion)} 为唯一主指标，从五个评分策略中选取 Bootstrap 均值最高者。'
        f'冻结策略为 <strong>{esc(selected)}</strong>，表内单位为{scale_label}。</p>'
        + _table(val_display)
        + '<p>获胜频率是在所有候选均有定义的重复中排名第一的比例。'
        '它描述固定评分下的选择稳定性。Test 复用冻结选择。</p>'
        + f'<h3>Test 主指标 · {esc(primary_label)}</h3><p>单位为{scale_label}。'
        f'{boot} 次处理组和对照组内的有放回配对重抽样。'
        '仅对主指标计算区间，各预算的区间为点态区间。'
        '有效重复数按单策略或共同有效配对计算。评分及处理概率固定。</p>'
        + f'<p>测试集共 {n_test:,} 行，其中 {esc(outcome)} 正例 {n_events:,} 个、对照组正例 {n_control_events:,} 个。</p>'
        + _table(test_display)
        + '<details><summary>相对冻结策略、Propensity 与 Random 的配对差值</summary>'
        + _table(comparison_display) + '</details></section>',
        '<section><h2>Ranking Performance · Test Evaluation 排序能力</h2>'
        + _image(
            paths["ranking"],
            "HT Gain、Uplift Gain 与 Traditional Qini 三类曲线",
            "横轴是触达比例；各虚线均为 q 乘以该曲线自己的全样本终点。图只抽取约 501 个点，面积沿用完整分辨率计算结果，不从图中重新积分。",
        )
        + '<div id="ranking-metrics">' + _table(ranking_display) + '</div>'
        + '<p>AUUC 对 Uplift Gain 的完整曲线积分。Traditional Qini Score 扣除自身端点参考线面积。'
        '上表统一乘以 10⁵，面积指标仍是排序指标。固定覆盖率的三种 Gain 见下节。'
        'Propensity 使用 T-XGBoost 预测的处理下转化概率。</p>'
        '<div class="formula" id="calculation-definitions"><h3>本报告的计算定义</h3>'
        '<p class="formula-intro">先计算每行的 HT 伪结果，再按固定排名累计；以下定义与结果表使用同一口径。</p>'
        + _equation(r'z_i=\frac{T_iY_i}{e}-\frac{(1-T_i)Y_i}{1-e}', '01 · 单行 HT 伪结果')
        + _equation(r'k=\lfloor qn\rfloor,\qquad G_{\mathrm{HT}}(q)=\frac{1}{n}\sum_{i\in\mathrm{top}\text{-}k}z_i',
                    '02 · 预算与累计增量')
        + _equation(r'\mathrm{Incremental\ responses\ per\ 100{,}000}=100{,}000\,G_{\mathrm{HT}}(q)',
                    '每 10 万名可触达对象的增量')
        + _equation(r'G_{\mathrm{random}}(q)=\frac{k}{n}\,\bar{z}', '03 · 随机政策的预期增量')
        + _equation(r'G(q)=\frac{k}{n}\left[\frac{R_1(q)}{N_1(q)}-\frac{R_0(q)}{N_0(q)}\right]',
                    'Uplift Gain：前缀两组响应率之差乘以覆盖比例')
        + _equation(r'\mathrm{AUUC}=\int_0^1G(q)\,dq',
                    'Uplift AUUC')
        + _equation(r'Q_{\mathrm{trad}}(q)=\frac{1}{n}\left[R_1(q)-R_0(q)\frac{N_1(q)}{N_0(q)}\right]',
                    '05 · 传统 Qini 曲线')
        + _equation(r'\mathrm{Qini\ score}=\int_0^1\left[Q_{\mathrm{trad}}(q)-q\,Q_{\mathrm{trad}}(1)\right]dq',
                    '06 · 传统 Qini score')
        + '<p class="formula-note"><var>N<sub>a</sub></var> 为已选前缀内各组人数，'
        '<var>R<sub>a</sub></var> 为对应正例数；<var>n</var> 为全体待评估人数，'
        '<var>q</var> 为预算比例，<var>e</var> 为报告使用的处理分配概率估计。</p>'
        '<p>Uplift Gain 仅在前缀同时包含两组时有定义；积分保留原点及这些有效前缀。'
        '传统 Qini 曲线从原点 (0,0) 开始，对所有 <var>N<sub>0</sub>(q) &gt; 0</var> 的前缀点作梯形积分，'
        '包括仅包含对照组时的零值点。尚无对照组的前缀曲线值未定义，不作为积分节点；'
        '积分从原点线性连接到首个保留点，原始曲线仍保留相应缺失值。'
        '未进行“完美排序面积”归一化，因此不能直接等同于其他库中同名的 normalized Qini score。</p>'
        '<p>相同分数用独立于 outcome 的固定随机优先级破同分；模型必须在被评估样本之外训练。</p>'
        '</div></section>',
        '<section><h2>Budget Sensitivity Analysis · 不同投放覆盖率的增量价值</h2>'
        '<p>Budget 表示投放覆盖率（Contact Fraction），不是货币金额；所有增量均以 10 万名总体可触达用户为分母。'
        '下表展示冻结策略、Random 和 Propensity；若冻结策略就是其中一个基线，仅展示一次。</p>'
        + '<div id="budget-sensitivity">' + _table(budget_key_display) + '</div>'
        + _image(paths["policy"], "HT 增量的覆盖率敏感性",
                 f"星号标识冻结策略；10%、20%、30%、100% 为关键情景，{primary_pct} 为预设的投放情景覆盖率，100% 为全量终点。连线仅辅助读图，未在评价点之间插值计算。")
        + '<details><summary>全部候选、全部覆盖率明细（含 2%、5%、50%）</summary>'
        + _table(budget_display) + '</details>'
        + '<p>同一固定评分及同分顺序下，Top 10% ⊂ Top 20% ⊂ Top 30% ⊂ Top 100%。'
        '从 10% 扩到 20% 加入排名 10%–20% 的人，从 20% 扩到 30% 加入排名 20%–30% 的人；'
        '选中人数按覆盖率乘以评价总人数向下取整。匿名特征不能用于编造这些人群的业务画像。'
        'Random 是相同人数的随机触达解析期望，不对应模型排序。辅助预算不用于 Test 后挑选最优商业预算。</p></section>',
        '<section><h2>Marginal Incremental Gain · 新增覆盖区间贡献</h2>'
        f'<p>保持冻结策略 <strong>{esc(selected)}</strong> 不变，以 ΔV = V(b) − V(a)、V(0) = 0 计算新增区间贡献。'
        '</p>'
        + '<div id="marginal-gain">' + _table(marginal_display) + '</div>'
        + f'<p>{marginal_findings}。{marginal_trend}</p>'
        '<p>这里仅报告点估计，未计算边际增量 CI，不能用两个累计 CI 的端点相减来构造区间。'
        '因此正、负和递减均为描述，不是显著性结论；30%–100% 宽达 70%，其总贡献不能直接与 10% 宽区间比较单位触达效率。'
        '没有实际成本和收入信息，本节只讨论估计增量转化，不能据此判断利润或 ROI。</p></section>',
        '<section><h2>Full Coverage Scenario · 全量投放情景</h2>'
        + _equation(r'G_{\mathrm{HT}}(1)=\widehat{\mathrm{ATE}}_{\mathrm{HT}},\qquad V(1)=100{,}000\,\widehat{\mathrm{ATE}}_{\mathrm{HT}}',
                    '全量覆盖与 HT-ATE')
        + f'<p>100% 覆盖表示对当前 Test 的全部 {n_test:,} 名可触达用户分配处理，估计增量为 '
        f'<strong>{_num(full_value)}</strong> / 10 万总体用户{full_interval}。'
        '所有策略选择相同的全体用户，因此具有同一 HT 终点；排序不会改变全量的价值。</p>'
        + f'<p>冻结策略与全量相比（同为每 10 万总体用户）：{full_comparisons}。'
        '扩大覆盖是否继续增加估计转化，需结合上一节各新增区间的实际符号判断，不预设越多越好。</p>'
        '<p>这是以当前测试人群和固定处理分配概率为前提的离线情景，HT-ATE 与差均值 ATE 在有限样本中可能不同。'
        '这些数字不代表已经实现的广告曝光、收入或利润，不能据此宣称找到了最优商业投放覆盖率。</p></section>',
        '<section><h2>Uplift Modeling · S/T/X 与 CATE 校准</h2>'
        f'<p>{calibration_note}</p>'
        + _image(
            paths["calibration"],
            "CATE 分箱校准",
            "只按预测 CATE 从低到高分为等人数箱。橙线为平均预测，蓝点为组间观察差值，蓝线为点对点 Wald 区间。",
        )
        + '<p>S 用 X 和 T 拟合一个结果模型；T 分组拟合两个结果模型；X 再用两组伪效应拟合回归器，按训练处理比例加权。X 的 CATE 来自两个效应回归器的加权预测。每个候选只在 Train 拟合一次，Validation 和 Test 复用已拟合模型。</p>'
        + _table(calibration_display)
        + '<p>没有个体真实反事实标签，不能在该数据上直接计算真实 ITE/CATE 的 PEHE。分箱观察差值只是组水平诊断；'
        '该测试结果没有用于重新拟合或校准模型，多个箱的区间也不是同时置信带。</p></section>',
        '<section><h2>Conclusion · 推断边界与下一步</h2>'
        '<p>全量/分箱 ATE 区间使用未合并方差的正态近似；ATE 双侧 p 值使用零假设下合并比例的 z 检验，'
        '因此两者并非严格有限样本对偶，稀疏分箱尤其需要谨慎解释。p 值不表示零假设为真的概率。</p>'
        '<p>C2ST 的 p 值使用 AUC 上尾置换检验：</p>'
        + _equation(r'p=\frac{1+\sum_{b=1}^{B}\mathbf{1}\{\mathrm{AUC}^{(b)}_{\mathrm{perm}}\geq\mathrm{AUC}_{\mathrm{obs}}\}}{B+1}',
                    '经验置换 P 值')
        + '<p>'
        f'本次 B = {b}，未校正 p 的最小可能值为 {1/(b+1):.4g}，最大为 1；两模型使用 Holm 校正。'
        'ATE 的渐近 p 值不是该离散网格上的值。政策表只报告点估计和区间，没有把 bootstrap 区间换算为未经定义的 p 值。</p>'
        '<p>公开匿名样本的离线 ITT 结论不能直接外推到当前线上人群、曝光效果或盈利效果。实际收益还需要成本、转化价值、'
        '可触达条件及线上随机验证。</p>'
        '<ol><li>在资源允许时扩到 500 万样本或全量，保留独立留出集；本报告没有宣称已完成该规模训练。</li>'
        '<li>本次已按既定多随机种子方案评估训练与选择稳定性；不能反复依据当前测试结果调参。</li>'
        '<li>对相同匿名特征向量按组拆分作敏感性分析，并比较结果；这不是用户级去重，结果见本报告 Stability Analysis。</li>'
        '<li>补充成本/价值场景与上线门槛，冻结候选政策后开展新的线上随机实验。</li></ol></section>',
        '<section class="provenance"><h2>复现记录</h2>'
        f'<p>来源 SHA256：<code>{esc(source["sha256"])}</code><br>'
        f'源文件大小：{int(source["size"]):,} bytes · seed：{int(manifest["seed"])} · '
        f'模型阶段累计耗时：{float(manifest["elapsed_seconds"]):.1f} s</p>'
        f'<p>{footer_versions}</p><p>详细参数、数据指纹与不确定性边界见 {esc(manifest_path.name)}；划分行号、CSV 表格、预测和模型保存在同一输出目录。'
        '此 HTML 的图片全部内嵌，无网络依赖；PNG 也单独保存在 figures/。</p>'
        '<p>方法假设：公开样本中近似恒定的处理分配概率，以及行独立、一致性和无干扰。C2ST 不能证明这些假设。</p>'
        '<p>参考：<a href="https://ailab.criteo.com/criteo-uplift-prediction-dataset/">Criteo AI Lab 数据集与版本说明</a>；'
        '<a href="https://arxiv.org/abs/1706.03461">Künzel et al. (2019), Meta-learners</a>。</p></section>',
    ]
    stability = json.loads((out / "stability" / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((out / "stability" / "protocol.json").read_text(encoding="utf-8"))
    aggregate = pd.read_csv(out / "stability/aggregate.csv")
    frequencies = pd.read_csv(out / "stability/selection_frequency.csv")
    stability_columns = ["split_method", "training_fraction", "policy",
                         "gain_ht_per_100k_mean", "gain_ht_per_100k_std", "vs_propensity_per_100k_mean"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for name in aggregate.policy.unique():
        rows = aggregate[(aggregate.policy == name) & (aggregate.split_method == "row_stratified")].sort_values("training_fraction")
        ax.plot(rows.training_fraction, rows.gain_ht_per_100k_mean, "o-", label=name)
    ax.set(xlabel="Fraction of development training partition", ylabel="HT gain per 100k eligible",
           title="Development stability: mean across configured seeds")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    stability_plot = figures / "stability.png"
    fig.savefig(stability_plot, dpi=160); plt.close(fig)
    assumptions = json.loads((out / "business" / "assumptions.json").read_text(encoding="utf-8"))
    plans = pd.read_csv(out / "business/hypothetical_ab_plans.csv")
    scenarios = pd.read_csv(out / "business/hypothetical_cost_scenarios.csv")
    extensions = [
        '<section><h2>Stability Analysis · 开发期敏感性分析</h2>'
        f'<p>{int(stability["experiments"])} 个配置、{int(stability["fitted_models"])} 次模型拟合；排除原 Test {int(stability["old_test_n_excluded"]):,} 行。'
        f'旧 Test 交集为零：{stability["all_old_test_row_overlaps_zero"]}；按特征分组的交集为零：{stability["all_group_split_feature_overlaps_zero"]}。</p>'
        f'<p>Seeds {esc(str(protocol["seeds"]))}；训练比例 {esc(str(protocol["training_fractions"]))}；比较行分层与特征分组划分。'
        f'各配置按 {esc(criterion)} 的 Bootstrap 均值选择。均值和标准差描述重复开发实验的波动，不是独立 Test 置信区间。相同匿名特征不代表同一用户；此前已看过历史 Test，排除其行不能抹去分析者已有知识。</p>'
        + _image(stability_plot, "开发期训练规模敏感性", "所有点来自已保存的 aggregate.csv；没有使用原 Test 重新选型。")
        + _table(aggregate[stability_columns]) + '<h3>Validation 选择频率</h3>' + _table(frequencies) + '</section>',
        '<section><h2>Planning · A/B 规划与假设经济情景</h2>'
        '<p>以下业务参数全部为假设，沿用原项目教学情景，不是 Criteo 已验证的线上转化率、收入或 ROI。'
        + ('价值区间按固定成本、转化价值和效果可迁移的假设换算 HT 区间。</p>'
           if assumptions["conditional_ht_interval"] else '本次业务情景只展示 HT 点估计。</p>')
        + '<pre>' + esc(json.dumps(assumptions["parameters"], ensure_ascii=False, indent=2)) + '</pre>'
        + _table(plans) + _table(scenarios) + '</section>',
    ]
    conclusion_index = next(i for i, section in enumerate(sections) if 'Conclusion ·' in section)
    sections[conclusion_index:conclusion_index] = extensions
    css = """
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#263746;font:15px/1.8 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    main{max-width:1200px;margin:auto;padding:44px 28px 60px}header{margin:0 0 30px;padding:0 8px}.eyebrow{font-size:12px;letter-spacing:.16em;color:#2171ad;font-weight:700}h1{font-size:36px;line-height:1.25;margin:12px 0}.subtitle{color:#5e7180}
    section{background:#fff;border:1px solid #e2e9ef;border-radius:8px;padding:28px;margin:20px 0}h2{font-size:23px;margin:0 0 16px}h3{font-size:18px;margin-top:24px}p{margin:12px 0}.lead{border-top:4px solid #2171ad}.key{font-size:21px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}.cards div{background:#f2f7fb;padding:16px;border-radius:5px}.cards span{display:block;color:#667887;font-size:13px}.cards strong{font-size:25px;font-weight:650}
    figure{margin:24px 0}img{width:100%;height:auto;display:block}figcaption{font-size:13px;color:#647887;margin-top:10px}.table-wrap{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th{background:#eff5f9;text-align:left;padding:11px 12px;border-bottom:2px solid #d5e3ee}td{padding:9px 12px;border-bottom:1px solid #e8eef3}tr:nth-child(even) td{background:#fafcfe}.formula{background:#f3f7fb;border-left:3px solid #2171ad;padding:16px 20px;margin:22px 0}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}.provenance{font-size:13px;color:#667887}.provenance h2{font-size:19px}li{margin:8px 0}
    .formula{border:1px solid #dce7ef;border-left:4px solid #2171ad;border-radius:8px;padding:24px;background:#f6f9fc}.formula h3{margin:0 0 8px}.formula-intro,.formula-note{color:#586e80;font-size:14px}.equation{margin:18px 0;padding:16px 18px;background:#fff;border:1px solid #e2eaf1;border-radius:6px;break-inside:avoid}.equation-label{font-size:12px;font-weight:650;letter-spacing:.04em;color:#587286;margin-bottom:8px}.equation-scroll{max-width:100%;overflow-x:auto;overflow-y:hidden;padding:2px 0}.equation-scroll:focus-visible{outline:2px solid #2171ad;outline-offset:3px}.math-equation{width:auto;max-width:none;height:auto;display:block}var{font-family:Georgia,"Times New Roman",serif}pre{overflow-x:auto}
    details{margin:18px 0;border:1px solid #dce7ef;border-radius:6px;padding:12px 16px}summary{cursor:pointer;color:#2171ad;font-weight:600}summary:focus-visible{outline:2px solid #2171ad;outline-offset:4px}
    @media(max-width:700px){.formula{padding:16px 12px}.equation{padding:12px 10px}}
    @media print{.equation-scroll{overflow:visible}.math-equation{max-width:100%;height:auto}.equation{border-color:#ddd}}
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
    report_path.write_text(document, encoding="utf-8")
    return report_path


def finalize_run(
    *, manifest_path: Path, report_path: Path, args, source, sample_n,
    assignment_probability, fitted, elapsed_seconds,
):
    #-------------
    # 函数名：finalize_run
    # 功能：写入实验复现 manifest，生成最终 HTML 报告，并仅在报告成功后将实验状态标记为 complete。
    # 输入参数：
    #     - manifest_path：本次实验 manifest.json 路径。
    #     - report_path：最终 HTML 报告路径。
    #     - args：本次 Pipeline 的命令行参数对象。
    #     - source：原始数据文件指纹。
    #     - sample_n：实际建模样本行数。
    #     - assignment_probability：用于 HT 评价的训练集处理比例代入值。
    #     - fitted：训练和 Validation 阶段返回的模型、元数据及冻结选择。
    #     - elapsed_seconds：本次 Pipeline 截至报告生成前的累计耗时秒数。
    # 输出参数：
    #     - 无；写入 manifest.json 并生成 report.html。
    #-------------
    manifest = {
        "assignment_probability": assignment_probability,
        "probability_source": "使用训练集的处理组比例作为分配概率的代入估计",
        "ci_method": (
            "在处理组和对照组内分别进行百分位 Bootstrap 重抽样，"
            "各策略共用同一次重抽样；固定已训练模型的评分和处理分配概率，"
            "每次重抽样后重新选择排名前 k 的样本"
        ),
        "uncertainty_excluded": [
            "模型训练与模型选择带来的不确定性",
            "处理分配概率估计带来的不确定性",
            "无法识别的用户依赖关系",
        ],
        "estimand": (
            "公开样本中，按处理分配计算的增量结果"
            "（类似 ITT 口径），以所有人均不接受处理为比较基准"
        ),
        "cost_assumption": (
            "假设每次触达成本相同、每个结果事件的价值相同；"
            "预算表示触达人群比例，而非货币金额"
        ),
        "ranking_metric_definition": RANKING_METRIC_DESCRIPTION,
        "diagnostic_models": ["logistic", "xgboost"],
        "primary_outcome": args.outcome,
        "selection": fitted["selection"],
        "models": fitted["model_metadata"],
        "assumptions": [
            "公开样本中，给定特征 X 的处理分配概率近似恒定",
            "样本行相互独立，满足一致性和个体间无干扰假设",
            "每次转化的价值相同，每次触达的成本相同",
        ],
        "ate_variance_definition": "Both inputs use Bernoulli sample variance with n-1 correction; pooled-null z test unchanged.",
        "status": "metrics_complete",  # 状态码：指标计算完成
        "source": source,
        "features": FEATURES,
        "seed": args.seed,
        "sample_n": sample_n,
        "arguments": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "statsmodels",
                "scikit-learn",
                "xgboost",
                "matplotlib",
                "joblib",
            )
        },
        "elapsed_seconds": elapsed_seconds,
    }
    write_result(manifest, manifest_path)
    build_report(manifest_path, report_path)
    manifest["status"] = "complete"
    write_result(manifest, manifest_path)

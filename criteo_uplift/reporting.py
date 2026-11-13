# 保存实验运行记录，读取已有结果生成图表和 HTML 报告。
# finalize_run 记录复现信息与完成状态；build_report 渲染结果。

from __future__ import annotations

import base64
import html
import importlib.metadata
import io
import json
import platform
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES
from .metrics import RANKING_METRIC_DESCRIPTION
from .pipeline_io import safe_output, write_result

_TITLE = "Incremental Conversion Optimization Under a Fixed Targeting Budget"

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


def _business_section(title, coverage, outcome, criterion):
    example = (
        ("A：高自然转化人群", 80, 85),
        ("B：高广告敏感人群", 10, 35),
    )
    rows = [(name, f"{control}%", f"{treated}%", f"+{treated - control} pp")
            for name, control, treated in example]
    table = _table(pd.DataFrame(
        rows, columns=["人群", "不处理转化概率", "处理后转化概率", "增量"],
    ))
    bars = "".join(
        '<div class="targeting-example-row">'
        f'<strong>{name}</strong>'
        f'<span>不处理 {control}% · 处理后 {treated}% · 增量 +{treated - control} pp</span>'
        '<div class="targeting-example-bar" aria-hidden="true">'
        f'<span class="targeting-baseline" style="width:{control}%"></span>'
        f'<span class="targeting-increase" style="width:{treated - control}%"></span>'
        '</div></div>'
        for name, control, treated in example
    )
    outcome_label = {"conversion": "转化", "visit": "访问"}[outcome]
    return (
        f'<section id="business"><h2>{html.escape(title)}</h2>'
        '<h3>1.1 广告投放的业务问题</h3>'
        '<p>以互联网平台上的电商广告为例，广告主希望带来更多下单等转化。平台先圈出符合投放条件的候选用户，'
        '再决定优先向谁分配广告资格。预算、广告库存和频次限制使可触达的人数有限，'
        '每用掉一个名额，就少一个触达其他候选用户的机会。</p>'
        '<p>传统 Response Propensity 模型预测用户在处理条件下的转化概率，并从高到低排序。'
        '这一分数回答“分配广告资格后，谁更可能转化”。例如，原本就打算下单的用户，即使没有获得广告资格也可能完成购买。'
        '评估广告带来的额外转化，需要比较同类用户在获得和未获得广告资格两种情景下的转化概率。</p>'
        '<p class="key">在触达规模有限的情况下，应该优先选择哪些用户，才能创造更多额外转化？</p>'
        '<h3>1.2 Response Propensity 与 Uplift 的直观比较</h3>'
        '<p><strong>示意数据，非 Criteo 实验结果。</strong> 假设有以下两类候选用户。</p>'
        + table
        + '<figure class="targeting-example" aria-label="两类用户的转化概率与增量">'
        '<div class="targeting-example-legend" aria-hidden="true">'
        '<span><i class="targeting-baseline"></i>不处理转化概率</span>'
        '<span><i class="targeting-increase"></i>分配广告资格带来的增量</span>'
        '</div>'
        + bars
        + '<div class="targeting-example-axis" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div>'
        '<figcaption>示意数据，非 Criteo 实验结果。两段条形相加得到处理后转化概率，pp 表示百分点。</figcaption>'
        '</figure>'
        '<p>Response Propensity 优先选择 A，因为 A 的处理后转化概率更高（85% 对 35%）。'
        'Uplift 优先选择 B，因为 B 在获得广告资格后增加的转化概率更大（25 个百分点对 5 个百分点）。'
        '如果单位触达成本与每次转化的价值相同，且一位用户的处理分配不会影响其他用户的结果，'
        '每分配 100 个名额给 B，预期比把同样的名额给 A 多带来 20 次增量转化。'
        '这就是按预期增量排序的业务动机。</p>'
        '<h3>1.3 因果增量与 CATE</h3>'
        '<p>用潜在结果框架描述这两个情景。<var>X</var> 表示处理前的用户特征。'
        '<var>Y(1)</var> 和 <var>Y(0)</var> 分别表示同一用户在获得和未获得广告处理资格时的潜在转化结果。'
        '在 Criteo 中，<code>treatment</code> 记录资格分配，<code>exposure</code> 记录实际曝光。'
        '本项目估计资格分配对结果的意向处理效应（ITT），处理组中未实际曝光的用户也计入该组。</p>'
        + _equation(r'\mathrm{ATE}=\mathbb{E}[Y(1)-Y(0)]', '平均处理效应（ATE）')
        + '<p>ATE 衡量对候选总体分配广告资格所带来的平均增量，可用来判断总体效果的大小。'
        '人群定向还需要了解这一效果如何随处理前特征变化。</p>'
        + _equation(r'\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x]', '条件平均处理效应（CATE）')
        + '<p>真实 CATE 是给定特征 <var>X</var> = <var>x</var> 时的平均增量。'
        'Uplift Modeling 利用处理前特征估计这一量，实际排序使用的是模型预测的 CATE。'
        '同一个体只经历一种分配，因此只能观察对应的结果。'
        '从观测数据估计 ATE 和 CATE，需要随机分配等实验设计支持，并依赖一致性、无干扰和两种分配都有可能等识别假设。</p>'
        '<h3>1.4 本项目的研究问题与定向策略</h3>'
        '<p>本项目使用 Criteo Uplift v2.1 公开随机实验数据。'
        f'将广告资格分配规模固定为候选总体的 <strong>{html.escape(coverage)}</strong>，'
        '这是项目预设的业务开发假设。两种策略在相同候选总体中各选出同样比例的用户。</p>'
        f'<ol><li><strong>Response Propensity Targeting</strong> 按处理条件下的预测{outcome_label}概率 '
        f'<code>P(Y=1 | X,T=1)</code> 排序。本项目复用 T-XGBoost 处理组模型的概率预测。'
        '处理分配倾向分数 <code>P(T=1 | X)</code> 描述的是获得处理资格的概率。</li>'
        '<li><strong>Uplift Targeting</strong> 按预测 CATE 排序。候选包括 S-Learner + Logistic、'
        'T-Learner + Logistic、T-Learner + XGBoost 和 X-Learner + XGBoost。</li></ol>'
        f'<p class="flow">在固定 {html.escape(coverage)} 广告资格分配规模下，Uplift Targeting 能否比传统 '
        f'Response Propensity Targeting 获得更多预期增量{outcome_label}？</p>'
        f'<p>两种策略使用同一无处理比较基准及同一<a href="#evaluation">评价口径</a>。'
        f'验证集以预设的{html.escape(criterion)}冻结策略，再在<a href="#test">独立测试集</a>比较固定覆盖率下的'
        f'增量{outcome_label}点估计及配对区间。结果用于判断是否值得开展线上验证。</p></section>'
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


def _overlap_section(out):
    #-------------
    # 功能：展示 Train 的经验覆盖结果，并与设计正值性及结果标签支持区别说明。
    # 输入参数：
    #     - out：当前报告的结果目录。
    # 输出参数：
    #     - 可嵌入诊断章节的 HTML。
    #-------------
    path = out / 'diagnostics/overlap_positivity.json'
    heading = '<h3>训练集覆盖与正值性</h3>'
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
            ("models", ("validation_metrics", "validation_bootstrap", "validation_pairwise")),
            ("evaluation", ("ate_summary", "test_metrics", "test_bootstrap", "test_pairwise", "curves")),
            ("data_preparation", ("split_summary",)),
        )
        for name in names
    }
    frames["ate_full"] = frames["ate_summary"].query("scope == 'full_dataset'")
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
    figures = safe_output(out / "figures")
    figures.mkdir(exist_ok=True)
    save_c2st_plot(out)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    paths = {
        name: safe_output(figures / f"{name}.png")
        for name in ("ate_full", "balance", "ranking", "policy")
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

        with plt.rc_context(
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
        balance_scope = "Full public data | raw float64"
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
    primary_pairs = test_pairs if area_primary else test_pairs[test_pairs.budget.eq(budget)]
    selected_pairs = primary_pairs[primary_pairs.policy.eq(selected)].set_index("reference")
    vs_propensity = selected_pairs.loc["propensity"] if selected != "propensity" else None
    vs_random = selected_pairs.loc["random"]
    uplift_vs_propensity = primary_pairs.loc[
        primary_pairs.policy.eq(selected_uplift) & primary_pairs.reference.eq("propensity")
    ].iloc[0]

    def paired_finding(row, reference, subject="冻结策略"):
        if row is None:
            return "Validation 冻结了 Response Propensity。同策略比较的点差与区间恒为 0。"
        low, high = row["ci_low"], row["ci_high"]
        interpretation = (
            "有效重复不足，无法判断差值方向" if pd.isna(low) else
            "区间完全高于零，支持增量优势" if low > 0 else
            "区间完全低于零，支持参考策略的增量优势" if high < 0 else
            "区间包含零，现有证据不足以区分策略效果"
        )
        return (
            f"{subject}相对 {reference} 的 {criterion} 点估计差值为 {_num(row['difference'] * 100_000)}，"
            f"配对 95% CI {_interval(row, scale=100_000)}。{interpretation}。"
        )

    propensity_finding = paired_finding(vs_propensity, "Response Propensity")
    random_finding = paired_finding(vs_random, "Random")
    uplift_finding = paired_finding(uplift_vs_propensity, "Response Propensity", esc(selected_uplift))
    test_comparisons = primary_pairs.loc[
        (primary_pairs.policy.eq(selected) & primary_pairs.reference.isin({"propensity", "random"}))
        | (primary_pairs.policy.eq(selected_uplift) & primary_pairs.reference.eq("propensity"))
    ]
    if vs_propensity is not None and vs_propensity.ci_low > 0:
        decision = "独立 Test 支持把获选策略带入与传统定向同覆盖率的线上随机验证。"
    elif vs_propensity is not None and vs_propensity.ci_high < 0:
        decision = "独立 Test 支持保留传统定向作为当前基线，并在线上复核策略差异。"
    elif selected == "propensity":
        decision = "本次证据支持保留传统响应倾向排序作为固定预算的候选投放方案，并以同覆盖率的线上随机实验验证 Uplift 挑战策略。"
    else:
        decision = "保留传统定向作为线上对照，使用同覆盖率的随机实验判断 Uplift 能否带来业务增量。"

    def point_estimates(frame):
        return pd.DataFrame({
            "策略": frame.policy,
            **{f"{metric_labels[key]} × 10⁵": frame[key].map(lambda v: _num(v * 100_000))
               for key in metric_labels},
        })

    ate_all = frames["ate_summary"]
    ate_display = pd.DataFrame(
        {
            "样本": ate_all["scope"].map({"full_dataset": "全量源数据", "test": "独立 Test"}),
            "指标": ate_all["outcome"],
            "对照率": ate_all["response_control"].map(lambda x: f"{100*x:.4f}%"),
            "处理率": ate_all["response_treatment"].map(lambda x: f"{100*x:.4f}%"),
            "ATE（百分点）": ate_all["ate"].map(lambda x: _num(100 * x, 4)),
            "95% CI（百分点）": ate_all.apply(
                lambda row: _interval(row, scale=100, digits=4), axis=1
            ),
            "相对增幅": ate_all["relative_uplift"].map(
                lambda x: "—" if pd.isna(x) else f"{100*x:.2f}%"
            ),
            "双侧 p": ate_all["p_value_text"] if "p_value_text" in ate_all else ate_all["p_value"].map(_p),
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
    def paired_table(frame):
        return _table(pd.DataFrame({
            "策略": frame.policy, "参考策略": frame.reference,
            "点估计差值": frame.difference.map(lambda x: _num(x * 100_000)),
            "配对 95% CI": frame.apply(lambda r: _interval(r, scale=100_000), axis=1),
            "有效配对": frame.valid_replicates,
        }))

    val_pairs = frames["validation_pairwise"]
    val_pairs = val_pairs.loc[val_pairs.policy.eq(selected)]
    test_display = test_display.loc[test_bootstrap.index if area_primary else test_bootstrap.index[test_bootstrap.budget.eq(budget)]]
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
    comparison_budgets = budget_display.loc[
        policies.policy.isin({selected, "random", "propensity"}) & ~policies.budget.isin(key_budgets)
    ]
    frozen = policies.loc[policies["policy"] == selected].set_index("budget").loc[key_budgets]

    # 固定排序的新增区间贡献等于两个累计值之差；不相减置信区间端点。
    values = np.r_[0., frozen["incremental_per_100k"].to_numpy()]
    marginal = np.diff(values)
    all_frozen = policies.loc[policies.policy.eq(selected)].sort_values("budget")
    all_values = all_frozen.incremental_per_100k.to_numpy()
    all_counts = all_frozen.n_selected.to_numpy()
    boundaries = [0., *all_frozen.budget]
    values_by_step = np.diff(np.r_[0., all_values])
    counts_by_step = np.diff(np.r_[0, all_counts])
    marginal_display = pd.DataFrame({
        "新增覆盖区间": [f"{a:.0%}–{b:.0%}" for a, b in zip(boundaries[:-1], boundaries[1:])],
        "新增覆盖比例（实际）": [f"{n / n_test:.2%}" for n in counts_by_step],
        "新增人数": counts_by_step,
        "新增估计转化 / 10 万总体用户（点估计）": [_num(v) for v in values_by_step],
    })
    marginal_findings = "；".join(
        f"{label} 为{'正' if value > 0 else '负' if value < 0 else '零'}增量 {_num(value)}"
        for label, value in zip(("0%–10%", "10%–20%", "20%–30%", "30%–100%"), marginal)
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
    outcome_learning = learning.loc[learning.outcome.eq(outcome)].set_index("model")
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
    overlap_data = overlap.get("fraction_matching_training_features", {})
    overlap_text = "、".join(f"{esc(str(k))} {v:.2%}" for k, v in overlap_data.items())
    srm_text = (
        f"处理组占 {srm['observed_ratio']:.4%}，外部参考为 {srm['expected_ratio']:.2%}，"
        f"二项检验 p = {_p(srm['pvalue'])}。该比较衡量公开样本对外部比例的偏离，"
        "原始实验 SRM 仍需依据各实验的设计分流比例与计数。"
    )
    balance_caption = "原始全量 float64 特征的 SMD 与 KS。0.1 是 SMD 的经验参考，KS p 值为探索性结果。"
    c2st_sampling_text = f"C2ST 从全量源数据的两臂分别等概率无放回抽取 {int(c2st_details['n_control']):,} 行"
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
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    stability_plot = figures / "stability.png"
    fig.savefig(stability_plot, dpi=160)
    plt.close(fig)
    assumptions = json.loads((out / "business" / "assumptions.json").read_text(encoding="utf-8"))
    plans = pd.read_csv(out / "business/hypothetical_ab_plans.csv")
    scenarios = pd.read_csv(out / "business/hypothetical_cost_scenarios.csv")
    stability_detail = pd.read_csv(out / "stability/longmetrics.csv")
    stability_detail = stability_detail.loc[
        stability_detail.policy.isin({selected, "propensity"}) | stability_detail.selected_on_validation,
        ["seed", "split_method", "training_fraction", "policy", "train_n",
         "validation_n", "gain_ht_per_100k", "vs_propensity_per_100k", "selected_on_validation"],
    ]
    model_settings = {
        name: details["parameters"] for name, details in manifest["models"].items()
    }
    # Publish a portable command, without machine-specific source or output paths.
    arguments = manifest["arguments"]
    reproducibility = (
        f"python run_pipeline.py --data data/raw/{shlex.quote(Path(source['path']).name)}"
        f" --sample-size {arguments['sample_size']} --outcome {outcome} --seed {manifest['seed']}"
        f" --primary-metric {primary_metric} --primary-budget {budget:.2f}"
        f" --bootstrap {boot} --permutations {b} --c2st-sample {arguments['c2st_sample']}"
        f" --trees {arguments['trees']}"
        f" --stability-seeds {' '.join(map(str, protocol['seeds']))}"
        f" --stability-fractions {' '.join(map(str, protocol['training_fractions']))}"
        f" --output artifacts/{shlex.quote(out.name)}"
    )
    ab_display = plans[["absolute_mde", "relative_mde", "n_policy_A", "n_policy_B", "n_total"]].copy()
    ab_display["absolute_mde"] = ab_display.absolute_mde.map(lambda x: f"{x * 100:.3f} pp")
    ab_display["relative_mde"] = ab_display.relative_mde.map(lambda x: f"{x:.0%}")
    economic_columns = ["value_per_conversion_assumed", "contact_cost_assumed",
                        "gross_incremental_value", "contact_cost_total", "net_incremental_value",
                        "net_ci_low_conditional", "net_ci_high_conditional", "break_even_contact_cost"]
    stability_stratified = aggregate.loc[aggregate.split_method.eq("row_stratified") &
                                         aggregate.policy.eq(selected)].sort_values("training_fraction")
    sensitivity = "、".join(f"{row.training_fraction:.0%} 训练量时为 {row.gain_ht_per_100k_mean:,.2f}"
                            for row in stability_stratified.itertuples())
    group_wins = frequencies.query("split_method == 'feature_group' and selected_count > 0")
    group_winners = "、".join(f"{esc(row.policy)}（{row.selected_count}/{row.runs}）" for row in group_wins.itertuples())
    scenario_example = scenarios.iloc[len(scenarios) // 2]
    best_areas = (
        f"Test 中 AUUC 最高的是 {esc(ranking.loc[ranking.auuc.idxmax(), 'policy'])}，"
        f"Traditional Qini Score 最高的是 {esc(ranking.loc[ranking.qini_score.idxmax(), 'policy'])}。"
        "这两项跨覆盖率排序统计承担辅助诊断，正式决策沿用预设的 Validation 标准。"
    )
    if selected == "propensity":
        test_relationship = "Validation 冻结了传统响应倾向排序。Test 中的 Uplift 候选结果仅用于说明模型表现。"
    else:
        test_relationship = "Validation 选出的 Uplift 策略在 Test 中保持冻结，配对差值衡量其相对传统排序的独立证据。"

    section_titles = {
        "summary": "00. 执行摘要",
        "business": "01. 业务问题与定向策略",
        "data": "02. 数据来源与准备",
        "diagnostics": "03. 实验诊断与因果识别",
        "ate": "04. 平均处理效应（ATE）",
        "modeling": "05. Uplift 建模",
        "evaluation": "06. 评估框架",
        "validation": "07. 验证与模型选择",
        "test": "08. 独立测试评估",
        "robustness": "09. 预算敏感性与稳健性",
        "conclusion": "10. 业务启示与结论",
        "appendix": "附录：技术细节与复现",
    }
    sections = [
        f'<header><p class="eyebrow">End-to-End Applied Data Science Case Study</p>'
        f'<h1>{_TITLE}</h1>'
        '<p class="subtitle">Causal Uplift Modeling, Experimental Diagnostics and Policy Evaluation on the Criteo Dataset</p></header>',
        f'<section class="lead" id="summary"><h2>{section_titles["summary"]}</h2>'
        f'<p>在固定覆盖预算内，选择能带来最多增量{"转化" if outcome == "conversion" else "访问"}的广告定向策略。'
        f'使用 Criteo 全量 {int(audit["n_rows"]):,} 行做审计与诊断，抽取 {int(manifest["sample_n"]):,} 行建模，'
        f'以独立 Test {n_test:,} 行评估。预设主指标为 <strong>{esc(criterion)}</strong>。</p>'
        f'<div class="cards"><div><span>Validation 获选策略</span><strong>{esc(selected)}</strong></div>'
        f'<div><span>Validation 最佳 Uplift 模型</span><strong>{esc(selected_uplift)}</strong></div>'
        f'<div><span>Test {primary_pct} HT 增量 / 10 万人</span>'
        f'<strong>{_num(selected_row["incremental_per_100k"])}</strong>'
        + (f'<span>95% CI {_interval(selected_row)}</span>' if primary_metric == "gain_ht" else "")
        + f'</div></div><p>{propensity_finding} {uplift_finding if selected == "propensity" else ""}</p><p>{random_finding}</p>'
        f'<p class="key">{decision}</p></section>',
        _business_section(section_titles["business"], primary_pct, outcome, criterion),
        f'<section id="data"><h2>{section_titles["data"]}</h2>'
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
        + _table(pd.DataFrame(audit["group_summary"]).rename(columns={
            "group": "分组", "n": "行数", "fraction": "占比", "conversion_positive": "转化数",
            "conversion_rate": "转化率",
        }).assign(占比=lambda t: t.占比.map(lambda x: f"{x:.2%}"),
                  转化率=lambda t: t.转化率.map(lambda x: f"{x:.4%}")))
        + '<h3>数据字段与建模划分</h3>'
        '<p>Criteo v2.1 公开数据来自广告增量实验。treatment 表示实验分配，conversion 和 visit 是二元结果，'
        'exposure 表示实际曝光，f0–f11 是匿名预处理协变量。'
        '划分已核验来源行号非空且唯一，三个集合两两互斥，并完整覆盖抽样行。</p>'
        + _table(split_display)
        + _image(figures / "sample_features.png", "十二个建模特征的分布",
                 "横轴为对应匿名特征值，纵轴为对数刻度的建模样本频数。尖峰和离散值有助于解释 KS 的并列值及特征向量重合。")
        + '</section>',
        f'<section id="diagnostics"><h2>{section_titles["diagnostics"]}</h2>'
        '<p>识别目标是公开样本中分配处理对结果的平均效应与排序政策价值。'
        '解释依赖一致性、无干扰、分配与潜在结果近似独立及各特征区域的处理概率远离 0 和 1。'
        'Criteo 的实验描述提供设计依据，以下检查衡量公开数据中的可观测差异。'
        '公开版本的抽样与隐私处理、未知实验分层和用户依赖限制了对原始随机机制的核验。</p>'
        '<h3>公开分流比例检查（SRM）</h3>'
        f'<p>{srm_text} 当前训练集处理比例 e = {e:.6f} 只作为 HT 评估的 plug-in 分配概率，并非已知实验设计参数。</p>'
        + '<h3>单特征分布平衡（SMD / KS）</h3><p>SMD 检查标准化均值差，KS 检查两组经验分布的最大距离。'
        '全量数据下应结合差异幅度与特征分布判断实际意义。</p>'
        + _image(
            paths["balance"],
            "处理前协变量平衡",
            balance_caption,
        )
        + f'<h3>联合分布可区分性（C2ST）</h3><p>{c2st_sampling_text}，固定70/30内部训练/留出划分；'
        f'在两部分内分别打乱 treatment，完整重拟合两类固定模型，共 {b} 次。诊断结果如下：</p>'
        + _table(c2_display)
        + _image(figures / "c2st_null.png", "C2ST 置换 AUC 分布",
                 f"柱高表示 {b} 次完整重拟合置换中落入各 AUC 区间的次数；橙线为真实标签的观测 AUC，灰色虚线为 0.5。"
                 "原始 p 按逐次 AUC 计算，未用直方图分箱近似；Holm 校正仅包含原定的两个分类器。")
        + f'<p>{c2st_conclusion}全量公开数据最大 |SMD| 为 {frames["balance"]["abs_smd"].max():.6f}；最大KS为 {max_ks_row["ks_statistic"]:.6f}（{esc(max_ks_row["feature"])}）。置换p值分辨率为 {1 / (b + 1):.3f}。</p>'
        + '<p>高于置换基线的 AUC 提示可观测分配差异，需要检查实验设计、抽样和特征时序；不显著不能证明随机化。'
        '该置换方案假设可交换分配；如存在未知实验分层或多个广告实验混合，需使用匹配实际设计的置换方案。</p>'
        + _overlap_section(out)
        + '<h3>结果可预测性与基线对照</h3>'
        + _table(learning_display)
        + f'<p>{esc(outcome)} 的 XGBoost AP 为 {_num(outcome_learning.loc["xgboost", "average_precision"], 4)}，'
        f'Dummy AP 为 {_num(outcome_learning.loc["dummy", "average_precision"], 4)}。'
        '这一差异衡量响应排序中的可学习信号。</p>'
        + '<p>上表使用 train 拟合、validation 评估，仅显示 whole 范围；按处理组分别训练的结果保存在 learnability.csv。'
        'Dummy 预测训练集正例率。LogLoss/Brier 越低越好，AP/ROC AUC 越高越好；AP 的基线取决于正例率。'
        '这些指标诊断结果模型，不评估个体因果效应；普通结果预测较弱不能推出不存在效应异质性。</p></section>',
        f'<section id="ate"><h2>{section_titles["ate"]}</h2><p>分别以全量源数据和独立 Test 估计分配处理的 ITT 效应。'
        'ATE 是处理组结果率减去对照组结果率；绝对效应用百分点表示，相对增幅以对照率为分母。</p>'
        + _table(ate_display)
        + _image(
            paths["ate_full"],
            "全量平均处理效应",
            "空心点为差均值，横线为 95% Wald 区间；两个指标采用各自横轴尺度，虚线表示零效应。",
        )
        + '<p>Conversion 与 Visit 对应不同漏斗阶段。全量估计给出实验的平均背景，Test 估计描述最终留出人群。'
        '总体平均值无法指出应优先触达哪些人，固定预算决策需要评价排序前缀中的增量。</p></section>',
        f'<section id="modeling"><h2>{section_titles["modeling"]}</h2>'
        '<p class="flow">Train 60% → 拟合结果与效应模型 → Validation 评分与冻结 → Test 加载模型并评分</p>'
        + _table(pd.DataFrame([
            ("S-Logistic", "单个 Logistic，输入 X 和 T", "分别代入 T=1 与 T=0，预测概率相减"),
            ("T-Logistic", "处理组与对照组各拟合一个 Logistic", "两个模型预测概率相减"),
            ("T-XGBoost", "处理组与对照组各拟合一个 XGBoost 分类器", "两个模型预测概率相减"),
            ("X-XGBoost", "两臂 XGBoost 分类器及两个伪效应回归器", "按训练处理比例加权两臂效应预测"),
            ("Response Propensity", "复用 T-XGBoost 处理组分类器", "按处理下结果概率排序"),
        ], columns=["策略", "训练结构", "评分"]))
        + '<p>Logistic 在 Train 内拟合中位数填补和标准化，采用 L2、C=1、LBFGS、最多 500 次迭代。'
        f'XGBoost 用 histogram、{arguments["trees"]} 棵树、深度 4、学习率 0.05、最小子节点权重 20、行采样 0.8、列采样 0.9。'
        '所有候选的正式训练使用同一随机种子。</p>'
        + _equation(r'D_1=Y-\hat{\mu}_0(X),\qquad D_0=\hat{\mu}_1(X)-Y,\qquad \hat{\tau}(X)=e\hat{\tau}_0(X)+(1-e)\hat{\tau}_1(X)',
                    'X-Learner 的伪效应与加权')
        + '<p>两个伪效应分别在对应处理臂回归，e 取 Train 处理比例。Validation 只负责评分与选型，四个模型随后直接交给独立 Test。</p></section>',
        f'<section id="evaluation"><h2>{section_titles["evaluation"]}</h2>'
        '<p>全曲线指标衡量排序在覆盖率变化时的整体表现。预算指标衡量 Top-k 的累计结果。'
        'Uplift Gain 用前缀内实际处理与对照响应率之差乘以覆盖率。Traditional Qini Gain 估计前缀内实际处理人数对应的增量。'
        'HT Gain 用固定分配概率逆概率加权，以全部 N 名可触达对象为分母，估计对整个前缀分配处理的增量。</p>'
        f'<p>预设主指标为 <strong>{esc(criterion)}</strong>。20% 是正式转化实验预先指定的业务覆盖率。'
        '每十万人是 HT 增量率的展示标度。Random 用同样的 k，取固定分配下的随机触达解析期望。'
        '完整曲线面积用所有 Top-k 前缀做梯形积分。绘图最多保留 501 个节点。</p>'
        f'<p>对主指标进行 {boot} 轮 Bootstrap。每轮在 Treatment 和 Control 内各自有放回抽样，五种策略共用同一批行。'
        '重抽样后沿固定评分顺序取 Top-k，逐轮求差，再取 2.5% 和 97.5% 分位数。'
        '区间以已训练评分和 e 为条件。各预算区间均为点态区间，模型训练及选型的不确定性未纳入。</p>'
        + '<div class="formula" id="calculation-definitions"><h3>本报告的计算定义</h3>'
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
        f'<section id="validation"><h2>{section_titles["validation"]}</h2>'
        f'<p>五种策略在 Validation 上统一评价。以下为 {primary_pct} 覆盖率的预算指标及两个完整曲线面积。'
        '主指标以 Bootstrap 均值排序，Random 仅作为参考。</p>'
        + _table(point_estimates(frames["validation_metrics"]))
        + _equation(r'\pi^*=\mathrm{argmax}_{\pi}\;\frac{1}{B}\sum_{b=1}^{B}\widehat{M}_{\pi}^{(b)}',
                    'Validation 选型规则')
        + _table(val_display)
        + f'<p>获选策略 <strong>{esc(selected)}</strong>，Uplift 候选中排名第一的是 <strong>{esc(selected_uplift)}</strong>。'
        f'下表以{scale_label}报告获选策略的配对差值。获胜频率来自所有候选均有定义的重复，表示固定评分下的波动。</p>'
        + paired_table(val_pairs)
        + '</section>',
        f'<section id="test"><h2>{section_titles["test"]}</h2>'
        f'<p>Test 共 {n_test:,} 行，其中 {esc(outcome)} 正例 {n_events:,} 个，对照组正例 {n_control_events:,} 个。'
        f'{test_relationship}</p>'
        + _table(point_estimates(policies.loc[policies.budget.eq(budget) & policies.policy.ne("random")]))
        + '<h3>主指标与配对推断</h3>'
        + _table(test_display)
        + f'<p>表内单位为{scale_label}。{propensity_finding} {random_finding}</p>'
        + (f'<p>{uplift_finding}</p>' if selected == "propensity" else "")
        + paired_table(test_comparisons)
        + '<h3>完整排序曲线</h3>'
        + _image(paths["ranking"], "HT Gain、Uplift Gain 与 Traditional Qini 三类曲线",
                 "横轴为触达比例。虚线连接该指标的原点与全量终点。面积计算使用完整前缀。")
        + '<div id="ranking-metrics">' + _table(ranking_display) + '</div>'
        + f'<p>{best_areas}</p></section>',
        f'<section id="robustness"><h2>{section_titles["robustness"]}</h2>'
        '<p>Budget 表示投放覆盖率（Contact Fraction），不是货币金额；所有增量均以 10 万名总体可触达用户为分母。'
        '下表展示冻结策略、Random 和 Propensity；若冻结策略就是其中一个基线，仅展示一次。</p>'
        + '<div id="budget-sensitivity">' + _table(budget_key_display) + '</div>'
        + _image(paths["policy"], "HT 增量的覆盖率敏感性",
                 f"星号标识冻结策略；10%、20%、30%、100% 为关键情景，{primary_pct} 为预设的投放情景覆盖率，100% 为全量终点。连线仅辅助读图，未在评价点之间插值计算。")
        + '<details><summary>辅助覆盖率明细（含 2%、5%、50%）</summary>'
        + _table(comparison_budgets) + '</details>'
        + '<p>同一固定评分及同分顺序下，Top 10% ⊂ Top 20% ⊂ Top 30% ⊂ Top 100%。'
        '从 10% 扩到 20% 加入排名 10%–20% 的人，从 20% 扩到 30% 加入排名 20%–30% 的人；'
        '选中人数按覆盖率乘以评价总人数向下取整。匿名特征不能用于编造这些人群的业务画像。'
        'Random 是相同人数的随机触达解析期望，不对应模型排序。辅助预算不用于 Test 后挑选最优商业预算。</p>'
        '<h3>扩大覆盖的边际增量</h3>'
        f'<p>保持冻结策略 <strong>{esc(selected)}</strong> 不变，以 ΔV = V(b) − V(a)、V(0) = 0 计算新增区间贡献。'
        '</p>'
        + '<div id="marginal-gain">' + _table(marginal_display) + '</div>'
        + f'<p>{marginal_findings}。{marginal_trend}</p>'
        '<p>这里仅报告点估计，未计算边际增量 CI，不能用两个累计 CI 的端点相减来构造区间。'
        '因此正、负和递减均为描述，不是显著性结论；30%–100% 宽达 70%，其总贡献不能直接与 10% 宽区间比较单位触达效率。'
        '没有实际成本和收入信息，本节只讨论估计增量转化，不能据此判断利润或 ROI。</p>'
        '<h3>全量投放情景</h3>'
        + _equation(r'G_{\mathrm{HT}}(1)=\widehat{\mathrm{ATE}}_{\mathrm{HT}},\qquad V(1)=100{,}000\,\widehat{\mathrm{ATE}}_{\mathrm{HT}}',
                    '全量覆盖与 HT-ATE')
        + f'<p>100% 覆盖表示对当前 Test 的全部 {n_test:,} 名可触达用户分配处理，估计增量为 '
        f'<strong>{_num(full_value)}</strong> / 10 万总体用户{full_interval}。'
        '所有策略选择相同的全体用户，因此具有同一 HT 终点；排序不会改变全量的价值。</p>'
        + f'<p>冻结策略与全量相比（同为每 10 万总体用户）：{full_comparisons}。'
        '扩大覆盖是否继续增加估计转化，需结合上一节各新增区间的实际符号判断，不预设越多越好。</p>'
        '<p>这是以当前测试人群和固定处理分配概率为前提的离线情景，HT-ATE 与差均值 ATE 在有限样本中可能不同。'
        '这些数字不代表已经实现的广告曝光、收入或利润，不能据此宣称找到了最优商业投放覆盖率。</p>'
        '<h3>开发集稳定性分析</h3>'
        f'<p>{int(stability["experiments"])} 个配置、{int(stability["fitted_models"])} 次模型拟合；排除原 Test {int(stability["test_n_excluded"]):,} 行。'
        f'正式 Test 交集为零：{stability["all_test_row_overlaps_zero"]}；按特征分组的交集为零：{stability["all_group_split_feature_overlaps_zero"]}。</p>'
        f'<p>Seeds {esc(str(protocol["seeds"]))}；训练比例 {esc(str(protocol["training_fractions"]))}；比较行分层与特征分组划分。'
        f'各配置按 {esc(criterion)} 的 Bootstrap 均值选择。均值和标准差描述重复开发实验的波动，不是独立 Test 置信区间。相同匿名特征不代表同一用户。</p>'
        + _image(stability_plot, "开发期训练规模敏感性", "所有点来自已保存的 aggregate.csv；没有使用原 Test 重新选型。")
        + _table(aggregate[stability_columns].round(2))
        + '<details><summary>各随机种子、训练规模与划分方式</summary>'
        + _table(stability_detail.round(2)) + '</details>'
        + '<h3>验证集策略获选频率</h3>' + _table(frequencies.round(3))
        + f'<p>特征分组划分下的获选策略与次数为 {group_winners}。'
        '不同划分改变开发评估的特征支持与正例构成，频率用于描述这种敏感性。</p></section>',
        f'<section id="conclusion"><h2>{section_titles["conclusion"]}</h2>'
        f'<p><strong>{decision}</strong> {uplift_finding}</p>'
        f'<p>{marginal_findings}。{marginal_trend} 行分层 Stability 中，冻结策略在不同训练规模下的平均 HT 增量 / 10 万人为 {sensitivity}。</p>'
        '<h3>线上策略 A/B</h3>'
        f'<p>将符合条件的用户随机分配至 {esc(selected_uplift)} 与 Response Propensity 两个策略臂，两组均按 {primary_pct} 覆盖。'
        '预设转化窗口与主指标，以全体分配用户为分母，记录触达率、成本、分流比例与污染。'
        f'下面的样本量以基线转化率 {assumptions["parameters"]["p0"]:.2%}、'
        f'双侧 α={assumptions["parameters"]["alpha"]:.2f}、功效 {assumptions["parameters"]["power"]:.0%} 为假设。</p>'
        + _table(ab_display)
        + '<h3>条件性经济价值</h3>'
        + _equation(r'\mathrm{Net\ Value}=V_{\mathrm{conversion}}\times\mathrm{Incremental\ Conversions}-c_{\mathrm{contact}}\times k',
                    '增量贡献价值扣除触达成本')
        + f'<p>以下情景使用 {assumptions["parameters"]["eligible_n"]:,} 名可触达对象和 {primary_pct} 覆盖，'
        '比较相同货币单位下的每次增量结果价值与每次触达成本。条件性区间仅传递固定评分 HT 区间，依赖效果可迁移的假设。</p>'
        + _table(scenarios[economic_columns].round(4))
        + f'<p>例如，每次增量结果价值 {_num(scenario_example.value_per_conversion_assumed)}、'
        f'每次触达成本 {_num(scenario_example.contact_cost_assumed, 3)} 时，净增量价值为 '
        f'{_num(scenario_example.net_incremental_value)}，盈亏平衡触达成本为 '
        f'{_num(scenario_example.break_even_contact_cost, 4)}。</p>'
        + '<p>推广边界包括广告活动与人群漂移、未知用户层依赖、稀少的对照转化、不可见的实验分层和真实成本差异。'
        '稳定性分析中重叠的开发样本用于刻画敏感性，独立 Test 提供一次留出验证。</p></section>',
        f'<section class="provenance" id="appendix"><h2>{section_titles["appendix"]}</h2>'
        + _equation(r'\widehat{\mathrm{ATE}}=\bar{Y}_1-\bar{Y}_0,\qquad '
                    r'\mathrm{SE}=\sqrt{s_1^2/N_1+s_0^2/N_0}', '差均值与未合并标准误')
        + _equation(r'p=\frac{1+\sum_{b=1}^{B}\mathbf{1}\{\mathrm{AUC}^{(b)}_{\mathrm{perm}}\geq\mathrm{AUC}_{\mathrm{obs}}\}}{B+1}',
                    'C2ST 经验置换 p 值')
        + f'<p>ATE 区间使用未合并方差的正态近似，双侧 p 值使用零假设下的合并比例 z 检验。'
        f'C2ST 的 {b} 轮置换为两类模型分别完整重拟合，未校正 p 的最小可能值为 {1/(b+1):.4g}，再做两模型 Holm 校正。</p>'
        f'<p>来源 SHA-256 <code>{esc(source["sha256"])}</code><br>'
        f'文件大小 {int(source["size"]):,} bytes · 随机种子 {int(manifest["seed"])} · '
        f'结果计算耗时 {float(manifest["elapsed_seconds"]):.1f} s</p>'
        f'<p>Python {esc(manifest["python"].split()[0])} · {footer_versions}</p>'
        + '<pre>' + esc(reproducibility) + '</pre>'
        + '<details><summary>模型配置与训练样本支持</summary>'
        + '<pre>' + esc(json.dumps(model_settings, indent=2)) + '</pre>'
        + _table(pd.DataFrame(next(iter(manifest["models"].values()))["outcome_support"]))
        + '</details>'
        + _table(pd.DataFrame([
            ("01", "input_checks.py / raw_scan.py / preparation.py", "来源 SHA-256、全量审计、抽样与划分"),
            ("02", "diagnostics.py", "SRM、平衡、C2ST、Overlap、可学习性"),
            ("03 / 05", "evaluation.py / metrics.py", "ATE、冻结评分、Test 指标与配对区间"),
            ("04", "models.py / training_validation.py", "训练、Validation 表与 selection.json"),
            ("06", "stability.py", "开发期协议、逐次指标与划分审计"),
            ("07 / 08", "planning.py / reporting.py", "业务情景、manifest.json 与独立 HTML"),
        ], columns=["阶段", "实现", "核验产物"]))
        + '<p>推断以固定模型评分、训练集处理比例及公开样本的近似独立性为条件。'
        '样本之间可能存在匿名实体依赖，且公开设计信息不足以核验全部随机分配机制。'
        '原始数据归 Criteo，数据派生成果适用 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>。</p>'
        '<p><a href="https://ailab.criteo.com/criteo-uplift-prediction-dataset/">Criteo AI Lab 数据集与基准论文</a> · '
        '<a href="https://arxiv.org/abs/1706.03461">Künzel et al. (2019), Meta-learners</a></p></section>',
    ]
    sections.insert(2, '<nav aria-label="报告目录"><ul>' + "".join(
        f'<li><a href="#{anchor}">{esc(label)}</a></li>' for anchor, label in section_titles.items()
    ) + '</ul></nav>')
    css = """
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#263746;font:15px/1.8 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    main{max-width:1200px;margin:auto;padding:44px 28px 60px}header{margin:0 0 30px;padding:0 8px}.eyebrow{font-size:12px;letter-spacing:.16em;color:#2171ad;font-weight:700}h1{font-size:36px;line-height:1.25;margin:12px 0}.subtitle{color:#5e7180}
    section{background:#fff;border:1px solid #e2e9ef;border-radius:8px;padding:28px;margin:20px 0}h2{font-size:23px;margin:0 0 16px}h3{font-size:18px;margin-top:24px}p{margin:12px 0}.lead{border-top:4px solid #2171ad}.key{font-size:21px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}.cards div{background:#f2f7fb;padding:16px;border-radius:5px}.cards span{display:block;color:#667887;font-size:13px}.cards strong{font-size:25px;font-weight:650}
    figure{margin:24px 0}img{width:100%;height:auto;display:block}figcaption{font-size:13px;color:#647887;margin-top:10px}.table-wrap{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th{background:#eff5f9;text-align:left;padding:11px 12px;border-bottom:2px solid #d5e3ee}td{padding:9px 12px;border-bottom:1px solid #e8eef3}tr:nth-child(even) td{background:#fafcfe}.formula{background:#f3f7fb;border-left:3px solid #2171ad;padding:16px 20px;margin:22px 0}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}.provenance{font-size:13px;color:#667887}.provenance h2{font-size:19px}li{margin:8px 0}
    .formula{border:1px solid #dce7ef;border-left:4px solid #2171ad;border-radius:8px;padding:24px;background:#f6f9fc}.formula h3{margin:0 0 8px}.formula-intro,.formula-note{color:#586e80;font-size:14px}.equation{margin:18px 0;padding:16px 18px;background:#fff;border:1px solid #e2eaf1;border-radius:6px;break-inside:avoid}.equation-label{font-size:12px;font-weight:650;letter-spacing:.04em;color:#587286;margin-bottom:8px}.equation-scroll{max-width:100%;overflow-x:auto;overflow-y:hidden;padding:2px 0}.equation-scroll:focus-visible{outline:2px solid #2171ad;outline-offset:3px}.math-equation{width:auto;max-width:none;height:auto;display:block}var{font-family:Georgia,"Times New Roman",serif}pre{overflow-x:auto}
    details{margin:18px 0;border:1px solid #dce7ef;border-radius:6px;padding:12px 16px}summary{cursor:pointer;color:#2171ad;font-weight:600}summary:focus-visible{outline:2px solid #2171ad;outline-offset:4px}
    a{color:#185983;text-underline-offset:.18em}nav ul{list-style:none;display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:0 18px;padding:0 14px}
    .flow{background:#eff5f9;border-left:3px solid #2171ad;padding:12px 16px;font-weight:600}section[id]{scroll-margin-top:16px}
    .key{font-size:17px;font-weight:650}.cards strong{overflow-wrap:anywhere}
    #business .table-wrap{overflow:visible}
    #business table{table-layout:fixed;white-space:normal}
    #business th,#business td{padding:8px;overflow-wrap:anywhere}
    #business th:first-child{width:30%}
    #business .math-equation{max-width:100%}
    #business .flow{overflow-wrap:anywhere}
    #business .targeting-example{padding:16px 20px;background:#f6f9fc;border:1px solid #dce7ef;border-radius:8px}
    .targeting-example-legend,.targeting-example-axis{display:flex;flex-wrap:wrap;gap:4px 16px;color:#586e80;font-size:13px}
    .targeting-example-legend i{display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:2px;vertical-align:middle}
    .targeting-example-row{display:grid;gap:2px;margin-top:12px;font-size:14px}
    .targeting-example-bar{display:flex;height:14px;overflow:hidden;background:#e3ebf2;border-radius:3px}
    .targeting-example-axis{justify-content:space-between;gap:0}
    .targeting-baseline{background:#2171ad}.targeting-increase{background:#e38636}
    @media(max-width:700px){.formula{padding:16px 12px}.equation{padding:12px 10px}}
    @media print{.equation-scroll{overflow:visible}.math-equation{max-width:100%;height:auto}.equation{border-color:#ddd}}
    @media(max-width:700px){main{padding:24px 12px}section{padding:18px}h1{font-size:28px}.cards{grid-template-columns:1fr}.cards strong{font-size:23px}}@media print{body{background:white}main{padding:0;max-width:none}section{border:0;padding:12px 0;break-inside:auto}figure,.cards{break-inside:avoid}.table-wrap{overflow:visible}table{white-space:normal;font-size:10px}h2,h3{break-after:avoid}}
    """
    document = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{_TITLE}</title><style>'
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
        "ate_variance_definition": "Unpooled Bernoulli sample variance with n-1 correction; pooled-null z test.",
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
                "threadpoolctl",
            )
        },
        "elapsed_seconds": elapsed_seconds,
    }
    write_result(manifest, manifest_path)
    build_report(manifest_path, report_path)
    manifest["status"] = "complete"
    write_result(manifest, manifest_path)

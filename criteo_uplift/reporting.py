# 保存实验运行记录，读取已有结果生成图表和 HTML 报告。
# finalize_run 记录复现信息与完成状态；build_report 渲染结果。

from __future__ import annotations

import base64
import html
import importlib.metadata
import io
import json
import platform
import re
from itertools import cycle
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES
from .metrics import RANKING_METRIC_DESCRIPTION
from .pipeline_io import safe_output, write_result

_TITLE = "固定投放预算下的增量转化优化"

_COLORS = {
    "random": "#7b8793",
    "response_xgboost": "#e38636", "response_logistic": "#b45c20",
    "s_xgboost": "#426b50", "x_logistic": "#9369a7",
    "s_logistic": "#8dbce1",
    "t_logistic": "#5594c5",
    "t_xgboost": "#2171ad",
    "x_xgboost": "#123f69",
}
_LABELS = {
    "random": "随机（Random）",
    "response_xgboost": "Response-XGBoost",
    "response_logistic": "Response-Logistic", "s_xgboost": "S-XGBoost", "x_logistic": "X-Logistic",
    "s_logistic": "S-Logistic",
    "t_logistic": "T-Logistic",
    "t_xgboost": "T-XGBoost",
    "x_xgboost": "X-XGBoost",
}


def _chart_text(chinese, english):
    from matplotlib.font_manager import fontManager, FontProperties

    for family in ("Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", "Arial Unicode MS"):
        if any(font.name == family for font in fontManager.ttflist):
            return chinese, {"fontproperties": FontProperties(family=[family, "DejaVu Sans"])}
    return english, {}


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
        label, font = _chart_text("频数（对数刻度）", "Count (log)")
        ax.set_ylabel(label, **font)
    title, font = _chart_text(f"建模样本的特征分布（{len(sample):,} 行）",
                              f"Feature distributions | modeling sample n = {len(sample):,}")
    fig.suptitle(title, **font)
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
        label, font = _chart_text(f"观测值 {row.roc_auc:.4f}", f"Observed {row.roc_auc:.4f}")
        ax.axvline(row.roc_auc, color="#ba6b24", label=label)
        ax.axvline(.5, color="#718096", linestyle="--", linewidth=1, label="AUC = 0.5")
        ax.set_title(f"{row.model} · p = {row.pvalue:.3f} · Holm p = {row.pvalue_holm:.3f}")
        label, font = _chart_text("诊断留出集 ROC AUC", "Diagnostic holdout ROC AUC")
        ax.set_xlabel(label, **font)
        label, font = _chart_text("置换次数", "Permutation count")
        ax.set_ylabel(label, **font)
        ax.legend(frameon=False, prop=font.get("fontproperties"))
    title, font = _chart_text(
        f"C2ST 置换分布 · 全量数据两臂等量抽样（{details['n_control'] + details['n_treatment']:,} 行）",
        f"C2ST permutation distribution | balanced sample ({details['n_control'] + details['n_treatment']:,} rows)",
    )
    fig.suptitle(title, **font)
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


def _table(frame: pd.DataFrame, *, wide=False) -> str:
    #-------------
    # 函数名：_table
    # 功能：将 DataFrame 转换为带横向滚动容器的 HTML 表格。
    # 输入参数：
    #     - frame：需要嵌入报告的表格数据。
    # 输出参数：
    #     - HTML 表格字符串。
    #-------------
    frame = frame.rename(columns={
        'treatment': '处理组标识', 'n': '样本量', 'positive': '正例数', 'negative': '负例数',
        'single_class': '单类别训练', 'feature': '特征', 'smd': 'SMD', 'abs_smd': '|SMD|',
        'mean_control': '对照组均值', 'mean_treatment': '处理组均值', 'ks_statistic': 'KS 统计量',
        'ks_pvalue': 'KS p 值', 'n_treatment': '处理组人数', 'n_control': '对照组人数',
    })
    for column in frame.select_dtypes(include='bool').columns:
        frame[column] = frame[column].map({True: '是', False: '否'})
    labels = cycle(frame.columns)
    markup = frame.to_html(index=False, border=0, escape=True, na_rep="—")
    markup = re.sub(r'<td>', lambda _: '<td data-label="' + html.escape(str(next(labels)), quote=True) + '">', markup)
    return (
        ('<div class="table-wrap wide" tabindex="0" role="region" aria-label="可横向滚动的明细表">'
         if wide else '<div class="table-wrap">')
        + markup
        + "</div>"
    )


def _image(path: Path, title: str, caption: str) -> str:
    #-------------
    # 函数名：_image
    # 功能：将 PNG 文件编码为 Base64，并生成带替代文本和图注的 HTML figure。
    # 输入参数：
    #     - path：PNG 图像文件路径。
    #     - title：图标题及图像替代文本。
    #     - caption：显示在图像下方的说明文字。
    # 输出参数：
    #     - 包含内嵌图片和图注的 HTML 字符串。
    #-------------
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}">'
        f'<figcaption><strong>{html.escape(title)}</strong> · {caption}</figcaption></figure>'
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
    content = (
        f'<section id="business"><h2>{html.escape(title)}</h2>'
        '<h3>1.1 广告投放的业务问题</h3>'
        '<p>以互联网平台上的电商广告为例，广告主希望带来更多下单等转化。平台先圈出符合投放条件的候选用户，'
        '再决定优先向谁分配广告资格。预算、广告库存和频次限制使可触达的人数有限，'
        '每用掉一个名额，就少一个触达其他候选用户的机会。</p>'
        '<p>传统 Response Propensity 模型预测用户在处理条件下的转化概率，并从高到低排序。'
        '这一分数回答“分配广告资格后，谁更可能转化”。例如，原本就打算下单的用户，即使没有获得广告资格也可能完成购买。'
        '评估广告带来的额外转化，需要比较同类用户在获得和未获得广告资格两种情景下的转化概率。</p>'
        '<p class="key">在触达规模有限的情况下，应该优先选择哪些用户，才能创造更多额外转化？</p>'
        '<h3>1.2 转化倾向与增量定向的直观比较</h3>'
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
        '增量建模利用处理前特征估计这一量，实际排序使用的是模型预测的 CATE。'
        '同一个体只经历一种分配，因此只能观察对应的结果。'
        '从观测数据估计 ATE 和 CATE，需要随机分配等实验设计支持，并依赖一致性、无干扰和两种分配都有可能等识别假设。</p>'
        + _equation(r'\max_{\pi}\;\mathbb{E}[\pi(X)\tau(X)],\quad \mathbb{E}[\pi(X)]=0.20', '固定覆盖率下的业务目标')
        + '<h3>1.4 本项目的研究问题与定向策略</h3>'
        '<p>本项目使用 Criteo Uplift v2.1 公开随机实验数据。'
        f'将广告资格分配规模固定为候选总体的 <strong>{html.escape(coverage)}</strong>，'
        '这是项目预设的业务开发假设，不代表 Criteo 官方真实广告预算。各策略在相同候选总体中各选出同样比例的用户。</p>'
        f'<ol><li><strong>转化倾向定向</strong> 按处理条件下的预测{outcome_label}概率 '
        f'<code>P(Y=1 | X,T=1)</code> 排序。分别使用 Logistic 与 XGBoost 的处理组结果模型，均仅在训练集的处理组拟合。'
        '处理分配倾向分数 <code>P(T=1 | X)</code> 描述的是获得处理资格的概率。</li>'
        '<li><strong>增量定向</strong> 按预测 CATE 排序。S/T/X-Learner 各搭配 Logistic 与 XGBoost，共六种 CATE 评分。'
        'X-Logistic 第二阶段使用连续线性回归。</li>'
        '<li><strong>随机定向</strong> 使用同覆盖率随机分配的期望增量，不参加模型选型。</li></ol>'
        f'<p class="flow">在固定 {html.escape(coverage)} 广告资格分配规模下，增量定向能否比传统'
        f'转化倾向定向获得更多预期增量{outcome_label}？</p>'
        f'<p>各策略使用同一无处理比较基准及同一<a href="#evaluation">评价口径</a>。'
        f'验证集以预设的{html.escape(criterion)}冻结策略，再在<a href="#test">独立测试集</a>比较固定覆盖率下的'
        f'增量{outcome_label}点估计及配对区间。结果用于判断是否值得开展线上验证。</p></section>'
    )

    return content


def _policy_comparison(summary, selected, criterion, unit, n):
    values = summary[["point_estimate", "ci_low", "ci_high"]].to_numpy() * 100_000
    finite = values[np.isfinite(values)]
    low, high = finite.min(initial=0), finite.max(initial=0)
    step = 10 ** np.floor(np.log10((high - low) / 4)) if high > low else 1
    low, high = np.floor(low / step) * step, np.ceil(high / step) * step
    if high == low:
        high = low + step

    def position(value):
        return 100 * (value - low) / (high - low)

    rows = []
    for row, (point, left, right) in zip(summary.itertuples(), values):
        name = html.escape(_LABELS.get(row.policy, row.policy))
        frozen = row.policy == selected
        interval = (f'<span class="policy-ci" style="left:{position(left):.6f}%;'
                    f'width:{position(right)-position(left):.6f}%"></span>'
                    if np.isfinite([left, right]).all() else "")
        marker = (f'<span class="policy-dot" style="left:{position(point):.6f}%"></span>'
                  if np.isfinite(point) else "")
        rows.append(
            f'<div class="policy-row{" frozen" if frozen else ""}" data-policy="{html.escape(row.policy)}">'
            f'<strong>{name}{"（冻结）" if frozen else ""}</strong>'
            f'<div class="policy-track" aria-hidden="true" style="--zero:{position(0):.6f}%">{interval}{marker}</div>'
            f'<span class="policy-value">{_num(point)} <small>{_interval(row._asdict(), scale=100_000)}</small></span></div>'
        )
    return (
        '<figure id="primary-policy-chart" class="policy-chart">'
        f'<h3>策略比较 · {html.escape(criterion)}</h3><p class="chart-unit">{html.escape(unit)}</p>'
        + "".join(rows)
        + f'<div class="policy-axis" aria-hidden="true"><span>{low:g}</span><span>{(low + high)/2:g}</span><span>{high:g}</span></div>'
        f'<figcaption><strong>独立测试集（{n:,} 行）</strong> · 圆点为点估计，横线为各策略的 95% Bootstrap 区间。'
        '策略差异依据配对区间判断。</figcaption></figure>'
    )


def _overlap_section(out, summary):
    #-------------
    # 功能：展示 训练集的经验覆盖结果，并与设计正值性及结果标签支持区别说明。
    # 输入参数：
    #     - out：当前报告的结果目录。
    # 输出参数：
    #     - 可嵌入诊断章节的 HTML。
    #-------------
    heading = '<h3>训练集覆盖与正值性</h3>'
    strata = pd.read_csv(out / 'diagnostics/overlap_strata.csv')
    counts = strata.assign(nonempty_min=strata.min_arm_n.where(~strata.empty_bin))
    by_feature = counts.groupby('feature', sort=False).agg(
        分箱数=('bin', 'size'), 缺臂箱数=('missing_arm', 'sum'),
        人数不足箱数=('insufficient_arm_count', 'sum'), 非空箱最少臂人数=('nonempty_min', 'min'))
    display = by_feature.reset_index().rename(columns={'feature': '特征'})
    reference = summary['expected_treatment_rate']
    reference_text = '未提供' if reference is None else f"{reference:.6f}；观察减参考 = {summary['observed_minus_expected']:.6f}"
    status = {'adequate': '未见预设覆盖警告', 'warning': '存在覆盖警告', 'pass': '通过经验检查', 'checked_no_major_warning': '未见主要边际覆盖警告', 'checked_with_warnings': '存在边际覆盖警告'}.get(summary['status'], summary['status'])
    content = (heading + f"<p>范围：训练集；经验状态：<strong>{html.escape(status)}</strong>。"
               f"共 {summary['n']:,} 人，处理组 {summary['n_treatment']:,} 人，对照组 {summary['n_control']:,} 人；"
               f"观察处理比例 {summary['observed_treatment_rate']:.6f}，对照比例 {summary['observed_control_rate']:.6f}。</p>"
               f"<p>参考分配概率：{reference_text}。来源：{html.escape(summary['probability_source'])}。</p>"
               f"<p>每个特征最多 {summary['rules']['requested_bins']} 个分位数箱，重复边界合并、缺失单独成箱。"
               f"每箱任一臂少于 {summary['rules']['min_arm_count']} 人时提示警告；这是经验阈值，不是正值性的数学判定。"
               f"缺臂 {summary['missing_arm_bins']} 箱，人数不足 {summary['weak_bins']} 箱。分箱边界仅从 训练集得到，未使用 验证集/测试集。</p>"
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
    positivity = json.loads((out / "diagnostics/overlap_positivity.json").read_text(encoding="utf-8"))
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
    response_key = "response_xgboost"
    response_label = _LABELS[response_key]
    selected = selection["selected_policy"]
    selected_uplift = selection["selected_uplift_model"]
    selected_label = html.escape(_LABELS.get(selected, selected))
    uplift_label = html.escape(_LABELS.get(selected_uplift, selected_uplift))
    budget = float(selection["primary_budget"])
    outcome = manifest["primary_outcome"]
    outcome_cn = {"conversion": "转化", "visit": "访问"}[outcome]
    ht_unit = f"每 10 万名全部合资格用户的增量{outcome_cn}"
    e = float(manifest["assignment_probability"])
    primary_metric = selection["primary_metric"]
    metric_labels = {
        "auuc": "AUUC", "qini_score": "Traditional Qini Score",
        "gain_ht": "HT Gain", "gain_plugin": "Uplift Gain", "qini_traditional": "Traditional Qini Gain",
    }
    area_primary = primary_metric in ("auuc", "qini_score")
    primary_label = metric_labels[primary_metric]
    criterion = f"完整曲线的 {primary_label}" if area_primary else f"{budget:.0%} 覆盖率的 {primary_label}"
    scale_label = ht_unit if primary_metric == "gain_ht" else "指标 × 10⁵"
    test_bootstrap = frames["test_bootstrap"]
    test_pairs = frames["test_pairwise"]
    policies = frames["test_metrics"].copy()
    full = policies[policies.budget.eq(1.)].iloc[0]
    random = policies[policies.policy.eq('random')].copy()
    if set(policies.policy) != {*selection['candidates'], 'random'}:
        raise ValueError('报告需要八种模型与 Random 的完整真实结果')
    ranking = policies.drop_duplicates('policy')
    policies["incremental_per_100k"] = policies["gain_ht_per_100k"]
    random_gain = random.set_index("budget").gain_ht_per_100k
    propensity_gain = policies[policies.policy.eq(response_key)].set_index("budget").gain_ht_per_100k
    policies["vs_random_per_100k"] = policies.incremental_per_100k - policies.budget.map(random_gain)
    policies["vs_propensity_per_100k"] = policies.incremental_per_100k - policies.budget.map(propensity_gain)
    if primary_metric == "gain_ht":
        intervals = test_bootstrap[["policy", "budget", "ci_low", "ci_high"]].copy()
        intervals[["ci_low", "ci_high"]] *= 100_000
        policies = policies.merge(intervals, on=["policy", "budget"], validate="one_to_one")
    else:
        policies[["ci_low", "ci_high"]] = np.nan
    selected_row = policies.loc[policies.budget.eq(budget) & policies.policy.eq(selected)].iloc[0]
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
            title, font = _chart_text("全量公开样本的平均处理效应（ATE）",
                                      "Average treatment effects in the full public sample")
            fig.suptitle(title, fontsize=13, **font)

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
                label, font = _chart_text({"conversion": "转化", "visit": "访问"}[ate_outcome],
                                          ate_outcome.capitalize())
                ax.set_yticks([0], [label], **font)
                ax.tick_params(axis='y', length=0, pad=12)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

                label, font = _chart_text("平均处理效应（百分点）", "Average treatment effect (percentage points)")
                ax.set_xlabel(label, **font)
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
        n_balance = int(balance.iloc[0]["n_treatment"] + balance.iloc[0]["n_control"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        axes[0].barh(balance["feature"], balance["abs_smd"], color="#2171ad", height=.65)
        axes[0].axvline(.1, color="#e38636", linestyle="--", lw=1.5)
        label, font = _chart_text("标准化均值差绝对值（|SMD|）", "Absolute standardized mean difference (|SMD|)")
        axes[0].set_xlabel(label, **font)
        axes[1].barh(balance["feature"], balance["ks_statistic"], color="#24858b", height=.65)
        label, font = _chart_text("双侧 KS 统计量", "Two-sided KS statistic")
        axes[1].set_xlabel(label, **font)
        for ax in axes:
            ax.grid(axis="x")
        title, font = _chart_text(f"全量样本单特征平衡（{n_balance:,} 行）",
                                  f"Marginal balance on the full sample ({n_balance:,} rows)")
        fig.suptitle(title, **font)
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
                    label="q × endpoint")
        for ax in axes:
            ax.axhline(0, color="#c4ced7", lw=0.8)
            label, font = _chart_text("候选用户覆盖率", "Coverage of eligible users")
            ax.set_xlabel(label, **font)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=0.7)
            ax.legend(frameon=False, fontsize=8, loc="best", ncol=3)
        for ax, title, ylabel in zip(
            axes, ("HT Gain", "Uplift Gain", "Traditional Qini"),
            (ht_unit, "Uplift Gain × 100,000", "Traditional Qini × 100,000"),
        ):
            ax.set_title(title)
            label, font = _chart_text(ylabel, "Incremental outcomes per 100,000 eligible users"
                                      if title == "HT Gain" else ylabel)
            ax.set_ylabel(label, **font)
        save(fig, "ranking")

        fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
        for name, group in policies.groupby("policy", sort=False):
            group = group.sort_values("budget")
            ax.plot(
                group["budget"],
                group["incremental_per_100k"],
                color=_COLORS.get(name, "#2171ad"),
                marker="o",
                markersize=3,
                lw=2.4 if name == selected else 1.3,
                linestyle="--" if name == "random" else "-",
                label=("Random" if name == "random" else _LABELS.get(name, name)) + (" *" if name == selected else ""),
            )
        ax.axvline(budget, color="#a7b3be", lw=1, ls=":", label=f"{budget:.0%}")
        ax.set_xticks(sorted({0., .1, .2, .3, .5, 1., budget}))
        title, font = _chart_text("覆盖率与 HT Gain", "Coverage and HT Gain")
        ax.set_title(title, **font)
        label, font = _chart_text("候选用户覆盖率", "Coverage of eligible users")
        ax.set_xlabel(label, **font)
        label, font = _chart_text(ht_unit, "Incremental outcomes per 100,000 eligible users")
        ax.set_ylabel(label, **font)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(frameon=False, fontsize=8, ncol=2)
        ax.grid(alpha=0.7)
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
    primary_summary = test_bootstrap if area_primary else test_bootstrap[test_bootstrap.budget.eq(budget)]
    # Display the frozen strategy first. Keep the remaining candidates in the saved order.
    primary_summary = pd.concat([
        primary_summary[primary_summary.policy.eq(selected)],
        primary_summary[primary_summary.policy.ne(selected)],
    ])
    train_control = frames["split_summary"].query("split == 'train' and treatment == 0").iloc[0]
    control_n = int(train_control["n"])
    control_events = int(train_control[event_column])
    selected_pairs = primary_pairs[primary_pairs.policy.eq(selected)].set_index("reference")
    vs_propensity = selected_pairs.loc[response_key] if selected != response_key else None
    vs_random = selected_pairs.loc["random"]
    uplift_vs_propensity = primary_pairs.loc[
        primary_pairs.policy.eq(selected_uplift) & primary_pairs.reference.eq(response_key)
    ].iloc[0]

    def paired_finding(row, reference, subject):
        low, high = row["ci_low"], row["ci_high"]
        interpretation = (
            "有效重复不足，无法判断差值方向" if pd.isna(low) else
            "区间完全高于零，支持增量优势" if low > 0 else
            "区间完全低于零，支持参考策略的增量优势" if high < 0 else
            "区间包含零，现有证据不足以区分策略效果"
        )
        return (
            f"{subject}相对 {reference} 的点估计差值为 {row['difference'] * 100_000:+.2f}，"
            f"配对 95% CI {_interval(row, scale=100_000)}。{interpretation}。"
        )

    uplift_finding = paired_finding(uplift_vs_propensity, response_label, uplift_label)
    propensity_finding = paired_finding(vs_propensity, response_label, selected_label) if vs_propensity is not None and selected != selected_uplift else ""
    test_comparisons = primary_pairs.loc[
        (primary_pairs.policy.eq(selected) & primary_pairs.reference.isin({response_key, "response_logistic", "random"}))
        | (primary_pairs.policy.eq(selected_uplift) & primary_pairs.reference.eq(response_key))
    ]
    decision = f"保留验证集预先冻结的 {selected_label}；在固定覆盖率的线上随机实验中验证可迁移性。"

    def point_estimates(frame):
        return pd.DataFrame({
            "策略": frame.policy,
            **{f"{metric_labels[key]} × 10⁵": frame[key].map(lambda v: _num(v * 100_000))
               for key in ("gain_ht", "gain_plugin", "qini_traditional", "auuc", "qini_score")},
        })

    ate_all = frames["ate_summary"]
    ate_display = pd.DataFrame(
        {
            "样本": ate_all["scope"].map({"full_dataset": "全量源数据", "test": "独立测试集"}),
            "指标": ate_all["outcome"].map({"conversion": "转化", "visit": "访问"}),
            "对照率": ate_all["response_control"].map(lambda x: f"{100*x:.4f}%"),
            "处理率": ate_all["response_treatment"].map(lambda x: f"{100*x:.4f}%"),
            "ATE（百分点）": ate_all["ate"].map(lambda x: _num(100 * x, 4)),
            "95% CI（百分点）": ate_all.apply(
                lambda row: _interval(row, scale=100, digits=4), axis=1
            ),
            "标准误（百分点）": (ate_all["standard_error"] * 100).map(lambda v: _num(v, 5)),
            "Z 统计量": ate_all["z_statistic"].map(lambda v: _num(v, 3)),
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
            "常数基线 LogLoss": c2["dummy_log_loss"].map(lambda x: _num(x, 5)),
            "置换 p": c2["pvalue"].map(_p),
            "Holm p（两模型）": c2["pvalue_holm"].map(_p),
        }
    )
    # 只展示主指标的不确定性。面积按 10⁵ 缩放，Gain 使用每十万总体用户标度。
    val = frames["validation_bootstrap"].query("policy != 'random'").sort_values(
        "bootstrap_mean", ascending=False, kind="stable",
    )
    val_display = pd.DataFrame({
        "策略": val.policy.map(lambda key: _LABELS.get(key, key)),
        "点估计": val.point_estimate.map(lambda x: _num(x * 100_000)),
        "Bootstrap 均值": val.bootstrap_mean.map(lambda x: _num(x * 100_000)),
        "95% CI": val.apply(lambda r: _interval(r, scale=100_000), axis=1),
        "获胜频率": val.win_frequency.map(lambda x: _num(x * 100, 1) + "%" if pd.notna(x) else "—"),
        "有效重复": val.valid_replicates,
        "冻结": val.policy.map(lambda name: "是" if name == selected else ""),
    })
    test_display = pd.DataFrame({
        "策略": primary_summary.policy.map(lambda key: _LABELS.get(key, key)),
        "点估计": primary_summary.point_estimate.map(lambda x: _num(x * 100_000)),
        "Bootstrap 均值": primary_summary.bootstrap_mean.map(lambda x: _num(x * 100_000)),
        "95% CI": primary_summary.apply(lambda r: _interval(r, scale=100_000), axis=1),
        "有效重复": primary_summary.valid_replicates,
    })
    core = policies[policies.budget.eq(budget)].set_index('policy').loc[primary_summary.policy]
    test_display = pd.DataFrame({
        '策略名称': [_LABELS.get(name, name) + ('（冻结）' if name == selected else '') for name in core.index],
        'HT Gain @20% × 10⁵': core.gain_ht_per_100k.map(_num).to_numpy(),
        '95% CI': [_interval(row) for _, row in core.iterrows()],
        **{label: (core[column] * 100_000).map(_num).to_numpy() for column, label in (
            ('gain_plugin', 'Uplift Gain × 10⁵'), ('qini_traditional', 'Traditional Qini Gain × 10⁵'),
            ('auuc', 'AUUC × 10⁵'), ('qini_score', 'Qini Score × 10⁵'))},
    })
    # Self comparison is exactly zero, not an independent significance test.
    if selected in ('response_logistic', 'response_xgboost'):
        self_row = dict(policy=selected, reference=selected, difference=0., ci_low=0., ci_high=0., valid_replicates=selection['n_bootstrap'])
        test_comparisons = pd.concat([test_comparisons, pd.DataFrame([self_row])], ignore_index=True)
    def paired_table(frame):
        return _table(pd.DataFrame({
            "策略": frame.policy.map(lambda key: _LABELS.get(key, key)),
            "参考策略": frame.reference.map(lambda key: _LABELS.get(key, key)),
            "点估计差值": frame.difference.map(lambda x: "—" if pd.isna(x) else f"{x * 100_000:+.2f}"),
            "配对 95% CI": frame.apply(lambda r: _interval(r, scale=100_000), axis=1),
            "有效配对": frame.valid_replicates,
        }))

    val_pairs = frames["validation_pairwise"]
    val_pairs = val_pairs.loc[val_pairs.policy.eq(selected)]
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
        "HT 对 Response-XGBoost 差值": policies.vs_propensity_per_100k.map(_num),
    })
    if primary_metric == "gain_ht":
        budget_display["HT 点态 95% CI"] = policies.apply(_interval, axis=1)
    key_budgets = [.1, .2, .3, .5, 1.]
    key_rows = []
    for name in dict.fromkeys([selected, "random", response_key]):
        for q in key_budgets:
            key_rows.extend(policies.index[(policies["policy"] == name) & np.isclose(policies["budget"], q)])
    budget_key_display = budget_display.loc[key_rows].drop(columns=["Uplift Gain × 10⁵", "Traditional Qini Gain × 10⁵"])
    budget_key_display = budget_key_display.drop(columns="HT 对 Response-XGBoost 差值") if selected == response_key else budget_key_display
    comparison_budgets = budget_display.loc[
        policies.policy.isin({selected, "random", response_key}) & ~policies.budget.isin(key_budgets)
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
        for label, value in zip(("0%–10%", "10%–20%", "20%–30%", "30%–50%", "50%–100%"), marginal)
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
            "指标": learning["outcome"].map({"conversion": "转化", "visit": "访问"}),
            "模型": learning["model"],
            "LogLoss": learning["log_loss"].map(lambda x: _num(x, 5)),
            "AP": learning["average_precision"].map(lambda x: _num(x, 5)),
            "Brier": learning["brier"].map(lambda x: _num(x, 5)),
            "ROC AUC": learning["roc_auc"].map(lambda x: _num(x, 4)),
        }
    )
    overlap_data = overlap.get("fraction_matching_training_features", {})
    overlap_text = "、".join(f"{dict(validation='验证集', test='测试集').get(k, k)} {v:.2%}" for k, v in overlap_data.items())
    srm_text = (
        f"处理组占 {srm['observed_ratio']:.4%}，外部参考为 {srm['expected_ratio']:.2%}，"
        f"Pearson 卡方检验 p = {_p(srm['pvalue'])}。该比较衡量公开样本对外部比例的偏离，"
        "原始实验 SRM 仍需依据各实验的设计分流比例与计数。"
    )
    balance_caption = "原始全量 float64 特征的 SMD 与 KS。0.1 是 SMD 的经验参考，KS p 值为探索性结果。"
    c2st_sampling_text = f"C2ST 从全量源数据无放回抽样，对照组 {int(c2st_details['n_control']):,} 行，处理组 {int(c2st_details['n_treatment']):,} 行"
    balance = frames["balance"]
    max_smd = balance.loc[balance.abs_smd.idxmax()]
    max_ks = balance.loc[balance.ks_statistic.idxmax()]
    diagnostic_overview = pd.DataFrame([
        ("SRM", f"公开全量 · 处理组 {srm['observed_ratio']:.4%}，外部参考 {srm['expected_ratio']:.2%}，Pearson p = {_p(srm['pvalue'])}",
         "该检验比较公开数据与外部参考比例。原实验的分流质量需依据原设计判断。"),
        ("SMD / KS", f"公开全量 · 最大 |SMD| {max_smd.abs_smd:.6f}（{max_smd.feature}），最大 KS {max_ks.ks_statistic:.6f}（{max_ks.feature}）",
         "汇总的是逐特征差异幅度，需结合联合分布检查。"),
        ("C2ST", f"等量抽样 · {int(c2st_details['n_control']) + int(c2st_details['n_treatment']):,} 行。"
         + "；".join(f"{row.model} AUC {_num(row.roc_auc, 4)}，Holm p = {_p(row.pvalue_holm)}" for row in c2.itertuples()) + "。",
         "以 5% 为探索性参考，"
         + ("、".join(c2.loc[c2.pvalue_holm.lt(.05), "model"]) + " 检出可观测分配信号。"
            if c2.pvalue_holm.lt(.05).any() else "两个分类器均未检出可观测分配信号。")),
        ("覆盖与正值性", f"训练集 · 缺臂 {positivity['missing_arm_bins']} 箱，低于经验阈值 {positivity['weak_bins']} 箱",
         f"阈值为每臂 {positivity['rules']['min_arm_count']} 人。逐特征分箱只能检查边际覆盖。"),
        ("结果可学习性", f"验证集 · {outcome_cn} AP：XGBoost {_num(outcome_learning.loc['xgboost', 'average_precision'], 4)}，Dummy {_num(outcome_learning.loc['dummy', 'average_precision'], 4)}",
         "结果预测提供响应排序的信息，CATE 排序的价值仍需用策略评估检验。"),
    ], columns=["诊断", "样本与观察", "读数含义"])
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
    split_display["划分"] = split_display["划分"].replace({"train": "训练集", "validation": "验证集", "test": "测试集"})
    footer_versions = ", ".join(
        f"{esc(k)} {esc(str(v))}" for k, v in manifest.get("versions", {}).items()
    )
    stability = json.loads((out / "stability" / "report.json").read_text(encoding="utf-8"))
    stability_enabled = stability['status'] == 'complete'
    protocol = {'seeds': [], 'training_fractions': []}
    if stability_enabled:
        protocol = json.loads((out / "stability" / "protocol.json").read_text(encoding="utf-8"))
        aggregate = pd.read_csv(out / "stability/aggregate.csv")
        frequencies = pd.read_csv(out / "stability/selection_frequency.csv")
        stability_columns = ["split_method", "training_fraction", "policy",
                             "gain_ht_per_100k_mean", "gain_ht_per_100k_std", "vs_propensity_per_100k_mean"]
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for name in aggregate.policy.unique():
            rows = aggregate[(aggregate.policy == name) & (aggregate.split_method == "row_stratified")].sort_values("training_fraction")
            ax.plot(rows.training_fraction, rows.gain_ht_per_100k_mean, "o-", label=_LABELS.get(name, name))
        title, font = _chart_text("开发集稳定性 · 跨随机种子均值", "Development stability: mean across seeds")
        ax.set_title(title, **font)
        label, font = _chart_text("开发训练子集比例", "Fraction of development training partition")
        ax.set_xlabel(label, **font)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        label, font = _chart_text(ht_unit, "Incremental outcomes per 100,000 eligible users")
        ax.set_ylabel(label, **font)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        stability_plot = figures / "stability.png"
        fig.savefig(stability_plot, dpi=160)
        plt.close(fig)
        stability_detail = pd.read_csv(out / "stability/longmetrics.csv")
        stability_detail = stability_detail.loc[
            stability_detail.policy.isin({selected, response_key}) | stability_detail.selected_on_validation,
            ["seed", "split_method", "training_fraction", "policy", "train_n",
             "validation_n", "gain_ht_per_100k", "vs_propensity_per_100k", "selected_on_validation"],
        ]
    assumptions = json.loads((out / "business" / "assumptions.json").read_text(encoding="utf-8"))
    plans = pd.read_csv(out / "business/hypothetical_ab_plans.csv")
    scenarios = pd.read_csv(out / "business/hypothetical_cost_scenarios.csv")
    training_display = pd.DataFrame([
        {'策略': _LABELS.get(name, name), '验证评分标准差': _num(details.get('validation_score_std'), 7),
         '训练警告': '；'.join(details.get('warnings', [])) or '无',
         '训练状态': '已拟合', '拟合秒数': _num(details['seconds']),
         '来源': _LABELS.get(details.get('fit_source', name), name)}
        for name, details in manifest['models'].items()
    ])
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
    if stability_enabled:
        reproducibility += ' --run-stability'
    if not stability_enabled:
        reproducibility = reproducibility.replace(' --stability-seeds  --stability-fractions ', ' --skip-stability')
    if arguments.get('protocol'):
        reproducibility += f" --stage {arguments['stage']} --protocol artifacts/{Path(arguments['protocol']).name}"
    ab_display = plans[["absolute_mde", "relative_mde", "n_policy_A", "n_policy_B", "n_total"]].copy()
    ab_display["absolute_mde"] = ab_display.absolute_mde.map(lambda x: f"{x * 100:.3f} pp")
    ab_display["relative_mde"] = ab_display.relative_mde.map(lambda x: f"{x:.0%}")
    ab_display = ab_display.rename(columns={
        "absolute_mde": "绝对 MDE", "relative_mde": "相对 MDE", "n_policy_A": "策略 A",
        "n_policy_B": "策略 B", "n_total": "总样本量",
    })
    economic_columns = ["value_per_conversion_assumed", "contact_cost_assumed",
                        "gross_incremental_value", "contact_cost_total", "net_incremental_value",
                        "net_ci_low_conditional", "net_ci_high_conditional", "break_even_contact_cost"]
    economic_display = scenarios[economic_columns].round(4)
    economic_display.columns = [
        "单次转化价值", "单位触达成本", "增量总价值", "触达总成本",
        "净增量价值", "条件区间下限", "条件区间上限", "盈亏平衡单位成本",
    ]
    sensitivity, group_winners = '本轮未进行额外重训，不能据此推断训练稳定性', ''
    if stability_enabled:
        stability_stratified = aggregate.loc[aggregate.split_method.eq("row_stratified") &
                                             aggregate.policy.eq(selected)].sort_values("training_fraction")
        sensitivity = "、".join(f"{row.training_fraction:.0%} 训练量时为 {row.gain_ht_per_100k_mean:,.2f}"
                                for row in stability_stratified.itertuples())
        group_wins = frequencies.query("split_method == 'feature_group' and selected_count > 0")
        group_winners = "、".join(f"{esc(row.policy)}（{row.selected_count}/{row.runs}）" for row in group_wins.itertuples())
    scenario_example = scenarios.iloc[len(scenarios) // 2]
    best_areas = (
        f"测试集中 AUUC 最高的是 {esc(ranking.loc[ranking.auuc.idxmax(), 'policy'])}，"
        f"Traditional Qini Score 最高的是 {esc(ranking.loc[ranking.qini_score.idxmax(), 'policy'])}。"
        "这两项跨覆盖率排序统计承担辅助诊断，正式决策沿用预设的 验证集标准。"
    )
    uplift_ci = _interval(uplift_vs_propensity, scale=100_000)
    unit_note = f"以下差值单位为{scale_label}。"
    random_reference = primary_summary.loc[primary_summary.policy.eq("random")].iloc[0]
    outcome_rate = n_events / n_test

    section_titles = {
        "summary": "00. 执行摘要",
        "business": "01. 业务问题与定向策略",
        "data": "02. 数据来源与准备",
        "diagnostics": "03. 实验诊断与因果识别",
        "ate": "04. 平均处理效应（ATE）",
        "modeling": "05. 因果增量建模",
        "evaluation": "06. 评估框架",
        "validation": "07. 验证与模型选择",
        "test": "08. 独立测试评估",
        "robustness": "09. 预算敏感性与稳健性",
        "conclusion": "10. 业务启示与结论",
        "appendix": "附录：技术细节与复现",
    }
    stability_html = '<h3>开发集稳定性分析</h3><p>' + esc(stability.get('reason', '')) + '本轮没有新增跨种子或训练规模重训结果，不以历史结果替代九策略证据。</p></section>'
    if stability_enabled:
        stability_html = (
        '<h3>开发集稳定性分析</h3>'
        f'<p>{int(stability["experiments"])} 个配置、{int(stability["fitted_models"])} 次模型拟合；排除原测试集 {int(stability["test_n_excluded"]):,} 行。'
        f'正式测试集 交集为零：{stability["all_test_row_overlaps_zero"]}；按特征分组的交集为零：{stability["all_group_split_feature_overlaps_zero"]}。</p>'
        f'<p>Seeds {esc(str(protocol["seeds"]))}；训练比例 {esc(str(protocol["training_fractions"]))}；比较行分层与特征分组划分。'
        f'各配置按 {esc(criterion)} 的 Bootstrap 均值选择。均值和标准差描述重复开发实验的波动，不是独立测试集 置信区间。相同匿名特征不代表同一用户。</p>'
        + _image(stability_plot, "开发期训练规模敏感性", "所有点来自已保存的 aggregate.csv；没有使用原测试集 重新选型。")
        + '<details><summary>训练规模、特征分组与获选频率明细</summary>'
        + _table(aggregate[stability_columns].round(2), wide=True)
        + '<details><summary>各随机种子、训练规模与划分方式</summary>'
        + _table(stability_detail.round(2), wide=True) + '</details>'
        + '<h3>验证集策略获选频率</h3>' + _table(frequencies.round(3), wide=True) + '</details>'
        + f'<p>特征分组划分下的获选策略与次数为 {group_winners}。'
        '不同划分改变开发评估的特征支持与正例构成，频率用于描述这种敏感性。</p></section>'
        )
    stage = manifest.get('arguments', {}).get('stage', '')
    stage_title = {'development': '开发实验报告 · 500 万行', 'full': '正式实验报告 · 全量数据'}.get(stage, '')
    protocol_note = ''
    if 'protocol' in sample_report:
        frozen_protocol = sample_report['protocol']
        protocol_note = (f'<p class="key">预冻结正式测试集 {frozen_protocol["test_n"]:,} 行；排除历史建模记录 '
                         f'{frozen_protocol["historical_rows_excluded"]:,} 行。' + esc(frozen_protocol['limitations']) + '</p>')
        if stage == 'development':
            protocol_note += '<p>本报告的测试集是开发样本内部留出评价；全部 500 万开发行与预冻结正式测试集无交集。开发结果不构成全量正式测试集证据。</p>'
    sections = [
        f'<header><p class="eyebrow">应用数据科学研究报告</p>'
        f'<h1>{_TITLE}</h1><p class="subtitle">{stage_title}</p>'
        '<p class="subtitle">基于 Criteo 数据的因果增量建模、实验诊断与投放策略评价</p></header>',
        f'<section class="lead" id="summary"><h2>{section_titles["summary"]}</h2>'
        f'<p class="decision"><strong>{decision}</strong></p>'
        f'<p>预设决策口径为 <strong>{esc(criterion)}</strong>。'
        f'本次建模 {int(manifest["sample_n"]):,} 行，在 {n_test:,} 行留出测试样本上评估。</p>'
        f'<div class="cards"><div><span>预设覆盖率</span><strong>{primary_pct}</strong>'
        '<span>候选用户中获得广告资格的比例</span></div>'
        f'<div><span>冻结策略的测试集 HT Gain</span><strong>{_num(selected_row["incremental_per_100k"])}</strong>'
        + (f'<span>95% CI {_interval(selected_row)}</span>' if primary_metric == "gain_ht" else "")
        + f'<span>{esc(ht_unit)}</span></div>'
        f'<div><span>相对随机（Random）</span><strong>{vs_random.difference * 100_000:+.2f}</strong>'
        f'<span>配对 95% CI {_interval(vs_random, scale=100_000)}</span><span>{esc(scale_label)}</span></div></div>'
        f'<p>验证集最优 Uplift 候选为 <strong>{uplift_label}</strong>。独立测试相对 {response_label} '
        f'的点差为 <strong>{uplift_vs_propensity.difference * 100_000:+.2f}</strong>，配对 95% CI {uplift_ci}。'
        + ('区间包含零，本次决策沿用验证集冻结的策略。'
           if uplift_vs_propensity.ci_low <= 0 <= uplift_vs_propensity.ci_high
           else f'独立测试沿用已冻结的 {selected_label}。')
        + '</p><p class="print-note">打印版展示核心结果。诊断、辅助指标、公式和情景明细可在离线 HTML 中展开。</p></section>',
        _business_section(section_titles["business"], primary_pct, outcome, criterion),
        f'<section id="data"><h2>{section_titles["data"]}</h2>' + protocol_note +
        f'<p>源文件 <code>{source_file}</code> 共 {int(audit["n_rows"]):,} 行；本次建模样本 {int(manifest["sample_n"]):,} 行，'
        f'抽样方法：{("保留全部源数据行" if manifest["arguments"]["sample_size"] == 0 else "按固定种子无放回均匀随机抽样")}，随机种子 {int(manifest["seed"])}。固定采用 训练集 / 验证集 / 测试集 = 60% / 20% / 20%，'
        '按 treatment × 主结果联合分层；模型均只在 train 拟合，没有 train + validation 重拟合。</p>'
        '<p>研究对象是公开样本中 treatment <strong>分配</strong>对结果的 ITT 风格效应。模型仅用 f0–f11 处理前特征；'
        'exposure、visit、conversion 不作为特征，也不按曝光或访问行为筛选人群。转化倾向在本报告中表示 '
        '<code>P(Y=1 | X,T=1)</code> 的结果倾向排序，不表示处理分配概率。</p>'
        f'<p>全量扫描发现：缺失/非有限数 {int(audit["missing_or_nonfinite"]):,}，非二元标签 {int(audit["invalid_binary_labels"]):,}，'
        f'对照组曝光 {int(audit["control_exposed"]):,}，转化但未访问 {int(audit["conversion_without_visit"]):,}；'
        f'全字段重复指纹 {int(audit["duplicate_row_fingerprints"]):,}。重复指纹只是记录特征诊断，没有 用户 ID，因此保留原行，不能直接推断重复用户。</p>'
        f'<p>与训练集匿名特征向量相同的比例：{overlap_text}。该检查基于建模时 float32 表示；相同向量不等于同一用户，也不能排除未知用户依赖。</p>'
        + _table(pd.DataFrame(audit["group_summary"]).rename(columns={
            "group": "分组", "n": "行数", "fraction": "占比", "conversion_positive": "转化数",
            "conversion_rate": "转化率",
        }).assign(分组=lambda t: t.分组.replace({'Control': '对照组', 'Treatment': '处理组', 'Total': '总体'}), 占比=lambda t: t.占比.map(lambda x: f"{x:.2%}"),
                  转化率=lambda t: t.转化率.map(lambda x: f"{x:.4%}")))
        + '<h3>数据字段与建模划分</h3>'
        '<p>Criteo v2.1 公开数据来自广告增量实验。treatment 表示实验分配，conversion 和 visit 是二元结果，'
        'exposure 表示实际曝光，f0–f11 是匿名预处理协变量。'
        '划分已核验来源行号非空且唯一，三个集合两两互斥，并完整覆盖抽样行。</p>'
        + _table(split_display)
        + '<details><summary>建模样本特征分布</summary>'
        + _image(figures / "sample_features.png", "十二个建模特征的分布",
                 "横轴为对应匿名特征值，纵轴为对数刻度的建模样本频数。尖峰和离散值有助于解释 KS 的并列值及特征向量重合。")
        + '</details></section>',
        f'<section id="diagnostics"><h2>{section_titles["diagnostics"]}</h2>'
        '<p>识别目标是公开样本中分配处理对结果的平均效应与排序政策价值。'
        '解释依赖一致性与无干扰（SUTVA）、分配与潜在结果近似独立及各特征区域的处理概率远离 0 和 1。'
        'Criteo 的实验描述提供设计依据，以下检查衡量公开数据中的可观测差异。'
        '公开版本的抽样与隐私处理、未知实验分层和用户依赖限制了对原始随机机制的核验。</p>'
        '<h3>诊断总览</h3>'
        + '<div id="diagnostic-overview">' + _table(diagnostic_overview) + '</div>'
        + '<details><summary>实验诊断明细</summary>'
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
        + f'<p>{c2st_conclusion}置换 p 值分辨率为 {1 / (b + 1):.3f}。</p>'
        + '<p>高于置换基线的 AUC 提示可观测分配差异，需要检查实验设计、抽样和特征时序；不显著不能证明随机化。'
        '该置换方案假设可交换分配；如存在未知实验分层或多个广告实验混合，需使用匹配实际设计的置换方案。</p>'
        + _overlap_section(out, positivity)
        + '<h3>结果可预测性与基线对照</h3>'
        + _table(learning_display)
        + '<p>上表使用训练集拟合、验证集评估，展示总体范围；按处理臂分别训练的明细保存在可学习性结果文件中。'
        'Dummy 预测训练集正例率。LogLoss/Brier 越低越好，AP/ROC AUC 越高越好；AP 的基线取决于正例率。'
        '这些指标衡量结果预测能力。</p></details></section>',
        f'<section id="ate"><h2>{section_titles["ate"]}</h2><p>分别以全量源数据和独立测试集 估计分配处理的 ITT 效应。'
        'ATE 是处理组结果率减去对照组结果率；绝对效应用百分点表示，相对增幅以对照率为分母。</p>'
        + _table(ate_display, wide=True)
        + _image(
            paths["ate_full"],
            "全量平均处理效应",
            "空心点为差均值，横线为 95% Wald 区间；两个指标采用各自横轴尺度，虚线表示零效应。",
        )
        + '<p>转化与访问对应不同漏斗阶段。全量估计给出实验的平均背景，测试集估计描述留出人群。'
        '总体平均值无法指出应优先触达哪些人，固定预算决策需要评价排序前缀中的增量。</p></section>',
        f'<section id="modeling"><h2>{section_titles["modeling"]}</h2>'
        '<p class="flow">训练集 60% → 拟合结果与效应模型 → 验证集评分与冻结 → 测试集加载模型并评分</p>'
        + _table(pd.DataFrame([
            (label, structure, score) for label, structure, score in [
                ('S-Logistic / S-XGBoost', '单个结果分类器，输入 X 与 T', 'f(X,1) − f(X,0)'),
                ('T-Logistic / T-XGBoost', '两臂分别拟合结果分类器', 'μ̂₁(X) − μ̂₀(X)'),
                ('X-Logistic', '两臂 Logistic + 两臂线性回归', '连续伪效应回归，加权 CATE'),
                ('X-XGBoost', '两臂 XGBClassifier + 两臂 XGBRegressor', '连续伪效应回归，加权 CATE'),
                ('Response-Logistic / Response-XGBoost', '复用对应 T-Learner 的 训练集处理组模型', 'μ̂₁(X)，不是 CATE'),
                ('Random', '不训练，不参加选型', '同覆盖率随机分配的期望收益'),
            ]], columns=['策略', '训练结构', '评分']))
        + '<p>Logistic 在 训练集内拟合中位数填补和标准化，采用 L2、C=1、LBFGS、最多 500 次迭代。'
        f'XGBoost 用直方图树算法、{arguments["trees"]} 棵树、深度 4、学习率 0.05、最小子节点权重 20、行采样 0.8、列采样 0.9。'
        '所有候选的正式训练使用同一随机种子。</p>'
        + _equation(r'D_1=Y-\hat{\mu}_0(X),\qquad D_0=\hat{\mu}_1(X)-Y,\qquad \hat{\tau}(X)=e\hat{\tau}_0(X)+(1-e)\hat{\tau}_1(X)',
                    'X-Learner 的伪效应与加权')
        + '<p>两个伪效应分别在对应处理臂回归，e 取 训练集处理比例。验证集只负责评分与选型，冻结的模型随后直接交给独立测试集。</p></section>',
        f'<section id="evaluation"><h2>{section_titles["evaluation"]}</h2>'
        f'<p class="flow">预设主指标 · {esc(criterion)}</p>'
        f'<p>HT Gain 衡量按排名给前 {primary_pct} 的用户分配广告资格，相对全部不分配时增加的预期{outcome_cn}。'
        f'分母为全部候选用户，按{esc(ht_unit)}展示。Random 是同覆盖率随机触达的解析期望。'
        'AUUC、Traditional Qini Score 与其他预算用于辅助理解排序表现。</p>'
        f'<p>采用 {boot} 轮处理臂内有放回的配对 Bootstrap。每轮各策略共用重采样行，沿固定评分排序重取 Top-k，'
        '并直接用逐轮策略差的 2.5% 与 97.5% 分位数形成配对区间。区间以固定评分与训练集分配概率估计为条件，'
        '各预算区间为点态区间，训练、选型和分配概率估计的不确定性未纳入。</p>'
        '<details id="calculation-definitions"><summary>展开指标定义与公式</summary><div class="formula">'
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
        '未进行“完美排序面积”归一化，因此不能直接等同于其他库中同名的 归一化 Qini Score。</p>'
        '<p>相同分数用独立于 outcome 的固定随机优先级破同分；模型必须在被评估样本之外训练。</p>'
        '</div></details></section>',
        f'<section id="validation"><h2>{section_titles["validation"]}</h2>'
        f'<p>在验证集按预设的 {esc(criterion)} Bootstrap 均值降序排序，表内单位为{scale_label}。'
        f'冻结 <strong>{selected_label}</strong>。Uplift 候选中，<strong>{uplift_label}</strong> 排名第一。'
        'Random 作为分析参照。</p>'
        + _table(val_display)
        + '<details><summary>验证集配对区间、辅助指标与选型公式</summary>'
        + _equation(r'\pi^*=\mathrm{argmax}_{\pi}\;\frac{1}{B}\sum_{b=1}^{B}\widehat{M}_{\pi}^{(b)}',
                    '验证集选型规则')
        + paired_table(val_pairs)
        + _table(point_estimates(frames["validation_metrics"]), wide=True)
        + f'<p>表内主指标与配对差值的单位为{scale_label}。获胜频率按各次固定评分重采样统计。</p></details></section>',
        f'<section id="test"><h2>{section_titles["test"]}</h2>'
        f'<p>独立测试集共 {n_test:,} 行，其中{outcome_cn}正例 {n_events:,} 个，对照组正例 {n_control_events:,} 个。'
        f'本节评估验证集已冻结的 <strong>{selected_label}</strong>。</p>'
        + _policy_comparison(primary_summary, selected, criterion, scale_label, n_test)
        + '<h3>同预算的配对差值</h3>'
        + f'<p>{unit_note}{uplift_finding} {propensity_finding}</p>'
        + '<div id="test-pairwise">' + paired_table(test_comparisons) + '</div>'
        + '<details><summary>全部策略的同预算配对差值</summary>' + paired_table(primary_pairs) + '</details>'
        + f'<p>随机策略在相同评价口径下的点估计为 {_num(random_reference.point_estimate * 100_000)}。'
        '表中相对随机的区间来自同批重采样的配对差。</p>'
        + '<h3>九策略核心比较</h3>'
        + '<div id="nine-strategies">' + _table(test_display) + '</div>'
        + '<p>HT Gain 与其区间单位为每 10 万名全部合资格用户的增量转化。其余列统一乘以 10⁵ 展示，但 Traditional Qini 的估计目标和尺度与 HT 不同，不直接比较大小。自比较差值及区间恒为零。</p>'
        + '<details><summary>辅助分析 · AUUC、Qini 与完整排序曲线</summary>'
        + _image(paths["ranking"], "完整排序曲线", "独立测试集。横轴为候选用户覆盖率，虚线为各指标自己的随机参照。面积使用完整排序前缀。")
        + '<div id="ranking-metrics">' + _table(ranking_display) + '</div>'
        + _table(point_estimates(policies.loc[policies.budget.eq(budget) & policies.policy.ne("random")]), wide=True)
        + f'<p>{best_areas}</p></details></section>',
        f'<section id="robustness"><h2>{section_titles["robustness"]}</h2>'
        f'<p>预算按候选用户覆盖率定义。下表按{ht_unit}展示冻结策略与参考策略。</p>'
        + '<div id="budget-sensitivity">' + _table(budget_key_display) + '</div>'
        + _image(paths["policy"], "HT 增量的覆盖率敏感性",
                 f"独立测试集。星号为冻结策略，竖线为预设覆盖率 {primary_pct}。连线连接保存的预算点估计。")
        + '<details><summary>辅助覆盖率与各口径明细（含 2%、5%、50%）</summary>'
        + _table(pd.concat([budget_display.loc[key_rows], comparison_budgets]), wide=True) + '</details>'
        + '<p>同一固定评分及同分顺序下，Top 10% ⊂ Top 20% ⊂ Top 30% ⊂ Top 100%。'
        '从 10% 扩到 20% 加入排名 10%–20% 的人，从 20% 扩到 30% 加入排名 20%–30% 的人；'
        '选中人数按覆盖率乘以评价总人数向下取整。匿名特征不能用于编造这些人群的业务画像。'
        'Random 是相同人数的随机触达解析期望，不对应模型排序。辅助预算不用于 测试后挑选最优商业预算。</p>'
        '<h3>扩大覆盖的边际增量</h3>'
        f'<p>在冻结策略 <strong>{selected_label}</strong> 下，以 ΔV = V(b) − V(a)、V(0) = 0 计算新增区间贡献。'
        '</p>'
        + '<div id="marginal-gain">' + _table(marginal_display) + '</div>'
        + f'<p>{marginal_findings}。{marginal_trend}</p>'
        '<p>这里仅报告点估计，未计算边际增量 CI，不能用两个累计 CI 的端点相减来构造区间。'
        '因此正、负和递减均为描述，不是显著性结论；30%–50% 与 50%–100% 的区间宽度分别为 20% 与 50%，总贡献不能直接与 10% 宽区间比较单位触达效率。'
        '没有实际成本和收入信息，本节只讨论估计增量转化，不能据此判断利润或 ROI。</p>'
        '<h3>全量投放情景</h3>'
        + _equation(r'G_{\mathrm{HT}}(1)=\widehat{\mathrm{ATE}}_{\mathrm{HT}},\qquad V(1)=100{,}000\,\widehat{\mathrm{ATE}}_{\mathrm{HT}}',
                    '全量覆盖与 HT-ATE')
        + f'<p>100% 覆盖表示对当前测试集 的全部 {n_test:,} 名可触达用户分配处理，估计增量为 '
        f'<strong>{_num(full_value)}</strong> / 10 万总体用户{full_interval}。'
        '所有策略选择相同的全体用户，因此具有同一 HT 终点；排序不会改变全量的价值。</p>'
        + f'<p>冻结策略与全量相比（同为每 10 万总体用户）：{full_comparisons}。'
        '扩大覆盖是否继续增加估计转化，需结合上一节各新增区间的实际符号判断，不预设越多越好。</p>'
        '<p>这是以当前测试人群和固定处理分配概率为前提的离线情景，HT-ATE 与差均值 ATE 在有限样本中可能不同。'
        '这些数字不代表已经实现的广告曝光、收入或利润，不能据此宣称找到了最优商业投放覆盖率。</p>'
        + stability_html,
        f'<section id="conclusion"><h2>{section_titles["conclusion"]}</h2>'
        '<h3>研究发现</h3>'
        f'<p><strong>{decision}</strong> {unit_note}{uplift_finding} '
        '测试集用来估计冻结策略的表现。验证集的选型规则在测试前已固定。</p>'
        f'<p>{marginal_trend} 该比较来自固定评分的描述性点估计。'
        '相同覆盖人数在真实竞价中可能对应不同花费，线上资源口径需要单独确认。</p>'
        '<h3>可能原因与不确定性</h3>'
        f'<p>独立测试集{outcome_cn}正例占 {outcome_rate:.3%}，其中对照组只有 {n_control_events:,} 个正例。'
        f'训练集对照组有 {control_n:,} 行、{control_events:,} 个正例。'
        '稀疏事件可能限制模型估计不处理时的结果概率，尤其是进一步按特征细分人群时。'
        'Uplift 需要估计两种处理状态之间的差异，模型误差可能传递到排序。'
        '当前结果无法确定这些机制各自解释了多少差异。配对区间跨零时，不能据此判定策略等效。</p>'
        f'<p>训练稳定性：{sensitivity}。'
        '测试集区间仅衡量给定已拟合评分与固定分配概率估计时的评估波动。</p>'
        '<h3>下一步线上验证</h3>'
        f'<p>按稳定用户 ID 将全部符合条件的用户随机分到 {uplift_label} 与 转化倾向策略两个实验臂。'
        f'各臂按各自入组人群的 {primary_pct} 分配广告资格，以全部随机入组用户的固定窗口转化率作主指标。'
        '预先冻结模型、观察窗、预算口径、样本量和停止规则。记录实际曝光与花费，并监控 SRM、跨组污染、频次和负反馈。'
        '两臂均值差估计线上策略差异。若要估计相对完全无广告的绝对增量，另设不处理留出组。</p>'
        f'<p>样本量情景假设基线转化率 {assumptions["parameters"]["p0"]:.2%}、'
        f'双侧 α={assumptions["parameters"]["alpha"]:.2f}、功效 {assumptions["parameters"]["power"]:.0%}。</p>'
        + _table(ab_display)
        + '<details><summary>条件性经济价值情景</summary>'
        + _equation(r'\mathrm{Net\ Value}=V_{\mathrm{conversion}}\times\mathrm{Incremental\ Conversions}-c_{\mathrm{contact}}\times k',
                    '增量贡献价值扣除触达成本')
        + f'<p>以下情景使用 {assumptions["parameters"]["eligible_n"]:,} 名可触达对象和 {primary_pct} 覆盖，'
        '比较相同货币单位下的每次增量结果价值与每次触达成本。条件性区间仅传递固定评分 HT 区间，依赖效果可迁移的假设。</p>'
        + _table(economic_display, wide=True)
        + f'<p>例如，每次增量结果价值 {_num(scenario_example.value_per_conversion_assumed)}、'
        f'每次触达成本 {_num(scenario_example.contact_cost_assumed, 3)} 时，净增量价值为 '
        f'{_num(scenario_example.net_incremental_value)}，盈亏平衡触达成本为 '
        f'{_num(scenario_example.break_even_contact_cost, 4)}。</p>'
        + '</details><p>推广边界包括广告活动与人群漂移、未知用户层依赖、稀少的对照转化、不可见的实验分层和真实成本差异。'
        '正式测试集 的独立性边界见数据章节，不能排除未留存的历史使用或用户层依赖。</p></section>',
        f'<section class="provenance" id="appendix"><h2>{section_titles["appendix"]}</h2>'
        + _equation(r'\widehat{\mathrm{ATE}}=\bar{Y}_1-\bar{Y}_0,\qquad '
                    r'\mathrm{SE}=\sqrt{s_1^2/N_1+s_0^2/N_0}', '差均值与未合并标准误')
        + _equation(r'p=\frac{1+\sum_{b=1}^{B}\mathbf{1}\{\mathrm{AUC}^{(b)}_{\mathrm{perm}}\geq\mathrm{AUC}_{\mathrm{obs}}\}}{B+1}',
                    'C2ST 经验置换 p 值')
        + f'<p>ATE 区间使用未合并方差的正态近似，双侧 p 值使用零假设下的合并比例 z 检验。'
        f'C2ST 的 {b} 轮置换为两类模型分别完整重拟合，未校正 p 的最小可能值为 {1/(b+1):.4g}，再做两模型 Holm 校正。</p>'
        f'<p>来源 SHA-256 <code>{esc(source["sha256"])}</code><br>'
        f'文件大小 {int(source["size"]):,} 字节 · 随机种子 {int(manifest["seed"])} · '
        f'结果计算耗时 {float(manifest["elapsed_seconds"]):.1f} s</p>'
        f'<p>Python {esc(manifest["python"].split()[0])} · {footer_versions}</p>'
        + '<pre>' + esc(reproducibility) + '</pre>'
        + '<details><summary>模型配置与训练样本支持</summary>'
        + _table(training_display)
        + '<pre>' + esc(json.dumps(model_settings, indent=2)) + '</pre>'
        + _table(pd.DataFrame(next(iter(manifest["models"].values()))["outcome_support"]))
        + '</details>'
        + _table(pd.DataFrame([
            ("01", "input_checks.py / raw_scan.py / preparation.py", "来源 SHA-256、全量审计、抽样与划分"),
            ("02", "diagnostics.py", "SRM、平衡、C2ST、Overlap、可学习性"),
            ("03 / 05", "evaluation.py / metrics.py", "ATE、冻结评分、测试集指标与配对区间"),
            ("04", "models.py / training_validation.py", "训练、验证集结果表与 selection.json"),
            ("06", "stability.py", "开发期协议、逐次指标与划分审计"),
            ("07 / 08", "planning.py / reporting.py", "业务情景、manifest.json 与独立 HTML"),
        ], columns=["阶段", "实现", "核验产物"]))
        + '<p>推断以固定模型评分、训练集处理比例及公开样本的近似独立性为条件。'
        '样本之间可能存在匿名实体依赖，且公开设计信息不足以核验全部随机分配机制。'
        '原始数据归 Criteo，数据派生成果适用 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>。</p>'
        '<p><a href="https://ailab.criteo.com/criteo-uplift-prediction-dataset/">Criteo AI Lab 数据集与基准论文</a> · '
        '<a href="https://arxiv.org/abs/1706.03461">Künzel 等（2019）：元学习器</a></p></section>',
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
    .decision{color:#183c56;font-size:19px}.cards{gap:12px}
    .table-wrap table{white-space:normal}
    th,td{overflow-wrap:anywhere;vertical-align:top}
    .table-wrap.wide table{min-width:620px}
    .table-wrap:focus-visible{outline:2px solid #2171ad;outline-offset:2px}
    #diagnostic-overview th:first-child{width:18%}
    #diagnostic-overview th:nth-child(2){width:43%}
    #test-pairwise td:nth-child(3),#test-pairwise td:last-child{white-space:nowrap}
    .policy-chart{--policy-columns:minmax(150px,1.15fr) minmax(120px,2.5fr) minmax(160px,1fr);padding:18px 20px;border:1px solid #dce7ef;border-radius:8px;background:#f6f9fc}
    .policy-chart h3{margin:0}.chart-unit{color:#586e80;margin:0 0 12px;font-size:13px}
    .policy-row,.policy-axis{display:grid;grid-template-columns:var(--policy-columns);gap:12px;align-items:center;padding:7px 10px}
    .policy-row.frozen{background:#e7eff7;border-radius:4px;color:#183c56}
    .policy-track{position:relative;height:22px;background:#edf2f6;border-radius:3px;margin-inline:6px}
    .policy-track::before{content:"";position:absolute;left:var(--zero);height:100%;border-left:1px dashed #8195a4}
    .policy-ci,.policy-dot{position:absolute;top:50%;background:#467297;transform:translateY(-50%)}
    .policy-ci{height:3px}.policy-dot{width:10px;height:10px;border:1px solid white;border-radius:50%;transform:translate(-50%,-50%)}
    .frozen .policy-ci,.frozen .policy-dot{background:#155686}
    .policy-value{font-variant-numeric:tabular-nums;line-height:1.5}.policy-value small{display:block;color:#4f6678}
    .policy-axis{color:#4f6678;font-size:12px}.policy-axis span{grid-area:1/2;margin-inline:6px}
    .policy-axis span:nth-child(2){justify-self:center}.policy-axis span:last-child{justify-self:end}
    .print-note{display:none}
    #nine-strategies th:nth-child(2),#nine-strategies td:nth-child(2){font-weight:700;color:#17557d}
    #nine-strategies tr:first-child td{background:#edf5fa}
    @media screen and (max-width:700px){
      #nine-strategies table,#nine-strategies tbody,#nine-strategies tr{display:block;width:100%}
      #nine-strategies thead{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
      #nine-strategies tr{border:1px solid #d9e3eb;margin:12px 0;padding:10px;border-radius:5px}
      #nine-strategies td{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:12px;padding:6px 0;overflow-wrap:anywhere}
      #nine-strategies td::before{content:attr(data-label);font-weight:600;color:#526a7e}

      main{padding:24px 12px}section{padding:16px}h1{font-size:28px}.cards{grid-template-columns:1fr}.cards strong{font-size:23px}
      .formula{padding:12px}.equation{padding:12px 10px}.math-equation{max-width:100%}
      th,td{padding:7px 6px}
      #diagnostic-overview table,#diagnostic-overview tbody,#diagnostic-overview tr,#diagnostic-overview td{display:block}
      #diagnostic-overview thead,#test-pairwise thead{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}
      #diagnostic-overview tr{border-bottom:1px solid #dce7ef;padding:8px 10px}
      #diagnostic-overview td{padding:2px 0;border:0;background:transparent}
      #diagnostic-overview td:first-child{font-weight:700;color:#183c56}
      #test-pairwise table,#test-pairwise tbody{display:block}
      #test-pairwise tr{display:grid;grid-template-columns:1fr 1fr;gap:2px 10px;padding:9px 10px;border-bottom:1px solid #dce7ef}
      #test-pairwise td{padding:0;border:0;background:transparent;overflow-wrap:normal}
      #test-pairwise td:last-child{grid-column:1/-1;color:#4f6678;font-size:12px}
      #test-pairwise td:last-child::before{content:"有效配对 "}
      #test-pairwise td:nth-child(2)::before{content:"对比 "}
      .policy-chart{padding:12px}.policy-row,.policy-axis{grid-template-columns:minmax(0,1fr) auto;gap:4px 8px}
      .policy-row strong{font-size:13px}.policy-track{grid-column:1/-1;grid-row:2}
      .policy-value{grid-column:2;grid-row:1;text-align:right;font-size:12px}
      .policy-axis span{grid-area:1/1/1/-1}
    }
    @page{size:A4;margin:14mm 15mm}
    @media print{
      body{background:white;font-size:10pt;line-height:1.55;print-color-adjust:exact}
      main{padding:0;max-width:none}header{padding:0;margin:0 0 14pt}h1{font-size:24pt}
      section{border:0;padding:8pt 0;margin:6pt 0}.lead{border-top:2pt solid #2171ad}
      h2,h3,summary,thead{break-after:avoid}p{orphans:3;widows:3}
      figure,.cards,.equation,tr{break-inside:avoid}figure>img{max-height:140mm;object-fit:contain}
      .table-wrap,.equation-scroll{overflow:visible}.table-wrap.wide table{min-width:0}
      table{font-size:8.5pt;table-layout:fixed}th,td{padding:5pt}thead{display:table-header-group}
      #ate .table-wrap,#modeling .table-wrap,#validation .table-wrap,#test-pairwise{break-inside:avoid}
      .math-equation{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}
      nav,details:not([open]){display:none}.print-note{display:block;color:#4f6678;font-size:9pt}
      .policy-chart{--policy-columns:minmax(100px,1.2fr) minmax(95px,2fr) minmax(95px,1fr)}
    }
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
    if stage == "development":
        document = document.replace("独立测试", "开发留出测试")
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
    import resource
    manifest = {
        "resource_usage": {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)},
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

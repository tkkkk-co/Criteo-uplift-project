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

_TITLE = "基于随机实验的广告增量评估与投放策略优化"

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

def save_full_feature_iqr_plot(summary, path):
    """Plot full-source feature summaries on a single raw-value axis."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    features = summary['features']
    positions = np.arange(len(features), 0, -1)
    normal = [(y, row) for y, row in zip(positions, features) if row['iqr_status'] == 'normal']
    stats = [dict(
        label=row['feature'], q1=row['q1'], med=row['median'], q3=row['q3'],
        whislo=row['whislo'], whishi=row['whishi'], fliers=row['fliers'],
    ) for _, row in normal]
    fig = Figure(figsize=(12, 6.9), facecolor='white')
    ax = fig.add_axes((.09, .15, .70, .70))
    box_color, line_color, median_color, flier_color = '#c5deef', '#276287', '#253b51', '#bc611f'
    ax.bxp(
        stats, positions=[y for y, _ in normal], vert=False, widths=.46,
        patch_artist=True, manage_ticks=False,
        boxprops={'facecolor': box_color, 'edgecolor': line_color},
        whiskerprops={'color': line_color}, capprops={'color': line_color},
        medianprops={'color': median_color, 'linewidth': 1.7},
        flierprops={'marker': '.', 'markersize': 3, 'markeredgewidth': 0,
                    'markerfacecolor': flier_color, 'alpha': .65},
    )
    for y in positions[1::2]:
        ax.axhspan(y - .5, y + .5, color='#f4f7fa', zorder=0)
    ax.set_yticks(positions, [row['feature'] for row in features])
    ax.set_ylim(.5, len(features) + .5)
    ax.set_xlim(
        min(row['min'] for row in features) - 4,
        max(row['max'] for row in features) + 4,
    )
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#dce5ec', linewidth=.8)
    ax.tick_params(axis='both', length=0, pad=8, colors=median_color)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#a3b3c1')

    title, font = _chart_text('全量匿名特征 IQR 箱线图', 'Full-source feature IQR box plots')
    fig.text(.09, .955, title, fontsize=19, color=median_color, **font)
    subtitle, font = _chart_text(
        f'Criteo Uplift v2.1 · {summary["n"]:,} 行 · 原始数值，共用横轴',
        f'Criteo Uplift v2.1 · {summary["n"]:,} rows · shared raw-value axis',
    )
    fig.text(.09, .906, subtitle, fontsize=11, color='#526679', **font)
    label, font = _chart_text('实际最小值 – 最大值', 'Observed min – max')
    ax.text(1.035, 1.015, label, transform=ax.transAxes, fontsize=10, color='#526679', **font)
    for y, row in zip(positions, features):
        if row['iqr_status'] == 'degenerate':
            ax.vlines(row['concentration_value'], y - .23, y + .23,
                      color=median_color, linewidth=1.7)
            left = row['below_concentration_fraction']
            right = row['above_concentration_fraction']
            directions = [
                (zh, en) for fraction, zh, en in (
                    (left, f'左侧 {left:.2%}', f'left {left:.2%}'),
                    (right, f'右侧 {right:.2%}', f'right {right:.2%}'),
                ) if fraction
            ] or [('无偏离', 'no deviations')]
            zh, en = map(' / '.join, zip(*directions))
            note, note_font = _chart_text(
                f'退化 IQR · 偏离集中值（{zh}）',
                f'Degenerate IQR · deviation ({en})',
            )
            ax.text(.98, y, note, transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=9, color='#526679', **note_font)
        ax.text(1.035, y, f'{row["min"]: .3f}   –   {row["max"]: .3f}',
                transform=ax.get_yaxis_transform(), va='center',
                fontfamily='DejaVu Sans Mono', fontsize=9, color=median_color)

    handles = [
        Patch(facecolor=box_color, edgecolor=line_color),
        Line2D([], [], color=median_color, linewidth=1.7),
        Line2D([], [], color=line_color, marker='|'),
        Line2D([], [], color=flier_color, marker='.', linestyle='none'),
    ]
    labels = [
        _chart_text('Q1–Q3 箱体', 'Q1–Q3 box')[0],
        _chart_text('中位数／集中值', 'Median / concentration')[0],
        _chart_text('1.5×IQR 须线', '1.5×IQR whiskers')[0],
        _chart_text('IQR 异常值位置', 'IQR outlier locations')[0],
    ]
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.075, .025),
               ncol=4, frameon=False, prop=font.get('fontproperties'))

    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
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
    """Wrap display TeX in a keyboard-scrollable container."""
    return (
        f'<div class="equation"><div class="equation-label">{html.escape(label)}</div>'
        f'<div class="equation-scroll" tabindex="0" role="group" aria-label="{html.escape(label)}">'
        rf'<div class="math-equation">\[{html.escape(expression)}\]</div></div></div>'
    )


def _policy_comparison(summary, selected, criterion, unit, n):
    summary = summary.sort_values("point_estimate", ascending=False, kind="stable")
    values = summary[["point_estimate", "ci_low", "ci_high"]].to_numpy() * 100_000
    finite = values[np.isfinite(values)]
    low, high = finite.min(initial=0), finite.max(initial=0)
    step = 10 ** np.floor(np.log10((high - low) / 4)) if high > low else 1
    low, high = np.floor(low / step) * step, np.ceil(high / step) * step
    if high == low:
        high = low + step

    random_values = values[summary.policy.eq("random")]
    model_values = values[summary.policy.ne("random")]
    broken = False
    if random_values.size and model_values.size and np.isfinite(values).all():
        gap_low = np.ceil(random_values.max() / 5) * 5
        gap_high = np.floor(model_values.min() / 5) * 5
        broken = gap_high - gap_low >= 20

    def position(value):
        if broken:
            if value <= gap_low:
                return 12 * (value - low) / (gap_low - low)
            return 18 + 82 * (value - gap_high) / (high - gap_high)
        return 100 * (value - low) / (high - low)

    ticks = [low, (low + high) / 2, high]
    if broken:
        ticks = [low, gap_low, gap_high, (gap_high + high) / 2, high]
    axis = "".join(
        f'<span style="left:{position(tick):.6f}%">{tick:g}</span>' for tick in ticks
    )
    break_mark = '<span class="policy-break">∕∕</span>' if broken else ""
    scale_note = (f'横轴省略 {gap_low:g}–{gap_high:g} 的空白区间，断轴两侧分别为线性刻度。'
                  if broken else "")

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
            f'<div class="policy-track" aria-hidden="true" style="--zero:{position(0):.6f}%">{break_mark}{interval}{marker}</div>'
            f'<span class="policy-value">{_num(point)} <small>{_interval(row._asdict(), scale=100_000)}</small></span></div>'
        )
    return (
        '<figure id="primary-policy-chart" class="policy-chart">'
        f'<h3>策略比较 · {html.escape(criterion)}</h3><p class="chart-unit">{html.escape(unit)}</p>'
        + "".join(rows)
        + '<div class="policy-axis-row"><div></div>'
        f'<div class="policy-axis" aria-hidden="true">{axis}{break_mark}</div><div></div></div>'
        f'<figcaption><strong>独立 Test Set（{n:,} 行）</strong> · 圆点为点估计，横线为各策略的 95% Bootstrap 区间。'
        f'{scale_note}策略差异依据配对区间判断。</figcaption></figure>'
    )

def _paired_gain_card(response_name, uplift_name, response_value, uplift_value, pair):
    """显示两种策略的业务收益与配对置信区间。"""
    difference = float(pair.difference) * 100_000
    low = float(pair.ci_low) * 100_000
    high = float(pair.ci_high) * 100_000
    radius = max(abs(low), abs(high), 1.0)
    left = 50 + 44 * low / radius
    right = 50 + 44 * high / radius
    point = 50 + 44 * difference / radius
    note = ('配对区间跨越零点。本阶段沿用 Validation Set 冻结策略，后续以线上实验检验挑战策略的业务价值。'
            if low <= 0 <= high else
            '配对区间位于零点一侧。策略切换仍遵循预设选型规则及后续线上验证。')
    return (
        '<figure class="paired-gain-card">'
        '<figcaption>固定 20% 覆盖率 · HT Gain（每 10 万名全部合资格用户）</figcaption>'
        '<div class="paired-gain-values">'
        f'<div><span>{html.escape(response_name)} · Response 基线</span><strong>{response_value:.2f}</strong></div>'
        f'<div><span>{html.escape(uplift_name)} · 最佳 Uplift 候选</span><strong>{uplift_value:.2f}</strong></div>'
        '</div>'
        '<div class="paired-delta"><b>配对收益差（Uplift − Response）</b>'
        f'<strong>{difference:+.2f}</strong><span>95% CI [{low:.2f}, {high:.2f}]</span></div>'
        '<div class="paired-track" aria-hidden="true"><span class="paired-zero"></span>'
        f'<span class="paired-ci" style="left:{left:.3f}%;width:{right-left:.3f}%"></span>'
        f'<span class="paired-point" style="left:{point:.3f}%"></span></div>'
        '<div class="paired-scale"><span>Uplift 较低</span><span>零差异</span><span>Uplift 较高</span></div>'
        f'<p class="paired-note">{note}</p>'
        '</figure>'
    )

def _paragraph(text, css=''):
    return f'<p class="{css}">{text}</p>'


def _details(anchor, title, content):
    return f'<details id="{anchor}" class="appendix-topic"><summary>{title}</summary><div class="appendix-content">{content}</div></details>'


def _section(anchor, title, content):
    return f'<section id="{anchor}"><h2>{title}</h2>{content}</section>'


def _json(value):
    def portable(item):
        if isinstance(item, dict): return {k: portable(v) for k, v in item.items()}
        if isinstance(item, list): return [portable(v) for v in item]
        if isinstance(item, str) and item.startswith('/'):
            parts = Path(item).parts
            return '/'.join(parts[parts.index('artifacts'):]) if 'artifacts' in parts else Path(item).name
        return item
    value = portable(value)
    return '<pre>' + html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>'


def _gain_plot(curves, names, *, columns=('gain_ht',)):
    """在内存中绘图，报告重建不改写历史运行目录。"""
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter
    fig = Figure(figsize=(10, 3.8 * len(columns)), constrained_layout=True)
    axes = fig.subplots(len(columns), 1, squeeze=False)[:, 0]
    labels = {'gain_ht': 'HT Gain / 10 万名合资格用户', 'gain_plugin': 'Uplift Gain × 10⁵',
              'qini_traditional': 'Traditional Qini Gain × 10⁵'}
    x_label, font = _chart_text('候选用户覆盖率', 'Eligible population coverage')
    for ax, column in zip(axes, columns):
        for name in names:
            rows = curves[curves.policy.eq(name)].sort_values('fraction')
            ax.plot(rows.fraction, rows[column].where(rows.fraction.ne(0), 0) * 1e5,
                    color=_COLORS[name], label=_LABELS[name], lw=2)
        first = curves[curves.policy.eq(names[0])].sort_values('fraction')
        ax.plot([0, 1], [0, first[column].iloc[-1] * 1e5], '--', color=_COLORS['random'], label='Random')
        ax.axvline(.2, color='#a1afba', ls=':', lw=1)
        ax.set_xlabel(x_label, **font)
        y_label, y_font = _chart_text(labels[column], 'HT Gain per 100,000 eligible users' if column == 'gain_ht' else labels[column])
        ax.set_ylabel(y_label, **y_font)
        ax.set_xticks([.02, .05, .1, .2, .3, .5, 1.])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, ncol=3, frameon=False)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150)
    return '<img alt="覆盖率与累计策略增量" src="data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode() + '">'


def _c2st_plot(null, scores):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 3.3), constrained_layout=True)
    axes = fig.subplots(1, len(scores), squeeze=False)[0]
    x_label, x_font = _chart_text('诊断留出集 ROC AUC', 'Diagnostic holdout ROC AUC')
    y_label, y_font = _chart_text('置换次数', 'Permutation count')
    for ax, row in zip(axes, scores.itertuples()):
        ax.hist(null.loc[null.model.eq(row.model), 'roc_auc'], bins=24, color='#497da4')
        ax.axvline(row.roc_auc, color='#ba6b24')
        ax.set_title(f'{row.model} · AUC {row.roc_auc:.4f}')
        ax.set_xlabel(x_label, **x_font)
        ax.set_ylabel(y_label, **y_font)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150)
    return '<img alt="C2ST 置换分布与观测值" src="data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode() + '">'


def _metric_table(metrics, bootstrap=None):
    if bootstrap is None:
        return _table(pd.DataFrame({
            '策略': metrics.policy.map(_LABELS), '覆盖率': metrics.budget.map(lambda v: f'{v:.0%}'),
            '选中人数': metrics.n_selected, 'HT Gain × 10⁵': (metrics.gain_ht * 1e5).map(_num),
            'Uplift Gain × 10⁵': (metrics.gain_plugin * 1e5).map(_num),
            'AUUC × 10⁵': (metrics.auuc * 1e5).map(_num),
            'Qini Score × 10⁵': (metrics.qini_score * 1e5).map(_num),
        }), wide=True)
    joined = metrics.merge(bootstrap, on=['policy', 'budget'], validate='one_to_one', suffixes=('', '_bootstrap'))
    return _table(pd.DataFrame({
        '策略': joined.policy.map(_LABELS), '覆盖率': joined.budget.map(lambda v: f'{v:.0%}'),
        '选中人数': joined.n_selected, 'HT Gain × 10⁵': (joined.gain_ht * 1e5).map(_num),
        '95% CI': joined.apply(lambda r: _interval(r, scale=1e5), axis=1),
        'Bootstrap 均值 × 10⁵': (joined.bootstrap_mean * 1e5).map(_num),
        '有效重复': joined.valid_replicates,
        'Uplift Gain × 10⁵': (joined.gain_plugin * 1e5).map(_num),
        'Traditional Qini Gain × 10⁵': (joined.qini_traditional * 1e5).map(_num),
        'AUUC × 10⁵': (joined.auuc * 1e5).map(_num),
        'Qini Score × 10⁵': (joined.qini_score * 1e5).map(_num),
    }), wide=True)


def _pair_table(frame, *, compact=False):
    data = {
        '策略': frame.policy.map(_LABELS), '参考策略': frame.reference.map(_LABELS),
        '差值 / 10 万人': frame.difference.map(lambda x: f'{x * 1e5:+.2f}'),
        '配对 95% CI': frame.apply(lambda r: _interval(r, scale=1e5), axis=1),
    }
    if not compact:
        data['覆盖率'] = frame.budget.map(lambda v: f'{v:.0%}')
        data['Bootstrap 差值均值 × 10⁵'] = (frame.bootstrap_mean_difference * 1e5).map(_num)
        data['有效配对'] = frame.valid_replicates
    return _table(pd.DataFrame(data), wide=not compact)


def _gain_comparison(primary):
    ranked = primary[['gain_ht', 'gain_plugin']].sort_values('gain_ht', ascending=False)
    return pd.DataFrame({
        '策略': ranked.index.map(_LABELS),
        'HT Gain': (ranked.gain_ht * 1e5).map(_num),
        'Uplift Gain': (ranked.gain_plugin * 1e5).map(_num),
        'HT 排名': ranked.gain_ht.rank(ascending=False, method='min').astype(int),
        'Uplift 排名': ranked.gain_plugin.rank(ascending=False, method='min').astype(int),
    })


def _dataset_audit_section(out, manifest, audit):
    """Render chapter 02 from the saved full-run audit and reviewed figures."""
    iqr = json.loads((out / 'diagnostics/full_feature_iqr.json').read_text())
    n = audit['n_rows']
    if iqr['n'] != n:
        raise ValueError('02 全量 IQR 行数与保存的审计结果不一致')
    features = {row['feature']: row for row in iqr['features']}
    degenerate = sum(row['iqr_status'] == 'degenerate' for row in features.values())
    treatment, control = (audit['groups'][arm] for arm in ('1', '0'))
    conversion_t, conversion_c = (group['conversion'] for group in (treatment, control))
    visit_t, visit_c = (group['visit'] for group in (treatment, control))
    conversions = conversion_t['positive'] + conversion_c['positive']
    visits = visit_t['positive'] + visit_c['positive']
    duplicates = audit['duplicate_row_fingerprints']
    duplicate_fraction = audit['duplicate_row_fraction']

    schema = _table(pd.DataFrame([
        ('处理前协变量 X', 'f0–f11', '12 个匿名数值型用户特征'),
        ('处理分配 T', 'treatment', '广告实验分配标识（1 = Treatment，0 = Control）'),
        ('观测结果 Y', 'conversion', '是否发生转化，项目主要分析目标（CVR）'),
        ('辅助行为变量', 'visit', '是否访问广告主网站，用于辅助行为分析'),
    ], columns=['数据类别', '字段', '定义']))
    audit_table = _table(pd.DataFrame([
        ('样本量（占比）', f'{treatment["n"]:,}（{treatment["n"] / n:.0%}）',
         f'{control["n"]:,}（{control["n"] / n:.0%}）', f'{n:,}'),
        ('Conversion 正例数', f'{conversion_t["positive"]:,}', f'{conversion_c["positive"]:,}', f'{conversions:,}'),
        ('Conversion 正例率（CVR）', f'{conversion_t["rate"]:.4%}', f'{conversion_c["rate"]:.4%}', f'{conversions / n:.4%}'),
        ('Visit 正例数', f'{visit_t["positive"]:,}', f'{visit_c["positive"]:,}', f'{visits:,}'),
        ('Visit 正例率', f'{visit_t["rate"]:.4%}', f'{visit_c["rate"]:.4%}', f'{visits / n:.4%}'),
        ('缺失值 / 非有限值', '—', '—', f'{audit["missing_or_nonfinite"]:,}'),
        ('非法二元标签', '—', '—', f'{audit["invalid_binary_labels"]:,}'),
        ('Conversion=1 但 Visit=0', '—', '—', f'{audit["conversion_without_visit"]:,}'),
        ('完整16字段行内重复的多余记录', '—', '—', f'{duplicates:,}（{duplicate_fraction:.2%}）'),
    ], columns=['审计项目', 'Treatment', 'Control', '全量结果']))
    return _section('dataset-audit', '02｜数据集与数据审计',
        _paragraph('本章围绕 Criteo Uplift v2.1 公开广告实验数据，完成数据集结构说明、全量数据质量审计及匿名特征分布分析。通过全量数据扫描，确认处理前协变量、处理分配和观测结果的完整性，评估处理组与对照组的样本规模及转化事件分布，并结合匿名特征的分布形态和取值范围，为后续因果识别与 Uplift 建模提供数据基础。')
        + '<h3>2.1 数据集结构</h3>'
        + _paragraph('数据来源为 Criteo AI Lab 发布的 <a href="https://ailab.criteo.com/criteo-uplift-prediction-dataset/">Criteo Uplift Prediction Dataset</a>。'
                     f'本项目使用 Criteo Uplift v2.1，原始数据包含 {n:,} 条记录、{audit["n_columns"]} 个字段。')
        + schema
        + _paragraph('原始数据还包含 exposure 字段，记录实际广告曝光情况。本项目以 treatment 定义处理分配，不使用实际曝光重新定义处理组，也不将 exposure 作为处理前协变量或主要分析结果。')
        + _paragraph('项目以 conversion 作为主要观测结果，围绕广告资格分配对转化率（CVR）的增量影响开展研究。')
        + '<h3>2.2 全量数据审计</h3>'
        + _paragraph('本项目采用流式分批读取方式，对原始 CSV 进行全量数据审计。Pipeline 按每批 250,000 行读取数据，逐批完成字段结构、数值有效性及标签合法性检查，并累计全量审计结果。')
        + _paragraph('全量审计结果如下。')
        + '<h4>表 2-1｜全量数据审计结果</h4>' + audit_table
        + '<h4>样本规模与转化分布</h4>'
        + _paragraph(f'处理组与对照组占比分别为 {treatment["n"] / n:.0%} 和 {control["n"] / n:.0%}，样本规模存在明显差异。')
        + _paragraph(f'从主要业务指标 CVR 来看，处理组产生 {conversion_t["positive"]:,} 个转化正例，对照组产生 {conversion_c["positive"]:,} 个转化正例，全体样本共包含 {conversions:,} 个转化事件。')
        + _paragraph('虽然 Conversion 属于稀疏结果，但全量样本仍提供了数万级正例事件，对照组也保有数千个正例，为后续二元转化模型拟合及处理组、对照组响应比较提供了可用的样本基础。')
        + _paragraph(f'处理组 CVR 为 {conversion_t["rate"]:.4%}，高于对照组的 {conversion_c["rate"]:.4%}。这一差异反映了原始样本中的组间转化表现，为后续增量效果识别提供分析基础，实际因果效果仍需结合后续实验诊断及模型评价确定。')
        + _paragraph('Visit 用于辅助观察用户访问行为，项目主要业务优化目标仍然是 Conversion。')
        + '<h4>数据完整性与重复记录</h4>'
        + _paragraph('全量数据未发现缺失值、非有限值、非法二元标签及 Conversion 与 Visit 的逻辑异常，说明原始数据具备较好的字段完整性和标签有效性。')
        + _paragraph('重复记录审计基于完整 16 个字段生成行指纹，统计内容完全重复记录产生的多余行数。同一完整行指纹出现 k 次，计入 k-1 条多余记录。')
        + _paragraph(f'全量识别出 {duplicates:,} 条多余重复行，占原始记录的 {duplicate_fraction:.2%}。')
        + _paragraph('由于匿名特征缺少用户级唯一标识，无法确认这些记录究竟来自同一用户的真实重复记录，还是不同用户恰好具有相同的 16 字段取值。因此，本项目保留原始记录，不根据完整行重复直接删除样本。')
        + '<h3>2.3 匿名特征分布与取值范围</h3>'
        + _paragraph('数据集包含 12 个匿名数值型处理前协变量。本节基于全量原始数据展示特征的四分位分布、实际取值范围及统计异常值，并结合原有 EDA 图分析各特征的分布形态。')
        + _image(out / 'figures/full_feature_iqr.png', '图 2-1｜全量匿名特征 IQR 箱线图',
                 f'数据来源：原始 CSV 全量 {n:,} 行。橙点表示正常 IQR 的异常值位置，退化 IQR 行标注偏离集中值的方向和比例。')
        + _paragraph(f'从全量统计结果来看，12 个特征中有 {degenerate} 个呈现退化 IQR，即 Q1、中位数与 Q3 均集中于同一取值，说明其中央分布具有明显的单点集中现象。其余 {len(features) - degenerate} 个特征具有非零 IQR，可以使用 1.5×IQR 规则识别潜在统计异常值。')
        + _paragraph(f'其中，f6 和 f8 分别识别出 {features["f6"]["outlier_count"]:,} 条和 {features["f8"]["outlier_count"]:,} 条 IQR 异常值，占全量样本的 {features["f6"]["outlier_fraction"]:.4%} 和 {features["f8"]["outlier_fraction"]:.4%}；f0 和 f2 未发现超出 IQR 边界的观测。')
        + _paragraph('对于退化 IQR 特征，本项目统计其偏离集中值的观测数量及比例，不将其直接认定为异常值。该处理保留了特征的实际分布信息，避免集中取值导致的异常值误判。')
        + _image(out / 'figures/sample_features.png', '图 2-2｜12 个匿名特征 EDA 分布图',
                 f'数据来源：本次正式运行的建模样本，共 {manifest["sample_n"]:,} 行。')
        + _paragraph('结合 EDA 分布图，各特征表现出不同的分布形态：f0 呈现多峰及较宽分布；f1、f4、f7、f9、f10 具有明显的右侧延伸；f3、f5、f11 主要表现为左侧延伸；f6 具有较宽的连续分布，而 f8 高度集中于较窄的取值区间。')
        + _paragraph('总体来看，12 个匿名特征存在明显的集中取值、偏态及长尾现象，特征间的分布结构具有较大差异。这些统计特征为后续模型拟合与比较提供了数据分布依据，也为分析模型对不同用户特征区间的响应差异提供了基础。由于特征缺少公开业务语义，当前分析主要限定于统计分布层面，具体业务含义需要结合后续模型结果进一步解释。')
        + _details('dataset-audit-engineering', '工程实现｜全量数据审计与报告组装',
            _paragraph('<code>run_pipeline.py::run()</code> → <code>criteo_uplift/preparation.py::prepare_sample()</code> → <code>criteo_uplift/raw_scan.py::scan_raw_data()</code>。扫描逐批调用 <code>validate_chunk()</code>，由 <code>AuditAccumulator</code> 和 <code>DiagnosticsAccumulator</code> 累计全量审计与特征统计。')
            + _paragraph('<code>criteo_uplift/reporting.py::build_report()</code> 调用 <code>_dataset_audit_section()</code>，读取保存的全量审计、IQR 统计和图表，组装正式 HTML 的第 02 章。')))


def _replace_dataset_audit(document, section):
    previous = re.compile(r'<section id="(?:dataset-audit|feature-eda)">.*?</section>', re.DOTALL)
    document, count = previous.subn(lambda _: section, document)
    if count == 1:
        return document
    anchor = '<section id="identification">'
    if count or document.count(anchor) != 1:
        raise ValueError('无法唯一定位正式报告的第 02 章')
    return document.replace(anchor, section + anchor, 1)



def _identification_section(audit, srm, balance, c2st, c2st_null, positivity, learnability):
    labels = {'dummy': 'Dummy', 'logistic': 'Logistic Regression', 'xgboost': 'XGBoost'}
    scopes = {'whole': '整体', 'treatment': '处理组', 'control': '对照组'}
    conversion_t = audit['groups']['1']['conversion']['positive']
    conversion_c = audit['groups']['0']['conversion']['positive']
    smd = balance.loc[balance.abs_smd.idxmax()]
    ks = balance.loc[balance.ks_statistic.idxmax()]
    permutations = int(c2st.permutations.iloc[0])
    arm_n = int(c2st.sample_n.iloc[0]) // 2
    c2st_rows = [('Dummy', '0.5000', '—', '—', '—')] + [
        (labels[r.model], f'{r.roc_auc:.4f}', f'{r.null_auc_mean:.6f}',
         f'{r.pvalue:.3f}', f'{r.pvalue_holm:.3f}') for r in c2st.itertuples()
    ]
    prediction_tables = []
    for outcome in ('conversion', 'visit'):
        rows = []
        for scope, label in scopes.items():
            scores = learnability.loc[
                learnability.outcome.eq(outcome) & learnability.scope.eq(scope)
            ].set_index('model').reindex(labels)
            rows.append((label, *(
                ' / '.join(scores[col].map(lambda value: _num(value, digits)))
                for col, digits in [('roc_auc', 3), ('log_loss', 6), ('brier', 6)]
            )))
        prediction_tables.append(_table(pd.DataFrame(rows, columns=[
            '范围', 'ROC AUC ↑', 'LogLoss ↓', 'Brier ↓'])))
    weak_logistic_signal = c2st.set_index('model').pvalue_holm.lt(.05).to_dict() == {'logistic': True, 'xgboost': False}
    interpretation = (
        '在从全量数据抽取的等臂样本上，Logistic Regression 和 XGBoost 的观测 AUC 均接近 0.5。'
        'Holm 校正后，Logistic Regression 检出弱的处理分配区分信号，XGBoost 在本次设置下未检出显著信号。'
        if weak_logistic_signal
        else '结合观测 AUC 与 Holm 校正后的 p 值，分别判断区分幅度与统计证据。'
    )

    return _section('identification', '03｜数据因果识别诊断',
        _paragraph('第 01 章将研究目标定义为广告资格分配对转化的增量影响，第 02 章审计了公开样本的字段、分组规模与转化事件。'
                   '本章核对公开样本的分配计数——SRM，检查处理前协变量平衡——SMD、KS 和 C2ST，'
                   '评估两臂覆盖——Overlap / Positivity，以及处理前特征对结果变量的预测能力——Learnability。')
        + '<h3>3.1 公开样本分配比例核对（SRM）</h3>'
        + _paragraph(r'本项目以公开资料中的 85% / 15% 为外部参考比例，对全量公开样本的两臂人数进行 Pearson 卡方拟合优度检验。'
                     r'原假设为 \(H_0:P(T=1)=0.85\)，备择假设为 \(H_1:P(T=1)\ne0.85\)。'
                     '<a href="https://arxiv.org/html/2111.10106#S3">公开数据的分配比例说明</a>。')
        + _table(pd.DataFrame([
            ('总样本量', f'{srm["n_treatment"] + srm["n_control"]:,}'),
            ('处理组人数', f'{srm["n_treatment"]:,}'), ('对照组人数', f'{srm["n_control"]:,}'),
            ('实际处理组占比', f'{srm["observed_ratio"]:.6%}'),
            ('参考处理组占比', f'{srm["expected_ratio"]:.0%}'),
            ('Pearson χ²', f'{srm["chi2"]:.9f}'), ('自由度', 1), ('p 值', f'{srm["pvalue"]:.6f}'),
        ], columns=['指标', '结果']))
        + _paragraph(('公开样本的整体分配计数未显示相对参考比例的显著偏离。'
                      if srm['pvalue'] >= .05 else '公开样本的整体分配计数显著偏离参考比例。')
                     + '原始线上实验的分流机制需要结合实验设计判断。')
        + '<h3>3.2 协变量平衡</h3><h4>3.2.1 标准化均值差（SMD）</h4>'
        + _paragraph('在全量公开样本上，将各特征的两组均值之差除以两组方差平均值的平方根。')
        + _equation(r'\mathrm{SMD}_j=\frac{\bar X_{1j}-\bar X_{0j}}{\sqrt{(s_{1j}^2+s_{0j}^2)/2}}', '标准化均值差')
        + _paragraph(f'{len(balance)} 个处理前特征中，最大绝对 SMD 为 <strong>{smd.abs_smd:.6f}</strong>，对应 {smd.feature}。')
        + '<h4>3.2.2 Kolmogorov–Smirnov 检验（KS）</h4>'
        + _paragraph('若处理分配与处理前特征独立，各特征在两臂中的边际分布应相同。KS 对全量公开样本逐特征比较两组经验累积分布函数。')
        + _equation(r'D_j=\sup_x\left|\widehat F_{1j}(x)-\widehat F_{0j}(x)\right|', '经验累积分布的最大距离')
        + _paragraph(f'最大 KS 统计量为 <strong>{ks.ks_statistic:.6f}</strong>，出现在 {ks.feature}，'
                     f'对应约 {100 * ks.ks_statistic:.2f} 个百分点的经验累计比例差。双侧渐近检验中，'
                     f'{len(balance)} 个特征有 {int(balance.ks_pvalue.lt(.05).sum())} 个未经多重校正的 p 值低于 0.05。'
                     '全量样本使检验对细小差异也很敏感，因此以 KS 统计量描述差异幅度，将未校正的显著性结果作为探索性信息。')
        + '<h4>3.2.3 分类器双样本检验（C2ST）</h4>'
        + _paragraph(r'C2ST 检查处理前特征能否为预测处理分配提供信息，原假设为 \(P(T\mid X)=P(T)\)。'
                     f'本项目从全量公开样本中按处理臂分别等概率无放回抽样，每臂 {arm_n:,} 条。'
                     '用 Logistic Regression 和 XGBoost 预测处理标签，在 C2ST 内部留出集上评价 ROC AUC。'
                     '常数基线对所有留出样本给出相同分数，ROC AUC 为 0.5。')
        + _paragraph(f'在固定的拟合和留出子集内分别打乱处理标签，保持各自的两臂人数，每次重新拟合分类器。重复 {permutations} 次得到置换零分布，'
                     '以观测 AUC 的单侧上尾计算经验 p 值，再对两个分类器作 Holm 校正。该方案依赖子集内标签可交换。'
                     '若原实验采用区组或分层分配，设计层面的随机化检验需要保留相应的分配约束。')
        + _paragraph(r'记 \(A_{\mathrm{obs}}\) 为真实标签下的留出 AUC，\(A_b^{\mathrm{perm}}\) 为第 \(b\) 次置换的留出 AUC，'
                     rf'置换次数 \(B={permutations}\)。')
        + _equation(r'\widehat p=\frac{1+\sum_{b=1}^{B}\mathbf{1}\{A_b^{\mathrm{perm}}\ge A_{\mathrm{obs}}\}}{B+1}', '单侧经验 p 值')
        + _table(pd.DataFrame(c2st_rows, columns=['分类器', '留出 ROC AUC', '置换 AUC 均值', '经验 p 值', 'Holm 校正 p 值']))
        + '<figure>' + _c2st_plot(c2st_null, c2st)
        + '<figcaption>图 3-1｜等臂样本上的 C2ST 置换 AUC 分布。竖线标示真实处理标签下的观测 AUC。</figcaption></figure>'
        + '<h4>协变量平衡综合判断</h4>'
        + _paragraph(('<strong>Criteo 公开样本的处理前协变量总体平衡较好。</strong>'
                      if weak_logistic_signal and smd.abs_smd < .10 and ks.ks_statistic < .02 else '')
                     + 'Criteo 公开数据来源于随机增量试验。下列诊断评估公开样本中两臂的协变量差异幅度。')
        + _paragraph(f'在全量公开样本中，最大绝对 SMD 为 {smd.abs_smd:.4f}'
                     + ('，低于常用的 0.10 经验参考线，两臂的特征均值差异较小。'
                        if smd.abs_smd < .10 else '，达到或超过常用的 0.10 经验参考线。')
                     + f'KS 检验有 {int(balance.ks_pvalue.lt(.05).sum())}/{len(balance)} 个特征的未校正 p 值低于 0.05。'
                     '大样本下，微小的分布偏差也容易达到统计显著，因此以 KS 统计量衡量差异幅度。'
                     f'最大 KS 统计量为 {ks.ks_statistic:.4f}，对应两臂经验累计比例最多相差约 {100 * ks.ks_statistic:.2f} 个百分点。')
        + _paragraph(interpretation)
        + _details('c2st-method', 'C2ST 工程实现',
            '<ul><li><strong>抽样与划分。</strong> 从全量公开数据按处理臂等量抽样，取 12 个处理前特征，按处理标签分层划分 70% 拟合和 30% 留出。</li>'
            '<li><strong>拟合与置换。</strong> 固定划分，先计算真实标签下的留出 AUC，再分别置换拟合与留出标签，每轮重新拟合并记录 AUC。</li>'
            '<li><strong>保存结果。</strong> 汇总观测 AUC、置换 AUC、经验 p 值与 Holm 校正结果。</li></ul>'
            + _paragraph('<code>run_pipeline.py</code> 将 <code>criteo_uplift/raw_scan.py</code> 的 <code>scan_raw_data</code> 输出'
                         '传给 <code>criteo_uplift/diagnostics.py</code> 的 <code>run_diagnostics</code>，后者调用 <code>c2st</code> 完成检验。'))
        + '<h3>3.3 单变量覆盖与正值性（Overlap / Positivity）</h3>'
        + _paragraph(r'正值性要求目标特征区域满足 \(0\lt P(T=1\mid X=x)\lt 1\)。'
                     '项目使用 Train Set 逐特征检查经验覆盖，每个特征最多划分 10 个分位数箱，合并重复切点。任一处理臂少于 20 人时标记为覆盖薄弱。')
        + _table(pd.DataFrame([
            ('实际分箱数', positivity['strata_n']), ('空分箱数', positivity['empty_bins']),
            ('缺少任一处理臂的分箱数', positivity['missing_arm_bins']), ('覆盖薄弱的分箱数', positivity['weak_bins']),
            ('非空箱中较少一臂的最小样本数', f'{positivity["min_nonempty_arm_count"]:,}'),
        ], columns=['指标', '结果']))
        + _paragraph(('所有单变量分箱均有两臂样本，为比较这些特征区间内的两臂响应提供了样本基础。'
                      if not (positivity['empty_bins'] or positivity['missing_arm_bins'])
                      else '分箱计数用于定位两臂样本覆盖不足的特征区间。')
                     + '高维联合支持和设计层面的条件分配概率仍待核实。')
        + '<h3>3.4 结果变量可学习性</h3>'
        + _paragraph(f'第 02 章的全量审计记录了处理组 {conversion_t:,} 个转化正例、对照组 {conversion_c:,} 个转化正例。'
                     '本节进一步检查 12 个处理前特征对主要结果 Conversion 和辅助结果 Visit 的预测能力。'
                     '分别在整体与两臂训练样本上拟合，并在对应 Validation Set 评价。Dummy 使用对应训练样本的正例率作为固定预测概率。'
                     'ROC AUC 衡量排序区分能力，LogLoss 和 Brier Score 衡量概率预测误差。'
                     '各单元格按 Dummy / Logistic Regression / XGBoost 顺序列出。')
        + '<h4>表 3-1｜Conversion 在 Validation Set 上的结果预测表现</h4>' + prediction_tables[0]
        + '<h4>表 3-2｜Visit 在 Validation Set 上的结果预测表现</h4>' + prediction_tables[1]
        + _paragraph('转化预测在两臂均优于常数基线时，分别拟合两条响应面具备经验支持。Visit 补充展示处理前特征对访问行为的预测表现。'
                     'Uplift 排序和策略增益由后续章节的留出评价衡量。')
        + '<h3>3.5 因果识别依据与诊断边界</h3>'
        + _paragraph('广告资格分配效应的识别依据来自原始增量实验的随机分配设计。公开样本诊断说明分配计数、可观测特征和样本覆盖的实际表现。')
        + _table(pd.DataFrame([
            ('一致性（Consistency）', r'按第 01 章的处理定义，\(Y_i=Y_i(T_i)\)。潜在结果与观测结果采用同一广告资格分配定义，Conversion 是主要观测结果。'),
            ('可交换性', r'原实验的随机分配提供设计依据。SRM 核对整体分配计数，SMD、KS 和 C2ST 提供 \(X\) 与 \(T\) 关系的观测证据。潜在结果的可交换性依赖实验设计。'),
            ('正值性（Positivity）', '单变量分箱诊断提供经验覆盖证据。设计分配概率和高维联合支持仍待核实。'),
            ('稳定单元处理值假设（SUTVA）', '资格分配效应的解释依赖处理版本定义与用户之间无干扰。处理版本及用户间干扰仍需结合投放机制判断。'),
        ], columns=['识别条件', '对应依据与适用范围']))
        + _paragraph('后续总体效应分析以原实验的随机分配设计为依据，估计公开样本中广告资格分配对 Conversion 的平均效应。'
                     'Uplift 建模基于两臂中观测到的转化预测信号拟合响应面，并通过留出样本评价增量定向策略。'))



def _uplift_methods_section(assignment_probability):
    notation = [
        (r'\(\mu(x,t)\)', r'\(\mathbb E[Y\mid X=x,T=t]\)，共享响应模型的估计目标'),
        (r'\(\mu_0(x)\)', r'\(\mathbb E[Y(0)\mid X=x]\)，未分配资格时的转化响应面'),
        (r'\(\mu_1(x)\)', r'\(\mathbb E[Y(1)\mid X=x]\)，分配资格时的转化响应面'),
        (r'\(\tau(x)\)', r'\(\mu_1(x)-\mu_0(x)\)，条件平均处理效应（CATE）'),
        (r'\(e(x)\)', r'\(P(T=1\mid X=x)\)，给定处理前特征的资格分配概率'),
        (r'\(\widehat e_{\mathrm{train}}\)', r'\(\frac{1}{n_{\mathrm{train}}}\sum_{i\in\mathrm{Train}}T_i\)，恒定分配概率近似下使用的 Train Set 处理比例'),
        (r'\(\widetilde D_i^{\,0}\)', r'\(\widehat\mu_1(X_i)-Y_i\)，在对照组构造的伪效应标签'),
        (r'\(\widetilde D_i^{\,1}\)', r'\(Y_i-\widehat\mu_0(X_i)\)，在处理组构造的伪效应标签'),
        (r'\(\widehat D_t(x)\)', r'在 \(T=t\) 组以 \(\widetilde D^{\,t}\) 为标签、以 \(X\) 为输入拟合得到的条件均值预测'),
    ]
    scores = [
        ('S-Learner', r'\(\widehat\mu(X_i,1)-\widehat\mu(X_i,0)\)'),
        ('T-Learner', r'\(\widehat\mu_1(X_i)-\widehat\mu_0(X_i)\)'),
        ('X-Learner', r'\(\widehat e_{\mathrm{train}}\widehat D_0(X_i)+(1-\widehat e_{\mathrm{train}})\widehat D_1(X_i)\)'),
        ('Response', r'\(\widehat\mu_1(X_i)\)'),
        ('Random', r'\(U_i,\quad U_i\stackrel{\mathrm{iid}}{\sim}\operatorname{Unif}(0,1)\)'),
    ]
    evaluation = [
        (r'\(N\)', '评价集全部合资格记录数'),
        (r'\(X_i,T_i,Y_i\)', '第 i 条记录的处理前特征、实际资格分配与观测转化结果'),
        (r'\(S_k\)', r'按策略分数选出的 Top-\(k\) 集合'),
        (r'\(N_T(k),N_C(k)\)', r'两臂人数，分别为 \(\sum_{i\in S_k}T_i\) 和 \(\sum_{i\in S_k}(1-T_i)\)'),
        (r'\(Y_T(k),Y_C(k)\)', r'两臂观测转化数，分别为 \(\sum_{i\in S_k}T_iY_i\) 和 \(\sum_{i\in S_k}(1-T_i)Y_i\)'),
    ]
    metrics = [
        ('HT Gain @20% · 主指标', r'\(G_{\mathrm{HT}}(q_k)=\frac1N\sum_{i\in S_k}Z_i\)，取 \(k=\lfloor0.20N\rfloor\)',
         '向排序前 20% 分配资格，相对全员不分配资格，为全体合资格用户带来的预期增量转化'),
        ('Uplift Gain · 辅助', r'\(G_{\mathrm U}(q_k)=q_k\left[\frac{Y_T(k)}{N_T(k)}-\frac{Y_C(k)}{N_C(k)}\right]\)',
         '以入选人群的两臂转化率差衡量同覆盖率下的增量收益'),
        ('AUUC · 辅助', r'\(\int_0^1G_{\mathrm U}(q)\,\mathrm dq\)', '概括从低覆盖率到全覆盖的累计增量表现'),
        ('Qini Score · 辅助', r'\(\int_0^1[Q(q)-qQ(1)]\,\mathrm dq\)', '概括全覆盖率范围内相对随机排序参考线的增益'),
    ]
    methods_url = 'https://causalml.readthedocs.io/en/latest/methodology.html'
    return _section('uplift-evaluation', '05｜Uplift 模型与评价口径',
        _paragraph('第 04 章估计了广告资格分配的平均转化效应。在资格名额有限时，决策进一步取决于优先向哪些用户分配资格。'
                   '本章定义 Uplift、Response 和 Random 策略的排序方式，以及同覆盖率下的增量收益评价口径。')
        + '<h3>5.1 Uplift 模型与策略基线</h3>'
        + _paragraph(r'沿用第 04 章的定义，\(X\) 表示处理前特征，\(T\) 表示广告资格分配，\(Y\) 表示观测转化结果。')
        + '<h4>表 5-1｜模型记号</h4>' + _table(pd.DataFrame(notation, columns=['记号', '定义']))
        + '<h4>表 5-2｜策略及排序分数</h4>' + _table(pd.DataFrame(scores, columns=['策略', '排序分数（由高到低）']))
        + _paragraph(r'<strong>S-Learner</strong> 合并两臂拟合响应模型，对同一用户分别代入 \(T=1\) 和 \(T=0\) 计算预测差。'
                     '共享拟合使两臂可以共同学习转化与特征的关系。在约 85%∶15% 的样本结构下，这能减轻单独拟合对照响应面时面临的样本量限制。'
                     '共享参数和正则化可能压缩处理相关信号。对于本项目未显式加入交互项的 Logistic，效应评分为')
        + _equation(r'\widehat\tau_S(x)=\sigma(\widehat\eta(x)+\widehat\beta_T)-\sigma(\widehat\eta(x))', 'S-Logistic 的效应评分')
        + _paragraph(r'其中 \(\sigma\) 为 Logistic 函数，\(\widehat\beta_T\) 表示原始 \(T\) 从 0 到 1 对线性预测量的拟合改变量。'
                     'L2 正则化可能将这一改变量向零收缩，使两种分配下的预测概率接近。用户间的效应评分差异来自 Logistic 函数在不同基准预测值处的变化。'
                     r'S-XGBoost 可以在不同特征区间内使用 \(T\) 分裂，从而表达资格分配效应随用户特征的变化。'
                     r'如果所有拟合树都未使用 \(T\) 分裂，同一用户在两种分配状态下的预测相同，模型给出的效应评分为零。'
                     f'<a href="{methods_url}#s-learner">CausalML 官方 S-Learner 方法</a>')
        + _paragraph('<strong>T-Learner</strong> 分别拟合两臂观测转化率，在第 03 章讨论的识别条件下估计两条响应面。'
                     '两臂可以各自学习转化与特征的关系，因此适合表达不同的响应模式。效应评分的误差等于两条响应面的估计误差之差。'
                     '本项目对照组较小、转化正例更少，对照响应面的估计误差会直接影响用户排序。'
                     f'<a href="{methods_url}#t-learner">CausalML 官方 T-Learner 方法</a>')
        + _paragraph(r'<strong>X-Learner</strong> 先拟合两条响应面，再用另一臂的响应预测构造本臂的伪效应标签，并分别回归其条件均值。'
                     r'对照组伪标签可以利用大样本处理组拟合的 \(\widehat\mu_1(x)\)，处理组伪标签则依赖样本较少的对照响应面 \(\widehat\mu_0(x)\)。'
                     r'项目以 \(\widehat e_{\mathrm{train}}\) 加权，对 \(\widehat D_0(x)\) 赋予更大的组合权重。'
                     '对照组的第二阶段回归仍受本臂样本量约束，第一阶段预测误差也会传入伪标签。'
                     f'<a href="{methods_url}#x-learner">CausalML 官方 X-Learner 方法</a>')
        + _paragraph('<strong>Response</strong> 复用对应 T-Learner 的处理组模型，按获得资格后的预测转化率排序，用于比较响应定向与增量定向的实际收益。'
                     '<strong>Random</strong> 提供同等资格名额下随机分配的参照，项目直接计算其期望收益。')
        + '<h3>5.2 评价设定与指标体系</h3>'
        + _paragraph(r'各策略在同一评价集上比较。给定目标覆盖率 \(q\)，按分数降序选取 \(k=\lfloor qN\rfloor\) 条记录，实际覆盖率为 \(q_k=k/N\)。'
                     '遇到同分时，各策略共用由固定随机种子生成的并列排序优先值。主评价覆盖率预设为 20%。')
        + '<h4>表 5-3｜策略评价记号</h4>' + _table(pd.DataFrame(evaluation, columns=['记号', '定义']))
        + _equation(r'Z_i=\frac{T_iY_i}{\widehat e_{\mathrm{train}}}-\frac{(1-T_i)Y_i}{1-\widehat e_{\mathrm{train}}}', '项目使用的 HT 加权结果')
        + _equation(r'Q(q_k)=\frac1N\left[Y_T(k)-Y_C(k)\frac{N_T(k)}{N_C(k)}\right]', '传统 Qini 曲线')
        + '<h4>表 5-4｜主指标与辅助指标</h4>' + _table(pd.DataFrame(metrics, columns=['指标', '计算口径', '业务理解']))
        + _paragraph(r'<strong>HT Gain @20%</strong> 直接对应固定资格名额下的决策目标，即哪套策略能够为全体合资格用户带来更多增量转化。'
                     r'分母统一为 \(N\)，各策略均可换算成每 10 万名合资格用户的增量转化。'
                     r'同覆盖率下，Random 的期望 HT Gain 为 \(q_kG_{\mathrm{HT}}(1)\)，用于衡量定向相对随机分配的增量。')
        + _paragraph(r'HT 的逆概率权重在理论上使用真实的条件分配概率 \(e(X_i)\)。策略固定且因果识别条件成立时，'
                     r'使用真实分配概率可使加权结果的条件期望等于 \(\tau(X_i)\)。本项目采用恒定分配概率近似，'
                     rf'代入 Train Set 处理比例 \(\widehat e_{{\mathrm{{train}}}}={assignment_probability:.6f}\)。'
                     '该数值也用于 X-Learner 的组合权重。第 03 章的 SRM 核对支持公开样本的整体处理比例与 85% 参考值相符，'
                     '恒定概率近似的适用性仍取决于实际分配机制。')
        + _paragraph('Uplift Gain 用入选前缀内的两臂转化率作差，可核对固定覆盖率下的策略比较对估计口径是否敏感。'
                     'AUUC 概括多个覆盖率上的累计表现，Qini Score 用随机参考线衡量整体排序增益。'
                     '两项面积指标均汇总 20% 以外的覆盖范围，因此用于评价整体排序。第 06 章按预设的 HT Gain @20% 冻结策略。')
        + _paragraph('Uplift Gain 要求前缀内两臂均有样本，传统 Qini 要求前缀内有对照样本。'
                     '面积指标从原点出发，按有定义的前缀节点做梯形积分。结果表将 HT Gain 和 Uplift Gain 乘以 100,000，'
                     '以每 10 万名合资格用户的增量转化数展示。'))


def _ate_section(out, ate):
    full = ate.loc[ate.scope.eq('full_dataset')].set_index('outcome')
    conversion = full.loc['conversion']
    rows = []
    for label, row in [('转化（Conversion）', conversion), ('访问（Visit）', full.loc['visit'])]:
        p_value = row.p_value_text
        scientific = re.fullmatch(r'(\d+\.\d+)e([+-]?\d+)', str(p_value))
        if scientific:
            mantissa, exponent = scientific.groups()
            p_value = rf'\({mantissa}\times10^{{{int(exponent)}}}\)'
        rows.append((label, f'{row.response_treatment:.4%}', f'{row.response_control:.4%}',
                     f'{row.relative_uplift:+.2%}', f'{row.z_statistic:.4f}', p_value))
    figure = out / 'figures/ate_full.png'
    return _section('ate-main', '04｜平均处理效应：广告资格分配的 ITT',
        '<h3>4.1 估计目标与估计方法</h3>'
        + _paragraph(r'沿用第 01 章的定义，\(T=1\) 表示获得广告投放资格，\(T=0\) 表示未获得资格。本章以转化（Conversion）为主要结果，'
                     '以访问（Visit）为辅助结果。处理按原始资格分配定义，获得资格后未实际曝光的记录仍归入处理组。'
                     '因此，本章的平均处理效应（ATE）对应资格分配的意向处理效应（ITT）。')
        + _paragraph(r'令 \(Y(1)\) 和 \(Y(0)\) 分别表示获得和未获得资格时的潜在结果，\(\widehat p_t\) 表示分配组 \(t\) 的观测结果率。'
                     '项目使用第 02 章的全量审计计数计算两臂结果率，并以率差估计平均分配效应。')
        + _equation(r'\tau_{\mathrm{ITT}}=\mathbb E[Y(1)-Y(0)],\qquad\widehat\tau_{\mathrm{ITT}}=\widehat p_1-\widehat p_0',
                    '平均资格分配效应与率差估计')
        + _paragraph(r'95% 置信区间使用两组各自的样本方差计算标准误，并采用正态近似构造。双侧两比例 \(Z\) 检验以两组结果率相等为原假设，'
                     '使用零假设下的合并比例计算检验标准误。')
        + _details('ate-method', '估计公式与工程实现',
            _paragraph(r'设 \(n_t\) 和 \(r_t\) 分别为分配组 \(t\) 的样本量和正例数，则 \(\widehat p_t=r_t/n_t\)。点估计的标准误和 95% 置信区间为')
            + _equation(r'\widehat{\mathrm{SE}}=\sqrt{\frac{\widehat p_1(1-\widehat p_1)}{n_1-1}'
                        r'+\frac{\widehat p_0(1-\widehat p_0)}{n_0-1}},'
                        r'\qquad\widehat\tau_{\mathrm{ITT}}\pm z_{0.975}\widehat{\mathrm{SE}}', '标准误与置信区间')
            + _paragraph(r'在 \(H_0:p_1=p_0\) 下，令 \(\widehat p=(r_1+r_0)/(n_1+n_0)\)。双侧检验统计量为')
            + _equation(r'Z=\frac{\widehat p_1-\widehat p_0}{\sqrt{\widehat p(1-\widehat p)(1/n_1+1/n_0)}}', '双侧检验统计量')
            + _paragraph('<code>run_pipeline.py</code> 对 Conversion 和 Visit 分别调用 <code>criteo_uplift/evaluation.py</code> 中的 '
                         '<code>ate_summary_from_audit</code>。该函数读取全量审计中的两臂人数和正例数，由 <code>_ate_summary_from_counts</code> '
                         '计算点估计、标准误、置信区间及双侧检验结果。结果保存于 <code>evaluation/ate_summary.csv</code>，分析范围标记为 <code>full_dataset</code>。'))
        + '<h3>4.2 全量样本的估计结果和业务解释</h3>'
        + _paragraph(f'全量公开样本包含 {int(conversion.n):,} 条记录，其中处理组 {int(conversion.n_treatment):,} 条、'
                     f'对照组 {int(conversion.n_control):,} 条。广告资格分配的转化 ITT 估计为 <strong>{conversion.ate * 100:+.4f} 个百分点</strong>，'
                     f'折合每 10 万名合资格用户约 <strong>{conversion.ate * 100000:,.0f} 次增量转化</strong>。')
        + '<h4>表 4-1｜全量公开样本的两臂结果率与效应检验</h4>'
        + _table(pd.DataFrame(rows, columns=['结果变量', '处理组结果率', '对照组结果率', '相对对照组增幅', 'Z 统计量', '双侧 p 值']))
        + (_image(figure, '图 4-1｜全量公开样本的平均处理效应与 95% 置信区间',
                  '圆点为点估计，横线为 95% 置信区间，竖虚线表示零效应。转化和访问面板各自使用对应的横轴刻度。')
           if figure.is_file() else _paragraph('各结果的点估计及置信区间保存于 evaluation/ate_summary.csv。'))
        + _paragraph(('转化和访问的效应区间均位于零点以上。' if full.ci_low.gt(0).all() else '')
                     + f'对照组 CVR 约为 {conversion.response_control:.3%}，因此相对转化增幅 {conversion.relative_uplift:.2%} 应结合基准率解读。'
                     f'以每 10 万名合资格用户计，估计增量约为 {conversion.ate * 100000:,.0f} 次转化，95% 置信区间约为 '
                     f'<strong>{conversion.ci_low * 100000:,.1f} 至 {conversion.ci_high * 100000:,.1f} 次</strong>。'
                     + ('访问率的正向差异补充说明了资格分配对访问行为的影响。' if full.loc['visit', 'ci_low'] > 0 else '')
                     + '转化增量仍是后续策略比较的主要业务指标。')
        + _paragraph('效应的因果解释依赖原实验的随机分配设计及其在公开样本中的适用性。公开数据经过非均匀抽样，'
                     '效应估计和区间均对应公开样本口径。总体 ITT 为衡量资格分配的平均转化价值提供基准。'
                     '固定覆盖率下，优先向哪些用户分配资格，则由第 05 章的 Uplift 建模与同覆盖率策略评价回答。'))



def _chapter_six(out, manifest, selection, matched, gain_figure, expansion):
    def csv(name):
        return pd.read_csv(out / name)

    selected, uplift = selection['selected_policy'], selection['selected_uplift_model']
    label, uplift_label = _LABELS[selected], _LABELS[uplift]
    response = matched['reference']
    response_label = _LABELS[response]
    split = csv('data_preparation/split_summary.csv')
    n_test = int(split.loc[split.split.eq('test'), 'n'].sum())
    test_label = '正式 Test Set' if manifest['arguments'].get('stage') == 'full' else '开发留出集'
    scopes = {'train': ('Train Set', '拟合模型与预处理'),
              'validation': ('Validation Set', '按预设指标选型'),
              'test': (test_label, '评价冻结策略')}
    split_rows = []
    for scope, (scope_label, role) in scopes.items():
        arms = split.loc[split.split.eq(scope)].set_index('treatment')
        split_rows.append((scope_label, f'{arms.n.sum():,}', f'{arms.loc[1, "conversion_positive"]:,}',
                           f'{arms.loc[0, "conversion_positive"]:,}', role))
    audit = json.loads((out / 'data_preparation/feature_overlap.json').read_text())
    checks = audit['row_id_checks']
    training_matches = audit['fraction_matching_training_features']
    leak_rows = [
        ('来源行号完整性', '行号非空且唯一，三集合计覆盖本次使用的记录'
         if checks['non_null_unique'] and checks['complete_coverage'] else '参见划分审计明细'),
        ('跨集合来源行号', '训练、验证和 Test Set 两两无重叠' if checks['pairwise_disjoint'] else '参见划分审计明细'),
        ('Validation Set 特征向量与 Train Set 匹配', f'{training_matches["validation"]:.2%}，分母为 Validation Set 记录数'),
        ('Test Set 特征向量与 Train Set 匹配', f'{training_matches["test"]:.2%}，分母为 Test Set 记录数'),
        ('Test Set 特征向量与 Validation Set 匹配', f'{audit["fraction_test_matching_validation_features"]:.2%}，分母为 Test Set 记录数'),
        ('特征与标签', '用户特征为 f0–f11，conversion 用作训练标签。exposure、visit 和来源行号均排除在评分特征之外。'),
        ('预处理拟合范围', 'Logistic 的中位数填补与标准化在 Train Set 拟合，Validation Set 和 Test Set 沿用训练参数。'),
    ]
    test_boot = csv('evaluation/test_bootstrap.csv')
    test_draws = int(test_boot.n_bootstrap.iloc[0])
    budget = selection['primary_budget']

    def results(metrics, boot, validation):
        rows = metrics.loc[metrics.budget.eq(budget)].copy()
        if not validation:
            rows = rows.merge(boot.loc[boot.budget.eq(budget)], on=['policy', 'budget'], validate='one_to_one')
        rows = rows.sort_values('gain_ht', ascending=False, kind='stable')
        roles = {uplift: '（最优 Uplift）', selected: '（冻结策略）'}
        data = {'策略': rows.policy.map(_LABELS) + rows.policy.map(roles).fillna('')}
        data['HT Gain @20%'] = rows.gain_ht_per_100k.map(_num)
        if not validation:
            data['HT 双侧 95% CI'] = rows.apply(lambda row: _interval(row, scale=1e5), axis=1)
        data.update({'Uplift Gain @20%': (rows.gain_plugin * 1e5).map(_num),
                     'AUUC × 10⁵': (rows.auuc * 1e5).map(_num),
                     'Qini Score × 10⁵': (rows.qini_score * 1e5).map(_num)})
        return _table(pd.DataFrame(data))

    val_metrics = csv('models/validation_metrics.csv')
    test_metrics = csv('evaluation/test_metrics.csv')
    test_pairs = csv('evaluation/test_pairwise.csv').query('budget == @budget').set_index(['policy', 'reference'], drop=False)
    key_pairs = test_pairs.loc[list(dict.fromkeys(((selected, 'random'), (uplift, response))))].reset_index(drop=True)
    vs_random = key_pairs.iloc[0]
    versus_response = test_pairs.loc[(uplift, response)]
    candidate_pairs = test_pairs.loc[
        test_pairs.reference.eq(response)
        & test_pairs.policy.isin(selection['candidates'])
        & test_pairs.policy.ne(response)
    ].sort_values('difference', ascending=False, kind='stable').reset_index(drop=True)
    comparison = (
        '两种策略的收益点估计接近，配对区间包含两个方向的小幅差异。'
        if versus_response.ci_low <= 0 <= versus_response.ci_high else
        '两种策略的配对区间位于零点同一侧，差异方向见表 6-5。'
    )
    candidate_analysis = ''
    if selected == response:
        other_candidates = candidate_pairs.loc[candidate_pairs.policy.ne(uplift)]
        below_zero = int(other_candidates.ci_high.lt(0).sum())
        candidate_analysis = _paragraph(
            f'<strong>其余候选的收益差异为替代方案评估提供依据。</strong>除 {uplift_label} 外，'
            f'{len(other_candidates)} 个候选中有 {below_zero} 个相对冻结策略的逐项区间完全低于零。'
            + ('其中 Logistic 候选的点估计损失约为每 10 万人 '
               f'{-other_candidates.loc[other_candidates.policy.str.contains("logistic"), "difference"].max() * 1e5:.2f}–'
               f'{-other_candidates.loc[other_candidates.policy.str.contains("logistic"), "difference"].min() * 1e5:.2f} 次转化。'
               if other_candidates.policy.str.contains('logistic').any()
               and other_candidates.loc[other_candidates.policy.str.contains('logistic'), 'difference'].lt(0).all() else '')
            + ''.join(
                f'{_LABELS[row.policy]} 的点估计损失为 {-row.difference * 1e5:.2f} 次／10 万人。'
                for row in other_candidates.itertuples()
                if row.policy in ('x_xgboost', 't_xgboost') and row.difference < 0
            )
            + '若未来考虑用其他模型替换冻结策略，应将这些收益差异与实际的推理成本、维护成本和更新频率一并评估。'
            '当前数据只量化转化收益，替代方案的成本优势仍需业务测量。'
        ) + _paragraph(
            '逐项区间用于描述每个候选相对共同参照的收益差异，整组比较未经同时覆盖率校正。'
            '例如，某个候选的区间端点接近零时，其差异判断对样本波动更敏感。'
            '这些比较用于解释 Validation Set 冻结策略在 Test Set 中的表现，线上是否替换策略还需结合业务收益门槛和实验验证。'
        )
    plot = test_boot.loc[test_boot.budget.eq(budget)].copy()
    plot = pd.concat([plot[plot.policy.eq(selected)], plot[plot.policy.ne(selected)]])
    ht = test_metrics.loc[test_metrics.policy.eq(selected)].set_index('budget')
    ht_fraction = ht.loc[budget, 'gain_ht'] / ht.loc[1., 'gain_ht'] if ht.loc[1., 'gain_ht'] else np.nan
    plugin_fraction = ht.loc[budget, 'gain_plugin'] / ht.loc[1., 'gain_plugin'] if ht.loc[1., 'gain_plugin'] else np.nan
    added_gain = (ht.loc[.30, 'gain_ht'] - ht.loc[budget, 'gain_ht']) * n_test
    added_slots = int(ht.loc[.30, 'n_selected'] - ht.loc[budget, 'n_selected'])
    split_intro = ('正式运行的 Train Set、Validation Set 和 Test Set 总体按 <strong>6∶2∶2</strong> 划分。'
                   '本轮开发前，从可追溯的历史建模记录之外，按 treatment × conversion 分层抽取约占全量 20% 的记录，冻结为正式 Test Set。'
                   '其余记录按相同组合分层，约 75% 用于训练、25% 用于验证。'
                   if manifest['arguments'].get('stage') == 'full' else '运行内的训练、验证和留出评价按处理分配与转化标签分层。')
    overlap = (
        '<figure class="overlap-figure"><div class="overlap-counts">'
        f'<div><span>仅 {uplift_label}</span><strong>{matched["only_a"]:,}</strong></div>'
        f'<div class="shared"><span>共同选中</span><strong>{matched["common"]:,}</strong><small>各自入选人数的 {matched["overlap"]:.2%}</small></div>'
        f'<div><span>仅 {response_label}</span><strong>{matched["only_b"]:,}</strong></div></div>'
        f'<figcaption>两策略各选 {matched["k"]:,} 条记录，按来源行号匹配入选集合。</figcaption></figure>'
    )
    expansion = re.sub(r'<p class="">.*?</p>', '', expansion, flags=re.DOTALL)
    ten_to_twenty = ht.loc[budget, 'gain_ht_per_100k'] - ht.loc[.10, 'gain_ht_per_100k']
    retained_gain = ht.loc[.10, 'gain_ht'] / ht.loc[budget, 'gain_ht'] if ht.loc[budget, 'gain_ht'] else np.nan
    rho = ('模型预测分数为常数时，Spearman 秩相关无定义。' if matched['spearman'] is None else
           f'两组排名的 Spearman 秩相关为 <strong>{matched["spearman"]:.4f}</strong>。')
    chapter = _section('methods', '06｜实验设计、策略冻结与正式结果',
        _paragraph('本章按第 05 章的统一口径比较九种策略。Train Set 用于拟合八个候选，Validation Set 按 HT Gain @20% 点估计选型，'
                   'Test Set 评价冻结后的策略，并分析入选人群画像、分层处理效应与扩量收益。')
        + '<h3>6.1 数据划分与泄露诊断</h3>' + _paragraph(split_intro)
        + '<h4>表 6-1｜数据划分与转化正例</h4>'
        + _table(pd.DataFrame(split_rows, columns=['数据范围', '样本量', '处理组转化正例', '对照组转化正例', '用途']))
        + '<h4>表 6-2｜数据泄露诊断</h4>'
        + _table(pd.DataFrame(leak_rows, columns=['检查项', '结果']))
        + _paragraph('特征匹配按模型实际使用的 float32 表示进行。第 02 章显示，部分匿名特征取值高度集中，不同来源行可能拥有相同的特征向量。'
                     '冻结测试行、核验跨集行号和在 Train Set 拟合预处理用于控制本轮模型开发中的行级泄露。'
                     '公开数据缺少可靠用户标识，无法识别匹配记录是否来自同一用户，因此未据此去重。')
        + '<h3>6.2 候选策略与模型训练</h3>'
        + _paragraph('S、T、X 三类 Meta-Learner 分别搭配 Logistic Regression 和 XGBoost，形成六个 Uplift 候选。'
                     '两种基础模型各对应一个 Response 候选，复用相应 T-Learner 的处理组响应模型。另设同覆盖率 Random 基线。')
        + _paragraph(r'<strong>Logistic Regression</strong> 使用训练内中位数填补与标准化，采用 L2 正则化、\(C=1\) 和 LBFGS，最大迭代次数为 500。'
                     r'S-Logistic 将 \(T\) 与处理前特征一同输入模型，再分别代入 \(T=1\) 和 \(T=0\) 计算效应评分。')
        + _paragraph(f'<strong>XGBoost</strong> 使用 {manifest["arguments"]["trees"]} 棵树、最大深度 4、学习率 0.05、'
                     '<code>min_child_weight=20</code>，行采样率为 0.8，列采样率为 0.9。')
        + _paragraph(r'<strong>X-Learner 第二阶段</strong>以连续伪效应为回归目标，X-Logistic 使用线性回归，X-XGBoost 使用树回归。'
                     rf'两臂预测按第 05 章的固定权重组合，Train Set 处理比例为 \(\widehat e_{{\mathrm{{train}}}}={selection["assignment_probability"]:.6f}\)。')
        + _details('training-implementation', '模型拟合、选型与评价实现',
            _paragraph('<code>run_pipeline.py</code> 调用 <code>criteo_uplift/training_validation.py</code> 的 <code>fit_and_select_models</code>。'
                       '<code>_fit_candidates</code> 调用 <code>criteo_uplift/models.py</code> 的 S、T 和 X-Learner，'
                       '<code>response_predict</code> 复用对应的处理组模型。')
            + _paragraph('验证评分传给 <code>criteo_uplift/metrics.py</code> 中的 <code>evaluate_ranking</code>，'
                         '选型结果和评价参数写入 <code>models/selection.json</code>。'
                         '<code>criteo_uplift/evaluation.py</code> 的 <code>evaluate_test_set</code> 加载已拟合模型和冻结设置完成测试评价。'))
        + '<h3 id="selection-flow">6.3 Validation Set 选型与策略冻结</h3>'
        + _paragraph('在 Validation Set 上直接计算九种策略的 HT Gain @20% 及三个辅助指标，'
                     '按主指标点估计降序排列，Random 提供同覆盖率参照。'
                     '同分沿用固定候选顺序，模型参数配置保持一致。')
        + '<h4>表 6-3｜Validation Set 九策略结果，按 HT Gain 点估计降序</h4>' + results(val_metrics, None, True)
        + _paragraph(f'<strong>{label}</strong> 按总体排名冻结为正式策略，<strong>{uplift_label}</strong> 在六个 Uplift 候选中排名最高。'
                     '候选选择只使用 Validation Set 主指标；正式 Test Set 用于评价策略收益及其不确定性。')

        + '<h3 id="results">6.4 正式 Test Set 评价与配对比较</h3>'
        + _paragraph(f'{test_label}共有 {n_test:,} 条记录，20% 覆盖率对应 {int(n_test * budget):,} 个资格名额。'
                     f'区间使用 {test_draws:,} 次处理臂内分层 Bootstrap。双侧 95% Bootstrap 百分位区间取第 2.5% 和第 97.5% 分位数。'
                     'HT Gain、对应区间和 Uplift Gain 均以每 10 万名合资格用户计。')
        + '<h4>表 6-4｜留出集九策略结果，按 HT Gain 点估计降序</h4>' + results(test_metrics, test_boot, False)
        + _policy_comparison(plot, selected, 'HT Gain @20%', '每 10 万名全部合资格用户的增量转化', n_test)
        + _paragraph('策略差值在每轮共用的重采样样本上计算，再对差值取第 2.5% 和第 97.5% 分位数。区间反映模型、评分和处理比例固定时的评价样本波动。')
        + '<h4>表 6-5｜20% 覆盖率下的关键策略配对比较</h4>' + _pair_table(key_pairs, compact=True)
        + _details('candidate-response-pairs', '各候选相对 Response-XGBoost 的配对比较',
            _paragraph(f'覆盖率为 20%，差值方向为“候选策略减 {response_label}”，单位为每 10 万名合资格用户的增量转化。'
                       f'使用 {test_draws:,} 次处理臂内分层配对 Bootstrap，取第 2.5% 和第 97.5% 分位数构造双侧 95% 百分位区间。'
                       '以下为逐项区间，未作多重比较校正。')
            + _pair_table(candidate_pairs, compact=True))
        + _paragraph(f'<strong>模型定向提高了固定名额的转化产出。</strong>在相同资格名额下，{label} 相对 Random 的 HT Gain 点差为 '
                     f'{vs_random.difference * 1e5:+.2f} 次／10 万名合资格用户，配对区间为 {_interval(vs_random, scale=1e5)}。'
                     + ('区间全在零以上，支持在本次公开样本中采用该定向策略提高资格名额的增量转化产出。'
                        if vs_random.ci_low > 0 else '定向收益的方向与精度应结合这一配对区间判断。'))
        + _paragraph(f'<strong>{uplift_label} 是与冻结策略收益接近的候选。</strong>它相对 {response_label} 的点差为 '
                     f'{versus_response.difference * 1e5:+.2f} 次／10 万人，配对区间为 {_interval(versus_response, scale=1e5)}。'
                     + comparison
                     + '当前结果未提供仅凭 Test Set 收益切换这两种策略的充分依据，因此继续采用 Validation Set 冻结选择。'
                     'Uplift 候选的增量建模目标仍有业务研究价值，后续可通过两种规则产生的不同入选人群，检验是否存在稳定的额外收益。')
        + candidate_analysis
        + '<h3 id="interpretation">6.5 入选人群重叠与用户画像分析</h3>' + overlap
        + _paragraph(f'{uplift_label} 按 CATE 估计值排序，{response_label} 按处理后的预测转化概率排序。'
                     f'两者 Top 20% 的共同入选比例为 {matched["overlap"]:.2%}，'
                     f'各自仅有 {matched["only_a_fraction_of_k"]:.2%} 的名额选择了不同记录。'
                     f'在全部 {n_test:,} 条 Test Set 记录上，' + rho
                     + '这表明两种预测目标在当前样本和覆盖率下形成了相近的投放名单与整体优先级。'
                     '本节进一步对比入选记录与全体候选的特征分布，描述模型优先覆盖的人群；收益差异见第 6.4 节。')
        + '<h3 id="budget-expansion">6.6 扩量增益与资格名额配置</h3>'
        + '<div id="chapter-six-gain-figure">' + gain_figure + '</div>'
        + '<div id="chapter-six-expansion">' + expansion + '</div>'
        + _paragraph('扩量判断以新增名额的边际转化回报为依据。扩量增益是沿用冻结策略排序、提高覆盖率时，'
                     '两个入选前缀的累计 HT Gain 之差。表中 Gain 及区间按每 10 万名合资格用户计，最后一列换算为每千个新增名额的回报。'
                     f'区间采用 {test_draws:,} 次处理臂内分层 Bootstrap，每轮共用同一份重采样、沿用固定评分重取两个前缀，'
                     '先计算累计收益差，再取差值的第 2.5% 和第 97.5% 分位数，构造双侧 95% 百分位区间。')
        + _paragraph(f'<strong>10% 是值得进一步评估的较低覆盖率方案。</strong>前 10% 的 HT Gain 为 '
                     f'{ht.loc[.10, "gain_ht_per_100k"]:.2f}，约保留前 20% 累计增量的 {retained_gain:.1%}，资格名额减半。'
                     f'从 10% 扩至 20% 的新增收益为 {ten_to_twenty:.2f} 次／10 万名合资格用户，'
                     '每千个新增名额约带来 0.474 次增量转化。该档配对区间为正，说明增加名额仍有转化回报，单位名额收益已明显低于前 10%。')
        + _paragraph('<strong>是否采用 10% 覆盖率取决于边际经济性。</strong>如果单次增量转化价值为 V，'
                     '则 10%–20% 这一档每千个新增名额的估计转化价值为 0.474V，应与同口径新增成本比较。'
                     '20% 是本项目预设的策略选型预算，10% 是由覆盖率分析提出的后续业务候选；'
                     '调整覆盖率还需在目标业务人群中验证。30% 之后两档的边际区间跨零，进一步扩量的收益方向更不确定。'
                     '公开数据缺少实际成本与转化金额，当前结果支持逐档评估，尚不足以确定利润最优覆盖率。'))



    from .report_followup import analysis_path, add_profiles_and_deciles
    experiment = analysis_path(out)
    return add_profiles_and_deciles(chapter, experiment) if experiment else chapter


def _report_header(research, *, development=False):
    """One contents list, derived from the actual numbered chapter headings."""
    chapters = re.findall(r'<section id="([^"]+)"><h2>(.*?)</h2>', research, re.DOTALL)
    entries = []
    for anchor, title in chapters:
        if anchor == 'summary':
            continue
        number, separator, label = title.partition('｜')
        if not separator:
            number, label = '', title
        entries.append(f'<li><a href="#{html.escape(anchor)}"><span class="toc-number">{number}</span><span>{label}</span></a></li>')
    return ('<header><p class="eyebrow">CRITEO UPLIFT PROJECT</p>'
            f'<h1>{html.escape(_TITLE)}</h1>'
            '<p class="report-subtitle">固定 20% 广告资格覆盖约束下的 Uplift Modeling：增量人群识别、策略价值评估与稳定性验证</p>'
            '<nav class="report-contents" aria-label="报告目录"><p class="toc-label">正文目录</p>'
            '<ol class="report-toc">' + ''.join(entries) + '</ol></nav></header>')


def _refresh_report_front_matter(document, *, development=False):
    document = re.sub(r'<section id="summary">.*?</section>', '', document, count=1, flags=re.DOTALL)
    document = re.sub(r'<aside id="workflow">.*?</aside>', '', document, count=1, flags=re.DOTALL)
    document = re.sub(r'<section id="appendix">.*?</section>', '', document, count=1, flags=re.DOTALL)
    document = document.replace('有效重复次数见附录 E。', '')
    document = document.replace('全部九策略曲线见附录。', '').replace('全部策略曲线见附录 G。', '')
    document = document.replace('附录 I 的成本和样本量仅为假设情景，实际方案应由', '实际方案应由')
    document = re.sub(r'<a href="#appendix[^"]*"[^>]*>.*?</a>', '', document, flags=re.DOTALL)
    research, = re.findall(r'<article id="research"[^>]*>(.*?)</article>', document, re.DOTALL)
    header = _report_header(research, development=development)
    document = re.sub(r'<header>.*?</header>', lambda _: header, document, count=1, flags=re.DOTALL)
    return re.sub(r'<title>.*?</title>', f'<title>{html.escape(_TITLE)}｜Criteo Uplift</title>', document, count=1, flags=re.DOTALL)


def build_report(manifest_path: Path, report_path: Path, *, dataset_audit_only=False, identification_only=False, uplift_methods_only=False, identification_and_ate_only=False, experiments_only=False, research_results_only=False, front_matter_only=False) -> Path:
    """从保存的实验工件生成报告和来源指纹。"""
    from .budget_analysis import BOOTSTRAP_REPEATS, evaluate_budget_margins
    from .planning import cost_value_scenario
    from .strategy_overlap import saved_score_overlap
    from .experiment_protocol import sha256_file
    manifest_path, report_path = Path(manifest_path), safe_output(report_path)
    out = manifest_path.parent
    if report_path.resolve() == (out / 'report.html').resolve() and report_path.exists():
        raise FileExistsError('历史运行报告不可覆盖；请指定 reports/ 下的新发布路径')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    read_json = lambda name: json.loads((out / name).read_text())
    read_csv = lambda name: pd.read_csv(out / name, dtype={'p_value_text': 'string'})
    manifest = read_json(manifest_path.name)
    audit = read_json('data_preparation/audit.json')
    if front_matter_only:
        document = _refresh_report_front_matter(report_path.read_text(encoding='utf-8'),
                                               development=manifest['arguments'].get('stage') == 'development')
        report_path.write_text(document, encoding='utf-8')
        from .report_followup import analysis_path, record_publication
        experiment = analysis_path(out)
        if experiment:
            record_publication(report_path, out, experiment)
        return report_path
    if experiments_only or research_results_only:
        document = report_path.read_text(encoding='utf-8')
        selection = read_json('models/selection.json')
        matched = json.loads((report_path.parent / 'strategy_overlap.json').read_text())
        start = document.index('<section id="methods">')
        next_sections = [document.index(anchor, start) for anchor in ('<section id="stability">', '<section id="decision">') if anchor in document[start:]]
        end = min(next_sections)
        previous = document[start:end]
        if '<div id="chapter-six-gain-figure">' in previous:
            gain, = re.findall(r'<div id="chapter-six-gain-figure">(<figure>.*?</figure>)</div>', previous, re.DOTALL)
            expansion, = re.findall(r'<div id="chapter-six-expansion">(.*?)</div><p class="">扩量判断', previous, re.DOTALL)
        else:
            gain, = re.findall(r'<figure><img alt="覆盖率与累计策略增量".*?</figure>', previous, re.DOTALL)
            expansion, = re.findall(r'<h3>每一档扩量带来的额外转化</h3>(.*?)</section>', previous, re.DOTALL)
            expansion = '<h4>表 6-6｜冻结策略逐档扩量的边际增益</h4>' + expansion
        section = _chapter_six(out, manifest, selection, matched, gain, expansion)
        if research_results_only:
            from .report_followup import analysis_path, chapter_seven, chapter_eight
            experiment = analysis_path(out)
            if experiment is None:
                raise ValueError('完整 06–08 更新需要已保存的正式重训与画像分析结果')
            end = document.index('</article>', end)
            section += chapter_seven(out, experiment) + chapter_eight()
        updated = document[:start] + section + document[end:]
        updated = _refresh_report_front_matter(updated, development=manifest['arguments'].get('stage') == 'development')
        report_path.write_text(updated, encoding='utf-8')
        from .report_followup import analysis_path, record_publication
        experiment = analysis_path(out)
        if experiment:
            record_publication(report_path, out, experiment)
        return report_path
    if uplift_methods_only:
        section = _uplift_methods_section(manifest['assignment_probability'])
        document = report_path.read_text(encoding='utf-8')
        previous = re.compile(r'<section id="uplift-evaluation">.*?</section>', re.DOTALL)
        document, count = previous.subn(lambda _: section, document)
        if count == 0:
            anchor = '<section id="methods">'
            if document.count(anchor) != 1:
                raise ValueError('无法唯一定位 Uplift 模型与评价口径章节的插入位置')
            document = document.replace(anchor, section + anchor, 1)
        elif count != 1:
            raise ValueError('报告中存在重复的 Uplift 模型与评价口径章节')
        report_path.write_text(document, encoding='utf-8')
        return report_path
    if dataset_audit_only:
        section = _dataset_audit_section(out, manifest, audit)
        document = report_path.read_text(encoding='utf-8')
        report_path.write_text(_replace_dataset_audit(document, section), encoding='utf-8')
        return report_path
    if identification_only or identification_and_ate_only:
        section = _identification_section(
            audit, read_json('diagnostics/srm.json'), read_csv('diagnostics/balance.csv'),
            read_csv('diagnostics/c2st.csv'), read_csv('diagnostics/c2st_null.csv'),
            read_json('diagnostics/overlap_positivity.json'),
            read_csv('diagnostics/learnability.csv'),
        )
        document = report_path.read_text(encoding='utf-8')
        pattern = r'<section id="identification">.*?</section>'
        existing, = re.findall(pattern, document, re.DOTALL)
        if identification_and_ate_only:
            document = re.sub(r'<section id="(?:ate-preview|ate-main)">.*?</section>', '', document, count=1, flags=re.DOTALL)
            section += _ate_section(out, read_csv('evaluation/ate_summary.csv'))
        report_path.write_text(document.replace(existing, section, 1), encoding='utf-8')
        return report_path
    selection = read_json('models/selection.json')
    sample = read_json('data_preparation/sample_report.json')
    split = read_csv('data_preparation/split_summary.csv')
    ate = read_csv('evaluation/ate_summary.csv')
    balance, c2st = read_csv('diagnostics/balance.csv'), read_csv('diagnostics/c2st.csv')
    c2st_null = read_csv('diagnostics/c2st_null.csv')
    positivity, srm = read_json('diagnostics/overlap_positivity.json'), read_json('diagnostics/srm.json')
    metrics, bootstrap = read_csv('evaluation/test_metrics.csv'), read_csv('evaluation/test_bootstrap.csv')
    pairs, curves = read_csv('evaluation/test_pairwise.csv'), read_csv('evaluation/curves.csv')
    val_metrics = read_csv('models/validation_metrics.csv')
    args = manifest['arguments']
    selected, uplift = selection['selected_policy'], selection['selected_uplift_model']
    response = 'response_xgboost'
    if selection['primary_metric'] != 'gain_ht' or selection['primary_budget'] != .2 or not selection['frozen_before_test']:
        raise ValueError('报告需要预冻结的 HT Gain @20% 协议')
    if set(metrics.policy) != {*selection['candidates'], 'random'}:
        raise ValueError('九策略结果不完整')
    development = args.get('stage') == 'development'
    stage_title = '500 万开发实验报告' if development else '全量正式实验报告'
    scope = '开发留出集' if development else '正式 Test Set'
    label, uplift_label, response_label = (_LABELS[k] for k in (selected, uplift, response))
    budget = .2
    score_path = out / 'evaluation/predictions.npz'
    margins = None
    if args.get('stage') == 'full':
        margins, refined_boot, refined_pairs = evaluate_budget_margins(score_path, selection, metrics)
        # 用同一批抽样更新报告中关键策略的区间，保留各记录的有效重复次数。
        for original, refined, index in (
            (bootstrap, refined_boot, ['policy', 'budget']),
            (pairs, refined_pairs, ['policy', 'reference', 'budget']),
        ):
            original.set_index(index, inplace=True)
            original.update(refined.drop(columns=['win_frequency'], errors='ignore').set_index(index))
            original.reset_index(inplace=True)
    primary = metrics[metrics.budget.eq(budget)].set_index('policy')
    primary_boot = bootstrap[bootstrap.budget.eq(budget)].set_index('policy')
    primary_pairs = pairs[pairs.budget.eq(budget)]
    frozen = metrics[metrics.policy.eq(selected)].set_index('budget')
    test_n = int(split.loc[split.split.eq('test'), 'n'].sum())
    value = float(primary.loc[selected, 'gain_ht_per_100k'])
    interval = _interval(primary_boot.loc[selected], scale=1e5)
    def comparison(policy, reference):
        if policy == reference:
            return pd.Series({'policy': policy, 'reference': reference, 'budget': budget,
                              'difference': 0., 'ci_low': 0., 'ci_high': 0., 'valid_replicates': selection['n_bootstrap']})
        return primary_pairs[(primary_pairs.policy == policy) & (primary_pairs.reference == reference)].iloc[0]
    vs_random = comparison(selected, 'random')
    uplift_pair = comparison(uplift, response)
    matched = saved_score_overlap(score_path, uplift, response,
                                  budget=budget, seed=manifest['seed'])
    write_result(matched, report_path.parent / 'strategy_overlap.json')
    if matched['n'] != test_n:
        raise ValueError('保存评分与测试样本数量不一致')
    marginal_10_20 = value - frozen.loc[.1, 'gain_ht_per_100k']
    marginal_20_30 = frozen.loc[.3, 'gain_ht_per_100k'] - value
    main = []
    example = (
        '<figure class="example">'
        '<h3>两种定向策略的排序差异</h3>'
        '<div class="example-row"><b>高自然转化人群</b>'
        '<div class="example-metrics"><span class="natural">80% 自然转化</span><span class="uplift">+5pp 增量</span></div>'
        '<div class="bar"><i style="width:80%"></i><em style="width:5%"></em></div>'
        '<p class="example-outcome">处理后转化率：85%</p></div>'
        '<hr class="example-divider">'
        '<div class="example-row"><b>高广告敏感人群</b>'
        '<div class="example-metrics"><span class="natural">10% 自然转化</span><span class="uplift">+25pp 增量</span></div>'
        '<div class="bar"><i style="width:10%"></i><em style="width:25%"></em></div>'
        '<p class="example-outcome">处理后转化率：35%</p></div>'
        '<figcaption>示意数据，非 Criteo 实验结果。蓝色为自然转化基础，橙色为广告处理带来的增量。</figcaption></figure>'
    )
    symbols = _table(pd.DataFrame([
        (r'\(N\)', '合资格候选用户总数'),
        (r'\(X_i\)', '用户处理前特征'),
        (r'\(T_i\)', '随机处理分配，1 为处理组，0 为对照组'),
        (r'\(Y_i(1), Y_i(0)\)', '用户在两种处理分配状态下的潜在转化结果'),
        (r'\(Y_i\)', '实际观测转化结果'),
        (r'\(\pi_i\)', '用户广告资格分配决策，取值为 0 或 1'),
        (r'\(K\)', '固定广告资格名额'),
        (r'\(\mu_t(x)\)', r'\(\mathbb{E}[Y(t)\mid X=x]\)，特征为 x 的用户在处理状态 t 下的条件平均转化概率'),
        (r'\(\tau(x)\)', r'\(\mathbb{E}[Y(1)-Y(0)\mid X=x]\)，即 CATE'),
    ], columns=['符号', '定义']))
    assumptions = _table(pd.DataFrame([
        ('固定候选人群', r'候选用户总数 \(N\) 固定'),
        ('固定资源容量', r'必须分配恰好 \(K\) 个广告资格名额'),
        ('单人投入成本一致', '每名用户获得广告资格的单位成本相同'),
        ('单次转化价值一致', '每次转化对应的业务价值相同'),
        ('无用户间处理干扰', '一名用户的处理分配不影响其他用户的潜在转化结果'),
    ], columns=['假设', '具体含义']))
    theory = (
        '<h3>1. 单个用户在分配策略下的潜在结果</h3>'
        + _paragraph(r'根据策略决策 \(\pi_i\)，用户对应的潜在结果为：')
        + _equation(r'Y_i(\pi_i)=(1-\pi_i)Y_i(0)+\pi_iY_i(1)', '策略分配下的潜在结果')
        + _paragraph('对处理前特征条件化取期望：')
        + _equation(r'\mathbb{E}[Y_i(\pi_i)\mid X_i]=(1-\pi_i)\mu_0(X_i)+\pi_i\mu_1(X_i)=\mu_0(X_i)+\pi_i\tau(X_i)', '单个用户的条件期望转化')
        + '<h3>2. 推广至全部候选用户</h3>'
        + _paragraph('利用期望的线性性质：')
        + _equation(r'\mathbb{E}[\sum_{i=1}^{N}Y_i(\pi_i)\mid X_{1:N}]=\sum_{i=1}^{N}\mu_0(X_i)+\sum_{i=1}^{N}\pi_i\tau(X_i)', '全部候选用户的总体预期转化')
        + _paragraph('得到正文中的总体预期转化目标。')
        + '<h3>3. Top-K 排序的最优性</h3>'
        + _paragraph('假设当前选中的用户 A 的 CATE 低于未选中的用户 B：')
        + _equation(r'\tau(X_A)<\tau(X_B)', '用户 A 与用户 B 的 CATE 比较')
        + _paragraph('将资格名额由 A 调整给 B，预期增量转化增加：')
        + _equation(r'\Delta=\tau(X_B)-\tau(X_A)>0', '替换后的预期增量')
        + _paragraph(r'因此，通过不断将较低 CATE 用户替换为较高 CATE 用户，最终得到 CATE 最高的 \(K\) 名用户，即固定覆盖率下的理论最优分配策略。')
        + '<h3>4. 理论最优策略</h3>'
        + _equation(r'\pi_i^*=\mathbf{1}\left(i\in\mathrm{TopK}\{\tau(X_1),\ldots,\tau(X_N)\}\right)', '固定覆盖率下的理论最优分配')
        + _paragraph(r'实际模型使用预测 CATE \(\hat{\tau}(X)\) 实现用户排序，其策略表现由增量效应估计与排序质量决定。')
    )
    main.append(_section('business', '01｜业务背景与增量转化目标',
        '<h3>1.1 业务背景与研究问题</h3>'
        + _paragraph('在互联网广告投放场景中，广告平台需要从合资格用户中选择广告投放对象。受广告预算、广告库存及触达频次限制，可分配的广告资格名额有限，因此需要优化有限资源下的用户选择策略。')
        + _paragraph('传统广告定向通常采用 Response Modeling，根据用户接受广告处理后的预测转化概率进行排序，优先选择更可能转化的用户。然而，用户的转化行为同时受到自然转化倾向与广告增量效果的影响。部分用户即使没有获得广告资格，也具有较高的自然转化概率；另一些用户虽然自然转化概率较低，却可能对广告处理具有更强的响应。')
        + _paragraph('因此，单纯依据处理后的转化概率进行排序，可能将有限的广告资源分配给自然转化倾向较高、增量效果较低的用户。')
        + _paragraph('<strong>本项目研究目标：在固定广告资格覆盖率下，识别不同用户群体的广告增量效果，优化目标人群选择，从而最大化预期增量转化人数。</strong>', 'lead')
        + '<h3>1.2 Response Targeting 与 Uplift Targeting</h3>'
        + _paragraph('两种策略的核心区别在于用户排序依据：')
        + '<ul><li><strong>Response Targeting：</strong>根据处理后的预测转化概率排序，优先选择更可能转化的用户。</li>'
        + '<li><strong>Uplift Targeting：</strong>根据广告处理带来的预测增量转化排序，优先选择从广告中获益更多的用户。</li></ul>'
        + example
        + _paragraph('在该示例中，Response 会优先选择高自然转化人群，Uplift 则会优先选择高广告敏感人群。同样分配 100 个广告资格名额，后者预期多产生 20 次增量转化。')
        + _paragraph('这体现了 Uplift Modeling 在有限广告资源分配中的业务价值：根据广告的增量贡献选择目标用户。')
        + '<h3>1.3 符号框架与固定覆盖率下的业务优化目标</h3>'
        + '<h4>（1）业务问题的数学抽象</h4>'
        + _paragraph(r'假设平台拥有 \(N\) 名合资格候选用户，需要从中选择 \(K\) 名用户分配广告资格。定义用户 \(i\) 的处理前特征为 \(X_i\)，处理分配为 \(T_i\)，潜在转化结果为 \(Y_i(1)\) 和 \(Y_i(0)\)。')
        + _paragraph(r'其中，\(Y_i(1)\) 表示用户被分配至广告处理组时的潜在转化结果，\(Y_i(0)\) 表示用户被分配至对照组时的潜在转化结果。')
        + _paragraph(r'设业务分配策略为 \(\pi_i\)，其取值为 0 或 1，分别表示不分配或分配广告资格。')
        + symbols
        + '<h4>（2）业务优化目标</h4>'
        + _paragraph('本项目将广告资格覆盖率固定为 20%，因此：')
        + _equation(r'K=\lfloor0.2N\rfloor', '固定广告资格名额')
        + _equation(
            r'\boxed{\begin{aligned}'
            r'\max_{\boldsymbol{\pi}}\quad'
            r'&\underbrace{\sum\limits_{i=1}^{N}\mu_0(X_i)}_{\text{自然转化基础}}'
            r'+\underbrace{\sum\limits_{i=1}^{N}\pi_i\tau(X_i)}_{\text{策略产生的预期增量转化}}\\'
            r'\text{s.t.}\quad &\sum_{i=1}^{N}\pi_i=K=\lfloor0.2N\rfloor\\'
            r'&\pi_i\in\{0,1\},\quad i=1,\ldots,N'
            r'\end{aligned}}',
            '总体预期转化人数目标与约束',
        )
        + '<p>其中：</p><ul><li>第一项表示全部候选用户在未获得广告资格时的预期自然转化人数。</li>'
        + '<li>第二项表示广告资格分配策略带来的预期增量转化人数。</li></ul>'
        + '<h4>（3）业务假设与模型约束</h4>'
        + _paragraph('上述优化模型建立在以下业务假设之上：')
        + assumptions
        + _paragraph('在这些条件下，最大化总体预期转化人数等价于最大化总体预期增量转化人数。')
        + '<h4>（4）由业务优化目标引出 CATE 估计</h4>'
        + _paragraph('由于候选用户群体固定，自然转化基础不随广告资格分配策略变化。因此，业务优化的关键在于估计每名用户的增量转化效果：')
        + _equation(r'\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x]=\mu_1(x)-\mu_0(x)', '条件平均处理效应（CATE）')
        + _paragraph('该量即为条件平均处理效应（Conditional Average Treatment Effect，CATE）。')
        + _paragraph(r'在真实 CATE 已知的情况下，按照 \(\tau(X_i)\) 从高到低选择 Top-K 用户，即可实现固定覆盖率下的理论最优分配。')
        + _paragraph(r'实际项目中，真实 CATE 无法直接观测，需要通过 Uplift 模型学习预测值 \(\hat{\tau}(X)\)，并根据预测增量效果进行用户排序。本项目采用固定 20% 覆盖率下的 HT-Gain 评价不同策略的增量转化表现。')
        + _paragraph('本项目以增量转化人数为优化目标。由于 Criteo 公开数据缺少用户级广告成本和转化金额，研究结论限定于固定资源条件下的增量转化优化。')
        + _details('theory-business-cate', '理论推导｜业务目标函数与 CATE 排序的关系', theory)
    ))
    if (out / 'diagnostics/full_feature_iqr.json').is_file():
        main.append(_dataset_audit_section(out, manifest, audit))
    main.append(_identification_section(
        audit, srm, balance, c2st, c2st_null, positivity, read_csv('diagnostics/learnability.csv'),
    ))
    main.append(_ate_section(out, ate))
    main.append(_uplift_methods_section(manifest['assignment_probability']))
    gain_comparison = _gain_comparison(primary)
    gain_figure = (
        '<figure>' + _gain_plot(curves, list(dict.fromkeys((selected, uplift, response))))
        + '<figcaption>同一留出总体、冻结评分与固定处理比例。</figcaption></figure>'
    )
    if margins is None:
        expansion = _paragraph(
            f'10%–20% 覆盖贡献 {marginal_10_20:.2f}，20%–30% 贡献 {marginal_20_30:.2f} 次增量转化／10 万名合资格用户。')
    else:
        expansion = (
            '<h4>表 6-6｜冻结策略逐档扩量的边际增益</h4>'
            + _paragraph('HT 边际 Gain、对应区间和 Uplift 边际 Gain 均以每 10 万名合资格用户计。')
            + '<div class="main-pairs">' + _table(pd.DataFrame({
                '新增覆盖': [f'{r.start:.0%}–{r.end:.0%}' for r in margins.itertuples()],
                'HT 边际 Gain': (margins.gain_ht * 1e5).map(_num),
                'HT 配对 95% CI': margins.apply(lambda r: _interval(r, scale=1e5), axis=1),
                'Uplift 边际 Gain': (margins.gain_plugin * 1e5).map(_num),
                '每千新增名额的 HT 增量': (margins.gain_ht * test_n / margins.n_added * 1000).map(lambda x: _num(x, 3)),
            })) + '</div>'
            + _paragraph(f'配对区间按 {BOOTSTRAP_REPEATS:,} 次处理臂内分层 Bootstrap 计算，在同一轮重采样内计算相邻覆盖率的累计收益差。')
        )
    main.append(_chapter_six(out, manifest, selection, matched, gain_figure, expansion))
    # Representative economic example is selected from saved scenarios, never refitted or optimized.
    assumptions = read_json('business/assumptions.json')
    scenarios = read_csv('business/hypothetical_cost_scenarios.csv')
    if margins is not None:
        scenarios = pd.DataFrame([
            cost_value_scenario(
                value, eligible_n=int(r.eligible_n), budget=r.budget,
                value_per_conversion=r.value_per_conversion_assumed, contact_cost=r.contact_cost_assumed,
                ci_low=primary_boot.loc[selected, 'ci_low'] * 1e5, ci_high=primary_boot.loc[selected, 'ci_high'] * 1e5,
            )
            for r in scenarios.itertuples()
        ])
    scenario = scenarios.iloc[len(scenarios)//2]
    expansion = ''
    if margins is not None:
        band = margins.loc[margins.start.eq(.1)].iloc[0]
        value_scale = scenario.eligible_n * scenario.value_per_conversion_assumed
        benefit = band.gain_ht * value_scale
        plugin_benefit = band.gain_plugin * value_scale
        added = int(scenario.eligible_n * .2) - int(scenario.eligible_n * .1)
        extra_cost = added * scenario.contact_cost_assumed
        net_limits = band[['ci_low', 'ci_high']] * value_scale - extra_cost
        expansion = _paragraph(
            f'同一假设下，从 10% 扩至 20% 需增加 {added:,} 个资格名额，成本增加 {extra_cost:,.2f}。'
            f'新增转化价值按 HT 估计为 {benefit:,.2f}，按 Uplift Gain 估计为 {plugin_benefit:,.2f}。'
            f'HT 边际净价值估计为 {benefit-extra_cost:+,.2f}，条件 95% CI [{net_limits.ci_low:,.2f}, {net_limits.ci_high:,.2f}]。'
            f'按 HT 点估计，每个新增资格名额的盈亏平衡成本为 {benefit/added:.4f}。'
             '此情景下 10%→20% 扩量降低净价值，后续应以真实成本与转化价值验证覆盖率边界。'
        )
    from .report_followup import analysis_path, chapter_seven, chapter_eight
    experiment = analysis_path(out)
    if experiment:
        main.extend([chapter_seven(out, experiment), chapter_eight()])
    else:
        main.append(_section('decision', '07｜策略推荐、扩量成本与线上验证',
            _paragraph(f'保留本阶段冻结策略 {label}，将 {uplift_label} 作为增量建模候选进行后续研究。'
                       + ('这是一项开发阶段结论，正式推荐应以全量实验的验证冻结和独立评价为准。' if development else
                          '全量正式推荐沿用 Validation Set 冻结的 Response-XGBoost，并将 S-XGBoost 保留为后续线上挑战策略。')
                       + '投放资格名额是本项目的资源口径，实际竞价花费需要另行验证。')
            + _paragraph(f'代表性经济情景：对 {int(scenario.eligible_n):,} 名合资格用户固定覆盖 {scenario.budget:.0%}，'
                         f'假设每次增量转化价值 {scenario.value_per_conversion_assumed:g} 个货币单位、每次触达成本 {scenario.contact_cost_assumed:g}，'
                         f'保存结果对应的条件净增量价值为 {scenario.net_incremental_value:,.2f}，盈亏平衡单位触达成本为 {scenario.break_even_contact_cost:.4f}。'
                         '上述净价值与盈亏平衡值是固定价值、成本及离线效果可迁移条件下的业务情景估算。'
                         '相关假设与情景结果保存在 business 目录。')
            + expansion
            + _paragraph(f'建议将 {uplift_label} 与 {response_label} 作为候选策略进行用户级随机对照。两臂分别在各自入组用户中按固定 20% 覆盖执行，'
                         '以全部随机入组用户的固定窗口转化率作为主指标，衡量真实策略间的增量差异。预先确定 MDE、样本量、观测窗口和停止规则，'
                         '监控 SRM、跨组污染、实际花费、频次、用户负反馈及业务护栏。达到计划样本量并完成观察窗口后，'
                         '只有转化率达到预先约定的改善标准、增量价值覆盖新增成本且护栏通过，才切换策略。证据不足时保留基线。'
                         '估计各策略相对完全无广告的绝对增量时，另设不投放留出组。线上实验尚未执行。')
            + _paragraph('两阶段报告承担不同职责：开发实验验证流程并选出 T-Logistic，全量实验用更大的训练和评价样本形成正式决策。'
                         '两阶段的训练规模、留出人群和拟合模型不同，因此分别呈现阶段内的相对策略收益。'
                         '全量评价增加正例支持并缩窄条件区间；跨种子重训与选型不确定性列为后续稳健性验证重点。')
            + _paragraph('本研究把“谁更可能转化”转化为可检验的“同样名额能创造多少额外转化”，并用预冻结流程约束决策。'
                         '当前证据支持优先实施有信息的定向，并以响应基线作为可解释、可复现的正式候选。'
                         '下一轮投入优先验证目标人群中的真实增量、实际投放成本与用户级实验结果，为规模化部署提供决策证据。')))
    research = ''.join(main)
    header = _report_header(research, development=development)
    if development:
        research = research.replace('独立 Test Set', '开发留出集')
    document = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{html.escape(_TITLE)}｜Criteo Uplift</title><style>{_REPORT_CSS}</style>{_MATHJAX}</head><body><main>'
                + header + '<article id="research" aria-label="研究正文">' + research + '</article>'
                + '<footer>Validation Set 选型已冻结 · <a href="#research">返回研究正文</a></footer></main>'
                + _NAV_SCRIPT + '</body></html>')
    document = _refresh_report_front_matter(document, development=development)
    report_path.write_text(document,encoding='utf-8')
    # Public provenance contains only hashes and relative artifact identifiers, no row-level records.
    inputs = [p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.csv')]
    provenance = {'original_run':out.name,'selected_policy':selected,
                  'selected_uplift_model':uplift,'inputs':{str(p.relative_to(out)):sha256_file(p) for p in sorted(inputs)},
                  'score_sha256':matched['source_sha256'],'report_sha256':sha256_file(report_path),
                  'generator_sha256':sha256_file(Path(__file__)),
                  'report_calculations': ['Top 20% overlap, Jaccard, raw-score Spearman']
                  + ([f'Paired HT bootstrap ({BOOTSTRAP_REPEATS} draws) for budget margins and key policies'] if margins is not None else [])}
    write_result(provenance,report_path.parent/'report_provenance.json')
    if experiment:
        from .report_followup import record_publication
        record_publication(report_path, out, experiment)
    return report_path


_MATHJAX = r"""
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    processEnvironments: false
  },
  chtml: {mtextInheritFont: true}
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
"""


_REPORT_CSS = """
:root{color-scheme:light;--ink:#263746;--blue:#21628e;--pale:#eef5fa;--line:#d9e4ed}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f7fa;color:var(--ink);font:16px/1.85 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1120px;margin:auto;padding:42px 28px}header{padding:12px 6px 24px}h1{font-size:35px;line-height:1.4;margin:10px 0}h2{font-size:25px;line-height:1.5;margin:0 0 20px}h3{font-size:19px;margin:28px 0 12px}p{margin:14px 0}a{color:var(--blue);text-underline-offset:3px}b,strong{font-weight:650}
.report-subtitle{font-size:17px;color:#607487;margin:12px 0 24px}.report-contents{margin-top:24px}.toc-label{font-size:13px;color:#607487;margin:0 0 8px}.report-toc{display:grid;grid-template-columns:1fr;column-gap:28px;list-style:none;padding:0;margin:0}.report-toc li{min-width:0}.report-toc a{display:flex;align-items:baseline;gap:12px;padding:10px 0;border-bottom:1px solid var(--line);font-size:14px;text-decoration:none}.report-toc a:hover{text-decoration:underline}.toc-number{flex:0 0 30px;color:#607487;font-variant-numeric:tabular-nums;font-size:12px}
@media(max-width:700px){.report-toc{grid-template-columns:1fr}.report-subtitle{font-size:15px}.report-toc a{font-size:13px}}
@media print{.report-contents{display:none}}
.eyebrow{font-size:12px;letter-spacing:.12em;color:var(--blue)}section,#workflow{border:1px solid var(--line);background:white;padding:30px;border-radius:8px;margin:24px 0;scroll-margin-top:18px}
.reading-entries,.chapter-nav{display:flex;flex-wrap:wrap;gap:12px;margin-top:20px}.reading-entries a{padding:8px 14px;background:var(--pale);border-radius:4px;text-decoration:none;font-weight:600}.chapter-nav a{font-size:14px}
.workflow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));list-style:none;padding:0;gap:12px;counter-reset:steps}.workflow li{counter-increment:steps;background:var(--pale);padding:12px;border-radius:4px}.workflow li:before{content:counter(steps) " → ";color:#607e96}.workflow a{font-size:14px;text-decoration:none}
.lead{font-size:19px}.cards,.split-flow,.overlap-counts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}.cards>div,.split-flow>div,.overlap-counts>div{background:var(--pale);padding:18px;border-radius:5px}.cards span,.cards small,.split-flow span,.overlap-counts span,.overlap-counts small{display:block;font-size:13px;color:#597185}.cards strong,.overlap-counts strong{font-size:27px;display:block}.split-flow b{display:block;font-size:15px}.shared{outline:2px solid #9fbbce}
figure{margin:24px 0}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#607487;line-height:1.7;margin:12px 0}.example{background:#fff;padding:26px 28px;border:1px solid #e5e7eb;border-radius:16px}.example h3{font-size:27px;line-height:1.45;margin:0 0 22px;color:#202428}.example-row{margin:0!important;padding:8px 0 24px}.example-row+.example-divider{margin:0 0 22px;border:0;border-top:1px solid #e7e9eb}.example-row b{display:block;font-size:19px;margin:0 0 16px;color:#202428}.example-metrics{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:10px;font-size:18px;font-weight:650}.example-metrics .natural{color:#73a0c4}.example-metrics .uplift{color:#c48645}.example .bar{height:26px;background:#e6edf2;display:flex;margin:0;border-radius:8px;overflow:hidden}.example .bar i{display:block;background:#73a0c4}.example .bar em{display:block;background:#c48645}.example .example-outcome{margin:7px 0 0;color:#6f7377;font-size:17px}.example figcaption{font-size:14px;color:#6f7377;line-height:1.7;margin:16px 0 0}.math-display{font-family:Georgia,"Times New Roman",serif;font-size:21px;line-height:1.9;text-align:center;white-space:normal;overflow-wrap:anywhere;color:#183c56;padding:2px 0}.theory-block{margin-top:22px}
.table-wrap{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:13px;line-height:1.65;background:#fff}th,td{text-align:left;vertical-align:top;padding:11px 10px;border-bottom:1px solid #dbe4eb;overflow-wrap:anywhere}th{background:#fff;border-bottom-color:#b9c8d4}tbody tr:last-child td{border-bottom:0}.wide table{min-width:980px;table-layout:auto}.wide td{min-width:85px}.wide th{min-width:100px}
.equation{background:#fff;margin:16px 0}details .equation{margin:8px 0}details .equation-label{display:none}.equation-label{font-size:13px;font-weight:600;color:#5c7790;margin-bottom:12px}.equation-scroll{overflow-x:auto}.math-equation{width:max-content;min-width:100%;white-space:nowrap;color:#183c56}
.policy-chart{--cols:minmax(140px,1fr) minmax(280px,2.7fr) minmax(130px,.85fr);background:#f6f9fc;border:1px solid var(--line);padding:20px;border-radius:5px}.chart-unit{font-size:13px;color:#5c7790}.policy-row{display:grid;grid-template-columns:var(--cols);gap:16px;align-items:center;padding:10px;font-size:14px}.policy-row.frozen{background:#e4eef6;border-radius:5px}.policy-track{position:relative;height:22px;background:#edf2f6}.policy-zero,.policy-ci,.policy-dot{position:absolute;top:50%;background:#4f7ca0}.policy-zero{top:0;bottom:0;width:1px;background:none;border-left:1px dashed #8ea2b3}.policy-ci{height:3px}.policy-dot{height:9px;width:9px;border-radius:50%;transform:translate(-50%,-50%);border:1px solid white}.policy-value{font-variant-numeric:tabular-nums}.policy-value small{display:block;font-size:12px;color:#5a7184}.policy-axis-row{display:grid;grid-template-columns:var(--cols);gap:16px;padding:0 10px}.policy-axis{position:relative;height:24px;margin:8px 0;font-size:12px;color:#607487}.policy-axis>span{position:absolute;transform:translateX(-50%)}.policy-break{position:absolute;left:15%;top:50%;transform:translate(-50%,-50%);background:#fff;color:#607487;padding:0 2px;font-size:13px}.policy-axis>.policy-break{left:15%;top:0;transform:translateX(-50%)}
.appendix-topic{border:1px solid var(--line);border-radius:5px;margin:15px 0;background:white}.appendix-topic>summary{padding:15px 18px;cursor:pointer;font-weight:650;color:#245f89;background:#f4f8fb}.appendix-content{padding:8px 18px 18px}.appendix-topic[open]>summary{border-bottom:1px solid var(--line)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.65;background:#f6f8fa;padding:16px;border-radius:4px}code{overflow-wrap:anywhere;font-size:.9em}footer{font-size:13px;text-align:center;color:#667c8d;padding:20px}.appendix-content .table-wrap{max-height:680px;overflow:auto}.appendix-content th{position:sticky;top:0;z-index:1}

.paired-gain-card{border:1px solid var(--line);border-radius:7px;background:#f8fafc;padding:22px 24px;margin:20px 0}
.paired-gain-card figcaption{margin:0 0 12px;color:#597185}
.paired-gain-values{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.paired-gain-values>div{border:1px solid #d9e4ed;background:white;border-radius:5px;padding:16px}
.paired-gain-values span{display:block;color:#5c7790;font-size:13px}
.paired-gain-values strong{font-size:27px;line-height:1.4;display:block}
.paired-delta{display:flex;gap:15px;align-items:baseline;flex-wrap:wrap;margin:20px 0 8px}
.paired-delta b{font-size:14px}.paired-delta strong{font-size:26px;color:#183c56}.paired-delta span{font-size:14px;color:#567087}
.paired-track{height:37px;position:relative;margin:8px 4px 0;border-bottom:1px solid #d8e1e9}
.paired-zero{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#899dad}
.paired-ci{position:absolute;top:17px;height:4px;border-radius:3px;background:#3f7eab}
.paired-point{position:absolute;top:11px;width:14px;height:14px;border-radius:50%;transform:translateX(-50%);background:#e38a38;border:2px solid white}
.paired-scale{display:flex;justify-content:space-between;color:#667c8d;font-size:12px;padding:6px 3px 0}
.paired-note{margin:12px 0 0;font-size:14px;line-height:1.7}

@media(max-width:700px){main{padding:20px 12px}section,#workflow{padding:18px;margin:18px 0}h1{font-size:28px}h2{font-size:23px}.workflow{grid-template-columns:repeat(2,minmax(0,1fr))}.cards,.split-flow,.overlap-counts{grid-template-columns:1fr}.policy-chart{padding:12px}.policy-row{grid-template-columns:minmax(0,1fr) minmax(100px,.9fr);gap:5px;padding:10px 4px}.policy-track{grid-row:2;grid-column:1/-1;margin-top:4px}.policy-value{text-align:right}.policy-axis-row{display:block;padding:0 4px}.policy-axis{margin:12px 0}.main-pairs table,.main-pairs tbody,.main-pairs tr,.main-pairs td{display:block;width:100%}.main-pairs thead{display:none}.main-pairs tr{padding:10px 0;border-bottom:1px solid #dbe4eb}.main-pairs tr:last-child{border-bottom:0}.main-pairs td{display:grid;grid-template-columns:1fr 1.4fr;gap:10px;border:0;padding:5px}.main-pairs td:before{content:attr(data-label);color:#58738a;font-weight:600}.appendix-content{padding:4px 10px 12px}th,td{padding:8px 5px;font-size:12px}.lead{font-size:17px}}
@media print{@page{size:A4;margin:15mm 14mm}body{background:white;font-size:10pt;line-height:1.6;print-color-adjust:exact}main{padding:0;max-width:none}section,#workflow{padding:10pt 0;border:0;margin:8pt 0}h1{font-size:24pt}h2{font-size:18pt}h3{font-size:13pt}h2,h3,thead{break-after:avoid}p{orphans:3;widows:3}.reading-entries,.chapter-nav,footer{display:none}.workflow{grid-template-columns:repeat(4,1fr)}figure,.equation,.cards,.split-flow,.overlap-counts,tr{break-inside:avoid}img{max-height:145mm;object-fit:contain}.table-wrap{overflow:visible}table{font-size:8pt}details:not([open]){display:none}.appendix-content .table-wrap{max-height:none;overflow:visible}.wide table{min-width:0;table-layout:fixed}.wide th,.wide td{min-width:0}.policy-chart{--cols:130px minmax(130px,2fr) 120px;padding:10pt}.policy-row{font-size:9pt;gap:10px}.policy-axis{margin:6pt 0}}
"""

_NAV_SCRIPT = '''<script>
(function(){
  function reveal(hash){
    var node=document.getElementById(hash.slice(1));
    if(!node)return;
    for(var p=node;p;p=p.parentElement){if(p.tagName==='DETAILS')p.open=true;}
  }
  document.addEventListener('click',function(e){var a=e.target.closest('a[href^="#"]');if(a)reveal(a.hash);});
  window.addEventListener('hashchange',function(){reveal(location.hash);});
  reveal(location.hash);
})();
</script>'''
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
    #     - assignment_probability：用于 HT 评价的 Train Set 处理比例代入值。
    #     - fitted：训练和 Validation 阶段返回的模型、元数据及冻结选择。
    #     - elapsed_seconds：本次 Pipeline 截至报告生成前的累计耗时秒数。
    # 输出参数：
    #     - 无；写入 manifest.json 并生成 report.html。
    #-------------
    import resource
    manifest = {
        "resource_usage": {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)},
        "assignment_probability": assignment_probability,
        "probability_source": "使用 Train Set 的处理组比例作为分配概率的代入估计",
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

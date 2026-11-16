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

_TITLE = "固定广告资格覆盖率下的增量转化优化"

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
        f'<figcaption><strong>独立测试集（{n:,} 行）</strong> · 圆点为点估计，横线为各策略的 95% Bootstrap 区间。有效重复次数见附录 E。'
        '策略差异依据配对区间判断。</figcaption></figure>'
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
    note = ('配对区间跨越零点。本阶段沿用验证集冻结策略，后续以线上实验检验挑战策略的业务价值。'
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


def _metric_table(metrics, bootstrap):
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


def build_report(manifest_path: Path, report_path: Path) -> Path:
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
    selection = read_json('models/selection.json')
    audit = read_json('data_preparation/audit.json')
    sample = read_json('data_preparation/sample_report.json')
    split = read_csv('data_preparation/split_summary.csv')
    ate = read_csv('evaluation/ate_summary.csv')
    balance, c2st = read_csv('diagnostics/balance.csv'), read_csv('diagnostics/c2st.csv')
    c2st_null = read_csv('diagnostics/c2st_null.csv')
    positivity, srm = read_json('diagnostics/overlap_positivity.json'), read_json('diagnostics/srm.json')
    metrics, bootstrap = read_csv('evaluation/test_metrics.csv'), read_csv('evaluation/test_bootstrap.csv')
    pairs, curves = read_csv('evaluation/test_pairwise.csv'), read_csv('evaluation/curves.csv')
    val_metrics, val_boot = read_csv('models/validation_metrics.csv'), read_csv('models/validation_bootstrap.csv')
    val_pairs = read_csv('models/validation_pairwise.csv')
    args = manifest['arguments']
    selected, uplift = selection['selected_policy'], selection['selected_uplift_model']
    response = 'response_xgboost'
    if selection['primary_metric'] != 'gain_ht' or selection['primary_budget'] != .2 or not selection['frozen_before_test']:
        raise ValueError('报告需要预冻结的 HT Gain @20% 协议')
    if set(metrics.policy) != {*selection['candidates'], 'random'}:
        raise ValueError('九策略结果不完整')
    development = args.get('stage') == 'development'
    stage_title = '500 万开发实验报告' if development else '全量正式实验报告'
    scope = '开发留出集' if development else '正式测试集'
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
    full_ate = ate[(ate.scope == 'full_dataset') & (ate.outcome == 'conversion')].iloc[0]
    train_c = split[(split.split == 'train') & (split.treatment == 0)].iloc[0]
    test_c = split[(split.split == 'test') & (split.treatment == 0)].iloc[0]
    events = int(split.loc[split.split.eq('test'), 'conversion_positive'].sum())
    full_gain = frozen.loc[1., 'gain_ht_per_100k']
    captured = value / full_gain if full_gain != 0 else np.nan
    marginal_10_20 = value - frozen.loc[.1, 'gain_ht_per_100k']
    marginal_20_30 = frozen.loc[.3, 'gain_ht_per_100k'] - value
    inference = ('配对区间跨越零点，当前留出评价显示两策略收益接近；后续可通过线上实验检验细微差异的业务价值。'
                 if uplift_pair.ci_low <= 0 <= uplift_pair.ci_high else
                 '该配对区间未包含零，方向应结合点估计解读；正式策略仍保持验证集冻结结果。')
    intro = ('本报告用于检查完整流程、训练诊断与开发期策略表现。验证集冻结 T-Logistic，内部留出评价全部来自预冻结正式测试集之外。'
             if development else '本报告回答固定广告资格分配规模下的投放选择问题。正式策略是在验证集预先冻结的 Response-XGBoost，正式测试集只承担效果评价。')
    metric_cards = f'''<div class="cards"><div><span>预设覆盖率</span><strong>20%</strong><small>固定业务假设</small></div>
    <div><span>冻结策略 HT Gain</span><strong>{value:.2f}</strong><small>95% CI {interval}</small></div>
    <div><span>相对 Random 的配对差值</span><strong>{vs_random.difference*1e5:+.2f}</strong><small>95% CI {_interval(vs_random, scale=1e5)}</small></div></div>'''
    main = []
    main.append(_section('summary', '01｜研究结论与业务决策',
        _paragraph(f'<strong>固定覆盖 20% 合资格用户，保留验证集冻结的 {label}。</strong> {intro}', 'lead')
        + metric_cards
        + _paragraph(f'原始数据共 {audit["n_rows"]:,} 行，本阶段建模 {manifest["sample_n"]:,} 行，{scope} {test_n:,} 行。'
                     f'上述 HT Gain 与区间单位均为每 10 万名<strong>全部合资格用户</strong>的预期增量转化。'
                     '它衡量固定资格覆盖策略相对全员不分配资格、在全部合资格用户范围内带来的额外转化。')
        + _paragraph(f'定向策略相对随机分配获得更高增量，是本研究最直接的业务价值。'
                     f'验证集最佳 Uplift 候选为 {uplift_label}，在{scope}上相对 {response_label} 的点差为 '
                     f'{uplift_pair.difference*1e5:+.2f}，配对 95% CI {_interval(uplift_pair, scale=1e5)}。{inference}')
        + _paragraph('项目交付了经过验证集选型和留出评价的策略推荐，并形成了可执行的线上实验方案。'
                     '按预设验证集规则保留冻结策略，并在目标业务人群通过同覆盖率随机实验比较挑战策略。'
                     '<a href="#decision">查看最终业务建议</a>。')))
    flow = [('business','业务问题'),('identification','数据与因果识别'),('ate-main','ATE'),('methods','CATE / Response 建模'),
            ('selection-flow','验证集冻结'),('results','测试集策略评价'),('interpretation','结果解释'),('decision','业务决策')]
    workflow = '<aside id="workflow"><h2>项目研究流程</h2><ol class="workflow">' + ''.join(
        f'<li><a href="#{anchor}">{title}</a></li>' for anchor,title in flow) + '</ol><p>沿正文顺序阅读即可复述项目；蓝色附录链接用于核验细节。</p></aside>'
    example = '<figure class="example"><div><b>高自然转化人群</b><span>80% → 85%　增量 +5 个百分点</span><div class="bar"><i style="width:80%"></i><em style="width:5%"></em></div></div><div><b>高广告敏感人群</b><span>10% → 35%　增量 +25 个百分点</span><div class="bar"><i style="width:10%"></i><em style="width:25%"></em></div></div><figcaption>示意数据，非 Criteo 实验结果。蓝色为不处理转化概率，橙色为处理分配带来的增量；两段之和为处理后转化概率。</figcaption></figure>'
    main.append(_section('business', '02｜为什么研究增量转化',
        _paragraph('广告平台需要从合资格用户中选择目标人群。预算、广告库存和频次限制使每个触达名额都有机会成本。'
                   '传统 Response 根据处理条件下的转化概率 μ₁(X) 排序，适合回答“获得广告资格后谁更可能购买”。'
                   'Uplift 进一步估计广告资格分配给不同用户的增量，直接服务于有限资格名额下的资源分配。')
        + example
        + _paragraph('示例中，Response 会优先选择高自然转化人群，因为 85% 高于 35%；Uplift 会优先选择广告敏感人群，因为 25 个百分点高于 5 个百分点。'
                     '同样增加 100 个处理名额，后者预期多带来 20 次增量转化。因此需要在真实数据中比较不同评分策略的实际增量收益。')
        + _equation(r'\mathbb{E}[\mathrm{Total\ Conversions}\mid X]=\sum_i\mu_0(X_i)+\sum_i\pi_i\tau(X_i)', '总体预期转化：自然转化基础 + 策略产生的增量')
        + _paragraph('当候选人群固定时，第一项不随策略变化。若覆盖人数固定、单位触达成本与转化价值相同、用户间不存在处理干扰，且真实 CATE 已知，'
                     '按真实 CATE 取 Top K 同时最大化总体预期增量转化和总体预期转化人数。若允许少于 K 人且部分效应为负，约束和最优规则会不同；本项目固定人数。')
        + _paragraph('实际能用于排序的是预测 CATE，误差可能来自有限正例、模型设定与反事实估计。理论最优性的落地依赖拟合误差、正例支持与反事实响应面的估计质量。'
                     '本项目将 20% 覆盖率设为事先确定的资格资源约束；通过实验数据比较策略价值，检验增量建模能否兑现理论动机。')))
    overview = _table(pd.DataFrame([
        ('原始公开记录', f'{audit["n_rows"]:,} 行'), ('本阶段建模 / 留出评价', f'{manifest["sample_n"]:,} / {test_n:,} 行'),
        ('处理组 / 对照组占比', f'{srm["observed_ratio"]:.2%} / {1-srm["observed_ratio"]:.2%}'),
        ('公开数据处理组转化率 / 正例', f'{full_ate.response_treatment:.4%} / {audit["groups"]["1"]["conversion"]["positive"]:,}'),
        ('公开数据对照组转化率 / 正例', f'{full_ate.response_control:.4%} / {audit["groups"]["0"]["conversion"]["positive"]:,}'),
        ('留出转化正例 / 对照正例', f'{events:,} / {int(test_c.conversion_positive):,}'),
    ], columns=['范围', '真实数据概览']))
    diag_rows = [
        ('SRM', f'对 85% 外部参考的 Pearson p={srm["pvalue"]:.4f}', '核对公开样本的处理分配比例'),
        ('SMD / KS', f'最大 |SMD|={balance.abs_smd.max():.4f}；最大 KS={balance.ks_statistic.max():.4f}', '单变量差异幅度较小，仍需联合诊断'),
        ('C2ST', '；'.join(f'{r.model} AUC {r.roc_auc:.4f}，Holm p={r.pvalue_holm:.3f}' for r in c2st.itertuples()), '统计可检出性与实际差异幅度分别解读'),
        ('覆盖与正值性', f'训练集缺臂箱 {positivity["missing_arm_bins"]}；薄弱箱 {positivity["weak_bins"]}', '检查单变量分箱中的两臂观测支持'),
    ]
    main.append(_section('identification', '03｜数据如何支持因果评价',
        _paragraph('项目使用 Criteo Uplift v2.1 公开广告增量实验数据。treatment 表示广告资格分配，exposure 表示分配后的实际曝光；'
                   'conversion 是主要结果，visit 是辅助结果。只有匿名处理前特征 f0–f11 进入排序模型，既不按曝光筛选用户，也不把曝光或结果标签作为输入。'
                   '这是避免处理后选择偏差与结果泄漏的第一步。')
        + overview
        + _paragraph('Criteo 为保护隐私对公开数据进行了非均匀抽样，原始实验的增量水平无法从公开数据恢复。'
                     '本文的统计结论以公开样本为限。<a href="https://huggingface.co/datasets/criteo/criteo-uplift">Criteo 数据卡</a>')
        + _paragraph('用 Y(1) 与 Y(0) 表示同一用户在获得和未获得广告资格时的潜在结果。每人只能观察其中之一。'
                     '随机分配在实验设计层面支持可交换性，还需要处理定义一致、无干扰（SUTVA）及目标人群中两种分配均有可能。'
                     '按原始资格分配保留全部处理组样本，将资格分配的 ITT 定义为主要因果目标。')
        + _table(pd.DataFrame(diag_rows, columns=['诊断', '保存结果', '解释边界']))
        + _paragraph('联合采用 SRM、SMD/KS、C2ST 与分箱覆盖检查公开样本中的可观察分配特征。'
                     'Logistic C2ST 检出轻微可分信号，XGBoost C2ST 则接近随机水平；本项目结合效应量与多重检验结果解读该诊断。'
                     f'HT 使用训练集处理比例 e={manifest["assignment_probability"]:.6f} 作为固定概率的代入估计；若真实分配概率随特征变化，恒定概率近似会限制识别。'
                     '<a href="#appendix-diagnostics">诊断范围、置换分布与全部明细</a>保留在附录。')
        + '<div id="ate-main"><h3>从总体增量效果到用户定向决策</h3>'
        + _equation(r'\mathrm{ATE}=\mathbb{E}[Y(1)-Y(0)],\qquad\tau(x)=\mathbb{E}[Y(1)-Y(0)\mid X=x]', 'ATE 与 CATE：总体平均增量与条件平均增量')
        + _paragraph(f'全部 {int(full_ate.n):,} 行的对照转化率为 {full_ate.response_control:.4%}，处理转化率为 {full_ate.response_treatment:.4%}。'
                     f'分配 ITT 为 <strong>+{full_ate.ate*100:.4f} 个百分点</strong>，95% CI {_interval(full_ate,scale=100,digits=4)} 个百分点，'
                     f'相对增幅 {full_ate.relative_uplift:.2%}。全量统计来自原始审计计数，未用开发抽样替代；该总体分析不参与模型选择。')
        + _paragraph('ATE 量化公开样本的平均资格分配效果，CATE 建模与排序评价进一步支持有限名额下的用户选择。'
                     '固定 20% 决策需要估计异质性并评价排序前缀。与此同时，转化稀疏且对照组较小，反事实响应面可用的信息明显少于总行数给人的直觉。'
                     '<a href="#appendix-data">两阶段划分、ATE 的标准误、Z 检验和完整区间</a>可进一步核验。') + '</div>'))
    methods = _table(pd.DataFrame([
        ('S-Learner', '单个结果模型共享两臂信息，输入 X 与 T', 'f̂(X,1) − f̂(X,0)'),
        ('T-Learner', '两臂分别拟合响应面', 'μ̂₁(X) − μ̂₀(X)'),
        ('X-Learner', '先拟合响应面，再回归两臂连续伪效应', '按训练处理比例组合效应预测'),
        ('Response', '只用训练集处理组拟合；复用对应 T 模型', 'μ̂₁(X)（处理后响应概率）'),
        ('Random', '不训练、不参加选型', '同覆盖率随机分配的期望收益'),
    ], columns=['方法', '作用', '评分']))
    counts = split.groupby('split', sort=False).n.sum()
    main.append(_section('methods', '04｜如何建模、冻结并评价策略',
        _paragraph('比较六种 CATE 模型与两种 Response 模型，可以分别检验学习器结构和基础模型的作用。S/T/X 各搭配 Logistic 与 XGBoost，'
                   'Response 的两种版本提供传统响应定向基线，Random 则回答定向本身是否有价值。所有策略使用相同特征、划分、覆盖率和评价单位。')
        + methods
        + _paragraph('S/T/X 体现不同的偏差—方差与样本利用取舍。S 共享样本信息，可能减少估计方差，但处理效应也可能被整体拟合目标弱化；'
                     'T 分开估计两条响应面，灵活性更高，也可能累积两组误差；X 增加伪效应学习步骤，能利用两臂信息，却不会自动消除第一阶段误差。'
                     f'本阶段训练集对照组只有 {int(train_c.conversion_positive):,} 个转化正例，结构灵活性与可用信息之间的权衡尤其重要。')
        + _paragraph('Logistic 的中位数填补与 StandardScaler 都只在训练数据上拟合。S-Learner 把处理指示 T 作为额外输入，也经过标准化；'
                     '采用 L2 正则化、C=1 和最多 500 次 LBFGS 迭代。标准化改变受惩罚系数的尺度与数值条件，使不同量纲特征的惩罚更可比；'
                     '模型仍需从数据中学习处理效应差异。X-Logistic 第二阶段是连续线性回归，X-XGBoost 第二阶段是连续回归树。'
                     '<a href="#appendix-models">全部参数、训练状态与正例支持</a>可展开核对。')
        + f'<div class="split-flow" id="selection-flow"><div><b>训练集 {counts["train"]:,}</b><span>拟合模型及预处理</span></div><div><b>验证集 {counts["validation"]:,}</b><span>HT Gain @20% 的 Bootstrap 均值选型</span></div><div><b>{scope} {counts["test"]:,}</b><span>加载冻结评分规则，最终评价</span></div></div>'
        + _paragraph('正式测试行号在新增开发之前固定，并排除可追溯历史建模行。500 万开发样本全部来自该测试集之外，正式实验再使用预冻结测试行。'
                     '提前留出是为了限制开发反馈回流；用验证集确定模型，避免把测试集的偶然领先当成可推广收益。'
                     '数据使用范围通过可追溯行号与冻结指纹记录；匿名记录的用户级隔离及未留存历史访问留作研究边界，详见附录。')
        + _equation(r'\widehat{G}_{\mathrm{HT}}(\pi)=\frac{1}{N}\sum_i\pi(X_i)\left[\frac{T_iY_i}{e}-\frac{(1-T_i)Y_i}{1-e}\right]', '唯一主业务指标：HT Gain @20%')
        + _paragraph('HT 仅累计策略选中用户的逆概率加权结果，分母始终是全部 N 名合资格评价用户。乘以 100,000 后，就是每十万名候选用户的预期增量转化。'
                     '取 K=floor(0.20N)，不足一人的偏差来自整数预算约束；不要求选中集合内的实际处理比例恰好等于总体比例。')
        + _paragraph(f'验证集选型与全策略留出评价使用 {selection["n_bootstrap"]} 次处理臂内分层配对 Bootstrap。'
                     + (f'正式报告的关键策略比较和扩量区间使用 {BOOTSTRAP_REPEATS:,} 次重复。' if margins is not None else '')
                     + '每轮所有策略共用同一份重采样行，固定模型与评分，重取各自 Top K。'
                     '同轮配对差值保留策略评价的相关性，使差异区间直接对应模型间的收益比较。'
                     '95% CI 条件于已拟合模型、评分和处理比例；重训、选型及分配概率估计的额外不确定性在附录列明。'
                     '<a href="#appendix-definitions">指标公式</a>与<a href="#appendix-pairs">完整配对比较</a>用于进一步审查。')))
    plot_summary = bootstrap[bootstrap.budget.eq(.2)].copy()
    plot_summary = pd.concat([plot_summary[plot_summary.policy.eq(selected)], plot_summary[plot_summary.policy.ne(selected)]])
    key_pairs = pd.DataFrame([comparison(selected,'random'), comparison(uplift,response), comparison(selected,'response_logistic')]).drop_duplicates(['policy','reference'])
    if margins is not None:
        key_pairs = key_pairs.iloc[:2]
    gain_comparison = _gain_comparison(primary)
    budget_summary = _paragraph(
        f'固定 20% 覆盖的 HT Gain 为 {value:.2f}，全覆盖为 {full_gain:.2f}，两者点估计比值约 {captured:.1%}。'
    )
    if margins is None:
        budget_summary += _paragraph(
            f'10%–20% 覆盖贡献 {marginal_10_20:.2f}，20%–30% 贡献 {marginal_20_30:.2f} 次增量转化／10 万名合资格用户。'
            '这些边际结果为描述性点估计。'
        )
    else:
        primary_uplift = primary.gain_plugin * 1e5
        plugin_full = frozen.loc[1., 'gain_plugin']
        plugin_capture = (f'，约为全覆盖的 {frozen.loc[.2, "gain_plugin"]/plugin_full:.1%}'
                          if plugin_full != 0 else '')
        key_labels = [_LABELS[name] for name in dict.fromkeys((selected, uplift, 'random'))]
        budget_summary += (
            _paragraph(f'Uplift Gain 下，冻结策略的 20% 覆盖收益为 {primary_uplift[selected]:.2f}{plugin_capture}。'
                       f'{uplift_label} 相对 {label} 的差值由 HT 的 {comparison(uplift, selected).difference * 1e5:+.2f} '
                       f'变为 Uplift Gain 的 {primary_uplift[uplift]-primary_uplift[selected]:+.2f}。')
            + _table(gain_comparison.iloc[:, :3].set_index('策略').loc[key_labels].reset_index())
            + _paragraph('HT Gain 按训练集处理比例加权，Uplift Gain 用覆盖前缀内两臂转化率之差乘覆盖率。'
                         '两种口径均将定向策略排在 Random 之前，前部覆盖所占增量比例对口径更敏感。'
                         '<a href="#appendix-budget">查看九策略排名对照</a>。')
            + '<h3>每一档扩量带来的额外转化</h3>'
            + _paragraph('下表中两种边际 Gain 及 HT 区间均按每 10 万名合资格用户展示。'
                         'Uplift Gain 列是累计前缀估计的差值。')
            + '<div class="main-pairs">' + _table(pd.DataFrame({
                '新增覆盖': [f'{r.start:.0%}–{r.end:.0%}' for r in margins.itertuples()],
                'HT 边际 Gain': (margins.gain_ht * 1e5).map(_num),
                'HT 配对 95% CI': margins.apply(lambda r: _interval(r, scale=1e5), axis=1),
                'Uplift 边际 Gain': (margins.gain_plugin * 1e5).map(_num),
                '每千新增名额的 HT 增量': (margins.gain_ht * test_n / margins.n_added * 1000).map(lambda x: _num(x, 3)),
            })) + '</div>'
            + _paragraph(f'配对区间按 {BOOTSTRAP_REPEATS:,} 次处理臂内分层 Bootstrap 计算。每次用同一批抽样行重取各档 Top K，'
                         '先相减再取分位数。区间为逐档的条件区间。20% 是预设覆盖率，边际结果可用于评估扩量条件。')
        )
    main.append(_section('results', '05｜三个核心研究发现',
        '<h3>定向策略相对随机分配的增量优势</h3>'
        + _policy_comparison(plot_summary, selected, 'HT Gain @20%', '每 10 万名全部合资格用户的增量转化', test_n)
        + _paragraph(f'{label} 的点估计为 {value:.2f}，Random 为 {primary.loc["random","gain_ht_per_100k"]:.2f}。'
                     f'二者配对差为 {vs_random.difference*1e5:+.2f}，95% CI {_interval(vs_random,scale=1e5)}。'
                     '结果为固定覆盖率下采用模型定向、优先选择高增量价值前缀提供了直接量化证据。'
                     '图中的横线是各策略自身的区间，策略差异仍应依据下表的配对区间。')
        + '<h3>Uplift 与 Response：配对收益比较</h3>'
         + _paired_gain_card(response_label, uplift_label, float(primary.loc[response, 'gain_ht_per_100k']), float(primary.loc[uplift, 'gain_ht_per_100k']), uplift_pair)
        + '<div class="main-pairs">' + _pair_table(key_pairs, compact=True) + '</div>'
        + _paragraph(f'{uplift_label} 与 {response_label} 的点估计分别为 {primary.loc[uplift,"gain_ht_per_100k"]:.2f} 与 '
                     f'{primary.loc[response,"gain_ht_per_100k"]:.2f}。{inference} '
                     f'本阶段的策略选择由验证集的预设 Bootstrap 均值规则确定，最终仍为 {label}。'
                     '策略决策遵循验证集预设规则，模型差异采用配对 Bootstrap 区间评价。<a href="#appendix-results">完整九策略验证与测试结果</a>保留在附录。')
        + '<h3>定向收益集中在排序前部，扩量依据边际经济性判断</h3>'
        + '<figure>' + _gain_plot(curves, list(dict.fromkeys([selected,uplift,response])))
        + '<figcaption>同一留出总体、冻结评分与固定处理概率。覆盖率为事先预设的敏感性分析；全部九策略曲线见附录。</figcaption></figure>'
        + budget_summary))
    overlap_fig = f'''<figure class="overlap-figure"><div class="overlap-counts"><div><span>仅 {uplift_label}</span><strong>{matched['only_a']:,}</strong><small>各自入选人数的 {matched['only_a_fraction_of_k']:.2%}</small></div><div class="shared"><span>共同选中</span><strong>{matched['common']:,}</strong><small>各自入选人数的 {matched['overlap']:.2%}</small></div><div><span>仅 {response_label}</span><strong>{matched['only_b']:,}</strong><small>各自入选人数的 {matched['only_b_fraction_of_k']:.2%}</small></div></div><figcaption>两策略各选 {matched['k']:,} 行，合并去重为 {matched['union']:,} 行；原始评分、共享随机破同分优先级，种子 {matched['seed']}。行号用于集合识别，不等同于可识别用户身份。</figcaption></figure>'''
    rho = '未定义（常数评分）' if matched['spearman'] is None else f'{matched["spearman"]:.4f}'
    main.append(_section('interpretation', f'06｜选人重叠 {matched["overlap"]:.2%}：解释相近收益',
        _paragraph('为分析两类策略收益接近的原因，本项目比较其 Top 20% 入选集合与整体评分排序。'
                   '所有重叠指标均从冻结评分与来源行号直接计算。')
        + overlap_fig
        + _paragraph(f'本阶段 {uplift_label} 与 {response_label} 的 <strong>Overlap@20% 为 {matched["overlap"]:.2%}</strong>，'
                     f'Jaccard 为 {matched["jaccard"]:.4f}，全体原始评分的 Spearman 相关为 {rho}。'
                     'Overlap 直接衡量同预算下的共同入选比例，Jaccard 描述集合相似性，Spearman 则补充整体评分排序的一致程度。')
        + _paragraph('观察发现一：两种评分目标在当前数据下形成了高度相似的资格分配名单。'
                     f'当前共同入选比例 {matched["overlap"]:.2%} 直接刻画两类评分策略在既定覆盖率下的决策一致性。'
                     '这一选人行为与收益接近的观察结果一致，具体机制仍可在独有入选集合上继续检验。')
        + _paragraph('观察发现二：两策略在覆盖率边界仍保留一小部分不同的入选记录，构成可进一步验证的差异人群。'
                     '项目已量化两策略各自的独有入选集合；下一步可针对差异集合评估增量收益是否具有稳定价值。'
                     '对照组正例支持与匿名特征的信息范围是进一步解释模型差异的重点诊断方向。')
        + _paragraph('业务解释：Response 按处理后响应概率排序，Uplift 按预测增量排序，两者都用统一的 HT Gain @20% 检验策略收益。'
                     '本次结果表明，响应倾向策略在这一业务目标下也形成了较强的增量排序表现；'
                     '最终以统一业务指标与配对比较决定策略，并以学习器差异辅助解释结果。'
                     '<a href="#appendix-overlap">集合分母、算法与评分指纹</a>可独立核验。')))
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
    main.append(_section('decision', '07｜策略推荐、扩量成本与线上验证',
        _paragraph(f'保留本阶段冻结策略 {label}，将 {uplift_label} 作为增量建模候选进行后续研究。'
                   + ('这是一项开发阶段结论，正式推荐应以全量实验的验证冻结和独立评价为准。' if development else
                      '全量正式推荐沿用验证集冻结的 Response-XGBoost，并将 S-XGBoost 保留为后续线上挑战策略。')
                   + '投放资格名额是本项目的资源口径，实际竞价花费需要另行验证。')
        + _paragraph(f'代表性经济情景：对 {int(scenario.eligible_n):,} 名合资格用户固定覆盖 {scenario.budget:.0%}，'
                     f'假设每次增量转化价值 {scenario.value_per_conversion_assumed:g} 个货币单位、每次触达成本 {scenario.contact_cost_assumed:g}，'
                     f'保存结果对应的条件净增量价值为 {scenario.net_incremental_value:,.2f}，盈亏平衡单位触达成本为 {scenario.break_even_contact_cost:.4f}。'
                     '上述净价值与盈亏平衡值是固定价值、成本及离线效果可迁移条件下的业务情景估算。'
                     '<a href="#appendix-economics">全部经济敏感性与样本量情景</a>保留在附录。')
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
    # 第二层按审查问题组织，默认折叠；不向原始工件写入新文件。
    appendix = []
    stage_splits = []
    for run_name, stage_name in [('development_conversion_5000000_seed42','500 万开发'),
                                 ('final_conversion_full_seed42','全量正式')]:
        path = out.parent / run_name / 'data_preparation/split_summary.csv'
        if path.exists():
            stage_splits.append(pd.read_csv(path).assign(阶段=stage_name))
    split_detail = pd.concat(stage_splits, ignore_index=True) if stage_splits else split.assign(阶段=stage_title)
    split_detail = split_detail.rename(columns={'split':'划分','treatment':'处理组标识','n':'样本量',
        'conversion_positive':'转化正例','visit_positive':'访问正例'})
    split_detail['划分'] = split_detail['划分'].map({'train':'训练集','validation':'验证集','test':'留出评价'})
    ate_display = pd.DataFrame({
        '总体': ate.scope.map({'full_dataset':'全量源数据','test':scope}),
        '结果': ate.outcome.map({'conversion':'转化','visit':'访问'}),
        '样本量': ate.n, '对照率': ate.response_control.map(lambda v:f'{v:.4%}'),
        '处理率': ate.response_treatment.map(lambda v:f'{v:.4%}'),
        'ATE（百分点）': (ate.ate * 100).map(lambda v:_num(v,4)),
        '标准误（百分点）': (ate.standard_error * 100).map(lambda v:_num(v,6)),
        '95% CI（百分点）': ate.apply(lambda r:_interval(r,scale=100,digits=4),axis=1),
        '相对增幅': ate.relative_uplift.map(lambda v:f'{v:.2%}'),
        'Z 统计量': ate.z_statistic.map(lambda v:_num(v,4)), '双侧 p': ate.p_value_text,
    })
    appendix.append(_details('appendix-data','A｜数据审计、两阶段划分与 ATE',
        _paragraph('所有 ATE 保留原结果文件的统计口径：未合并、含 n−1 修正的 Bernoulli 样本方差用于标准误与正态近似区间；'
                   '两比例 Z 检验使用零假设下的合并比例。极小 p 值保留原文本表示，不显示成精确的零。')
        + _table(ate_display,wide=True) + _table(split_detail)
        + _details('appendix-protocol','冻结协议与当前划分审计',_json(sample.get('protocol', {'说明':'当前为合成测试运行'}))
                   + _json(read_json('data_preparation/feature_overlap.json')))
        + _details('appendix-audit','全部审计字段与数据质量计数',_json(audit))
        + _image(out/'figures/sample_features.png','建模特征分布','原始运行保存的特征图；未重算或替换原文件。')))
    appendix.append(_details('appendix-diagnostics','B｜实验诊断、C2ST 与结果可学习性',
        _paragraph(f'全量 SMD/KS 来自原始 float64 特征。KS p 值为探索性结果。C2ST 两臂等量抽样，'
                   f'每模型 {int(c2st.permutations.iloc[0])} 次完整置换重拟合，两个模型使用 Holm 校正。'
                   '覆盖诊断仅在训练集按预先规则分箱，不删除稀疏箱或改变训练目标。')
        + _table(balance,wide=True) + _table(c2st,wide=True)
        + '<figure>' + _c2st_plot(c2st_null,c2st) + '<figcaption>原保存置换 AUC 分布；竖线为观测 AUC，未重新运行检验。</figcaption></figure>'
        + _details('appendix-c2st-raw','C2ST 全部置换记录与运行配置',_table(c2st_null,wide=True)+_json(read_json('diagnostics/c2st_details.json')))
        + _details('appendix-support','SRM、分箱覆盖与正值性明细',_json(srm)+_json(positivity)+_table(read_csv('diagnostics/overlap_strata.csv'),wide=True))
        + _paragraph('可学习性使用训练集拟合、验证集评价结果预测，不是 CATE 真值检验。')
        + _table(read_csv('diagnostics/learnability.csv'),wide=True)))
    appendix.append(_details('appendix-models','C｜模型、参数、训练状态与正例支持',
        _equation(r'\hat\tau_S(x)=\hat f(x,1)-\hat f(x,0),\quad\hat\tau_T(x)=\hat\mu_1(x)-\hat\mu_0(x)', 'S/T-Learner 的评分')
        + _equation(r'D_1=Y-\hat\mu_0(X),\quad D_0=\hat\mu_1(X)-Y,\quad\hat\tau_X=e\hat\tau_0+(1-e)\hat\tau_1', 'X-Learner 的伪效应与组合权重')
        + _paragraph('两组伪效应均采用处理减对照的方向，权重 e 使用训练处理比例。Response 仅复用对应 T-Learner 的处理组模型，独立保存策略名称与评分。'
                     'Logistic 默认 C=1、L2、LBFGS、最多 500 次迭代，训练内中位数填补与标准化；'
                     f'XGBoost {args["trees"]} 棵树、深度 4、学习率 0.05、min_child_weight=20、subsample=0.8、colsample_bytree=0.9、hist 算法、4 线程。'
                     '以下包含实际覆写参数、训练警告、组件类型、收敛迭代与树分裂数。')
        + _json(manifest['models']) + _json(read_json('stability/report.json'))))
    definitions = [
        (r'z_i=T_iY_i/e-(1-T_i)Y_i/(1-e)', 'HT 单行伪结果：固定训练分配比例'),
        (r'k=\lfloor qN\rfloor,\quad G_{HT}(q)=N^{-1}\sum_{i\in top-k}z_i', 'HT Gain：全部合资格人数为分母'),
        (r'G_{random}(q)=(k/N)\bar z', 'Random：相同整数人数下的期望 HT Gain'),
        (r'G(q)=\frac{k}{N}\left(\frac{R_1(k)}{N_1(k)}-\frac{R_0(k)}{N_0(k)}\right)', 'Uplift Gain：按前缀实际分组比例估计'),
        (r'Q(k)=\frac{1}{N}\left(R_1(k)-R_0(k)\frac{N_1(k)}{N_0(k)}\right)', 'Traditional Qini Gain：对应前缀实际处理人数，尺度不同于 HT'),
        (r'\mathrm{AUUC}=\int_0^1G(q)dq,\quad\mathrm{Qini\ Score}=\int_0^1Q(q)dq-\frac{Q(1)}{2}', '全覆盖排序面积：不进行完美排序归一化'),
        (r'\hat\pi=\mathrm{argmax}_{\pi}\;B^{-1}\sum_{b=1}^{B}\widehat G^{(b)}_{HT,validation}(0.2)', '验证集按预设 Bootstrap 均值选型，Random 不参加'),
        (r'\widehat{ATE}=\bar Y_1-\bar Y_0,\quad SE=\sqrt{s_1^2/N_1+s_0^2/N_0}', 'ATE 与未合并标准误'),
    ]
    appendix.append(_details('appendix-definitions','D｜指标定义、公式与不确定性',
        ''.join(_equation(expr,title) for expr,title in definitions)
        + _paragraph('N₁、N₀ 表示前缀内两臂人数，R₁、R₀ 表示对应转化数。Uplift Gain 在前缀包含两臂时定义；'
                     '传统 Qini 在前缀至少含对照组时定义，仅有对照组的前缀为零。积分从原点连接至首个有效前缀，使用全部前缀的梯形面积；'
                     '图表抽点仅用于展示，未重算面积。所有表内 Gain 和面积乘 10⁵ 展示，Qini 的不同目标不能与 HT 直接比较数值大小。')
        + _paragraph('同分使用独立于结果的共享随机优先级。Bootstrap 按处理臂内有放回抽样，保持两臂样本量，固定模型评分与 e，再选择重采样总体中的 Top K。'
                     '单策略区间取有效重复的百分位数，配对区间取两策略共同有效重复的逐轮差值。各预算区间为点态区间，不是同时置信带；'
                     '不把测试集重采样获胜频率作为再次选型证据。')))
    appendix.append(_details('appendix-results','E｜九策略验证与测试结果（全部预设预算）',
        _paragraph(f'验证集冻结 <strong>{label}</strong>，最佳 Uplift 为 <strong>{uplift_label}</strong>。'
                   '下表将点估计、主指标区间与辅助指标合并，避免重复展示九策略大表。')
        + '<h3>验证集：预设 20% 覆盖率</h3>' + _metric_table(val_metrics,val_boot)
        + _details('appendix-selection','冻结记录与验证选中频率',_json(selection)+_table(read_csv('models/validation_selection_frequency.csv'),wide=True))
        + '<h3>留出评价：全部预设覆盖率</h3>' + _metric_table(metrics,bootstrap)))
    appendix.append(_details('appendix-pairs','F｜完整配对 Bootstrap 比较',
        _paragraph('差值方向始终为“策略减参考策略”。下表列出全部两两比较和预算，并逐行标记有效配对次数。区间单位为每 10 万名全部合资格用户的增量转化。'
                   '原文件不保存自比较，自比较差值与区间按定义为零。')
        + '<h3>验证集</h3>' + _pair_table(val_pairs)
        + '<h3>留出评价</h3>' + _pair_table(pairs)))
    appendix.append(_details('appendix-budget','G｜预算敏感性与辅助累计曲线',
        '<h3>Top 20% 收益口径与排名</h3>'
        + _paragraph('两列 Gain 均以每 10 万名合资格用户的增量转化表示。')
        + _table(gain_comparison)
        + _paragraph('预算敏感性覆盖七档。<a href="#appendix-results">九策略全部预算点、区间与面积明细见附录 E</a>，此处不再重复表格。'
                   '三类曲线使用各自真实保存终点，Random 为对应终点参考线。辅助面积不替代 HT Gain @20% 选型。')
        + '<figure>' + _gain_plot(curves,list(selection['candidates']),columns=('gain_ht','gain_plugin','qini_traditional'))
        + '<figcaption>全部模型的三类累计曲线。横轴为候选总体覆盖比例；HT 与 Uplift Gain 按不同方式估计同一覆盖策略的增量，传统 Qini Gain 对应前缀中的实际处理人数。</figcaption></figure>'))
    appendix.append(_details('appendix-overlap','H｜Top 20% 集合、Spearman 与评分溯源',
        _equation(r'\mathrm{Overlap}=|A\cap B|/K,\quad\mathrm{Jaccard}=|A\cap B|/|A\cup B|', '重叠与并集相似度的不同分母')
        + _paragraph('两策略均使用 metrics.py 的规则：按评分降序，共享种子生成的随机优先级破同分，K=floor(0.20N)。'
                     'Spearman 使用完整原始评分的平均并列秩，不人为加入破同分噪声。不报告其大样本 p 值，也不把描述性相关性解释成因果机制。')
        + _json(matched)))
    appendix.append(_details('appendix-economics','I｜经济敏感性与线上样本量规划',
        _equation(r'V_{net}=v\cdot N\widehat G_{HT}-c\cdot K,\quad c_{break-even}=v\cdot N\widehat G_{HT}/K', '条件净价值与盈亏平衡单位成本')
        + _paragraph('以下为假设情景。固定转化价值、固定触达成本与效果可迁移是额外假设；区间只传递 HT 条件区间，未包括价格和成本的不确定性。'
                     'A/B 样本量按全部入组用户的策略转化率差规划，不能把离线增量估计当作已经观测到的线上基线率。')
        + _json(assumptions) + _table(scenarios,wide=True)
        + _table(read_csv('business/hypothetical_ab_plans.csv'),wide=True)))
    rebuild_command = (
        'python -c "from pathlib import Path; from criteo_uplift.reporting import build_report; '
        f"p=Path('artifacts/{out.name}'); build_report(p/'manifest.json', Path('reports/{'development' if development else 'final'}/report.html'))\""
    )
    appendix.append(_details('appendix-reproducibility','J｜运行配置、复现与限制',
        _paragraph('重建报告从保存工件计算选人重叠和秩相关。全量正式报告同时从冻结评分计算边际区间与关键策略的配对区间。')
        + '<pre>' + html.escape(rebuild_command) + '</pre>'
        + _json({'数据SHA256':manifest['source']['sha256'],'随机种子':manifest['seed'],
                 '原始运行耗时秒':manifest['elapsed_seconds'],'运行库版本':manifest['versions'],
                 '原始运行参数':{k:Path(v).name if k in ('data','output','protocol') and v else v for k,v in args.items()},
                 '估计目标':manifest['estimand'],'区间方法':manifest['ci_method'],
                 '未覆盖的不确定性':manifest['uncertainty_excluded']})
        + _paragraph('匿名数据无法识别用户级依赖，公开实验的全部分层与分配机制不可见。单变量平衡、可学习性、排序相关及选人重叠都不能替代识别假设。'
                     '正式模型依据验证集冻结记录确定，业务验证方案见第七章。')
        + _paragraph('数据来源：<a href="https://ailab.criteo.com/criteo-uplift-prediction-dataset/">Criteo AI Lab</a>。'
                     '原始数据及派生成果遵循 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>。')))
    titles = [('summary','研究结论'),('business','业务问题'),('identification','数据与因果识别'),
              ('methods','建模与评价'),('results','核心发现'),('interpretation','策略解释'),('decision','业务决策')]
    header = f'''<header><p class="eyebrow">CRITEO UPLIFT PROJECT</p><h1>固定投放预算下的增量转化研究</h1><p>{stage_title} · 正文约 10–15 分钟 · 技术附录可展开</p>
    <nav class="reading-entries" aria-label="阅读入口"><a href="#research">阅读研究正文</a><a href="#appendix">查看技术附录</a><a href="#workflow">项目研究流程</a><a href="#decision">核心结果与业务建议</a></nav>
    <nav class="chapter-nav" aria-label="正文目录">{''.join(f'<a href="#{key}">{title}</a>' for key,title in titles)}</nav></header>'''
    research = ''.join(main)
    if development:
        research = research.replace('独立测试集', '开发留出集')
    document = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{stage_title}｜Criteo Uplift</title><style>{_REPORT_CSS}</style></head><body><main>'
                + header + workflow + '<article id="research" aria-label="研究正文">' + research + '</article>'
                + '<section id="appendix"><h2>技术附录｜按主题核验完整证据</h2><p>默认折叠。正文中的附录链接会自动展开对应主题；长表格可在容器内横向滚动，页面本身保持适配。</p>'
                + ''.join(appendix) + '</section><footer>验证集选型已冻结 · <a href="#research">返回研究正文</a></footer></main>'
                + _NAV_SCRIPT + '</body></html>')
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
    return report_path


_REPORT_CSS = """
:root{color-scheme:light;--ink:#263746;--blue:#21628e;--pale:#eef5fa;--line:#d9e4ed}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f7fa;color:var(--ink);font:16px/1.85 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1120px;margin:auto;padding:42px 28px}header{padding:12px 6px 24px}h1{font-size:35px;line-height:1.4;margin:10px 0}h2{font-size:25px;line-height:1.5;margin:0 0 20px}h3{font-size:19px;margin:28px 0 12px}p{margin:14px 0}a{color:var(--blue);text-underline-offset:3px}b,strong{font-weight:650}
.eyebrow{font-size:12px;letter-spacing:.12em;color:var(--blue)}section,#workflow{border:1px solid var(--line);background:white;padding:30px;border-radius:8px;margin:24px 0;scroll-margin-top:18px}
.reading-entries,.chapter-nav{display:flex;flex-wrap:wrap;gap:12px;margin-top:20px}.reading-entries a{padding:8px 14px;background:var(--pale);border-radius:4px;text-decoration:none;font-weight:600}.chapter-nav a{font-size:14px}
.workflow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));list-style:none;padding:0;gap:12px;counter-reset:steps}.workflow li{counter-increment:steps;background:var(--pale);padding:12px;border-radius:4px}.workflow li:before{content:counter(steps) " → ";color:#607e96}.workflow a{font-size:14px;text-decoration:none}
.lead{font-size:19px}.cards,.split-flow,.overlap-counts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:22px 0}.cards>div,.split-flow>div,.overlap-counts>div{background:var(--pale);padding:18px;border-radius:5px}.cards span,.cards small,.split-flow span,.overlap-counts span,.overlap-counts small{display:block;font-size:13px;color:#597185}.cards strong,.overlap-counts strong{font-size:27px;display:block}.split-flow b{display:block;font-size:15px}.shared{outline:2px solid #9fbbce}
figure{margin:24px 0}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#607487;line-height:1.7;margin:12px 0}.example{background:#f8fafc;padding:18px;border:1px solid var(--line);border-radius:5px}.example>div{margin:14px 0}.example b,.example span{display:block}.example span{font-size:14px}.bar{height:16px;background:#e6edf2;display:flex;margin-top:7px}.bar i{background:#73a0c4}.bar em{background:#c48645}
.table-wrap{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:13px;line-height:1.65}th,td{text-align:left;vertical-align:top;padding:11px 10px;border-bottom:1px solid #e0e8ef;overflow-wrap:anywhere}th{background:var(--pale)}tr:nth-child(even){background:#fafcfe}.wide table{min-width:980px;table-layout:auto}.wide td{min-width:85px}.wide th{min-width:100px}
.equation{background:#f3f7fa;border:1px solid var(--line);padding:18px;margin:22px 0;border-radius:5px}.equation-label{font-size:13px;font-weight:600;color:#5c7790;margin-bottom:12px}.equation-scroll{overflow-x:auto}.math-equation{width:auto;max-width:100%;height:auto;display:block}
.policy-chart{--cols:minmax(150px,1.3fr) minmax(200px,2fr) minmax(140px,1fr);background:#f6f9fc;border:1px solid var(--line);padding:20px;border-radius:5px}.chart-unit{font-size:13px;color:#5c7790}.policy-row{display:grid;grid-template-columns:var(--cols);gap:16px;align-items:center;padding:10px;font-size:14px}.policy-row.frozen{background:#e4eef6;border-radius:5px}.policy-track{position:relative;height:22px;background:#edf2f6}.policy-zero,.policy-ci,.policy-dot{position:absolute;top:50%;background:#4f7ca0}.policy-zero{top:0;bottom:0;width:1px;background:none;border-left:1px dashed #8ea2b3}.policy-ci{height:3px}.policy-dot{height:9px;width:9px;border-radius:50%;transform:translate(-50%,-50%);border:1px solid white}.policy-value{font-variant-numeric:tabular-nums}.policy-value small{display:block;font-size:12px;color:#5a7184}.policy-axis{display:flex;justify-content:space-between;margin:8px calc(25% + 12px) 8px calc(30% + 2px);font-size:12px;color:#607487}
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

@media(max-width:700px){main{padding:20px 12px}section,#workflow{padding:18px;margin:18px 0}h1{font-size:28px}h2{font-size:23px}.workflow{grid-template-columns:repeat(2,minmax(0,1fr))}.cards,.split-flow,.overlap-counts{grid-template-columns:1fr}.policy-chart{padding:12px}.policy-row{grid-template-columns:minmax(0,1fr) minmax(100px,.9fr);gap:5px;padding:10px 4px}.policy-track{grid-row:2;grid-column:1/-1;margin-top:4px}.policy-value{text-align:right}.policy-axis{margin:12px 4px}.main-pairs table,.main-pairs tbody,.main-pairs tr,.main-pairs td{display:block;width:100%}.main-pairs thead{display:none}.main-pairs tr{padding:10px 0;border-bottom:1px solid var(--line)}.main-pairs td{display:grid;grid-template-columns:1fr 1.4fr;gap:10px;border:0;padding:5px}.main-pairs td:before{content:attr(data-label);color:#58738a;font-weight:600}.appendix-content{padding:4px 10px 12px}th,td{padding:8px 5px;font-size:12px}.equation{padding:12px}.lead{font-size:17px}}
@media print{@page{size:A4;margin:15mm 14mm}body{background:white;font-size:10pt;line-height:1.6;print-color-adjust:exact}main{padding:0;max-width:none}section,#workflow{padding:10pt 0;border:0;margin:8pt 0}h1{font-size:24pt}h2{font-size:18pt}h3{font-size:13pt}h2,h3,thead{break-after:avoid}p{orphans:3;widows:3}.reading-entries,.chapter-nav,footer{display:none}.workflow{grid-template-columns:repeat(4,1fr)}figure,.equation,.cards,.split-flow,.overlap-counts,tr{break-inside:avoid}img{max-height:145mm;object-fit:contain}.table-wrap{overflow:visible}table{font-size:8pt}details:not([open]){display:none}.appendix-content .table-wrap{max-height:none;overflow:visible}.wide table{min-width:0;table-layout:fixed}.wide th,.wide td{min-width:0}.policy-chart{--cols:130px minmax(130px,2fr) 120px;padding:10pt}.policy-row{font-size:9pt;gap:10px}.policy-axis{margin:6pt 140px 6pt 140px}}
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

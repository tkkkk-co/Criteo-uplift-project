"""Readable distribution summaries from saved full ECDFs and profile statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties


def quantiles_from_ecdf(frame, n):
    values = frame.value.to_numpy()
    cumulative = np.rint(frame.cdf.to_numpy() * n).astype(np.int64)
    positions = (n - 1) * np.array([.1, .25, .5, .75, .9])
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    low_values = values[np.searchsorted(cumulative, lower + 1)]
    high_values = values[np.searchsorted(cumulative, upper + 1)]
    return low_values + (positions - lower) * (high_values - low_values)


def draw_summary(output, population, title, filename, profiles, ecdf, record):
    profile = profiles.loc[profiles.population.eq(population)].set_index('feature')
    counts = {'all_test': record['test_n'], population: int(profile.n.iloc[0])}
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
    fig = Figure(figsize=(13.5, 7.5), facecolor='white')
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=.65, wspace=.30)
    labels = {'all_test': '全体 Test', population: '入选人群'}
    colors = {'all_test': '#718399', population: '#c56d26'}
    records = []
    for column, feature in enumerate(['f8', 'f9', 'f3']):
        ax = fig.add_subplot(grid[0, column])
        summary = []
        for name, y_position in [('all_test', 2), (population, 1)]:
            frame = ecdf.loc[ecdf.feature.eq(feature) & ecdf.population.eq(name)]
            q10, q25, median, q75, q90 = quantiles_from_ecdf(frame, counts[name])
            mean = float(profile.loc[feature, 'population_mean' if name == 'all_test' else 'selected_mean'])
            summary.append({'label': labels[name], 'q1': q25, 'med': median, 'q3': q75,
                            'whislo': q10, 'whishi': q90, 'fliers': []})
            records.append({'population': name, 'feature': feature, 'n': counts[name],
                            'p10': q10, 'p25': q25, 'median': median,
                            'p75': q75, 'p90': q90, 'mean': mean})
            ax.plot(mean, y_position, marker='D', color=colors[name], markersize=6, zorder=4)
            ax.annotate(f'均值 {mean:.2f}', (mean, y_position), xytext=(0, -25),
                        textcoords='offset points', ha='center', fontproperties=font, fontsize=10,
                        color=colors[name])
        artists = ax.bxp(summary, positions=[2, 1], orientation="horizontal", widths=.32,
                         showfliers=False, patch_artist=True,
                         medianprops={'color': '#263746', 'linewidth': 1.6})
        for index, name in enumerate(['all_test', population]):
            artists['boxes'][index].set(facecolor=colors[name], edgecolor=colors[name], alpha=.24)
            for line in artists['whiskers'][2*index:2*index+2] + artists['caps'][2*index:2*index+2]:
                line.set(color=colors[name], linewidth=1.4)
        ax.set_yticks([2, 1], ['全体 Test', '入选人群'], fontproperties=font, fontsize=10)
        ax.set_ylim(.40, 2.55)
        ax.set_title(feature, fontsize=15, fontweight='bold')
        ax.set_xlabel('特征取值', fontproperties=font, fontsize=10)
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    share_ax = fig.add_subplot(grid[1, :])
    high = profile.loc['f9']
    proportions = [100*high.test_fraction_above_q75, 100*high.selected_fraction_above_q75]
    share_ax.barh([1, 0], proportions, height=.42, color=[colors['all_test'], colors[population]])
    for position, value in zip([1, 0], proportions):
        share_ax.text(value+1.5, position, f'{value:.2f}%', va='center', fontsize=13, fontweight='bold')
    share_ax.set_yticks([1, 0], ['全体 Test', '入选人群'], fontproperties=font, fontsize=11)
    share_ax.set_xlim(0, 100)
    share_ax.set_xticks([0, 20, 40, 60, 80, 100], ['0%', '20%', '40%', '60%', '80%', '100%'])
    share_ax.set_title(f'f9 > {high.test_q75:.2f} 的记录占比（阈值为全体 Test 的第75百分位点）',
                       fontproperties=font, fontsize=13, loc='left', pad=15)
    share_ax.spines[['top', 'right', 'left']].set_visible(False)
    share_ax.tick_params(axis='y', length=0)
    share_ax.grid(axis='x', alpha=.15)
    share_ax.set_axisbelow(True)
    fig.suptitle(title, fontproperties=font, fontsize=18, y=.985)
    fig.text(.5, .92, f'全体 Test n={counts["all_test"]:,}    入选人群 n={counts[population]:,}',
             fontproperties=font, fontsize=11, ha='center', color='#526577')
    fig.text(.5, .035,
             '箱体表示中间50%取值，箱内线为中位数，须表示第10–90百分位区间，菱形为均值。\n'
             '各组均使用完整数据计算；同一特征共享数值轴。分位区间外的记录保留在统计中。',
             fontproperties=font, fontsize=10, ha='center', color='#526577', linespacing=1.6)
    fig.subplots_adjust(left=.10, right=.965, top=.82, bottom=.15)
    figure_path = output / 'figures' / filename
    fig.savefig(figure_path, dpi=180)
    return records


def main(output):
    profiles = pd.read_csv(output / 'feature_profiles.csv')
    ecdf = pd.read_csv(output / 'feature_ecdf.csv')
    record = json.loads((output / 'analysis_record.json').read_text())
    records = []
    configurations = [
        ('flow_stable', '四轮选型均入选的人群画像', 'flow_stable_feature_summary.png'),
        ('formal_response_xgboost', 'Response-XGBoost｜原正式模型 Top 20% 人群画像', 'formal_response_feature_summary.png'),
        ('formal_s_xgboost', 'S-XGBoost｜原正式模型 Top 20% 人群画像', 'formal_s_feature_summary.png'),
    ]
    for population, title, filename in configurations:
        records.extend(draw_summary(output, population, title, filename, profiles, ecdf, record))
    pd.DataFrame(records).to_csv(output / 'profile_quantile_summary.csv', index=False)
    path = output / 'analysis_record.json'
    record = json.loads(path.read_text())
    record['profile_display'] = {
        'method': 'P25-P75 boxes, median lines, P10-P90 whiskers, mean diamonds, plus strict f9 > all-Test Q75 shares.',
        'data': 'Exact saved ECDF counts and all-record means; no resampling or row exclusion.',
        'main_figures': [f'figures/{filename}' for _, _, filename in configurations],
        'reproduce_command': f'python -m criteo_uplift.plot_profile_summary --output {output}',
    }
    record['figure_note'] = (
        'Main profile figures use quantile boxes, mean markers and high-f9 proportion bars. '
        'ECDFs are retained as supporting data.'
    )
    record['outputs'] = [str(file.relative_to(output)) for file in sorted(output.rglob('*')) if file.is_file()]
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)

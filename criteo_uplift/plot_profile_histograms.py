"""Draw aligned Test/Top-20% histograms from saved full-population bin counts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator

POPULATIONS = {
    'formal_response_xgboost': ('Response-XGBoost', 'formal_response_feature_histogram.png'),
    'formal_s_xgboost': ('S-XGBoost', 'formal_s_feature_histogram.png'),
    'flow_stable': ('四轮流程稳定入选人群', 'flow_stable_feature_histogram.png'),
}


def ranked_features(profiles, population):
    return (profiles.loc[profiles.population.eq(population)]
            .assign(absolute_standardized_mean_shift=lambda d: d.standardized_mean_shift.abs())
            .sort_values('absolute_standardized_mean_shift', ascending=False, kind='stable'))


def _quantile(frame, n, q):
    cumulative = np.rint(frame.cdf.to_numpy() * n).astype(np.int64)
    position = (n - 1) * q
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    values = frame.value.to_numpy()
    low = values[np.searchsorted(cumulative, lower + 1)]
    high = values[np.searchsorted(cumulative, upper + 1)]
    return float(low + (position - lower) * (high - low))


def display_counts(frame, n, boundaries):
    """Exact (<= lower), interior, (>= upper) counts from a full saved ECDF."""
    values = frame.value.to_numpy()
    cumulative = np.rint(frame.cdf.to_numpy() * n).astype(np.int64)
    positions = np.searchsorted(values, boundaries, side='right')
    positions[-1] = np.searchsorted(values, boundaries[-1], side='left')
    totals = np.where(positions > 0, cumulative[np.maximum(positions - 1, 0)], 0)
    counts = np.diff(np.r_[0, totals, n])
    assert (counts >= 0).all() and counts.sum() == n
    return counts


def main(output):
    output = Path(output)
    profiles = pd.read_csv(output / 'feature_profiles.csv')
    record = json.loads((output / 'analysis_record.json').read_text())
    test_n = int(record['test_n'])
    features = ranked_features(profiles, 'formal_response_xgboost').feature.head(3).tolist()
    groups = ['all_test', *POPULATIONS]
    # Read only the required features/populations; this uses saved distributions, no refit.
    parts = []
    for chunk in pd.read_csv(output / 'feature_ecdf.csv', chunksize=250000):
        part = chunk.loc[chunk.feature.isin(features) & chunk.population.isin(groups)]
        if len(part):
            parts.append(part)
    ecdf = pd.concat(parts, ignore_index=True)
    distributions, bounds = {}, {}
    for feature in features:
        limits = []
        for population in groups:
            frame = ecdf.loc[ecdf.feature.eq(feature) & ecdf.population.eq(population)].sort_values('value')
            n = test_n if population == 'all_test' else int(profiles.loc[profiles.population.eq(population), 'n'].iloc[0])
            distributions[feature, population] = (frame, n)
            if population != 'flow_stable':
                limits.append((_quantile(frame, n, .005), _quantile(frame, n, .995)))
        lower, upper = min(x[0] for x in limits), max(x[1] for x in limits)
        bounds[feature] = np.linspace(lower, upper, 19)  # 18 interior bars plus 2 tail bars.
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
    rankings, saved_bins = [], []
    for population, (label, filename) in POPULATIONS.items():
        rank = ranked_features(profiles, population)
        rankings.append(rank.assign(rank=np.arange(1, len(rank) + 1)))
        selected = rank.head(3).set_index('feature')
        assert selected.index.tolist() == features
        fig = Figure(figsize=(13, 7.4), facecolor='white')
        axes = fig.subplots(2, 3, sharex='col', sharey='col')
        for column, feature in enumerate(features):
            boundaries = bounds[feature]
            lower, upper = boundaries[0], boundaries[-1]
            width = boundaries[1] - lower
            centers = np.r_[lower - width / 2, (boundaries[:-1] + boundaries[1:]) / 2, upper + width / 2]
            percentages = {}
            for name in ['all_test', population]:
                frame, n = distributions[feature, name]
                counts = display_counts(frame, n, boundaries)
                percentages[name] = counts / n * 100
                for index, (count, percent, center) in enumerate(zip(counts, percentages[name], centers)):
                    saved_bins.append({'model': population, 'population': name, 'feature': feature, 'bin': index + 1,
                        'lower': float('-inf') if index == 0 else boundaries[index - 1],
                        'upper': float('inf') if index == len(counts) - 1 else boundaries[index],
                        'count': int(count), 'percent': float(percent), 'plot_center': center, 'plot_width': width})
            ymax = min(100., (np.ceil(max(v.max() for v in percentages.values()) / 10) + 1) * 10)
            ticks = np.r_[centers[0], np.linspace(lower, upper, 5)[1:-1], centers[-1]]
            digits = 2 if feature == 'f8' else 1
            labels = [f'≤{lower:.{digits}f}', *[f'{x:.{digits}f}' for x in ticks[1:-1]], f'≥{upper:.{digits}f}']
            for row, (name, color, group) in enumerate([
                ('all_test', '#718399', '全体 Test Set'),
                (population, '#c56d26', '流程稳定人群' if population == 'flow_stable' else 'Top 20%')
            ]):
                ax = axes[row, column]
                ax.bar(centers, percentages[name], width=width * .94, color=color, linewidth=0, zorder=3)
                mean = selected.loc[feature, 'population_mean' if row == 0 else 'selected_mean']
                ax.set_title(f'{feature} · {group}', fontproperties=font, fontsize=12, loc='left', pad=10)
                ax.axvline(mean, ymin=0, ymax=.86, color='#263746', linewidth=1.35,
                           linestyle=(0, (4, 3)), zorder=4)
                ax.text(mean, .89, f'均值 {mean:.2f}', transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontproperties=font, fontsize=10,
                        color='#263746', zorder=5,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .94, 'pad': 2})
                ax.set_xlim(lower - width, upper + width)
                ax.set_ylim(0, ymax)
                ax.set_ylabel('组内占比（%）', fontproperties=font, fontsize=10)
                ax.set_xlabel('特征取值（两端为尾部合并柱）', fontproperties=font, fontsize=9, labelpad=6)
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.set_xticklabels(labels)
                ax.tick_params(axis='x', labelbottom=True, labelsize=9, pad=6)
                ax.tick_params(axis='y', labelsize=9)
                ax.grid(axis='both', color='#dce3ea', linewidth=.65, zorder=0)
                ax.set_axisbelow(True)
                ax.spines[['top', 'right']].set_visible(False)
                ax.spines[['left', 'bottom']].set_color('#9ba9b4')
        fig.suptitle(f'{label}｜入选画像与全体 Test Set 对照', fontproperties=font, fontsize=17, y=.985)
        selected_label = '稳定入选人群' if population == 'flow_stable' else 'Top 20%'
        fig.text(.5, .93, f'全体 Test Set n={test_n:,}    {selected_label} n={int(selected.n.iloc[0]):,}',
                 ha='center', fontproperties=font, fontsize=11, color='#607487')
        fig.text(.5, .025, '上下共用分箱、刻度及网格；虚线为各组均值；两端柱合并尾部（≤ / ≥）。各组占比合计 100%，未剔除记录。',
                 ha='center', fontproperties=font, fontsize=10, color='#607487')
        fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.14, wspace=.28, hspace=.70)
        fig.savefig(output / 'figures' / filename, dpi=180)
    ranking_frame = pd.concat(rankings, ignore_index=True)
    ranking_frame.loc[ranking_frame.population.ne('flow_stable')].to_csv(output / 'formal_feature_ranking.csv', index=False)
    ranking_frame.loc[ranking_frame.population.eq('flow_stable')].to_csv(output / 'flow_stable_feature_ranking.csv', index=False)
    bin_frame = pd.DataFrame(saved_bins)
    bin_frame.loc[bin_frame.model.ne('flow_stable')].to_csv(output / 'formal_profile_display_bins.csv', index=False)
    bin_frame.loc[bin_frame.model.eq('flow_stable')].to_csv(output / 'flow_stable_display_bins.csv', index=False)
    record['formal_profile_display'] = {
        'method': 'Two aligned rows, 18 equal-width body bins plus two pooled tail bars, identical x/y ticks and grids, dashed group-mean markers with labels above each marker.',
        'body_range': 'Envelope of 0.5th–99.5th percentiles across all Test and both fixed-model selections; common bounds across models. Both tails retained in endpoint bars.',
        'features': 'Each fixed model ranked separately by absolute standardized mean shift; both top three are f8, f9, f3.',
        'source': 'Exact saved ECDF cumulative counts; complete populations, no refit or resampling.',
        'ranking': 'formal_feature_ranking.csv', 'display_counts': 'formal_profile_display_bins.csv',
        'main_figures': [f'figures/{name}' for key, (_, name) in POPULATIONS.items() if key != 'flow_stable'],
        'reproduce_command': f'python -m criteo_uplift.plot_profile_histograms --output {output}',
    }
    record['flow_profile_display'] = {
        'population': 'Intersection of the four validation-winner Top-20% memberships on the common Test Set.',
        'method': 'Same bins, coordinates, grids and labelled mean markers as the fixed-model histograms; all records retained.',
        'main_figure': 'figures/flow_stable_feature_histogram.png',
        'display_counts': 'flow_stable_display_bins.csv', 'ranking': 'flow_stable_feature_ranking.csv',
    }
    record['profile_display'] = {
        'method': 'Aligned two-row histogram comparisons for fixed-model and stable-flow populations.',
        'main_figures': [f'figures/{name}' for _, name in POPULATIONS.values()],
    }
    record['figure_note'] = 'Fixed-model and stable-flow profiles use aligned two-row histograms with pooled tails and labelled dashed mean markers.'
    record['outputs'] = sorted(set(record['outputs']) | {'formal_feature_ranking.csv', 'formal_profile_display_bins.csv',
        'flow_stable_feature_ranking.csv', 'flow_stable_display_bins.csv'} |
                               {f'figures/{name}' for _, name in POPULATIONS.values()})
    (output / 'analysis_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)

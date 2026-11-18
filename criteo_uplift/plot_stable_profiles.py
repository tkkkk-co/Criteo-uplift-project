"""Overlay normalized feature histograms for the common Test and stable selections."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogLocator, FuncFormatter

from .config import FEATURES
from .stability_analysis import load_test_features


def plot_profiles(source_run, experiment):
    protocol = json.loads((experiment / 'protocol.json').read_text())
    manifest = json.loads((source_run / 'manifest.json').read_text())
    seed_paths = [(seed, experiment / f'seed{seed}/test_predictions.npz') for seed in protocol['seeds']]
    seed_paths.append((manifest['seed'], source_run / 'evaluation/predictions.npz'))
    with np.load(seed_paths[-1][1]) as saved:
        row_ids = saved['row_id']
    n, k = len(row_ids), int(np.floor(.2 * len(row_ids)))
    stable = {name: np.ones(n, dtype=bool) for name in ['response_xgboost', 's_xgboost']}
    for seed, path in seed_paths:
        with np.load(path) as saved:
            np.testing.assert_array_equal(row_ids, saved['row_id'])
            priority = np.random.default_rng(seed).random(n)
            for name in stable:
                order = np.lexsort((priority, -saved[name]))
                chosen = np.zeros(n, dtype=bool)
                chosen[order[:k]] = True
                stable[name] &= chosen
    features = load_test_features(source_run, row_ids)
    recorded = pd.read_csv(experiment / 'stable_population_groups.csv').query("group == 'all_fits_selected'").set_index('policy')
    for name, mask in stable.items():
        assert int(mask.sum()) == recorded.loc[name, 'n']
    groups = [
        ('All Test Set', np.ones(n, dtype=bool), '#377ca8', '--'),
        ('Response-XGBoost stable', stable['response_xgboost'], '#df852c', '-'),
        ('S-XGBoost stable', stable['s_xgboost'], '#42765a', '-'),
    ]
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
    fig = Figure(figsize=(16, 11), facecolor='white')
    axes = fig.subplots(3, 4)
    records = []
    for feature, ax in zip(FEATURES, axes.flat):
        x = features[feature].to_numpy()
        edges = np.histogram_bin_edges(x, bins=40)
        for label, mask, color, linestyle in groups:
            counts, _ = np.histogram(x[mask], bins=edges)
            probability = counts / mask.sum() * 100
            ax.stairs(probability, edges, color=color, linestyle=linestyle, linewidth=1.7, label=label)
            for left, right, count, percent in zip(edges[:-1], edges[1:], counts, probability):
                records.append((feature, label, float(left), float(right), int(count), float(percent)))
            np.testing.assert_allclose(probability.sum(), 100)
        ax.set_title(feature, fontsize=13, loc='left', fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(.00001, 150)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.grid(axis='y', alpha=.15)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('特征取值', fontproperties=font, fontsize=9)
        if ax in axes[:, 0]:
            ax.set_ylabel('每箱样本占比（%，对数）', fontproperties=font, fontsize=10)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 4), useOffset=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    labels = [
        f'全体 Test Set（n={n:,}）',
        f'Response 稳定入选（n={stable["response_xgboost"].sum():,}）',
        f'S-XGBoost 稳定入选（n={stable["s_xgboost"].sum():,}）',
    ]
    fig.suptitle('稳定入选人群与全体 Test Set 的匿名特征分布', fontproperties=font, fontsize=20, y=.985)
    fig.legend(handles, labels, prop=font, loc='upper center', bbox_to_anchor=(.5, .953), ncol=3, frameon=False)
    fig.text(.5, .019,
             '稳定入选＝原正式模型与三个重训种子（17、29、83）均选入 Top 20%\n'
             '各特征共用全体 Test 的 40 个等宽分箱；各群体独立归一化至 100%；保留完整取值范围。',
             ha='center', fontproperties=font, fontsize=10, linespacing=1.6, color='#4c5d68')
    fig.subplots_adjust(left=.065, right=.985, bottom=.095, top=.89, hspace=.43, wspace=.25)
    output = experiment / 'figures'
    output.mkdir(exist_ok=True)
    path = output / 'stable_feature_overlay.png'
    fig.savefig(path, dpi=180)
    pd.DataFrame(records, columns=['feature', 'population', 'left', 'right', 'count', 'percent']).to_csv(
        output / 'stable_feature_overlay_bins.csv', index=False)
    (output / 'stable_feature_overlay.json').write_text(json.dumps({
        'scope': 'common frozen Test', 'test_n': n, 'seeds': [seed for seed, _ in seed_paths],
        'stable_n': {key: int(value.sum()) for key, value in stable.items()},
        'bins': 40, 'range': 'full Test min/max per feature',
        'normalization': 'count / population n × 100', 'y_axis': 'logarithmic percentage',
    }, ensure_ascii=False, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    plot_profiles(args.source_run, args.experiment)

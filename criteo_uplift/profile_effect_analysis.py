"""Persist policy-stable membership, anonymous profiles, and held-out effect bins."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from scipy.stats import norm

from .config import FEATURES
from .pipeline_io import write_result
from .stability_analysis import load_test_features

Z975 = norm.ppf(.975)


def wilson(events, n):
    p = events / n
    scale = 1 + Z975**2 / n
    middle = (p + Z975**2 / (2*n)) / scale
    half = Z975 * np.sqrt(p*(1-p)/n + Z975**2/(4*n*n)) / scale
    return middle-half, middle+half


def effect(n1, n0, r1, r0):
    p1, p0 = r1/n1, r0/n0
    lo1, hi1 = wilson(r1, n1)
    lo0, hi0 = wilson(r0, n0)
    delta = p1-p0
    return delta, delta-np.hypot(p1-lo1, hi0-p0), delta+np.hypot(hi1-p1, p0-lo0)


def analyze(source_run, experiment):
    output = experiment / 'profile_effect_analysis'
    output.mkdir(exist_ok=True)
    manifest = json.loads((source_run / 'manifest.json').read_text())
    policies = ['response_xgboost', 's_xgboost']
    selections = pd.read_csv(experiment / 'selections.csv').set_index('seed').winner.to_dict()
    reference_seed = manifest['seed']
    selections[reference_seed] = manifest['selection']['selected_policy']
    with np.load(source_run / 'evaluation/predictions.npz') as saved:
        ids, t, y = saved['row_id'], saved['t'], saved['y']
        scores = {name: saved[name] for name in policies}
    n, k = len(ids), int(.2*len(ids))
    flow_masks = {}
    official_masks = {}
    for seed, policy in selections.items():
        path = source_run / 'evaluation/predictions.npz' if seed == reference_seed else experiment / f'seed{seed}/test_predictions.npz'
        with np.load(path) as saved:
            np.testing.assert_array_equal(ids, saved['row_id'])
            order = np.lexsort((np.random.default_rng(seed).random(n), -saved[policy]))
            mask = np.zeros(n, dtype=bool)
            mask[order[:k]] = True
            flow_masks[seed] = mask
    stable = np.logical_and.reduce(list(flow_masks.values()))
    pair_rows = []
    for a,b in combinations(selections,2):
        pair_rows.append({'seed_a':a, 'policy_a':selections[a], 'seed_b':b, 'policy_b':selections[b],
                          'common_n':int((flow_masks[a]&flow_masks[b]).sum()),
                          'overlap':float((flow_masks[a]&flow_masks[b]).sum()/k)})
    write_result(pd.DataFrame(pair_rows), output/'flow_overlap.csv')
    membership = {'row_id':ids, 'flow_stable':stable}
    membership.update({f'flow_seed{seed}':mask for seed,mask in flow_masks.items()})
    bin_records = []
    whole = {'n1': int(t.sum()), 'n0':int((1-t).sum()),
             'r1':int((t*y).sum()), 'r0':int(((1-t)*y).sum())}
    whole_effect = effect(**whole)
    for policy in policies:
        order = np.lexsort((np.random.default_rng(reference_seed).random(n), -scores[policy]))
        chosen = np.zeros(n,dtype=bool)
        chosen[order[:k]]=True
        official_masks[policy]=chosen
        membership[f'formal_{policy}']=chosen
        for bins in [5,10]:
            bounds = np.floor(np.arange(bins+1)*n/bins).astype(int)
            labels = np.empty(n,dtype=np.int8)
            for b in range(bins):
                idx = order[bounds[b]:bounds[b+1]]
                labels[idx]=b+1
                tt, yy = t[idx], y[idx]
                counts={'n1':int(tt.sum()),'n0':int((1-tt).sum()),'r1':int((tt*yy).sum()),'r0':int(((1-tt)*yy).sum())}
                itt, lo, hi=effect(**counts)
                # Within-arm group/full dependence is retained in the contrast variance.
                contrast_var=0.
                for arm in [0,1]:
                    ng=counts[f'n{arm}']; rg=counts[f'r{arm}']; total=whole[f'n{arm}']
                    nr=total-ng; rr=whole[f'r{arm}']-rg
                    pg,pr=rg/ng,rr/nr
                    contrast_var += (nr/total)**2*(pg*(1-pg)/(ng-1)+pr*(1-pr)/(nr-1))
                contrast=itt-whole_effect[0]
                bin_records.append({'policy':policy,'bins':bins,'group':b+1,
                    'coverage_start':bounds[b]/n,'coverage_end':bounds[b+1]/n,
                    'n':len(idx),'n_treatment':counts['n1'],'n_control':counts['n0'],
                    'events_treatment':counts['r1'],'events_control':counts['r0'],
                    'rate_treatment':counts['r1']/counts['n1'],'rate_control':counts['r0']/counts['n0'],
                    'mean_score':float(scores[policy][idx].mean()),
                    'mean_predicted_cate':float(scores[policy][idx].mean()) if policy=='s_xgboost' else None,
                    'itt':itt,'ci_low':lo,'ci_high':hi,'whole_test_itt':whole_effect[0],
                    'itt_minus_whole':contrast,
                    'contrast_ci_low':contrast-Z975*np.sqrt(contrast_var),
                    'contrast_ci_high':contrast+Z975*np.sqrt(contrast_var)})
            membership[f'{policy}_bins{bins}']=labels
    np.savez_compressed(output/'membership_and_bins.npz',**membership)
    bins_frame=pd.DataFrame(bin_records)
    write_result(bins_frame,output/'effect_bins.csv')
    features=load_test_features(source_run,ids)
    profile_rows=[]; hist_rows=[]; cdf_rows=[]
    populations={'flow_stable':stable, **{f'formal_{name}':mask for name,mask in official_masks.items()}}
    for feature in FEATURES:
        x=features[feature].to_numpy(dtype=np.float64)
        edges=np.histogram_bin_edges(x,bins=40)
        mean,sd=float(x.mean()),float(x.std(ddof=1))
        q75=float(np.quantile(x,.75))
        values,counts=np.unique(x,return_counts=True)
        cumulative=np.cumsum(counts)/n
        # Full discrete ECDF is retained, including tied feature values.
        for value,cdf in zip(values,cumulative):
            cdf_rows.append((feature,'all_test',float(value),float(cdf)))
        for label,mask in populations.items():
            subset=x[mask]; sm=float(subset.mean())
            profile_rows.append({'population':label,'feature':feature,'n':int(mask.sum()),
                'population_mean':mean,'selected_mean':sm,'standardized_mean_shift':(sm-mean)/sd if sd else 0.,
                'mean_percentile_in_test':float((x<=sm).mean()),'test_q75':q75,
                'test_fraction_above_q75':float((x>q75).mean()),
                'selected_fraction_above_q75':float((subset>q75).mean())})
            val,cnt=np.unique(subset,return_counts=True)
            for value,cdf in zip(val,np.cumsum(cnt)/len(subset)):
                cdf_rows.append((feature,label,float(value),float(cdf)))
        for label,mask in {'all_test':np.ones(n,dtype=bool),**populations}.items():
            counts,_=np.histogram(x[mask],bins=edges)
            percentages=counts/mask.sum()*100
            np.testing.assert_allclose(percentages.sum(),100)
            for left,right,count,percent in zip(edges[:-1],edges[1:],counts,percentages):
                hist_rows.append((feature,label,float(left),float(right),int(count),float(percent)))
    profile_frame=pd.DataFrame(profile_rows)
    write_result(profile_frame,output/'feature_profiles.csv')
    hist=pd.DataFrame(hist_rows,columns=['feature','population','left','right','count','percent'])
    write_result(hist,output/'histogram_bins.csv')
    cdf=pd.DataFrame(cdf_rows,columns=['feature','population','value','cdf'])
    write_result(cdf,output/'feature_ecdf.csv')
    record={'test_n':n,'top20_n':k,'selected_policies':{str(seed):name for seed,name in selections.items()},
        'flow_stable_n':int(stable.sum()),'fraction_of_top20':float(stable.sum()/k),
        'fraction_of_test':float(stable.mean()),'whole_test_itt':whole_effect[0],
        'effect_ci':'Two-sided 95% Wilson-Newcombe difference of proportions; no multiplicity adjustment.',
        'group_vs_whole_ci':'Unpooled normal contrast interval accounting for nested group/full means within each arm.',
        'feature_profiles':'Descriptive means, Test percentile location, and strict above-Q75 proportions.',
        'bins':'Descending fixed formal scores; floor(q*N) cutoffs, shared tie priority seed42. Membership fixed.',
        'scope':'Fixed Test post-hoc descriptive analysis. No refit or Test-based selection.',
        'all_records_saved':True}
    write_result(record,output/'analysis_record.json')
    draw_figures(output,bins_frame,hist,cdf,profile_frame,record)
    from .plot_profile_summary import main as draw_profile_summaries

    draw_profile_summaries(output)
    from .plot_profile_histograms import main as draw_fixed_histograms
    draw_fixed_histograms(output)
    print((output/'analysis_record.json').read_text(),flush=True)


def draw_figures(output,bins_frame,hist,cdf,profiles,record):
    font=FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
    chosen=profiles.loc[profiles.population.eq('flow_stable')].assign(abs_shift=lambda d:d.standardized_mean_shift.abs())
    features=chosen.sort_values('abs_shift',ascending=False).feature.head(3).tolist()
    figures=output/'figures';figures.mkdir(exist_ok=True)
    # ECDF avoids logarithmic tails and crowded histogram spikes.
    fig=Figure(figsize=(13,4.6),facecolor='white')
    axes=fig.subplots(1,3)
    for feature,ax in zip(features,axes):
        for population,color,label in [('all_test','#718399','全体 Test Set'),('flow_stable','#c56d26','选型流程稳定人群')]:
            frame=cdf.loc[cdf.feature.eq(feature)&cdf.population.eq(population)]
            ax.step(frame.value,frame.cdf*100,where='post',color=color,label=label,linewidth=2)
        ax.set_title(feature,fontsize=14)
        ax.set_xlabel('特征取值',fontproperties=font)
        ax.set_ylabel('累计人群比例（%）',fontproperties=font)
        ax.set_ylim(0,103);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('稳定投放人群的特征分布：三项差异最大的特征',fontproperties=font,fontsize=16,y=.99)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,prop=font,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
    fig.subplots_adjust(top=.72,bottom=.17,wspace=.28,left=.065,right=.985)
    fig.savefig(figures/'flow_stable_feature_ecdf.png',dpi=180)
    fig=Figure(figsize=(13,7.4),facecolor='white')
    axes=fig.subplots(2,3)
    for row,(population,title) in enumerate([
        ('formal_response_xgboost','Response-XGBoost'),
        ('formal_s_xgboost','S-XGBoost'),
    ]):
        for feature,ax in zip(features,axes[row]):
            for group,color,label in [('all_test','#718399','全体 Test Set'),(population,'#c56d26','该模型 Top 20%')]:
                frame=cdf.loc[cdf.feature.eq(feature)&cdf.population.eq(group)]
                ax.step(frame.value,frame.cdf*100,where='post',color=color,label=label,linewidth=1.8)
            ax.set_title(f'{title} · {feature}',fontsize=11)
            ax.set_ylim(0,103)
            ax.set_xlabel('特征取值',fontproperties=font)
            ax.set_ylabel('累计人群比例（%）',fontproperties=font)
            ax.grid(alpha=.18)
            ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.suptitle('原正式模型的投放画像｜各自 Top 20% 与全体 Test 对照',fontproperties=font,fontsize=16,y=.99)
    fig.legend(handles,labels,prop=font,loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False)
    fig.subplots_adjust(top=.83,bottom=.08,hspace=.45,wspace=.28,left=.065,right=.985)
    fig.savefig(figures/'formal_selected_feature_ecdf.png',dpi=180)
    # Full EDA retained as a two-population overlay.
    fig=Figure(figsize=(14,9),facecolor='white');axes=fig.subplots(3,4)
    for feature,ax in zip(FEATURES,axes.flat):
        for population,color,label in [('all_test','#718399','全体 Test Set'),('flow_stable','#c56d26','选型流程稳定人群')]:
            frame=hist.loc[hist.feature.eq(feature)&hist.population.eq(population)]
            edges=np.r_[frame.left.to_numpy(),frame.right.iloc[-1]]
            ax.stairs(frame.percent,edges,color=color,label=label,linewidth=1.5)
        ax.set_title(feature,loc='left',fontsize=12)
        ax.set_yscale('log');ax.set_ylim(.00001,150);ax.tick_params(labelsize=8)
        ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('稳定投放人群与全体 Test Set：十二维 EDA',fontproperties=font,fontsize=17)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,prop=font,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.text(.5,.02,'共用完整取值范围及40个等宽分箱；每组占比独立归一化；纵轴为每箱占比（%，对数）。',ha='center',fontproperties=font,fontsize=10)
    fig.subplots_adjust(top=.85,bottom=.08,wspace=.3,hspace=.3)
    fig.savefig(figures/'flow_stable_full_eda.png',dpi=180)
    draw_deciles(output)


def draw_deciles(output):
    """Render only the saved decile point estimates; keep interval data intact."""
    output = Path(output)
    data = pd.read_csv(output / 'effect_bins.csv').query('bins == 10')
    record = json.loads((output / 'analysis_record.json').read_text())
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
    fig = Figure(figsize=(12, 4.9), facecolor='white')
    axes = fig.subplots(1, 2, sharey=True)
    for policy, ax, title in zip(['response_xgboost', 's_xgboost'], axes, ['Response-XGBoost', 'S-XGBoost']):
        frame = data.loc[data.policy.eq(policy)].sort_values('group')
        ax.plot(frame.group, frame.itt * 100, 'o-', color='#287c9b', lw=2, ms=5, label='观测 ITT')
        ax.axhline(record['whole_test_itt'] * 100, ls=':', color='#7f8994', label='全体 Test Set ITT')
        if policy == 's_xgboost':
            ax.plot(frame.group, frame.mean_predicted_cate * 100, 's--', color='#c56d26', lw=1.7, ms=4, label='平均预测 CATE')
        ax.axhline(0, color='#b8b8b8', lw=.7)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel('分数十分位组（1 最高，10 最低）', fontproperties=font)
        ax.set_ylabel('转化率差（百分点）', fontproperties=font)
        ax.grid(axis='y', alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(prop=font, frameon=False)
    fig.suptitle('十分层处理效应验证｜固定模型与 Test Set', fontproperties=font, fontsize=16)
    fig.text(.5, .025, '各层互不重叠；仅展示点估计，连线用于阅读，不代表连续效应或显著性。', ha='center', fontproperties=font, fontsize=10)
    fig.subplots_adjust(top=.84, bottom=.19, wspace=.20, left=.065, right=.985)
    fig.savefig(output / 'figures/effect_deciles.png', dpi=180)
    record['report_effect_display'] = 'Deciles only; point-estimate lines without intervals; original interval and quintile data retained outside report.'
    if 'figures/effect_deciles.png' not in record['outputs']:
        record['outputs'].append('figures/effect_deciles.png')
    write_result(record, output / 'analysis_record.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.source_run,args.experiment)

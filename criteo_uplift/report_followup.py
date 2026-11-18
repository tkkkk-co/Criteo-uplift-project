"""Report chapters based exclusively on persisted profile and repeated-fit results."""
from pathlib import Path
import json

import pandas as pd


def analysis_path(out):
    path = Path(out).parent / 'full_split_stability_test_20261117'
    return path if Path(out).name == 'final_conversion_full_seed42' and (path / 'profile_effect_analysis/analysis_record.json').exists() else None


def add_profiles_and_deciles(chapter, experiment):
    from .reporting import _paragraph as p, _table, _image, _details, _LABELS
    root = experiment / 'profile_effect_analysis'
    profiles = pd.read_csv(root / 'feature_profiles.csv')
    from .plot_profile_histograms import ranked_features
    response_rank = ranked_features(profiles, 'formal_response_xgboost')
    s_rank = ranked_features(profiles, 'formal_s_xgboost')
    features = response_rank.feature.head(3).tolist()
    assert features == s_rank.feature.head(3).tolist(), '两模型画像特征排序不一致，需要分别说明'
    rows = []
    for feature in features:
        r = profiles.loc[profiles.feature.eq(feature)].set_index('population')
        rows.append((feature, f'{r.population_mean.iloc[0]:.3f}',
                     f'{r.loc["formal_response_xgboost", "selected_mean"]:.3f}',
                     f'{r.loc["formal_s_xgboost", "selected_mean"]:.3f}',
                     f'{r.loc["formal_response_xgboost", "standardized_mean_shift"]:+.3f}',
                     f'{r.loc["formal_s_xgboost", "standardized_mean_shift"]:+.3f}'))
    profile = (
        p('对两个固定模型分别比较 Top 20% 与全体 Test Set 的十二维特征，按绝对标准化均值差从大到小排序。'
          '两个模型的前三项均为 f8、f9、f3，因此采用这三项作共同画像展示，表内按该顺序排列。')
        + p(r'对模型 \(m\) 的特征 \(j\)，标准化均值差为 '
            r'\(\Delta_j^{(m)}=(\bar X_{j,\mathrm{Top20}}^{(m)}-\bar X_{j,\mathrm{Test}})/s_{j,\mathrm{Test}}\)。'
            '分母是全体 Test Set 的特征样本标准差；正值表示入选均值较高，负值表示较低，绝对值表示偏离整体的幅度。')
        + '<h4>表 6-6｜固定策略 Top 20% 的前三项画像特征</h4>'
        + _table(pd.DataFrame(rows, columns=['匿名特征', '全体 Test 均值', 'Response 均值', 'S 均值', 'Response 标准化均值差', 'S 标准化均值差']))
        + _image(root / 'figures/formal_response_feature_histogram.png', '图 6-2｜Response-XGBoost 的入选特征分布',
                 '上排为全体 Test Set，下排为该模型 Top 20%；上下刻度与网格对齐，虚线及其上方标注对应各组均值。两端柱合并尾部，全部记录计入组内占比。')
        + _details('s-xgboost-profile', '图 6-3｜S-XGBoost 的入选特征分布',
            _image(root / 'figures/formal_s_feature_histogram.png', 'S-XGBoost 的入选特征分布',
                   '与上一图使用相同分箱和坐标口径；虚线上方标注各组均值，两排均显示横坐标。'))
        + p('<strong>两种规则均优先覆盖 f8、f3 较低、f9 较高的记录。</strong>'
            'f8 与 f3 的入选分布相对全体向较低取值移动，f9 则从低值集中区域向较高取值扩展。'
            '以 f9 为例，Response 与 S 入选均值分别约为 26.07、26.06，高于全体的 16.03。'
            'f9 高于全体第 75% 分位点（约 13.19）的记录，在全体 Test Set 中占 20.51%，'
            '在两模型入选集合中分别占 77.48% 和 77.27%。两种规则的主要特征方向一致，与其较高名单重叠率相呼应。')
        + p('这些画像将模型的资格分配结果转化为可检查的特征分布，可用于核验新流量的输入结构和入选名单是否偏离离线基准。'
            '展示特征依据事后均值差筛选，匿名字段尚不能转换为具体人口属性，差异幅度也不表示特征的因果作用。'
            '实际投放仍由完整十二维模型评分及 Top-K 规则决定，不能仅凭单项特征阈值替代模型。'))
    bins = pd.read_csv(root / 'effect_bins.csv').query('bins == 10')
    r = bins.query('policy == "response_xgboost"').set_index('group')
    s = bins.query('policy == "s_xgboost"').set_index('group')
    data = pd.DataFrame({
        '分数层（高→低）': [f'{i*10}%–{(i+1)*10}%' for i in range(10)],
        'Response 观测 ITT（pp）': (r.itt * 100).map(lambda x: f'{x:.4f}').tolist(),
        'S 观测 ITT（pp）': (s.itt * 100).map(lambda x: f'{x:.4f}').tolist(),
        'S 平均预测 CATE（pp）': (s.mean_predicted_cate * 100).map(lambda x: f'{x:.4f}').tolist(),
    })
    deciles = ('<h3 id="effect-deciles">6.6 十分层处理效应验证与扩量增益分析</h3>'
        + p('沿用同一冻结评分，本节从分段效应与累计收益两个角度检查增量集中程度。'
            '分层 ITT 描述各不重叠分数段的平均资格效应，累计 HT Gain 描述逐步扩大覆盖率时对全部候选的转化贡献；两者使用各自的估计口径。')
        + '<h4>十分层处理效应验证</h4>'
        + p('将原正式模型在 Test Set 的分数从高到低排序，按人数切成十个互不重叠的区间。'
            'Response 使用处理后的预测转化概率，S 使用预测 CATE；两种模型分别分层，因此同编号层的人群不完全相同。'
            '这里的 10%–20% 指第二个十分位组，不是累计 Top 20%。')
        + p(r'每层 \(B_j\) 的观测效应为该层处理组与对照组转化率之差：'
            r'\(\widehat\tau_j=\frac{\sum_{i\in B_j}T_iY_i}{\sum_{i\in B_j}T_i}'
            r'-\frac{\sum_{i\in B_j}(1-T_i)Y_i}{\sum_{i\in B_j}(1-T_i)}\)。'
            r'S 的预测曲线取同层的平均预测 CATE，即 \( |B_j|^{-1}\sum_{i\in B_j}\widehat\tau_S(X_i)\)。'
            '分层只依赖冻结分数；观测结果用于计算层内效应，不参与排序。')
        + _image(root / 'figures/effect_deciles.png', '图 6-4｜十分层观测 ITT 与平均预测 CATE',
                 '折线仅连接分层点估计，不表示连续效应或显著性。虚线基准为全体 Test Set ITT（0.1146 个百分点）；仅 S 展示预测 CATE。')
        + _details('effect-decile-table', '表 6-7｜十分层效应点估计（完整明细）', _table(data))
        + p('<strong>可分配的增量主要集中在最高分的前 10%。</strong>'
            'Response 和 S 首层观测 ITT 分别约为 0.8323 和 0.8225 个百分点，均明显高于全体 Test Set 的 0.1146 个百分点。'
            '第二层约为 0.0475 和 0.0442 个百分点，其余层的点估计进一步接近零。'
            '这支持将高分群体作为优先试点对象，也为后续评估较低覆盖率提供人群层面的解释。')
        + p('S 首层平均预测 CATE 约为 0.7707 个百分点，与观测 ITT 的量级接近；后续各层不严格单调，'
            '低转化率下的层内事件稀疏使小效应的点估计容易波动。图中未展示区间，因此不据此判断各层差异显著或模型已精确校准。'
            '这里验证的是分层后的效应集中程度；固定预算下的策略优劣仍以第 6.4 节的 HT Gain 与配对比较为依据。'))
    chapter = chapter.replace('<h3 id="budget-expansion">6.6 扩量增益与资格名额配置</h3>', profile + deciles + '<h4 id="budget-expansion">累计收益与扩量回报</h4>', 1)
    return chapter.replace('表 6-6｜冻结策略逐档扩量', '表 6-8｜冻结策略逐档扩量')


def chapter_seven(out, experiment):
    from .reporting import _paragraph as p, _table, _section, _details, _image, _LABELS
    root = experiment / 'profile_effect_analysis'
    record = json.loads((root / 'analysis_record.json').read_text())
    original = json.loads((out / 'models/selection.json').read_text())
    selections = pd.read_csv(experiment / 'selections.csv').set_index('seed')
    metrics = pd.read_csv(experiment / 'test_comparison.csv')
    validation = pd.concat([
        pd.read_csv(out / 'models/validation_metrics.csv').assign(seed=42),
        pd.read_csv(experiment / 'validation_comparison.csv'),
    ], ignore_index=True)
    rows = []
    for seed in [42, 17, 29, 83]:
        winner = original['selected_policy'] if seed == 42 else selections.loc[seed, 'winner']
        uplift = original['selected_uplift_model'] if seed == 42 else selections.loc[seed, 'best_uplift']
        v = validation.loc[validation.seed.eq(seed) & validation.policy.eq(winner) & validation.budget.eq(.2)].iloc[0]
        test = metrics.loc[metrics.seed.eq(seed) & metrics.policy.eq(winner)].iloc[0]
        rows.append(('42（正式）' if seed == 42 else str(seed), _LABELS[winner], _LABELS[uplift],
                     f'{v.gain_ht_per_100k:.2f}', f'{test.gain_ht_per_100k:.2f}', f'{test.gap_to_observed_best:.2f}'))
    selection_table = _table(pd.DataFrame(rows, columns=['Seed', 'Validation 首位策略', 'Validation 首位 Uplift',
        '首位策略 Validation HT', '同策略 Test HT', '距该轮 Test 最高点差']))
    wide = metrics.pivot(index='policy', columns='seed', values='gain_ht_per_100k')
    wide = wide.loc[wide[42].sort_values(ascending=False).index, [42, 17, 29, 83]]
    gains = pd.DataFrame({'策略': [_LABELS[x] for x in wide.index]})
    for seed in wide.columns:
        gains[f'{seed}（正式）' if seed == 42 else str(seed)] = wide[seed].map(lambda x: f'{x:.2f}').tolist()
    profiles = pd.read_csv(root / 'feature_profiles.csv').query('population == "flow_stable"').set_index('feature')
    f9 = profiles.loc['f9']
    pairs = pd.read_csv(root / 'flow_overlap.csv')
    fixed = pd.read_csv(experiment / 'selection_overlap.csv')
    fixed_rows = []
    for policy, group in fixed.groupby('policy'):
        fixed_rows.append((_LABELS[policy], f'{group.overlap.min():.2%}', f'{group.overlap.max():.2%}'))
    coverage = pd.read_csv(experiment / 'coverage_sensitivity.csv').query('policy == "response_xgboost" and metric == "HT"').set_index('seed')
    cov = pd.DataFrame({
        'Seed': ['42（正式）', '17', '29', '83'],
        '10% HT Gain': [f'{coverage.loc[s, "gain10_per100k"]:.2f}' for s in [42,17,29,83]],
        '20% HT Gain': [f'{coverage.loc[s, "gain20_per100k"]:.2f}' for s in [42,17,29,83]],
        '10% / 20% 收益': [f'{coverage.loc[s, "retention10_vs20"]:.2%}' for s in [42,17,29,83]],
        '10%–20% 每千新增名额的增量': [f'{coverage.loc[s, "margin10_20_per1000_slots"]:.3f}' for s in [42,17,29,83]],
    })
    gaps = metrics.query('policy == "response_xgboost"').gap_to_observed_best
    return _section('stability', '07｜稳定性分析：模型选型、收益差距、入选人群与覆盖率',
        p('稳定性分析分别检查模型名称、投放收益和入选人群是否随重新划分而改变。'
          '以正式 Seed 42 为参照，补充 Seed 17、29、83 三轮完整拟合；各轮在 Validation Set 独立选型，'
          '再把已拟合模型作用于同一正式 Test Set。')
        + '<h3>7.1 重复拟合方案与 Validation Set 选型</h3>'
        + p('固定 2,795,919 条 Test Set 记录，在其余 11,183,673 条记录中重新按 treatment × conversion 分层，'
            '以 75%∶25% 划分 Train Set 与 Validation Set，每轮拟合八个候选并保留 Random 参照。'
            '模型配置与 20% 预算保持一致，Validation Set 按 HT Gain 点估计选型，不做 Bootstrap。'
            '划分种子同时用于模型随机数，因此结果反映划分与训练随机性的共同影响。')
        + '<h4>表 7-1｜正式运行与三轮重复拟合的选型结果</h4>' + selection_table
        + p('表中 Gain 及点差均以每 10 万名合资格用户计。正式运行置于顶行。'
            '四轮中 Response-XGBoost 三次成为 Validation Set 首位，S-XGBoost 一次首位；'
            '六类 Uplift 候选内部的首位也发生变化。由此可见，接近的候选之间存在选型敏感性，'
            '不能把单轮排名解释为稳定的模型性能顺序。')
        + '<h3>7.2 排名波动是否改变投放收益</h3>'
        + '<h4>表 7-2｜同一 Test Set 上四轮模型的 HT Gain @20%</h4>' + _table(gains)
        + p('<strong>定向相对随机的优势稳定，Response 与 S 的差异仍小。</strong>'
            'S-XGBoost 在四轮共同 Test Set 上均为最高点估计；Response-XGBoost 约为 108.66–109.46，'
            'S-XGBoost 约为 109.50–110.54，而 Random 约为 22.90。'
            f'Response 距各轮观测最高点的差距均值为 {gaps.mean():.2f}、最大为 {gaps.max():.2f} 次／10 万人。'
            '因此，首位模型名称变化尚未表现为同量级的投放收益损失。这里的差距以观测最高点为参照，不是真实最优策略的遗憾值。')
        + p('S 的重复领先使其成为值得优先线上验证的挑战策略；正式运行的 S−Response 配对区间仍跨零，'
            '这两项证据分别反映重训方向和单次评价精度。它们不足以直接确定线上替换。'
            '四轮使用相同 Test Set，开发样本也相互重叠，不能将四次点估计领先当作四次独立成功验证，'
            '更不能把 Test Set 排名重新写入 Validation Set 冻结记录。')
        + '<h3>7.3 选型流程的稳定入选人群与画像</h3>'
        + p('每轮取 Validation Set 首位模型，在共同 Test Set 上选择 Top 20%，再对四份名单取交集，定义为“流程稳定入选人群”。'
            '对应模型为 Seed 42、17、29 的 Response-XGBoost 与 Seed 83 的 S-XGBoost。'
            '这一口径衡量从重新拟合、选型到资格分配的整体稳定性，不等于单个固定模型跨四轮的交集。')
        + _table(pd.DataFrame([
            ('每轮资格名额', f'{record["top20_n"]:,}'),
            ('四轮共同入选', f'{record["flow_stable_n"]:,}'),
            ('占单轮名额', f'{record["fraction_of_top20"]:.2%}'),
            ('占全体 Test Set', f'{record["fraction_of_test"]:.2%}'),
            ('两轮名单重叠范围（六对）', f'{pairs.overlap.min():.2%}–{pairs.overlap.max():.2%}'),
        ], columns=['口径', '结果']))
        + _image(root / 'figures/flow_stable_feature_histogram.png', '图 7-1｜流程稳定入选人群与全体 Test Set 的画像对照',
                 '上排为全体 Test Set，下排为四轮流程稳定入选人群。分箱、刻度与网格沿用第 6.5 节，虚线上方标注各组均值；全部记录纳入占比。')
        + p(f'<strong>稳定入选主要对应相近的特征区域，而非完全不同的人群。</strong>'
            f'f9 均值为 {f9.selected_mean:.2f}，这一取值约位于全体 Test Set 的第 {f9.mean_percentile_in_test:.1%} 分位位置；'
            f'超过全体 f9 第 75% 分位点的记录，在稳定人群中占 {f9.selected_fraction_above_q75:.2%}，'
            f'全体占比为 {f9.test_fraction_above_q75:.2%}。同时，稳定人群的 f8、f3 均值较低。'
            '这说明本次开发样本扰动后，大部分优先分配区域仍可重复识别，可作为后续流量画像和名单漂移的检查基准。')
        + p(r'特征差异按 \((\bar X_{\mathrm{stable}}-\bar X_{\mathrm{Test}})/s_{\mathrm{Test}}\) 标准化，'
            '分母为全体 Test Set 的特征标准差；绝对值越大，只表示该特征的入选均值偏离整体越多。'
            '这不是变量的因果效应、显著性检验或模型重要性。匿名字段取值有集中点，因此严格高于第 75% 分位点的整体占比未必恰好为 25%。')
        + p('四轮交集占单轮名额的 93.49%，说明此次扰动下大部分名额去向一致。'
            '但交集由既有 Test Set 和多轮分数事后定义，新样本上线时无法直接使用这份历史名单。'
            '实际交付仍是一个冻结模型及覆盖率规则，稳定画像用于验证规则的可重复性和识别漂移。')
        + _details('fixed-family-overlap', '补充｜固定模型类别跨四轮的名单重叠',
            _table(pd.DataFrame(fixed_rows, columns=['模型类别', '两轮重叠最小值', '两轮重叠最大值']))
            + p('此表固定 Response 或 S 模型类别，分别计算六对 Top 20% 名单的交集占各自名额的比例。'
                '它检查同类模型重训后的名单稳定性，作为整体选型流程结果的补充。'))
        + '<h3>7.4 覆盖率收益的重复性与分析边界</h3>'
        + '<h4>表 7-3｜Response-XGBoost 较低覆盖率的重复拟合结果</h4>' + _table(cov)
        + p('四轮中，Response 的前 10% 均保留前 20% 约 95.67%–97.32% 的 HT 累计增量。'
            '10%–20% 每千新增名额仅增加约 0.292–0.474 次转化，收益高度集中于首批名额的方向具有重复性。'
            '按 Uplift Gain 计算，前 10% 保留约 84.61%–86.04% 的前 20% 收益，集中程度随估计口径有所不同。'
            '两种口径均支持比较较低覆盖率的经济性，不宜将 95%以上这一单一比例直接解释为已证实可节省一半实际广告花费。')
        + p('以上为共同 Test Set 上的事后敏感性分析，新增轮次只报告点估计，不增加一轮新的选型或区间推断。'
            '本分析未覆盖时间漂移、跨平台迁移和真实用户重复观测；这些不确定性需要新的业务数据检验。')
        + _details('stability-records', '重训、评分与分析记录',
            p('<code>criteo_uplift/split_stability.py</code> 保存每轮八个模型、参数与训练状态、行号划分、Validation/Test 评分和选型记录；'
              '<code>stability_analysis.py</code> 汇总收益、覆盖率和同类模型名单；'
              '<code>profile_effect_analysis.py</code> 保存十分层、人群成员和画像；'
              '<code>plot_profile_histograms.py</code> 从已保存分布生成固定模型与流程稳定人群的柱状图。')
            + p('<code>artifacts/full_split_stability_test_20261117/</code> 为本章记录目录；'
                '<code>profile_effect_analysis/</code> 内保留原始分层计数与点估计、画像统计和成员行号。正文生成无需重新训练。')))


def chapter_eight():
    from .reporting import _paragraph as p, _section, _table
    decisions = pd.DataFrame([
        ('定向是否有价值', '同名额下，冻结策略优于 Random；跨四轮仍保持较大点估计差距。', '优先验证模型定向相对随机或现有规则的增量价值。'),
        ('Response 与 Uplift 如何取舍', '两者收益接近、名单高度重合；S 在重复拟合的共同 Test Set 上持续小幅领先。', '保留正式 Response 基线，以 S 为优先挑战策略，按线上净增量决定替换。'),
        ('是否保持 20% 覆盖', '前 10% 已承载大部分累计增量，后续名额单位回报下降。', '20% 保留为离线评价口径；10% 作为预算试点候选，与实际成本结合。'),
        ('交付什么人群规则', '大部分入选记录和主要匿名特征方向在重复拟合后保持一致。', '交付冻结模型、预处理和 Top-K 决策规则，以画像监测目标流量适用性。'),
    ], columns=['决策问题', '现有证据', '业务行动'])
    experiment = pd.DataFrame([
        ('入组与随机化', '从同一目标合资格流量入组，以可靠用户标识随机分配至 Response 或 S 策略臂，持续保持分组。'),
        ('资格分配', '在各策略臂内部按冻结分数排序，采用相同 20% 覆盖率、观察窗口和投放执行约束；同分规则预先确定。'),
        ('主要结果', '比较两策略臂全部随机入组用户的固定窗口转化率，按原始策略分组分析，包含未获得资格及获得资格但未曝光者。'),
        ('业务判据', '事先约定最小有业务价值的提升、样本量和观察期；用转化价值与实际新增成本核算净收益，决定是否替换。'),
        ('执行与护栏', '核验 SRM、跨臂污染、资格执行率及曝光率；监测花费、频次、延迟与用户负反馈，按计划一次完成主分析。'),
    ], columns=['环节', '方案'])
    return _section('decision', '08｜业务价值、交付与线上验证',
        p('本项目的业务产出是可执行的广告资格优先级，以及支撑策略选择和预算配置的离线证据。'
          '总体效应说明资格分配平均有正向转化价值，定向评价进一步说明相同名额应优先分给哪些记录。'
          '现有数据最明确支持模型定向优于随机分配；Response 与 Uplift 的额外差异需要继续验证。')
        + '<h3>8.1 从离线证据形成业务决策</h3>'
        + '<h4>表 8-1｜现有证据与决策建议</h4>' + _table(decisions)
        + p('正式交付基线沿用 Validation Set 冻结的 Response-XGBoost，保留 S-XGBoost 为挑战策略。'
            '这个选择保证原评价过程可追溯，并不意味着响应预测在业务上必然优于增量预测。'
            'S 在重复拟合中表现出的持续领先，应转化为下一步实验优先级；切换依据应来自新的目标业务流量。')
        + '<h3>8.2 模型与人群决策的交付</h3>'
        + p('交付包括已拟合模型、训练预处理参数、十二维特征规范、分数定义、同分规则和覆盖率。'
            '对于尚无转化标签的新样本，先使用处理前特征计算分数，再在约定批次内选取 Top-K 分配资格；'
            '上线后收集转化与实际曝光，用于评价和后续更新。模型推理不需要新样本的 Y。')
        + p('画像提供可审查的人群特征参照：高分及稳定入选记录更集中于 f8、f3 较低、f9 较高的区域。'
            '业务可据此检查输入分布、名单结构和覆盖率是否偏离离线基准。'
            '匿名字段缺少业务语义，尚不能直接形成“某年龄或某兴趣用户更有效”的营销结论；'
            '若迁移到实际场景，应先建立合法、稳定的字段映射，再验证画像与收益能否复现。')
        + '<h3>8.3 同覆盖率的策略 A/B Test</h3>'
        + '<h4>表 8-2｜Response 与 S 的线上比较方案</h4>' + _table(experiment)
        + p('此处的 ITT 是“被分配到哪种策略”的意向处理效应，比较单位为全部随机入组用户。'
            '它与第 04 章“是否获得广告资格”的 ITT 具有不同的处理定义。'
            '只比较两种策略各自入选用户的转化率，会把选人差异与策略收益混在一起，不能替代策略随机实验。')
        + p('两种规则共同选中的用户获得相同资格决策，切换收益主要来自名单差异部分。'
            '离线高重叠意味着预期策略差可能很小，因此样本量应围绕业务认可的策略效应差规划，'
            '不能直接套用资格分配相对不分配的总体 ITT。若同时需要估计绝对增量，可增加不授予资格的留出臂；'
            '若要检验定向价值，可增加同覆盖率 Random 策略臂。新增实验臂需结合流量与预算取舍。')
        + '<h3>8.4 预算试点与价值边界</h3>'
        + p('覆盖率与模型替换是两个决策。先在相同覆盖率下识别策略差异，可避免同时改变模型和预算而难以归因；'
            '随后可固定策略，在新的流量中比较 10% 与 20% 覆盖率，评估新增名额的转化回报与成本。'
            '前 10% 的高收益集中度是提出较低预算试点的依据，不代表 10% 已是最优覆盖率。')
        + p('按正式 Response 点估计，10%–20% 每千新增资格名额约增加 0.474 次转化。'
            '若单次增量转化价值为 V，则该档每千名额的估计毛增量价值为 0.474V；'
            '应与这批名额实际增加的媒体、服务及其他可归属成本比较。'
            '资格名额与广告花费并非一一对应，公开样本也缺少金额数据，因此当前结果支持资源优先级，不能直接证明真实 ROI 或利润最优。')
        + p('后续上线前需重新验证目标人群、特征采集时点、随机化与执行机制。'
            '第 07 章验证的是同一公开 Test Set 上的重训敏感性，对跨时间、跨业务和新流量的效果不能直接外推。'
            '本项目尚未实施线上实验；实际方案应由目标流量的基准率、业务阈值和成本重新确定。'))


def record_publication(report_path, out, experiment):
    """Attach the saved follow-up inputs to the published report provenance."""
    from .experiment_protocol import sha256_file
    report_path = Path(report_path)
    path = report_path.parent / 'report_provenance.json'
    record = json.loads(path.read_text()) if path.exists() else {}
    sources = ['protocol.json', 'selections.csv', 'test_comparison.csv',
               'validation_comparison.csv', 'coverage_sensitivity.csv', 'selection_overlap.csv',
               'profile_effect_analysis/analysis_record.json',
               'profile_effect_analysis/effect_bins.csv', 'profile_effect_analysis/feature_profiles.csv',
               'profile_effect_analysis/flow_overlap.csv', 'profile_effect_analysis/profile_quantile_summary.csv',
               'profile_effect_analysis/formal_feature_ranking.csv', 'profile_effect_analysis/formal_profile_display_bins.csv',
               'profile_effect_analysis/flow_stable_feature_ranking.csv', 'profile_effect_analysis/flow_stable_display_bins.csv']
    sources += [f'profile_effect_analysis/figures/{name}.png' for name in
                ['effect_deciles', 'formal_response_feature_histogram', 'formal_s_feature_histogram', 'flow_stable_feature_histogram']]
    record.update(report_sha256=sha256_file(report_path),
                  generator_sha256=sha256_file(Path(__file__).with_name('reporting.py')),
                  generator_dependencies={name: sha256_file(Path(__file__).with_name(name)) for name in
                      ['report_followup.py', 'plot_profile_histograms.py']},
                  supplemental_run=experiment.name,
                  supplemental_inputs={name: sha256_file(experiment / name) for name in sources},
                  report_revision={'chapters': ['06', '07', '08'],
                      'effect_display': 'Ten disjoint score strata, point-estimate lines only; no quintile results in report.',
                      'formal_selection': 'response_xgboost', 'new_training_required': False,
                      'source_records': 'Persisted model bundles, scores, memberships, summaries and figure inputs.'})
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')

# 生成扩展分析 Notebook 的代码与说明单元，不在生成过程中训练模型。

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks'
SETUP = '''from pathlib import Path
import sys, json
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'criteo_uplift').is_dir())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from criteo_uplift.runtime import configure_runtime
configure_runtime()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from criteo_uplift.pipeline_io import result_directory
FIG = ROOT/'artifacts/extension_figures'
FIG.mkdir(exist_ok=True)
'''


#-----------------------------------------------------------------------------------------------------------------------
# 函数：md —— 创建一个 Markdown 说明单元格。
# 输入：
#     s：单元格文本。
# 输出：
#     单元格：包含输入说明文本的 Notebook 单元格对象。
# 说明：创建一个 Markdown 说明单元格。
#-----------------------------------------------------------------------------------------------------------------------
def md(s):
    return nbf.v4.new_markdown_cell(s)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：code —— 创建一个 Python 代码单元格。
# 输入：
#     s：单元格文本。
# 输出：
#     单元格：包含输入代码的 Notebook 单元格对象。
# 说明：创建一个 Python 代码单元格。
#-----------------------------------------------------------------------------------------------------------------------
def code(s):
    return nbf.v4.new_code_cell(s)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：save —— 组合并保存 Notebook。
# 输入：
#     name：需要构造、读取或保存的对象名称。
#     cells：按顺序排列的 Notebook 单元格列表。
# 输出：
#     返回值：None；将单元格写入指定 Notebook 文件。
# 说明：组合并保存 Notebook。
#-----------------------------------------------------------------------------------------------------------------------
def save(name, cells):
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            'kernelspec': {
                'display_name': 'Python (causalml)',
                'language': 'python',
                'name': 'python3',
            }
        },
    )
    nbf.validate(nb)
    nbf.write(nb, OUT / name)


save(
    '05_metrics_worked_example.ipynb',
    [
        md(r'''# 05 · 五个指标：同一个 8 人例子，公式到代码

详细推导见 `因果推断理论/7.Metrics.md`。本页所有数值都是教学例子，不用于估计真实业务效果。

- **lift curve**：$L=R_1/N_1-R_0/N_0$，单位为选中人群转化率差。
- **uplift/gain curve**：$G=qL$，单位为每名待选人的累计增量。
- **AUUC**：$\int G(q)dq$，这里不做 perfect normalization。
- **Qini curve**：$Q=(R_1-R_0N_1/N_0)/n$。
- **Qini score**：$\int Q(q)dq-Q(1)/2$。

相对 lift $L/L(1)$、HT gain 与传统 gain 均另外标注，不能混成同一指标。'''),
        code(SETUP),
        code(
            '''from criteo_uplift.evaluation import uplift_curve, ranking_metrics
from fractions import Fraction as F
t = np.array([1,0,1,0,1,0,1,0])
y = np.array([1,0,1,0,0,1,1,0])
scores = np.arange(8,0,-1)/10
curve = uplift_curve(y,t,scores,e=.5)
metrics = ranking_metrics(y,t,scores,e=.5)
display(pd.DataFrame({'rank':np.arange(1,9),'score':scores,'treatment':t,'conversion':y}))
display(curve[['n_selected','fraction','n_treatment','n_control','responses_treatment','responses_control','cumulative_lift','relative_lift','gain_plugin','qini_traditional','gain_ht']])'''
        ),
        md(r'''## 精确分数复算：独立于浮点实现

累计曲线的原点取 0；第一人没有对照，传统 gain/Qini 初始点不可估，积分用原点连到 $q=1/4$。下方使用 Python `Fraction` 逐段计算，不依赖浮点四舍五入。

传统 AUUC 参照线是 $q/2$，Qini 参照线是 $q/4$。相同排序、不同纵轴，面积自然不同。'''),
        code(
            '''q = [F(0),F(1,4),F(3,8),F(1,2),F(5,8),F(3,4),F(7,8),F(1)]
g = [F(0),F(1,4),F(3,8),F(1,2),F(5,12),F(1,4),F(35,96),F(1,2)]
qini = [F(0),F(1,8),F(1,4),F(1,4),F(1,4),F(1,8),F(5,24),F(1,4)]
#-----------------------------------------------------------------------------------------------------------------------
# 函数：trapezoids —— 计算相邻曲线点之间的各段梯形面积。
# 输入：
#     x：按顺序排列的横坐标。
#     y：与横坐标对应的曲线值。
# 输出：
#     面积列表：每两个相邻点对应的一段梯形面积。
# 说明：保留各段面积，便于逐项核对手算示例。
#-----------------------------------------------------------------------------------------------------------------------
def trapezoids(x,y):
    return [(x[i]-x[i-1])*(y[i]+y[i-1])/2 for i in range(1,len(x))]
g_areas, q_areas = trapezoids(q,g), trapezoids(q,qini)
display(pd.DataFrame({'q_left':list(map(str,q[:-1])),'q_right':list(map(str,q[1:])),
                      'gain_trapezoid':list(map(str,g_areas)),'qini_trapezoid':list(map(str,q_areas))}))
exact = {'auuc_plugin':sum(g_areas),'auuc_plugin_centered':sum(g_areas)-F(1,4),
         'qini_score_traditional':sum(q_areas)-F(1,8), 'auuc_ht':F(3,8),'qini_ht':F(1,8)}
assert exact['auuc_plugin'] == F(81,256)
assert exact['qini_score_traditional'] == F(19,384)
for name,value in exact.items(): np.testing.assert_allclose(metrics[name],float(value),rtol=0,atol=1e-14)
display(pd.DataFrame([{'metric':k,'exact':str(v),'decimal':float(v),'code':metrics[k]} for k,v in exact.items()]))'''
        ),
        code('''fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
axes[0,0].plot(curve.fraction,curve.cumulative_lift,'o-',label='L = rate difference')
axes[0,0].axhline(.5,ls='--',color='gray',label='Overall L(1)')
axes[0,0].set(title='Cumulative lift',ylabel='Incremental rate among selected')
axes[0,1].plot(curve.fraction,curve.relative_lift,'o-',color='#d47a20',label='L / L(1)')
axes[0,1].axhline(1,ls='--',color='gray',label='Relative reference = 1')
axes[0,1].set(title='Relative lift (additional view)',ylabel='Multiple of overall effect')
axes[1,0].plot(list(map(float,q)),list(map(float,g)),'o-',label='Plug-in gain')
axes[1,0].fill_between(list(map(float,q)),list(map(float,g)),alpha=.15)
axes[1,0].plot([0,1],[0,.5],'--',color='gray',label='Endpoint reference')
axes[1,0].set(title='Uplift curve: AUUC = 81/256',ylabel='Incremental responses / eligible person')
axes[1,1].plot(list(map(float,q)),list(map(float,qini)),'o-',color='#167f67',label='Traditional Qini')
axes[1,1].plot([0,1],[0,.25],'--',color='gray',label='Endpoint reference')
axes[1,1].fill_between(list(map(float,q)),list(map(float,qini)),[float(x/4) for x in q],alpha=.15,color='#167f67')
axes[1,1].set(title='Qini curve: centered score = 19/384',ylabel='Qini count / total sample')
for ax in axes.flat:
    ax.set_xlabel('Selected population fraction');ax.legend(fontsize=9);ax.grid(alpha=.2)
fig.savefig(FIG/'worked_metrics.png',dpi=160)
plt.show()'''),
        md(
            r'''## HT 为什么不同？

$z=(2,0,2,0,0,-2,2,0)$。固定 $e=1/2$ 时，首行也可以计算 HT；传统组间差需要两臂。两种曲线在全体处理比例吻合时终点相同，中途依然可能不同。

下面两条曲线估计同一总体策略目标，但有限样本权重不同。不要把一条曲线的面积和另一条的随机参照混用。'''
        ),
        code('''fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
ax.plot(curve.fraction,curve.gain_ht,'o-',label='HT gain: AUUC=3/8')
ax.plot(list(map(float,q)),list(map(float,g)),'s--',label='Plug-in gain: AUUC=81/256')
ax.plot([0,1],[0,.5],':',color='gray',label='Endpoint reference')
ax.set(xlabel='Selected population fraction',ylabel='Gain per eligible person',title='Same ranking, different finite-sample estimators')
ax.legend();ax.grid(alpha=.2)
fig.savefig(FIG/'ht_vs_plugin.png',dpi=160);plt.show()'''),
        md(
            '''## 真实结果在下一页

这里的完美转化差来自人为构造的 8 行数据，不能推算实际收益。真实实验中我们使用 20% HT 主指标，并报告配对 bootstrap、AUUC/Qini、模型训练稳定性与事件数。全部曲线积分使用完整前缀，只有绘图做降采样。'''
        ),
    ],
)

save(
    '06_robustness_sql_and_decisions.ipynb',
    [
        md(
            '''# 06 · 稳定性、SQL 核对与业务决策

本页读取已经运行完成的独立扩展产物，不重训、不根据测试结果更换模型。原验证集选择仍为 `propensity`，最佳 uplift 候选为 S-Logistic。

稳定性研究从 80 万开发行重新划分，排除旧测试集 20 万行；这不能抹去此前已看过测试结果的分析者知识，因此它是后续开发诊断，不能包装成全新独立确认。'''
        ),
        code(SETUP),
        code('''ST=result_directory('stability_v1', ROOT)
report=json.loads((ST/'report.json').read_text())
long=pd.read_csv(ST/'longmetrics.csv')
agg=pd.read_csv(ST/'aggregate.csv')
audit=pd.read_csv(ST/'split_audit.csv')
display({k:report[k] for k in ['experiments','fitted_models','development_n','old_test_n_excluded','all_old_test_row_overlaps_zero','all_group_split_feature_overlaps_zero']})
display(audit)
cols=['split_method','training_fraction','policy','incremental_per_100k_mean','incremental_per_100k_std','vs_propensity_per_100k_mean','train_control_events_min','train_control_events_max']
display(agg[cols])'''),
        md(
            '''## 训练规模敏感性

三个 seed × 25% / 50% / 100% 的嵌套训练子集；每个 seed 的验证集固定。误差棒为三个划分的样本标准差，**不是 95% CI**。组间共用数据，不能当作三个独立实验。

另做三个全规模特征组隔离划分：相同 float32 匿名特征向量不跨训练/验证，保留全部行；这不是用户 ID 分组，也不能证明实际用户无泄漏。组隔离改变验证分布，不能用两种切分的值高低宣称隔离提升了模型。'''
        ),
        code(
            '''fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for name in ['propensity','s_logistic','t_logistic','t_xgboost','x_xgboost']:
    a=agg[(agg.split_method=='row_stratified') & (agg.policy==name)].sort_values('training_fraction')
    axes[0].errorbar(a.training_fraction,a.incremental_per_100k_mean,yerr=a.incremental_per_100k_std,marker='o',capsize=3,label=name)
    axes[1].plot(a.training_fraction,a.vs_propensity_per_100k_mean,'o-',label=name)
axes[0].set(title='Development learning curves: mean +/- SD',ylabel='20% HT gain / 100k eligible')
axes[1].axhline(0,color='gray',ls='--')
axes[1].set(title='Paired point differences, averaged over seeds',ylabel='Gain difference vs response / 100k eligible')
for ax in axes:
    ax.set_xlabel('Fraction of development training partition');ax.set_xticks([.25,.5,1]);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(FIG/'stability_learning_curves.png',dpi=160);plt.show()
x=long[long.policy=='x_xgboost']
print('X-Learner 胜过 response 的开发配置数:',int((x.vs_propensity_per_100k>0).sum()),'/',len(x))
print('注意：25% 规模 T-XGBoost 和 response 指标相等；选择频率含固定候选顺序的 tie-break，不能当作获胜证据。')'''
        ),
        md(
            '''## 冻结预测的 v2 指标与配对区间

增加了传统 plug-in AUUC、HT AUUC、两者中心化面积和传统 Qini 区间。只有已有测试预测被重算，未重新训练/选择；这不是一次新独立测试。

相对于 response 的区间包含 0 就不能宣称显著优于它。这些是固定模型、固定分配概率、给定臂样本量下的逐项区间，没有纳入训练和选择不确定性，也不是多个指标的同时区间。'''
        ),
        code(
            '''MET=result_directory('metrics_v2', ROOT)
display(pd.read_csv(MET/'ranking.csv'))
intervals=pd.read_csv(MET/'ranking_intervals.csv')
display(intervals)
chosen=intervals[intervals.metric=='qini_score_traditional'].copy()
fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
# 用线段直接绘制百分位区间端点；区间不一定包含点估计。
for i,row in enumerate(chosen.itertuples()):
    ax.plot([row.vs_propensity_ci_low,row.vs_propensity_ci_high],[i,i],color='#2476a8',lw=3)
    ax.plot(row.vs_propensity,i,'o',color='#173849')
ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(len(chosen)),chosen.policy)
ax.set(title='Traditional Qini difference vs response: paired 95% intervals',xlabel='Unnormalized, population-fraction axes')
ax.grid(axis='x',alpha=.2);fig.savefig(FIG/'qini_paired_intervals.png',dpi=160);plt.show()'''
        ),
        md('''## SQL：派生实验表、主键连接、条件聚合与窗口函数

这是从公开 CSV 派生的 SQLite 表；`row_id` 是原始行号，不是真实用户 ID，也没有伪造事件时间。SQL 演示唯一键/外键约束、1:1 完整连接、两臂 CVR、十分位分桶与预算排名。平分优先级与 Python 相同，便于逐项核对 HT。

完整查询在 `artifacts/sql_audit_v1/queries.sql`。'''),
        code('''SQL=result_directory('sql_audit_v1', ROOT)
display(pd.read_csv(SQL/'join_integrity.csv'))
display(pd.read_csv(SQL/'cvr.csv'))
display(pd.read_csv(SQL/'policy_values.csv'))
display(pd.read_csv(SQL/'score_bins.csv').head(12))
print((SQL/'queries.sql').read_text())'''),
        md(
            r'''## 上线前决策：需要多少样本？在什么成本假设下才值得？

以下是**假设场景**，不是 Criteo 的真实收入、成本或线上 CVR。

两臂均覆盖各自入组人群的 20%，按全部入组人群比较策略 CVR。假设基线策略 $p_B=0.003$、要检出 $\delta=0.0003$，双侧 5%、80% 功效。样本量分母不是被触达的 20%；覆盖率也不能直接拿来除 MDE。

净增价值 = 估计增量转化 × 假设每次转化贡献毛利 − 接触人数 × 假设单次成本。等预算且单位成本相同的两策略比较中，成本项相消。'''
        ),
        code(
            '''from criteo_uplift.planning import plan_policy_ab, cost_value_scenario
plans=pd.DataFrame([plan_policy_ab(.003,d) for d in [.00015,.0003,.0006]])
display(plans)
RUN=result_directory('conversion_1000000_seed42', ROOT)
pd_policy=pd.read_csv(RUN/'test_policies.csv')
selection=json.loads((RUN/'selection.json').read_text())
row=pd_policy[(pd_policy.policy==selection['selected_policy']) & np.isclose(pd_policy.budget,.2)].iloc[0]
scenarios=[]
for value in [1.,5.,10.]:
    for cost in [.001,.01,.05]:
        scenarios.append(cost_value_scenario(float(row.incremental_per_100k),budget=.2,value_per_conversion=value,contact_cost=cost,
                                            ci_low=float(row.ci_low),ci_high=float(row.ci_high)))
display(pd.DataFrame(scenarios))
print('单次接触盈亏平衡成本 = 增量转化 × 假设价值 / 接触人数；负值表示即使零成本也无正收益。')'''
        ),
        md('''## 当前可辩护的结论

1. 公共基准上，冻结的 response 策略在 20% 预算下高于解析随机策略；这不是线上收入增长。
2. S/T/X 已实装，复杂 X 并未优于强 response 基线；多次开发切分也未观察到反转。报告负结果是项目的一部分。
3. 少数臂事件数有限，特征匿名且用户/时间信息缺失。当前学习曲线不能单独证明“扩样就会解决”，尚未运行 500 万/全量模型或线上实验。
4. 下一步若有业务机会，冻结策略、候选池、转化窗口、同预算口径和停止规则，按真实用户随机开展策略 A/B；观察全部入组用户的转化与成本护栏。

面试复盘见 `因果推断理论/8.项目复盘与面试.md`。'''),
    ],
)
print('Created notebooks 05 and 06')

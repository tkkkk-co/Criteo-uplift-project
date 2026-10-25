# 读取已保存的扩展评价结果，生成补充图表和网页报告。

from pathlib import Path
import base64
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from criteo_uplift.runtime import configure_runtime

configure_runtime()
import pandas as pd
from criteo_uplift.planning import plan_policy_ab, cost_value_scenario
from criteo_uplift.pipeline_io import safe_output, write_json


#-----------------------------------------------------------------------------------------------------------------------
# 函数：table —— 将结果表转换成 HTML 表格。
# 输入：
#     frame：本次处理的数据表。
# 输出：
#     HTML：可以嵌入报告的表格字符串。
# 说明：将结果表转换成 HTML 表格。
#-----------------------------------------------------------------------------------------------------------------------
def table(frame):
    return (
        '<div class="table">'
        + frame.to_html(index=False, border=0, float_format=lambda x: f'{x:.6g}')
        + '</div>'
    )


#-----------------------------------------------------------------------------------------------------------------------
# 函数：figure —— 构造扩展报告的图片片段。
# 输入：
#     name：需要构造、读取或保存的对象名称。
#     alt：图片的替代说明文字。
# 输出：
#     HTML：图片及其替代说明组成的网页片段。
# 说明：构造扩展报告的图片片段。
#-----------------------------------------------------------------------------------------------------------------------
def figure(name, alt):
    data = base64.b64encode(
        (ROOT / 'artifacts/extension_figures' / name).read_bytes()
    ).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'


#-----------------------------------------------------------------------------------------------------------------------
# 函数：build —— 生成扩展评价的图表和网页。
# 输入：
#     无。
# 输出：
#     返回值：None；保存扩展图表和 HTML 报告。
# 说明：生成扩展评价的图表和网页。
#-----------------------------------------------------------------------------------------------------------------------
def build():
    st = ROOT / 'artifacts/stability_v1'
    report = json.loads((st / 'report.json').read_text())
    agg = pd.read_csv(st / 'aggregate.csv')
    full = agg[agg.training_fraction == 1][
        [
            'split_method',
            'policy',
            'incremental_per_100k_mean',
            'incremental_per_100k_std',
            'vs_propensity_per_100k_mean',
        ]
    ]
    met = ROOT / 'artifacts/metrics_v2'
    intervals = pd.read_csv(met / 'ranking_intervals.csv')
    sql = ROOT / 'artifacts/sql_audit_v1'
    run = ROOT / 'artifacts/conversion_1000000_seed42'
    selection = json.loads((run / 'selection.json').read_text())
    policies = pd.read_csv(run / 'test_policies.csv')
    row = policies[
        (policies.policy == selection['selected_policy']) & (policies.budget == 0.2)
    ].iloc[0]
    plans = pd.DataFrame([plan_policy_ab(0.003, d) for d in [0.00015, 0.0003, 0.0006]])
    scenarios = pd.DataFrame(
        [
            cost_value_scenario(
                float(row.incremental_per_100k),
                budget=0.2,
                value_per_conversion=v,
                contact_cost=c,
                ci_low=float(row.ci_low),
                ci_high=float(row.ci_high),
            )
            for v in [1.0, 5.0, 10.0]
            for c in [0.001, 0.01, 0.05]
        ]
    )
    planning = safe_output(ROOT / 'artifacts/planning_v1')
    planning.mkdir(exist_ok=True)
    plans.to_csv(planning / 'hypothetical_ab_plans.csv', index=False)
    scenarios.to_csv(planning / 'hypothetical_cost_scenarios.csv', index=False)
    write_json(
        {
            'hypothetical': True,
            'effect_source': str(run / 'test_policies.csv'),
            'frozen_selected_policy': selection['selected_policy'],
            'online_baseline_assumed': 0.003,
            'absolute_mde_scenarios': [0.00015, 0.0003, 0.0006],
            'economic_assumptions': 'Contribution margin per conversion 1/5/10; contact cost .001/.01/.05. Arbitrary sensitivity scenarios, not observed currency values.',
            'reference': 'https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.power_proportions_2indep.html',
        },
        planning / 'assumptions.json',
    )
    parts = [
        '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Criteo 项目扩展验证</title>
<style>body{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;line-height:1.7;background:#f3f5f8;color:#203246;margin:0}main{max-width:1120px;margin:auto;padding:36px}h1{font-size:34px;line-height:1.3}h2{margin-top:40px;border-top:1px solid #d6e0e8;padding-top:22px}p,li{max-width:1000px}.lead{font-size:19px}.box{padding:18px 24px;border-left:5px solid #18816a;background:#e9f4ef}.table{overflow:auto;background:white;margin:16px 0}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px 12px;border-bottom:1px solid #e3e8ee;text-align:right;white-space:nowrap}th{background:#e6eef4}img{width:100%;height:auto}figure{margin:24px 0;background:white;padding:12px}figcaption{font-size:13px;color:#526577}code{background:#e7edf2;padding:2px 5px}footer{font-size:13px;color:#526577;padding-top:24px}</style><main>
<p>2026-10-11 · 公开广告实验数据 · 离线研究</p><h1>从排序指标到可复核的策略决策</h1>
<p class="lead">本轮补齐指标教学、配对不确定性、多次训练验证、SQL 对账与业务实验规划。主目标保持为 20% 覆盖预算下的增量转化。</p>
<div class="box">首轮验证选择仍为 <b>response propensity</b>。复杂 X-Learner 尚未提供超越这一强基线的可靠证据。所有结果均为公开样本离线估计，没有线上收入或实际投放收益声明。</div>
<h2>1. 五个指标：同一个例子，口径明确</h2>
<p>累计 lift 是选中人群内两组转化率差；uplift/gain 为覆盖率乘该差。AUUC 是 gain 的面积；传统 Qini 以实际处理人数为尺度，Qini score 再扣除其端点参照线面积。8 人例子得到 AUUC=81/256、传统 Qini score=19/384；HT AUUC=3/8。纵轴不同，有限样本面积不同。</p>'''
    ]
    parts.append(
        figure(
            'worked_metrics.png',
            '8 人手算：累计 lift、相对 lift、uplift 面积与传统 Qini 面积。原点连到首个有效点是积分约定。',
        )
    )
    parts.append(
        figure('ht_vs_plugin.png', '固定分配概率 HT 与传统比率代入估计的有限样本差异。')
    )
    parts.append(
        '<h2>2. 排除旧测试集后的开发稳定性</h2><p>80 万开发行，排除旧 test 的 20 万行；3 seeds × 3 嵌套训练规模，加 3 个特征组隔离划分，共 12 个实验、48 次模型拟合。全部旧 test 行号重叠为零，group 划分特征哈希重叠为零。组划分不是用户 ID 隔离。</p>'
    )
    parts.append(
        figure(
            'stability_learning_curves.png',
            '误差棒是 3 个种子的样本标准差，不是置信区间。共享样本的重复划分不能当作独立试验。',
        )
    )
    parts.append(table(full))
    parts.append(
        '<p>完整训练规模的行级开发划分中，response 平均 127.51、S 126.67、X 99.03 次增量/每 10 万待选人。全部 12 个配置中 X 的点估计低于 response。25% 规模 T-XGBoost 与 response 得分相同，选择频率的平分次序不代表优势。特征组隔离改变验证分布，不可用数值高低证明某种切分带来收益。</p>'
    )
    parts.append(
        '<h2>3. AUUC / Qini 的配对区间</h2><p>200 次臂内共享重采样，固定模型、e、tie priorities；每次重新计算曲线及其端点。原始测试预测只作辅助指标扩展，没有重新选型。区间不包含训练/选择/未知用户相关性，不是多指标同时置信带。</p>'
    )
    parts.append(
        figure(
            'qini_paired_intervals.png',
            '与 response 的传统 Qini 配对差区间。区间含零，不能宣称超越基线。',
        )
    )
    parts.append(
        table(intervals[intervals.metric.isin(['auuc_ht', 'qini_score_traditional'])])
    )
    parts.append(
        '<h2>4. SQL 与 Python 核对</h2><p>SQLite 派生表展示主外键、一对一完整连接、条件聚合 CVR、窗口分桶与 top-20% 排名。20 万冻结测试行、5 种预测策略全部通过；同一独立随机平分规则下 HT 最大差约 9.52e−13。没有伪造用户 ID 或事件日志。</p>'
    )
    parts.append(table(pd.read_csv(sql / 'policy_values.csv')))
    parts.append(
        '<h2>5. 进入业务前：样本量和成本场景</h2><p>下表假设基线策略的全部入组用户 CVR 为 0.003，双侧 alpha=0.05、power=0.8，两策略各覆盖 20%。MDE 为绝对概率差。这些是规划输入，不是从公开数据识别的真实线上指标。</p>'
    )
    parts.append(
        table(
            plans[
                [
                    'p0_assumed',
                    'absolute_mde',
                    'relative_mde',
                    'n_policy_B',
                    'n_policy_A',
                    'n_total',
                    'approximate_achieved_power',
                ]
            ]
        )
    )
    parts.append(
        '<p>下表套用冻结策略的离线效果，但每次转化贡献毛利与接触成本都是人为指定的敏感性场景。净值区间条件于这些经济参数及可迁移性；不得解释成真实 ROI。</p>'
    )
    parts.append(
        table(
            scenarios[
                [
                    'value_per_conversion_assumed',
                    'contact_cost_assumed',
                    'net_incremental_value',
                    'net_ci_low_conditional',
                    'net_ci_high_conditional',
                    'break_even_contact_cost',
                ]
            ]
        )
    )
    parts.append(
        '''<h2>6. 交付与下一步</h2><p>学习顺序：<code>7.Metrics.md</code> → notebook 05 → notebook 06 → <code>8.项目复盘与面试.md</code>。正式线上验证需要真实用户随机单元、候选池、转化窗、成本和护栏日志；目前尚未执行线上 A/B。</p>
<footer>来源：工作区真实 CSV/JSON 与已执行 notebook。数据永久只读；20 份原笔记已备份。历史 Qini v1 与新 v2 的初始前缀积分规则分别披露，主 HT 指标不变。当前未运行 500 万/全量模型，也未将多次开发切分包装为新独立测试。</footer></main></html>'''
    )
    output = safe_output(ROOT / 'artifacts/extension_report.html')
    output.write_text('\n'.join(parts))
    print(output)


if __name__ == '__main__':
    build()

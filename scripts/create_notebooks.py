"""生成按学习顺序组织的 notebook；指标手算与扩展验证另见 create_extension_notebooks.py。"""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks'
SETUP = '''from pathlib import Path
import sys, json, os
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'criteo_uplift').is_dir())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from criteo_uplift.runtime import configure_runtime
configure_runtime()
import numpy as np
import pandas as pd
from IPython.display import display, Image, HTML
from criteo_uplift.results import result_directory
RUN = result_directory('conversion_1000000_seed42', ROOT)
print('读取结果目录:', RUN.relative_to(ROOT))
manifest = json.loads((RUN / 'manifest.json').read_text())
print('分析对象:', manifest['primary_outcome'], '| 随机样本:', f"{manifest['sample_n']:,}")
print('seed:', manifest['seed'], '| 主预算:', manifest['selection']['primary_budget'])
'''


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def save(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python (causalml)',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {'name': 'python', 'version': '3.11'},
    }
    nbf.write(nb, OUT / name)


save(
    '01_data_understanding.ipynb',
    [
        md(
            '''# 01 · 项目目标与数据理解

业务问题：在可触达用户比例有限、单次转化价值和成本相同的假设下，哪种排序策略能带来更多**增量转化**？主结果是 `conversion`，`visit` 用于辅助可学习性检查。主预算已确认 **20%**，其他预算为探索分析。

本轮完成全量质量审计 + 100 万行开发实验。学习顺序：01 数据 → 02 实验诊断与 ATE → 03 模型 → 04 策略。所有结果写入 `artifacts/`；`data/` 永远只读。学习 Markdown 已按新的编辑授权补充。

**运行方式**：使用现有 `causalml` Python 环境。以下单元读取本轮真实结果，不会自动重训。完整重跑入口见复现说明。没有本地原始数据时，自动读取 `reports/published/` 中的历史聚合快照；快照展示不等于重新拟合。'''
        ),
        code(SETUP),
        md(
            '''## 字段与可识别对象

- `f0`–`f11`：12 个匿名处理前特征，是唯一模型输入。
- `treatment`：处理分配 T；并不等于实际广告曝光。
- `conversion` / `visit`：两个结果指标 Y。
- `exposure`：实际曝光情况，仅做审计，不能加入 X 或据此筛选。

公开数据经过特殊子采样，无法还原原始线上增量；项目结论针对公开样本的处理分配效果。匿名字段没有业务语义，不能编造“年轻人/高消费人群”等解释。'''
        ),
        code(
            "audit = json.loads((RUN/'audit.json').read_text())\ndisplay(pd.DataFrame([{k:v for k,v in audit.items() if k not in ['feature_stats','groups']}]).T.rename(columns={0:'value'}))\ndisplay(pd.read_csv(RUN/'ate_full.csv'))"
        ),
        md(
            '''## 全量扫描与等概率抽样

用分块读取扫描 CSV 的全部 13,979,592 行；检查有限数值、二元标签、对照组曝光和 `conversion ⇒ visit`。给每行独立随机优先值，保留最小的 1,000,000 个，从全文件等概率无放回抽样。不会取文件头部充当随机样本，不会人为重配类别比例；有限随机样本比例允许自然波动。

审计使用原始 float64；模型样本使用 float32 降低内存。全字段重复使用 64-bit 指纹统计（有极小碰撞可能）；没有用户 ID，不能据此自动删除或认定同一用户。'''
        ),
        code('''from criteo_uplift.data import load_sample, FEATURES, LABELS
cache_path = ROOT/'artifacts/cache/sample_1000000_seed42.npz'
if cache_path.exists() and os.environ.get('CRITEO_RESULTS_MODE') != 'published':
    sample = load_sample(cache_path)
    display(sample.head())
    display(sample.groupby('treatment')[['conversion','visit','exposure']].agg(['count','sum','mean']))
    display(sample[FEATURES].describe().T)
else:
    sample = None
    print('公开快照模式：展示历史聚合统计；仓库不包含逐行数据。')
    display(pd.read_csv(RUN/'sample_group_summary.csv'))
    display(pd.read_csv(RUN/'sample_feature_summary.csv'))'''),
        code('''import matplotlib.pyplot as plt
if sample is None:
    display(Image(filename=str(RUN/'figures/sample_features.png')))
else:
    fig, axes = plt.subplots(3,4,figsize=(13,8))
    for feature, ax in zip(FEATURES,axes.flat):
        ax.hist(sample[feature],bins=40,color='#2476a8',log=True)
        ax.set_title(feature); ax.set_ylabel('Count (log)')
    fig.suptitle('Feature distributions · uniform 1M sample')
    fig.tight_layout(); plt.show()'''),
        md(
            '''## 数据划分与防止泄漏

采用 60% train / 20% validation / 20% test，按 `T × conversion` 分层以保持稀疏结果事件数。各集合的原始 `row_id` 独立保存。预处理只在训练集拟合；验证集选择策略；测试集用于固定方案的最终评估。

相同匿名特征可能属于不同用户，因此本轮采用行级划分，额外披露与训练集的特征重叠比例；后续特征组隔离敏感性验证已完成，见 notebook 06；本页保留首轮实验的结果。'''
        ),
        code(
            "display(pd.read_csv(RUN/'splits.csv'))\ndisplay(json.loads((RUN/'feature_overlap.json').read_text()))\ndel sample"
        ),
        md(
            '''## 可复现入口

在项目根目录运行：
```bash
python run_pipeline.py --output artifacts/conversion_repeat
```
指定新目录可避免混合不同实验。`--sample-size 5000000` 或 `--sample-size 0` 分别为 500 万 / 全量建模，需评估内存；本轮未运行它们。OpenMP 位于项目 `.runtime/openmp`，运行时自动加载，不改写原有 Conda 环境。

数据来源：[Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)。版本、参数、源文件 SHA256 见下一格。'''
        ),
        code(
            "display({'source': manifest['source'], 'versions': manifest['versions'], 'features': manifest['features']})"
        ),
    ],
)

save(
    '02_sanity_and_ate.ipynb',
    [
        md(
            '''# 02 · 实验可信度诊断与 ATE

本节回答：公开样本是否存在明显数据异常、X 能否预测 T、X 是否有结果预测信息、T/C 的平均结果差异是多少？可观测诊断不能证明所有因果识别假设成立。'''
        ),
        code(SETUP),
        md(r'''## ATE：差均值与不确定性

$$\hat\tau=\bar Y_1-\bar Y_0,\quad SE=\sqrt{s_1^2/n_1+s_0^2/n_0}.$$

95% CI 用非合并方差；双侧零效应检验使用零假设下的合并比例。绝对差以概率表示，乘 100 是百分点 pp，乘 100,000 是每十万人增量。相对提升为绝对差除以对照率。

全量 ATE 是描述性分析，不能直接解释成真实线上广告收益。'''),
        code(
            "ate = pd.read_csv(RUN/'ate_full.csv')\ndisplay(ate)\ndisplay(Image(filename=str(RUN/'figures/ate_full.png')))"
        ),
        md('''## SRM、协变量平衡与 C2ST

SRM 必须有外部给定的预设分流概率；不能拿观察到的 85% 当零假设再说通过。当前无设计参数，明确跳过。

SMD 衡量单变量均值差；KS 描述单变量分布差；C2ST 用 Logistic + 小型树模型检查联合分布的可区分性。仅取训练集中的 20,000 个随机样本，固定训练/留出划分，每次在各划分内置换 T 并完整重拟合。99 次置换，p 值最小粒度 0.01；两模型同时看，给出 Holm 校正。

不显著表示“未发现足够证据”，不能表述为证明随机化。'''),
        code(
            "display(json.loads((RUN/'srm.json').read_text()))\ndisplay(pd.read_csv(RUN/'balance.csv'))\ndisplay(pd.read_csv(RUN/'c2st.csv'))\ndisplay(Image(filename=str(RUN/'figures/balance.png')))"
        ),
        code(
            "null = pd.read_csv(RUN/'c2st_null.csv')\nobserved = pd.read_csv(RUN/'c2st.csv').set_index('model')\nimport matplotlib.pyplot as plt\nfig, axes=plt.subplots(1,2,figsize=(11,3.4))\nfor (name, group), ax in zip(null.groupby('model'), axes):\n    ax.hist(group.roc_auc,bins=16,color='#2476a8',alpha=.75)\n    ax.axvline(observed.loc[name,'roc_auc'],color='#e78b35',label='Observed AUC')\n    ax.set_title(name);ax.set_xlabel('Held-out assignment ROC AUC');ax.legend()\nplt.tight_layout();plt.show()"
        ),
        md('''## 可学习性

Dummy 使用训练集平均结果率；Logistic / HistGradientBoosting 只在训练集拟合，在 validation 比较 LogLoss、average precision、Brier、ROC-AUC。分别查看整个样本和两个 treatment 臂，并分别检查 visit / conversion。

Outcome 可预测不等于 CATE 可预测；这些分类指标不能替代 AUUC、Qini 或策略增量。'''),
        code("learning = pd.read_csv(RUN/'learnability.csv')\ndisplay(learning)"),
        md('''## 可独立复用的函数

```python
from criteo_uplift.diagnostics import c2st, covariate_balance, learnability, srm_summary
from criteo_uplift.evaluation import ate_summary
# c2st(train, FEATURES, sample_size=20_000, permutations=99, seed=42)
# ate_summary(frame['conversion'], frame['treatment'])
```

全量聚合量来源于流式扫描，不需要把 3 GB CSV 整体读进内存。'''),
    ],
)

save(
    '03_uplift_models.ipynb',
    [
        md(
            '''# 03 · CATE 模型与独立验证

候选：S-Logistic、T-Logistic、T-XGBoost、X-XGBoost；另与 random 和 response-propensity 排序比较。选择标准事先固定为 **20% 预算下 validation 的 HT 增量转化**，AUUC / Qini 为辅助。所有模型只用同一 train 拟合，未用测试集调参。'''
        ),
        code(SETUP),
        md(
            r'''## 模型对应的估计量

S-Learner：一个模型预测 $\mu(x,t)$，计算 $\mu(x,1)-\mu(x,0)$。Logistic baseline 只含 T 主效应，不含 T×X 交互；它在 log-odds 尺度上受限。

T-Learner：分别拟合两臂 $\hat\mu_0,\hat\mu_1$，相减得到 CATE。

X-Learner：先拟合 outcome 模型，用跨臂模型插补反事实：
$$D_0=\hat\mu_1(X_0)-Y_0,\quad D_1=Y_1-\hat\mu_0(X_1).$$
再拟合 $\hat\tau_0,\hat\tau_1$，组合为
$$\hat\tau(x)=e\hat\tau_0(x)+(1-e)\hat\tau_1(x).$$
此处 $e$ 为训练集 treatment 比例。未用类别权重、SMOTE 或未经校正的重采样概率计算差值。

“propensity”策略在本项目指 $P(Y=1|X,T=1)$ 的 response 排序；与 $P(T=1|X)$ 的 treatment propensity 概念不同。'''
        ),
        code(
            "display(pd.DataFrame({name: info['parameters'] for name,info in manifest['models'].items()}).T)\ndisplay(pd.read_csv(RUN/'validation_policies.csv')[['policy','budget','incremental_per_100k']])\ndisplay(json.loads((RUN/'selection.json').read_text()))"
        ),
        md(r'''## AUUC 与 Qini 的明确口径

HT 伪结果：$z_i=T_iY_i/e-(1-T_i)Y_i/(1-e)$。

按预测分数降序的前 k 行：$G(k/n)=\sum_{i\in top(k)}z_i/n$。`auuc_ht` 是 G 的面积；`qini_ht = auuc_ht - G(1)/2` 为中心化 HT 面积。

传统 Qini：$Q(k/n)=[R_1(k)-R_0(k)N_1(k)/N_0(k)]/n$。
`qini_score_traditional` 是 $\int_0^1[Q(f)-fQ(1)]df$；未按完美排序面积再归一化，不可直接与其他库的 normalized Qini 比较。本页读取历史 v1 结果（Qini 积分保留原点与双臂前缀）。当前 v2 保留仅对照前缀的零节点；细节和重新计算的结果见 notebook 05/06。分数相同使用固定独立随机优先值打破平局。

所有面积用完整排序计算，图表才降采样。'''),
        code(
            "display(pd.read_csv(RUN/'validation_ranking.csv'))\ndisplay(pd.read_csv(RUN/'test_ranking.csv'))\ndisplay(Image(filename=str(RUN/'figures/ranking.png')))"
        ),
        md('''## CATE 校准诊断

分箱比较预测平均 CATE 与实测组间差，CI 是描述性逐箱区间，不是同时置信带。没有个体真实反事实标签，不能报告真实 PEHE/ITE MSE。X-Learner 的最终 CATE 不必等于其第一阶段两模型概率差。

全局均值接近 ATE 也不能证明个体排序正确。测试集诊断结果不会用于重新校准当前模型。'''),
        code(
            "chosen_uplift=manifest['selection']['selected_uplift_model']\nprint('验证集选出的 uplift 模型（可能与最终政策不同）:',chosen_uplift)\ndisplay(pd.read_csv(RUN/f'calibration_{chosen_uplift}.csv'))\ndisplay(Image(filename=str(RUN/'figures/calibration.png')))"
        ),
        md('''## 模型复用

```python
from criteo_uplift.models import make_model
model = make_model('x_xgboost', random_state=42, n_estimators=150)
model.fit(train[FEATURES], train.treatment, train.conversion)
tau = model.predict(new_users[FEATURES])
```

已训练模型保存在当前结果目录 `.joblib` 文件中，仅加载本项目可信文件。给新特征文件打分并按冻结策略选出 20% 名单：

```bash
python score_policy.py --run artifacts/conversion_1000000_seed42 --input new_users.csv --output artifacts/scored_users.csv --budget 0.2
```

输入至少含 f0–f11，输出 source_row_id / score / selected。这里仅生成离线名单；输入文件保持只读，输出不可写入 data/。
参考：[Künzel et al., Meta-learners (2019)](https://arxiv.org/abs/1706.03461)。'''),
    ],
)

save(
    '04_policy_evaluation.ipynb',
    [
        md(
            '''# 04 · 投放策略离线评估

主预算是覆盖 20% 用户；同时展示 2%、5%、10%、50%、100%。这里是公开样本上的离线策略模拟，没有实际广告投放动作，也不将触达比例等同于真实广告金额。'''
        ),
        code(SETUP),
        md(
            '''## 固定策略，再看测试集

验证集已冻结最终政策。即使另一个模型在测试集点估计更高，也不能回头改选它并宣称独立验证成功。

“每十万人增量”以所有待选用户为分母；20% 预算代表其中 20,000 人被选中。random 是解析期望，不额外引入一次随机名单的噪声。'''
        ),
        code(
            "selection=json.loads((RUN/'selection.json').read_text())\nprint('冻结的最终政策:',selection['selected_policy'])\npolicies=pd.read_csv(RUN/'test_policies.csv')\nmain=policies[np.isclose(policies.budget,selection['primary_budget'])].copy()\ndisplay(main[['policy','n_selected','incremental_per_100k','ci_low','ci_high','vs_random_per_100k','vs_random_ci_low','vs_random_ci_high','vs_propensity_per_100k','vs_propensity_ci_low','vs_propensity_ci_high']])"
        ),
        md(
            '''## 成对 bootstrap

200 次按 treatment 臂分层重抽样，同一次抽样用于所有策略；冻结已学分数，每次重新选择 top-k。报告 percentile 95% CI，策略差值直接使用成对差值分布，不能靠两条独立 CI 是否重叠判断优劣。

当前 CI 条件于训练模型和估计的 e，不包含重新训练、模型选择、e 估计误差或未知用户依赖。多个辅助预算比较是探索性的，未作多重比较校正。'''
        ),
        code(
            "display(Image(filename=str(RUN/'figures/policy.png')))\ndisplay(policies[['policy','budget','incremental_per_100k','ci_low','ci_high']])"
        ),
        md(
            '''## 本轮结论如何读取

1. 以冻结政策在主预算的增量及成对差异 CI 为核心。
2. X-Learner 不保证比转化倾向策略好。若差异 CI 覆盖 0，应写“证据不足”，不要写“没有差异”或强行得出 uplift 胜出。
3. conversion 稀疏、control 更少，复杂模型可能不稳定；100 万开发样本的结果不等同全量建模最终结论。
4. 公布的预算表要求恰好选择 k 人；若业务改为“最多 k 人”，对 CATE 策略可增加只选预测正效应用户的规则，并在验证集确定，不能看测试结果临时加规则。
5. 下一阶段建议：多随机种子与特征分组敏感性分析已完成，见 notebook 06。扩样可作为后续预定检验；真实上线需独立 A/B 和真实成本数据。500 万/全量训练与线上实验尚未执行。'''
        ),
        code(
            "row=main[main.policy==selection['selected_policy']].iloc[0]\nprint(f\"冻结政策 {row.policy}：每十万待选用户增量 {row.incremental_per_100k:.1f}，95% CI [{row.ci_low:.1f}, {row.ci_high:.1f}]\")\nprint(f\"相对 random 差值 {row.vs_random_per_100k:.1f}，成对95% CI [{row.vs_random_ci_low:.1f}, {row.vs_random_ci_high:.1f}]\")\nprint('完整报告:', (RUN/'report.html').relative_to(ROOT))"
        ),
        md(
            '''## 学习笔记与版本

理论笔记中的标准误、MDE、C2ST 和版本比例已修正，见相应章节。本页展示历史冻结实验；新的 v2 指标和开发稳定性见 notebook 05/06。'''
        ),
    ],
)
print('Generated 4 notebooks in', OUT)

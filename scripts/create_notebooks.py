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

统一扫描入口为 `diagnostics.py` 顶部的 `scan_raw_data()`，返回字典中的 `sample` 是建模样本、`audit` 是全量审计，另有平衡检查和 C2ST 样本等结果。`audit_enabled` 独立控制全量审计，`diagnostics_enabled` 独立控制全量 SRM/SMD/KS；`sample_size`、`per_arm` 只控制建模和 C2ST 抽样。审计使用原始 float64；模型样本使用 float32 降低内存。全字段重复使用 64-bit 指纹统计（有极小碰撞可能）；没有用户 ID，不能据此自动删除或认定同一用户。'''
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
        md('# 02 · 实验可信度诊断与 ATE\n\n本节回答：\n- 公开样本是否存在明显数据异常；\n- X 能否预测 T；\n- X 是否有结果预测信息；\n- T/C 的ATE是多少？\n注：可观测诊断不能证明所有因果识别假设成立。'),
        code(SETUP),
        md('## ATE：差均值与不确定性\n\n$$\\hat\\tau=\\bar Y_1-\\bar Y_0,\\quad SE=\\sqrt{s_1^2/n_1+s_0^2/n_0}.$$\n\n- 95% CI 用非合并方差；\n- 双侧零效应检验使用零假设下的合并比例。\n- 绝对差以概率表示，乘 100 是百分点 pp，乘 100,000 是每十万人增量。相对提升为绝对差除以对照率。\n\n全量 ATE 是描述性分析，不能直接解释成真实线上广告收益。'),
        code('# 读取并展示已保存的全量 ATE 结果；本单元只调整展示，不重新计算统计量。\n\nate_df = pd.read_csv(RUN / \'ate_full.csv\')\nate = ate_df  # 保留现有表格使用的变量名\n\nate_display = ate.set_index(\'outcome\').T\ndisplay(\n    ate_display.style\n    .format(\'{:.6f}\')\n    .format(\n        \'{:.0f}\',\n        subset=pd.IndexSlice[[\'n\', \'n_control\', \'n_treatment\'], :]\n    )\n    .format(\n        \'{:.3e}\',\n        subset=pd.IndexSlice[[\'p_value\'], :]\n    )\n)\n\n# 从当前表格现场绘图；原始列保持比例单位，绘图副本转换为百分点（pp）。\nimport matplotlib.pyplot as plt\nfrom matplotlib.ticker import MaxNLocator\n\nate_plot = (\n    ate_df.set_index(\'outcome\')\n    .loc[[\'conversion\', \'visit\'], [\'ate\', \'ci_low\', \'ci_high\']]\n    .mul(100) #转换成pp值\n)\n\nwith plt.rc_context( # 使用with:设置临时的图片样式参数\n    {\n        \'font.family\': \'DejaVu Sans\',\n        \'font.size\': 11,\n        \'axes.spines.top\': False,\n        \'axes.spines.right\': False,\n        \'axes.spines.left\': False,\n        \'axes.edgecolor\': \'#C8D0D8\',\n        \'axes.labelcolor\': \'#344454\',\n        \'text.color\': \'#23313F\',\n        \'figure.facecolor\': \'white\',\n        \'axes.facecolor\': \'white\',\n    }\n):\n    fig, axes = plt.subplots(\n        2,\n        1,\n        figsize=(11, 5.8),\n        sharex=False,\n        constrained_layout=True,\n    )\n    fig.suptitle(\n        \'Absolute treatment effects in the full public sample\\n\'\n        \'95% confidence intervals | Separate x-axis scales | Dashed line: zero effect\',\n        fontsize=13,\n    )\n\n    for ax, (outcome, row), color in zip(\n        axes,\n        ate_plot.iterrows(),\n        [\'#2166AC\', \'#087F8C\'],\n    ):\n        ax.axvline(\n            0,\n            color=\'#536170\',\n            linestyle=\'--\',\n            linewidth=1.3,\n            zorder=2,\n        )\n        ax.errorbar(\n            x=[row[\'ate\']],\n            y=[0],\n            xerr=[\n                [row[\'ate\'] - row[\'ci_low\']],\n                [row[\'ci_high\'] - row[\'ate\']],\n            ],\n            fmt=\'o\',\n            color=color,\n            markersize=7,\n            elinewidth=2.5,\n            capsize=7,\n            capthick=2,\n            zorder=3,\n        )\n\n        # 每个子图独立缩放，但始终包含零点、点估计和完整 CI；保持线性坐标。\n        left = min(0.0, row[\'ci_low\'], row[\'ate\'])\n        right = max(0.0, row[\'ci_high\'], row[\'ate\'])\n        span = right - left\n        padding = 0.08 * span if span > 0 else 0.01\n        ax.set_xlim(\n            left - padding,\n            right + padding,\n        )\n\n        ax.set_ylim(-0.45, 0.85)\n        ax.set_yticks(\n            [0],\n            [outcome.capitalize()],\n        )\n        ax.tick_params(axis=\'y\', length=0, pad=12)\n        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))\n        \n        ax.set_xlabel(\'Absolute treatment effect (percentage points, pp)\')\n        ax.set_axisbelow(True)\n        ax.grid(axis=\'x\', color=\'#E6EBEF\', linewidth=0.8)\n\n        # 标注使用绘图区相对坐标，放在点和 CI 上方，避免遮挡。\n        ax.text(\n            0.98,\n            0.94,\n            f"ATE = {row[\'ate\']:+.4f} pp\\n"\n            f"95% CI: [{row[\'ci_low\']:.4f}, {row[\'ci_high\']:.4f}] pp",\n            transform=ax.transAxes,\n            ha=\'right\',\n            va=\'top\',\n            fontsize=12,\n            fontweight=\'bold\',\n            color=color,\n            linespacing=1.5,\n        )\n\n    plt.show()\n'),
        md('## 1. SRM：全量公开数据与 85% 参考比例的一致性\n\n全部诊断统一从原来的 `RUN` 目录读取。SRM原本就使用全量计数，继续使用Pearson卡方检验。\n\n85% 的独立依据是[数据集作者论文 Table 1 和 Section 3](https://arxiv.org/html/2111.10106#S3)。它是公开数据的参考Treatment占比；检验只检查发布计数与参考比例的一致性，不能据此宣布原始线上实验通过SRM或证明随机化。'),
        code("srm = json.loads(\n    (RUN / 'srm.json').read_text()\n)\ndisplay(\n    pd.DataFrame([srm])[\n        ['scope', 'n_treatment', 'n_control', 'observed_ratio',\n         'expected_ratio', 'chi2', 'pvalue', 'status']\n    ].style.format(\n        {'observed_ratio': '{:.10f}', 'expected_ratio': '{:.2f}',\n         'chi2': '{:.6e}', 'pvalue': '{:.6f}'}\n    )\n)\nprint(srm['interpretation'])\nprint('85%独立参考来源：', srm['reference_source'])"),
        md('## 2. SMD / KS：全量公开数据的单变量平衡\n\n直接读取原始 CSV 为 float64，分块验证全部 16 列，逐特征比较 Treatment 与 Control；没有使用 float32 建模缓存或删除记录。SMD 仍为两组均值差除以 `sqrt((s1² + s0²) / 2)`，方差使用 `ddof=1`；KS 仍为双侧两样本统计量。\n\n原来的 `balance.csv` 和 `figures/balance.png` 已原位更新为全量结果，表和图均标记分析范围。`|SMD| < 0.1` 只是常用经验参考，不是通用硬性通过标准。KS p 值未经多重校正，使用渐近近似；大量样本、重复取值和未知实体依赖都影响其解释，应重点观察差异大小。数值为 0 的 p 值可能是浮点下溢，不是数学上恰好为零。单变量平衡不代表联合分布一致。'),
        code('balance_public = pd.read_csv(\n    RUN / \'balance.csv\'\n)\nassert balance_public[\'scope\'].eq(\'full_published_dataset\').all()\nassert balance_public[\'n_treatment\'].eq(srm[\'n_treatment\']).all()\nassert balance_public[\'n_control\'].eq(srm[\'n_control\']).all()\ndisplay(\n    balance_public.drop(columns=\'source_sha256\').style.format(\n        {\'mean_treatment\': \'{:.6f}\', \'mean_control\': \'{:.6f}\',\n         \'smd\': \'{:+.6f}\', \'abs_smd\': \'{:.6f}\',\n         \'ks_statistic\': \'{:.6f}\', \'ks_pvalue\': \'{:.3e}\'}\n    )\n)\nmax_smd = balance_public.loc[balance_public[\'abs_smd\'].idxmax()]\nmax_ks = balance_public.loc[balance_public[\'ks_statistic\'].idxmax()]\nprint(\n    f"全量最大 |SMD|：{max_smd[\'feature\']} = {max_smd[\'abs_smd\']:.6f}；"\n    f"最大 KS：{max_ks[\'feature\']} = {max_ks[\'ks_statistic\']:.6f}"\n)\ndisplay(\n    Image(\n        filename=str(\n            RUN / \'figures/balance.png\'\n        )\n    )\n)\n'),
        md('## 3. C2ST：全量来源、两组等量抽样的联合可区分性诊断\n\n从全量原始数据分别对 T=0、T=1 做等概率无放回抽样，每组 10,000 行，共 20,000 行。这代表各组条件特征分布，**不保留总体 85:15 的处理比例**，也不是用全量样本训练分类器。\n\n复用现有 Logistic Regression 与小型 HistGradientBoosting，内部按 T 分层做 70/30 训练/留出划分；此划分与正式模型的 Train/Validation/Test 无关。标准化仅在诊断训练部分拟合。评价为留出 ROC AUC，以 0.5 为无区分能力基准；不根据结果翻转 AUC 方向。\n\n每次在两个内部集合各自置换 T，并完整重新拟合两个分类器。p=(1+置换 AUC≥观测 AUC 的次数)/(B+1)，对两模型做 Holm 校正。实际 B、p 值分辨率、组别数和来源如下。不显著只表示当前样本与分类器未发现足够证据，不能证明联合分布相同或随机化成立；未知分块、多实验和实体依赖仍限制置换可交换性。'),
        code('c2st_public = pd.read_csv(\n    RUN / \'c2st.csv\'\n)\nc2st_details = json.loads(\n    (RUN / \'c2st_details.json\').read_text()\n)\nassert c2st_public[\'scope\'].eq(\'full_source_arm_balanced_sample\').all()\ndisplay(\n    c2st_public[\n        [\'scope\', \'model\', \'sample_n\', \'n_control\', \'n_treatment\',\n         \'roc_auc\', \'pvalue\', \'pvalue_holm\', \'permutations\', \'pvalue_resolution\']\n    ].style.format(\n        {\'roc_auc\': \'{:.6f}\', \'pvalue\': \'{:.3f}\',\n         \'pvalue_holm\': \'{:.3f}\', \'pvalue_resolution\': \'{:.3f}\'}\n    )\n)\ndisplay(\n    pd.DataFrame(\n        [{key: c2st_details[key] for key in\n          [\'scope\', \'train_n_control\', \'train_n_treatment\',\n           \'holdout_n_control\', \'holdout_n_treatment\', \'seed\', \'metric\', \'multiplicity\']}]\n    )\n)\nprint(\'抽样方式：\', c2st_details[\'sampling\'])\nprint(\'实际置换次数：\', c2st_details[\'permutations\'])\nprint(\'p 值分辨率：\', c2st_details[\'pvalue_resolution\'])\nprint(\'数据来源：\', c2st_details[\'source\'][\'sha256\'])\n# 直接从原来的 c2st_null.csv 现场绘制置换零分布。\nnull = pd.read_csv(\n    RUN / \'c2st_null.csv\'\n)\nimport matplotlib.pyplot as plt\nfig, axes = plt.subplots(\n    1, 2, figsize=(11, 4), constrained_layout=True,\n)\nfor ax, row in zip(axes, c2st_public.itertuples()):\n    ax.hist(\n        null.loc[null[\'model\'] == row.model, \'roc_auc\'],\n        bins=24, color=\'#2171ad\', alpha=.8,\n    )\n    ax.axvline(\n        row.roc_auc, color=\'#ba6b24\', label=f\'Observed {row.roc_auc:.4f}\',\n    )\n    ax.set_title(f\'{row.model} | Holm p = {row.pvalue_holm:.3f}\')\n    ax.set_xlabel(\'Diagnostic holdout ROC AUC\')\n    ax.set_ylabel(\'Permutation count\')\n    ax.legend(frameon=False)\nfig.suptitle(\n    f"Full-source balanced C2ST | {c2st_details[\'n_control\']:,} control + "\n    f"{c2st_details[\'n_treatment\']:,} treatment\\n"\n    f"{c2st_details[\'permutations\']} complete-refit permutations; internal 70/30 split"\n)\nplt.show()'),
        md('## 4. 综合结论与边界\n\n以下摘要由本次产物动态生成。统计显著性不等于业务重要性；诊断没有发现异常也不代表完成随机化证明。公开数据经过发布处理，匿名字段不能验证用户级隔离或特征的真实生成时间；原始实验的分块、跨实验结构和实体依赖无法由这些字段恢复。\n\n本次属于已查看数据后的描述性诊断，不是新的独立测试集验证；结果不会反馈到正式模型选择、特征删除或切分调整。'),
        code('srm_statement = (\n    \'未检测到公开计数偏离85%参考比例的证据\'\n    if srm[\'pvalue\'] >= 0.05\n    else \'公开计数与85%参考比例存在统计差异，需要调查发布机制\'\n)\nprint(f"全量 SRM 参考检查：{srm_statement}（p={srm[\'pvalue\']:.6f}）。")\nprint(\n    f"全量单变量差异：最大 |SMD|={max_smd[\'abs_smd\']:.6f} ({max_smd[\'feature\']})，"\n    f"最大 KS={max_ks[\'ks_statistic\']:.6f} ({max_ks[\'feature\']})；"\n    \'请结合量级解释，不以显著性或0.1作为随机化通过标准。\'\n)\nfor row in c2st_public.itertuples():\n    finding = (\n        \'当前设计下检测到联合可区分性的证据，需调查\'\n        if row.pvalue_holm < 0.05\n        else \'当前样本规模与分类器未检测到足够联合可区分性证据\'\n    )\n    print(f"全量来源 C2ST / {row.model}：{finding}（Holm p={row.pvalue_holm:.3f}）。")\nprint(\'不能据此证明随机化、联合分布完全一致、用户级隔离或真实投放收益。\')\n'),
        md('## 可学习性\n\nDummy 使用训练集平均结果率；Logistic / HistGradientBoosting 只在训练集拟合，在 validation 比较 LogLoss、average precision、Brier、ROC-AUC。分别查看整个样本和两个 treatment 臂，并分别检查 visit / conversion。\n\nOutcome 可预测不等于 CATE 可预测；这些分类指标不能替代 AUUC、Qini 或策略增量。'),
        code("learning = pd.read_csv(RUN/'learnability.csv')\ndisplay(learning)"),
        md('## 原位更新诊断的运行入口\n\n在仓库根目录的项目Python环境中执行：\n\n```bash\npython -B run_pipeline.py --diagnostics-only\n```\n\n该命令直接更新原来 `RUN` 中的 `srm.json`、`balance.csv`、`c2st.csv`、`c2st_null.csv`、`c2st_details.json`，以及 `figures/balance.png` 和 `report.html`。不重新抽样建模、不重新切分或训练正式模型，不改ATE及策略评价。\n\n默认C2ST从原始全量来源每组抽10,000行，seed=42，499次完整重拟合置换。需要调整诊断计算规模时使用现有 `--c2st-sample`、`--permutations` 参数；`--output` 可定位已有的实验目录。\n\n代码主线：`diagnostics.py` 顶部的 `scan_raw_data()` 是唯一的全量扫描入口，统一完成读取、审计、建模抽样、C2ST 抽样及全量 SMD/KS、SRM。函数返回字典，包含 `sample`、`audit`、`balance`、`c2st_sample`、`srm`、`metadata`、`source`。完整运行复用这些结果，由同文件的 `update_experiment_diagnostics()` 运行 C2ST 并保存文件；四项任务独立控制：`audit_enabled` 控制全量审计，`diagnostics_enabled` 控制全量 SRM/SMD/KS，`sample_size` 控制建模抽样，`per_arm` 控制 C2ST 抽样。仅更新诊断时，显式设置 `sample_size=None`、`audit_enabled=False`、`diagnostics_enabled=True`，并指定 `per_arm`。`data.py` 提供路径保护、文件指纹、格式转换、缓存读写和数据划分等辅助函数。`reporting.py` 负责更新图表。Notebook统一通过 `RUN` 读取结果。'),
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

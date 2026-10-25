#-----------------------------------------------------------------------------------------------------------------------
# 定义数据列名、项目路径常量和训练、验证、测试集划分。
# 路径保护、文件指纹、JSON 和样本缓存读写统一放在 pipeline_io.py。
# 全量读取、审计、抽样和诊断统一由 diagnostics.py 的 scan_raw_data() 负责。
# 导入本模块不会自动读取原始数据或启动分析。
#-----------------------------------------------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

FEATURES = [f'f{i}' for i in range(12)]
LABELS = ['treatment', 'conversion', 'visit', 'exposure']
COLUMNS = FEATURES + LABELS

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
DEFAULT_DATA = DATA_ROOT / 'raw/criteo-uplift-v2.1.csv'


#-----------------------------------------------------------------------------------------------------------------------
# 函数：split_sample() —— 将样本划分为训练集、验证集和测试集。
# 输入：
#     df：全量数据抽样的100万样本，16列
#     outcome：用于分层的结果标签，可选 conversion 或 visit，默认 conversion。
#     seed：随机种子，默认 42。
# 输出：字典，各字段如下。
#     train：约占总样本 60% 的训练数据表。
#     validation：约占总样本 20% 的验证数据表。
#     test：约占总样本 20% 的测试数据表。
# 说明：按分组与结果标签的组合分层划分，使三部分尽量保持原样本中的组合比例。
#       三部分使用互不重叠的数据行；模型选择使用验证集，测试集用于最终评价。
#-----------------------------------------------------------------------------------------------------------------------
def split_sample(df, outcome='conversion', seed=42):
    if outcome not in ['conversion', 'visit']:
        raise ValueError('unknown outcome')
    idx = np.arange(len(df))
    strata = df.treatment.to_numpy() * 2 + df[outcome].to_numpy()
    train, holdout = train_test_split(
        idx,
        test_size=0.4,
        stratify=strata,
        random_state=seed,
    )
    val, test = train_test_split(
        holdout,
        test_size=0.5,
        stratify=strata[holdout],
        random_state=seed + 1,
    )
    return {
        name: df.iloc[pos].reset_index(drop=True)
        for name, pos in [('train', train), ('validation', val), ('test', test)]
    }

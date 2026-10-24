# 环境、数据、复现与版本

## 运行层次

| 路径 | 需要原始数据吗 | 得到什么 |
|---|---|---|
| 读README/公开CSV/Notebook保存输出 | 不需要 | 历史结果和解释 |
| 执行Notebook默认模式 | 不需要 | 重新展示/绘制公开聚合快照，手算重新计算 |
| run_demo.py / tests | 不需要 | 合成数据与小例子的真实代码验证 |
| run_pipeline.py | 需要官方Criteo文件 | 新的抽样、拟合、选择和test评价 |
| run_stability.py / SQL/指标扩展 | 需要本地缓存/逐行预测/来源文件 | 基于指定真实run的补充计算 |
| score_policy.py | 需要本地训练模型及新特征 | 冻结策略的离线名单 |

公开仓库不包含原始数据、逐行预测、样本行号或joblib模型。报告快照无法重训或复算bootstrap，因为它只有聚合证据。

## 创建隔离环境

推荐Python3.11；以下命令从项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks]'
python -m unittest discover -s tests -v
python scripts/run_demo.py --output artifacts/demo
python scripts/execute_notebooks.py
```

Windows使用 `.venv\Scripts\Activate.ps1` 激活。`python` 必须指向所选环境，不要把系统Python与已安装依赖的Conda环境混用。已有本机环境可 `conda activate causalml` 后运行，但公开项目不要求使用这个特定环境名。

`-e` 为editable安装，允许Python找到当前仓库的代码包；修改.py后不必每次重新复制安装。Notebook还保留根目录搜索以兼容直接从仓库运行。`scripts/execute_notebooks.py` 使用启动该脚本的解释器启动内核，避免误用另一个全局Python。

`pyproject.toml` 中是兼容范围；`requirements-tested.txt` 是本机直接依赖版本快照。如要尽量接近历史环境，可在新环境中先安装该快照再editable安装项目。它未锁定全部间接依赖，也不保证跨机器完全逐位相同。

## OpenMP与平台

XGBoost包含编译组件，运行依赖平台动态库。当前作者macOS缺失OpenMP时使用了本地 `.runtime/openmp/lib/libomp.dylib`；该二进制不上传。新机器应使用自身环境提供的库。

macOS可按自己的包管理器配置，例如安装Homebrew `libomp`，或者使用Conda-forge的XGBoost/OpenMP。具体平台方式见 [XGBoost官方安装文档](https://xgboost.readthedocs.io/en/stable/install.html)。需要显式指定已安装可信库时：

```bash
export CRITEO_OPENMP_LIBRARY=/实际安装位置/libomp.dylib
```

不需要此库的环境不用设置。Linux CI采用pip wheel与host平台依赖；CI文件存在不代表远端已运行，本次上传前的验证在macOS/Python3.11执行。

## 官方数据与只读约束

在[Criteo官方下载页](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)选择修正版，遵守CC BY-NC-SA4.0。自行获取并将CSV放入 `data/raw/criteo-uplift-v2.1.csv`。作者原始data目录内容、大小和mtime已验证未变；代码的所有结果写入artifacts，并解析符号链接以阻止绕过data输出保护。

历史源文件SHA256：

```text
e4d7c710ca1f38e523309d0f8a0745d1b53e7392d51f20d1088b6cfeaef222ef
```

历史1M主实验：seed42，train/validation/test=60/20/20，4个模型，XGBoost150trees，主budget=.2，bootstrap200，C2ST20k与99次置换。新run输出目录不应覆盖旧结果。

```bash
python run_pipeline.py --output artifacts/conversion_repeat
```

首次运行由 `diagnostics.py:scan_raw_data()` 在同一次 CSV 分块扫描中完成审计、建模抽样、C2ST 两组抽样和全量 SMD/KS、SRM；`update_experiment_diagnostics()` 复用扫描结果运行 C2ST 并保存诊断文件。后续仅在源 SHA 与配置匹配时复用建模样本和审计缓存，仍扫描原始数据准备全量诊断，但不重复建模抽样。SHA256 核验仍需读取原始文件字节，不能把一次 CSV 扫描理解为只有一次磁盘读取。扫描函数中的 `audit_enabled` 和 `diagnostics_enabled` 独立控制全量审计及 SRM/SMD/KS；`sample_size` 和 `per_arm` 只控制建模及 C2ST 抽样。缓存复用时显式关闭完整审计，而不是由抽样参数隐式决定。`--sample-size 0`支持全量模式，但本次未运行，不推荐把它当作第一次安装检查。

## 复算后续研究

以下示例针对原作者本地首轮run：

```bash
python run_stability.py --output artifacts/stability_repeat
python scripts/run_metrics_extension.py --output artifacts/metrics_repeat
python scripts/run_sql_audit.py --output artifacts/sql_repeat
```

对新run使用 `--base-run` / `--source` 和对应 `--cache`，以各自 `--help` 为准。不要把只有公开汇总的reports目录当作含逐行输入的run。

传统Qini v1/v2在首个缺臂区间的积分约定不同，历史文件保留，v2独立记录。20%HT和HT面积不变；新pipeline自动记录v2。所有面积采用全排序，不是从501/1001个绘图点近似。

## 对外快照与完整性

`reports/published/snapshot_manifest.json` 记录来源文件SHA和公开文件SHA。文本中的本机绝对路径已替换为 `[PROJECT_ROOT]`，因此两份文件哈希可以不同；数值和选择未改变。

```bash
python scripts/check_publication.py
python scripts/build_release.py
```

`export_public_results.py` 是作者用完整本地实验导出聚合快照的工具，不是克隆者必须执行的安装步骤。它白名单导出，不收录row-ID文件、预测数组和模型。

原始data完整性脚本依赖作者本地初始快照和备份；公开克隆没有这些输入，应使用check_publication验证公开结果，不以缺少私人备份判定代码错误。


## 仅原位更新实验可信度诊断

```bash
python -B run_pipeline.py --diagnostics-only
```

直接更新已有主实验目录的 `srm.json`、`balance.csv`、`c2st.csv`、`c2st_null.csv`、`c2st_details.json`，以及原来的平衡图和HTML报告。SMD/KS直接用原始全量float64特征；C2ST从原始全量来源组内等概率无放回抽样，默认两组各10,000行、499次完整重拟合置换并作Holm校正。SRM继续用全量计数与论文给出的85%公开参考比例。诊断不证明随机化，显著性不等于业务意义。

该模式不重新切分或拟合正式Uplift模型，不改变ATE、预测或策略评价。Notebook 2统一通过现有 `RUN` 读取这些结果。后续诊断优化直接更新这些原文件，不新增诊断版本目录或独立入口。

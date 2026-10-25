# 核验建模缓存并调用唯一的全量扫描入口，为建模和诊断准备数据。

from pathlib import Path
import json

from .data import ROOT
from .pipeline_io import safe_output, fingerprint, load_sample, save_sample, write_json
from .diagnostics import scan_raw_data


#-----------------------------------------------------------------------------------------------------------------------
# 函数：prepare_sample —— 核验缓存并准备建模样本和全量诊断。
# 输入：
#     args：本次运行的配置对象，包含数据路径、规模、种子和输出设置。
#     root：项目根目录，用于定位缓存或结果。
#     progress：接收进度文字的函数。
# 输出：
#     sample：建模样本表。
#     audit：全量审计字典。
#     source：原始文件指纹。
#     scanned：同一次扫描得到的可复用诊断结果。
# 说明：来源与配置一致时复用建模缓存，否则重新抽样；全量诊断在同一次扫描中准备。
#-----------------------------------------------------------------------------------------------------------------------
def prepare_sample(args, *, root: Path = ROOT, progress=None):
    # args 保留 run_pipeline.py 的命令行配置，供样本准备与诊断步骤使用。
    # - args.data:data的绝对路径
    # - args.sample_size:抽样规模，0=全量
    # - args.seed：控制抽样和模型拟合的随机种子。
    # - args.outcome:选择conversion或visit
    # - args.primary_budget:主策略最多触达的人群比例，默认20%
    # - args.c2st_sample:C2ST 从原始全量来源两组等量抽取的诊断样本规模
    # - args.permutations:C2ST 诊断的置换次数
    # - args.bootstrap:测试集策略评价的 bootstrap 次数
    # - args.trees:XGBoost 模型的树数量
    # - args.output（输出目录）

    if progress is None:
        progress = lambda message: print(message, flush=True)

    if args.c2st_sample < 20 or args.c2st_sample % 20:
        raise ValueError('C2ST sample_size must be >=20 and divisible by 20')
    if args.permutations < 1:
        raise ValueError('permutations must be positive')

    # 1. 定位共享抽样缓存及其元数据；缓存与某次实验的 out 结果目录分开管理。
    base = safe_output(Path(root) / 'artifacts')
    base.mkdir(exist_ok=True)
    cache = base / f'cache/sample_{args.sample_size}_seed{args.seed}.npz' # 缓存抽好的样本 下次直接调用不用再抽了
    metadata = cache.with_suffix('.json') # 缓存说明文件

    # 2. 核对原始文件内容、抽样规模和种子后才复用缓存。
    # fingerprint 仍会读取文件计算 SHA256；复用缓存不代表完全不读取原始文件。
    current = fingerprint(args.data)
    reuse = False
    if cache.exists() and metadata.exists():
        info = json.loads(
            metadata.read_text()
        )
        reuse = (
            info['source']['sha256'] == current['sha256']
            and info['sample_size'] == args.sample_size
            and info['seed'] == args.seed
        )
    # 3. 兼容早期已生成的 100 万行、seed=42 缓存：用旧的来源指纹验证。
    # 这是本项目的历史缓存兼容分支，不是通用抽样步骤。
    initial = base / 'source_fingerprint.json'
    if (
        not metadata.exists()
        and cache.exists()
        and initial.exists()
        and args.sample_size == 1_000_000
        and args.seed == 42
    ):
        reuse = (
            json.loads(initial.read_text())['sha256']
            == current['sha256']
        )

    # 4. 只进行一次 CSV 分块扫描，同时准备全量平衡、SRM 和 C2ST 样本。
    # 有建模缓存时跳过建模抽样及完整审计；没有缓存时在同一次扫描中一并完成。
    progress('统一扫描：全量检查 + C2ST 抽样' + ('（复用建模缓存）' if reuse else ' + 建模抽样与审计'))
    scanned = scan_raw_data(
        args.data,
        sample_size=None if reuse else args.sample_size,
        seed=args.seed,
        per_arm=args.c2st_sample // 2,
        audit_enabled=not reuse,
        diagnostics_enabled=True,
        progress=progress,
        source=current,
    )
    if reuse:
        progress('复用已校验来源 SHA256 的随机样本')
        sample = load_sample(cache)
        audit = json.loads(
            (
                metadata
                if metadata.exists()
                else base / 'full_audit.json'
            ).read_text()
        )
        if 'audit' in audit:
            audit = audit['audit']
        for key in ('n_treatment', 'n_control'):
            if audit[key] != scanned['metadata'][key]:
                raise ValueError(f'建模缓存审计与当前全量扫描不一致：{key}')
    else:
        sample, audit = scanned['sample'], scanned['audit']
        save_sample(sample, cache)

    # 建模样本由 sample 单独交给划分步骤，诊断结果无需继续持有这份大表。
    scanned.pop('sample')
    scanned.pop('audit')

    # 5. 保存缓存来源，并返回样本、审计、文件指纹和可直接复用的诊断数据。
    write_json(
        {
            'source': current,
            'sample_size': args.sample_size,
            'seed': args.seed,
            'audit': audit,
        },
        metadata,
    )
    return sample, audit, current, scanned


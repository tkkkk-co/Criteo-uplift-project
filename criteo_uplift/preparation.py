
from pathlib import Path
import json

from .data import ROOT, fingerprint
from .pipeline_io import safe_output, load_sample, save_sample, write_json
from .diagnostics import SAMPLE_CACHE_VERSION, scan_raw_data

AUDIT_CACHE_VERSION = 1


#-----------------------------------------------------------------------------------------------------------------------
# 函数：prepare_sample —— 核验缓存并准备建模样本和全量诊断。
# 输入：
#     args：本次运行的配置对象
'''
    - args.data:data的绝对路径
    - args.sample_size:抽样规模，0=全量
    - args.seed：控制抽样和模型拟合的随机种子。
    - args.outcome:选择conversion或visit
    - args.primary_budget:主策略最多触达的人群比例，默认20%
    - args.c2st_sample:C2ST 从原始全量来源两组等量抽取的诊断样本规模
    - args.permutations:C2ST 诊断的置换次数
    - args.bootstrap:测试集策略评价的 bootstrap 次数
    - args.trees:XGBoost 模型的树数量
    - args.output（输出目录）
'''
#     root：项目根目录，用于定位缓存或结果。
#     progress：接收进度文字的函数。
# 输出：
#     sample：建模样本表。
#     audit：全量审计字典。
#     source：原始文件指纹。
#     scanned：同一次扫描得到的可复用诊断结果。
# 说明：抽样缓存与全量审计分别校验和复用；缺失任务在同一次扫描中完成，全量诊断照常准备。
#-----------------------------------------------------------------------------------------------------------------------
def prepare_sample(args, *, root: Path = ROOT, progress=None):

    if progress is None:
        progress = lambda message: print(message, flush=True)

    if args.c2st_sample < 20 or args.c2st_sample % 20:
        raise ValueError('C2ST sample_size must be >=20 and divisible by 20')
    if args.permutations < 1:
        raise ValueError('permutations must be positive')

    # 1. 分开定位抽样结果、抽样说明和按原始内容指纹共享的全量审计。
    base = safe_output(Path(root) / 'artifacts')
    base.mkdir(exist_ok=True)
    cache = base / f'cache/sample_{args.sample_size}_seed{args.seed}.npz'
    metadata = cache.with_suffix('.json')
    current = fingerprint(args.data)
    audit_cache = base / f'cache/full_audit_{current["sha256"]}.json'

    # 2. 抽样缓存依赖：原始数据、抽样规模、seed 和抽样规则版本。
    reuse_sample = False
    if cache.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        report_fields = {
            'scope', 'sample_cache_version', 'source', 'source_n', 'sample_size',
            'sample_n', 'seed', 'sampling', 'feature_representation',
            'n_treatment', 'n_control',
        }
        reuse_sample = (
            report_fields.issubset(info)
            and info.get('scope') == 'modeling_sample'
            and info.get('source', {}).get('sha256') == current['sha256']
            and info.get('sample_size') == args.sample_size
            and info.get('seed') == args.seed
            and info.get('sample_cache_version') == SAMPLE_CACHE_VERSION
        )

    # 3. 全量审计依赖：原始内容、审计规则版本。
    reuse_audit = False
    if audit_cache.exists():
        saved_audit = json.loads(audit_cache.read_text())
        reuse_audit = (
            saved_audit.get('source', {}).get('sha256') == current['sha256']
            and saved_audit.get('audit_cache_version') == AUDIT_CACHE_VERSION
            and isinstance(saved_audit.get('audit'), dict)
        )

    # 4. 调用scan_raw_data() 完成全量审计、抽样（建模抽样、C2ST抽样）、诊断（SRM SMD KS）
    progress(
        '统一扫描：全量检查 + C2ST 抽样；'
        + ('复用建模缓存' if reuse_sample else '重新建模抽样') + '；'
        + ('复用全量审计' if reuse_audit else '重新全量审计')
    )
    scanned = scan_raw_data(
        args.data,
        sample_size=None if reuse_sample else args.sample_size,
        seed=args.seed,
        per_arm=args.c2st_sample // 2,
        audit_enabled=not reuse_audit,
        diagnostics_enabled=True,
        progress=progress,
        source=current,
    )

    # 5. 全量审计：复用与重新生成各自集中处理。
    if reuse_audit:
        # 5.1 复用审计：读取已核验来源和版本的缓存，并与本次扫描人数核对。
        audit = saved_audit['audit']
        for audit_key, scan_key in (
            ('n_rows', 'n'), ('n_treatment', 'n_treatment'), ('n_control', 'n_control'),
        ):
            if audit[audit_key] != scanned['metadata'][scan_key]:
                raise ValueError(f'全量审计缓存与当前扫描不一致：{audit_key}')
    else:
        # 5.2 重新审计：取得本次扫描生成的完整审计，并保存独立审计缓存。
        audit = scanned['audit']
        write_json(
            {'source': current, 'audit_cache_version': AUDIT_CACHE_VERSION, 'audit': audit},
            audit_cache,
        )

    # 6. 建模样本：复用与重新抽样各自集中处理，报告由扫描函数生成。
    expected_n = audit['n_rows'] if args.sample_size == 0 else min(args.sample_size, audit['n_rows'])
    if reuse_sample:
        # 6.1 复用样本：读取样本及其已有说明，不重新计算抽样报告。
        sample = load_sample(cache)
        sample_report = info
        if len(sample) != expected_n or sample_report['sample_n'] != len(sample):
            raise ValueError('抽样缓存行数与配置及原始数据规模不一致')
    else:
        # 6.2 重新抽样：直接取得扫描函数生成的样本和说明，一起保存。
        sample = scanned['sample']
        sample_report = scanned['sample_report']
        if len(sample) != expected_n:
            raise ValueError('抽样结果行数与配置及原始数据规模不一致')
        save_sample(sample, cache)
        write_json(sample_report, metadata)

    scanned['sample_report'] = sample_report
    # 诊断结果无需继续持有建模大表或重复持有审计字典。
    scanned.pop('sample')
    scanned.pop('audit')
    return sample, audit, current, scanned

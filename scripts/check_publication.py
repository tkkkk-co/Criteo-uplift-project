# 核验公开聚合快照的指纹，并检查可发布文件范围。

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (
    'criteo_uplift',
    'scripts',
    'tests',
    'notebooks',
    'docs',
    'reports',
    '.github',
    '因果推断理论',
    'ML',
)
TOP_FILES = (
    'README.md',
    'LICENSE',
    'NOTICE.md',
    '.gitignore',
    'pyproject.toml',
    'requirements-tested.txt',
    'run_pipeline.py',
    'score_policy.py',
)
SUFFIXES = {
    '.py',
    '.md',
    '.ipynb',
    '.csv',
    '.json',
    '.sql',
    '.html',
    '.png',
    '.yml',
    '.yaml',
    '.txt',
}


def publication_files(root=ROOT):
    #-------------
    # 函数名：publication_files
    # 功能：按允许发布的目录、顶层文件和扩展名枚举项目文件，并拒绝符号链接、越界文件和异常大文件。
    # 输入参数：
    #     - root：项目根目录。
    # 输出参数：
    #     - 排序后的可发布文件 pathlib.Path 列表。
    #-------------
    root = Path(root).resolve()
    candidates = [root / name for name in TOP_FILES]
    for directory in DIRECTORIES:
        candidates.extend(
            p
            for p in (root / directory).rglob('*')
            if p.is_file() and p.suffix in SUFFIXES
        )
    result = []
    for p in candidates:
        if not p.is_file():
            raise FileNotFoundError(p)
        rel = p.relative_to(root)
        if any(
            part in ('__pycache__', '.ipynb_checkpoints', '.DS_Store')
            for part in rel.parts
        ):
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError(
                f'发布文件不能为符号链接或位于项目外：{rel}'
            )
        if p.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f'发布文件大小超出预期：{rel}')
        result.append(p)
    return sorted(set(result))


def check(root=ROOT):
    #-------------
    # 函数名：check
    # 功能：校验公开聚合快照文件哈希、排除个体级/模型产物，并检查发布白名单范围。
    # 输入参数：
    #     - root：项目根目录。
    # 输出参数：
    #     - 发布核验摘要字典，包含快照文件数、发布文件数、总大小及数据范围标记。
    #-------------
    root = Path(root).resolve()
    public = root / 'reports/published'
    manifest = json.loads((public / 'snapshot_manifest.json').read_text())
    assert (
        manifest['raw_rows_exported'] is False
        and manifest['selection_changed'] is False
    )
    for record in manifest['files']:
        path = public / record['published']
        if not path.resolve().is_relative_to(public.resolve()):
            raise ValueError('快照路径超出公开目录')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record['published_sha256']:
            raise ValueError(f'已发布快照发生变化：{path}')
    for path in public.rglob('*'):
        if path.is_file() and (
            path.suffix in ('.npz', '.npy', '.joblib', '.pickle', '.pkl')
            or 'row_ids' in path.name
        ):
            raise ValueError(
                f'不得发布逐行数据或模型产物：{path}'
            )
    files = publication_files(root)
    assert not any(
        p.relative_to(root).parts[0] in ('data', 'artifacts', '.runtime', '.git')
        for p in files
    )
    size = sum(p.stat().st_size for p in files)
    result = {
        'snapshot_files_verified': len(manifest['files']),
        'publication_files': len(files),
        'publication_bytes': size,
        'raw_data_included': False,
        'model_arrays_included': False,
        'scope': 'allowlisted source, documentation, notebooks and aggregate evidence',
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    check()

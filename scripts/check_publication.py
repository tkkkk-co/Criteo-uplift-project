# 核验公开聚合快照的指纹，并检查可发布文件范围。

from pathlib import Path
import hashlib
import json
import sys

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
    'run_stability.py',
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：publication_files —— 枚举允许发布的项目文件。
# 输入：
#     root：项目根目录，用于定位缓存或结果。
# 输出：
#     文件列表：校验后可发布的文件路径。
# 说明：枚举允许发布的项目文件。
#-----------------------------------------------------------------------------------------------------------------------
def publication_files(root=ROOT):
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
                f'Symlink/out-of-project file excluded from publication: {rel}'
            )
        if p.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f'Unexpected large publication file: {rel}')
        result.append(p)
    return sorted(set(result))


#-----------------------------------------------------------------------------------------------------------------------
# 函数：check —— 核验公开快照和发布文件范围。
# 输入：
#     root：项目根目录，用于定位缓存或结果。
# 输出：
#     result：核验文件数、总大小及发布范围的概要字典。
# 说明：核验公开快照和发布文件范围。
#-----------------------------------------------------------------------------------------------------------------------
def check(root=ROOT):
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
            raise ValueError('Snapshot path escapes public directory')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record['published_sha256']:
            raise ValueError(f'Published snapshot changed: {path}')
    for path in public.rglob('*'):
        if path.is_file() and (
            path.suffix in ('.npz', '.npy', '.joblib', '.pickle', '.pkl')
            or 'row_ids' in path.name
        ):
            raise ValueError(
                f'Individual/model artifacts must not be published: {path}'
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

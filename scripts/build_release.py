# 按明确的发布文件列表生成项目 ZIP 包，不包含原始数据与模型缓存。

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import zipfile
from check_publication import check, publication_files
from criteo_uplift.pipeline_io import safe_output


#-----------------------------------------------------------------------------------------------------------------------
# 函数：build —— 按发布清单打包项目。
# 输入：
#     output：本次任务的新输出路径。
# 输出：
#     output：生成的 ZIP 文件路径。
# 说明：按发布清单打包项目。
#-----------------------------------------------------------------------------------------------------------------------
def build(output):
    output = safe_output(output)
    if output.exists():
        raise FileExistsError(output)
    check()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = publication_files()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path('criteo-uplift-project') / path.relative_to(ROOT))
    print(f'GitHub-ready ZIP: {output} ({output.stat().st_size:,} bytes)')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='按明确的发布文件列表生成项目 ZIP 包，不包含原始数据与模型缓存。')
    p.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'artifacts/releases/criteo-uplift-github-ready.zip',
    )
    build(p.parse_args().output)

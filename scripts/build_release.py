"""Build a small GitHub-ready source ZIP from an explicit publication list."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import zipfile
from check_publication import check, publication_files
from criteo_uplift.data import safe_output


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'artifacts/releases/criteo-uplift-github-ready.zip',
    )
    build(p.parse_args().output)

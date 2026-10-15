"""Verify data remains byte/mtime-identical; record authorized learning-note edits.

The user explicitly authorized all Markdown edits on 2026-10-11. Keep the
original immutable snapshot and historical verification; write a separate
report. Markdown backups preserve all twenty original files before editing.
"""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    before = json.loads((ROOT / 'artifacts/input_integrity_before.json').read_text())
    data_checks = []
    note_changes = []
    for record in before:
        path = ROOT / record['path']
        digest = sha(path)
        stat = path.stat()
        if record['path'].startswith('data/'):
            data_checks.append(
                {
                    'path': record['path'],
                    'sha256_unchanged': digest == record['sha256'],
                    'size_unchanged': stat.st_size == record['size'],
                    'mtime_unchanged': stat.st_mtime_ns == record['mtime_ns'],
                }
            )
        elif path.suffix == '.md':
            backup = ROOT / 'artifacts/note_backups_20261011' / record['path']
            note_changes.append(
                {
                    'path': record['path'],
                    'content_changed': digest != record['sha256'],
                    'original_backup_verified': backup.is_file()
                    and sha(backup) == record['sha256'],
                }
            )
    expected = {r['path'] for r in before if r['path'].startswith('data/')}
    actual = {
        str(p.relative_to(ROOT)) for p in (ROOT / 'data').rglob('*') if p.is_file()
    }
    report = {
        'data_all_unchanged': all(
            all(v for k, v in r.items() if k != 'path') for r in data_checks
        ),
        'data_file_set_unchanged': expected == actual,
        'data_checks': data_checks,
        'markdown_edits_authorized': True,
        'original_notes': note_changes,
        'all_original_note_backups_verified': all(
            r['original_backup_verified'] for r in note_changes
        ),
    }
    (ROOT / 'artifacts/input_integrity_authorized_notes.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    assert (
        report['data_all_unchanged']
        and report['data_file_set_unchanged']
        and report['all_original_note_backups_verified']
    ), report
    print(
        f"Data verified unchanged ({len(data_checks)} files, SHA256/size/mtime); {len(note_changes)} original notes safely backed up; Markdown edits authorized."
    )
    return report


if __name__ == '__main__':
    verify()

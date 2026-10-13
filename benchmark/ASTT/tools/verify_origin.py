"""Verify the complete, unmodified upstream snapshot against its SHA-256 manifest."""
import hashlib
import json
from pathlib import Path

ASTT_ROOT = Path(__file__).resolve().parents[1]


def verify_origin(root=ASTT_ROOT):
    manifest = json.loads((root / 'doc/ORIGIN_MANIFEST.json').read_text(encoding='utf-8'))
    origin = root / 'origin'
    actual = {p.relative_to(origin).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in origin.rglob('*') if p.is_file()}
    expected = manifest['files']
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    changed = sorted(name for name in expected.keys() & actual.keys()
                     if expected[name] != actual[name])
    if missing or extra or changed:
        raise ValueError(f'origin verification failed: missing={missing}, extra={extra}, changed={changed}')
    return manifest


if __name__ == '__main__':
    result = verify_origin()
    print(f'origin OK: {len(result["files"])} files; upstream commit {result["commit"]}')

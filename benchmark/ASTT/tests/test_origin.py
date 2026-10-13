"""Protect the upstream snapshot and ensure documentation tools cannot annotate it."""
import importlib.util
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.ASTT.tools.verify_origin import verify_origin


class OriginTests(unittest.TestCase):
    def test_snapshot_is_complete_and_unchanged(self):
        manifest = verify_origin()
        self.assertEqual(manifest['commit'], '1d05305a94fb47b33897c7aa085c5351bcbbc8f9')
        self.assertEqual(len(manifest['files']), 18)

    def test_annotation_tool_leaves_origin_untouched(self):
        spec = importlib.util.spec_from_file_location('file_index', PROJECT_ROOT / 'tools/update_file_index.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in verify_origin()['files']:
            path = PROJECT_ROOT / 'benchmark/ASTT/origin' / name
            before = path.read_bytes()
            module.annotate(path, 'This comment must never be written to origin')
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()

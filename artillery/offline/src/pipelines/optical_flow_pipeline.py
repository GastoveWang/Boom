"""Traditional optical-flow artillery-impact detection pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artillery.offline.src.pipelines.main import main as _run_pipeline


def main(argv=None):
    """Run optical-flow detection with this method's own CLI options."""
    return _run_pipeline(default_detector="motion", allow_detector_selection=False, argv=argv)


if __name__ == "__main__":
    main()

"""Compatible live entrypoint; implementation is split under boom_online/."""
import sys
from pathlib import Path

ONLINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boom.online.cli import main
from boom.online.pipeline import run
from boom.online.configuration import build_arg_parser, detector_config_from_args
from boom.online.positioning import StaticPose, static_pose_from_args, estimate_position
from boom.online.storage import EventStore, _format_optional
from boom.online.logging_setup import LOGGER_NAME, configure_logging
from boom.online.display import _draw_confirmed_box

if __name__ == "__main__":
    main()

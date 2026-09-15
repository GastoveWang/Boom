"""Single offline composition entrypoint for all detector-specific launchers."""
from artillery.common.defaults import DETECTOR_BACKEND
from .cli import build_arg_parser, validate_args
from ..runtime.runner import run


def main(default_detector=DETECTOR_BACKEND, *, allow_detector_selection=True, argv=None):
    args = build_arg_parser(default_detector, allow_detector_selection=allow_detector_selection).parse_args(argv)
    validate_args(args)
    return run(args)

"""
==============================================================================
Boom 煙霧辨識 - 光流運動學與膨脹比分析模組 (boom.detection.optical_flow)
==============================================================================

提供無人機飛行時之相機運動補償、影格時間差分與連通域徑向膨脹比分析：
- InstantSmokeDustDetector: 核心即時煙霧偵測器，實作 BaseSmokeDetector 介面
- DetectorConfig: 偵測器門檻與相機濾波參數設定
- run_video / run_batch: 離線分析執行輔助函式
==============================================================================
"""

from .types import (
    DetectorConfig,
    FrameBundle,
    HomographyResult,
    RadialFlowResult,
    BlobCandidate,
    EventTrack,
    ConfirmedEvent,
    OpticalCandidateEvent,
    ConfirmedTrackSnapshot,
    EventMemory,
    VideoRunSummary,
    DetectionArtifact,
)
from .detector import InstantSmokeDustDetector
from .runner import run_video, run_batch, build_arg_parser, config_from_args

__all__ = [
    "DetectorConfig",
    "FrameBundle",
    "HomographyResult",
    "RadialFlowResult",
    "BlobCandidate",
    "EventTrack",
    "ConfirmedEvent",
    "OpticalCandidateEvent",
    "ConfirmedTrackSnapshot",
    "EventMemory",
    "VideoRunSummary",
    "DetectionArtifact",
    "InstantSmokeDustDetector",
    "run_video",
    "run_batch",
    "build_arg_parser",
    "config_from_args",
]

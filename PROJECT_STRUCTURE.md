# Project Structure

Boom is a wilderness smoke detection and localization system for drones in flight. Its goal is to highlight smoke regions in real time and estimate the ground coordinates of suspected smoke sources.

The system is organized as a clean, extensible **Modular Monolith** under `src/boom/` with clear boundary separation between domain detection, localization, runtime orchestration, UI rendering, configuration, and tools.

---

## Directory Overview

```
Boom/
├── configs/
│   └── default.yaml             # Centralized YAML configuration
├── src/
│   └── boom/                    # Core Python package (pip install -e .)
│       ├── core/                # Pure domain entities and geodesy
│       │   ├── events.py        # ConfirmedEvent, LoggedEvent, GeoReference
│       │   └── coordinates.py   # WGS84 / TWD97 bidirectional conversions
│       ├── interfaces/          # Abstract contracts (Dependency Inversion)
│       │   ├── detector.py      # BaseSmokeDetector
│       │   ├── localizer.py     # BaseLocalizer
│       │   ├── matcher.py       # BaseFeatureMatcher
│       │   └── sensor.py        # BaseFrameSource
│       ├── config/              # YAML config loader & defaults
│       │   ├── loader.py        # load_config, get_config, ConfigFacade
│       │   └── defaults.py      # Structured defaults & MapMatchingConfig
│       ├── detection/           # Smoke detection algorithms
│       │   ├── pidnet/          # PIDNet-S semantic segmentation & tracking
│       │   ├── optical_flow/    # Ego-motion compensation & growth ratio
│       │   ├── fusion/          # Optical onset + PIDNet confirmation
│       │   └── factory.py       # create_detector unified builder
│       ├── localization/        # Drone pose tracking & geo projection
│       │   ├── localizer.py     # MapLocalizer facade (implements BaseLocalizer)
│       │   ├── geo_map.py       # GeoTIFF raster mosaic and coordinates
│       │   ├── nearby.py        # SuperPoint + LightGlue nearby tile search
│       │   ├── photo.py         # Drone EXIF/XMP pose parsing
│       │   └── visual_odometry.py # Visual odometry tracking
│       ├── ui/                  # Visualization and multi-panel composition
│       │   ├── composition.py   # Main video + map panel frame merger
│       │   ├── drawing.py       # Detection bounding boxes & text
│       │   ├── panels.py        # Status & coordinate HUD cards
│       │   └── theme.py         # Colors and typography
│       ├── runtime/             # Offline runtime lifecycle orchestration
│       │   ├── runner.py        # Main frame loop coordinator
│       │   ├── inputs.py        # Video/photo resolution & auto-pairing
│       │   ├── output.py        # Video encoding & coordinate log writer
│       │   ├── events.py        # Event confirmation lifetime & state
│       │   └── progress.py      # FPS and ETA terminal progress
│       ├── online/              # Drone real-time flight onboard system
│       │   ├── spinnaker_camera.py # FLIR/Teledyne SpinView driver
│       │   ├── pipeline.py      # Real-time multi-threaded capture loop
│       │   ├── storage.py       # Onboard event storage
│       │   └── runner.py        # Real-time runner
│       └── pipelines/           # Execution entry points
│           ├── main.py          # Unified offline pipeline coordinator
│           └── cli.py           # Command-line parser & validators
├── models/                      # Neural network weights
│   ├── pretrained/              # sam_sup_pidnet_s.pt, yolo26m.pt
│   └── checkpoints/             # Custom fine-tuned weights
├── tools/                       # Dataset processing and training utilities
│   ├── split_dataset.py         # 7:3 balanced dataset split (with GUI)
│   ├── train_yolo.py            # YOLO fine-tuning script
│   ├── video_to_frames.py       # Video frame extraction utility
│   └── README.md
├── benchmarks/                  # External baseline comparisons
│   ├── mod_ir/                  # MOD-IR paper implementation
│   └── README.md                # Evaluation protocols
├── tests/                       # Comprehensive pytest suite
├── run_offline.py               # Root offline execution launcher
└── run_realtime.py              # Root real-time onboard execution launcher
```

---

## Quick Execution Commands

### 1. Offline Video Inference
```powershell
# PIDNet semantic segmentation (default)
python run_offline.py pidnet --video data/video/1440-1080/DJI_0070_1440-1080.mp4

# Optical flow motion analysis
python run_offline.py motion --video data/video/1440-1080/DJI_0070_1440-1080.mp4

# Two-stage staged fusion
python run_offline.py fusion --video data/video/1440-1080/DJI_0070_1440-1080.mp4

# Interactive GUI display
python run_offline.py motion --video data/video/1440-1080/DJI_0070_1440-1080.mp4 --display
```

### 2. Drone Onboard Real-time Detection
```powershell
python run_realtime.py --exposure-us 5000 --display
```

### 3. Run Automated Tests
```powershell
python -m pytest tests
```

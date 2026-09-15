# Project Structure

Boom is a wilderness smoke detection and localization system for drones in flight. Its goal is to highlight smoke regions in real time and estimate the ground coordinates of suspected smoke sources.

Boom separates shared services from runtime orchestration and detector implementations.
See `artillery/README.md` for module responsibilities and execution commands.

## Current mapping

- `artillery/common/`
  - `defaults.py`: shared defaults, paths and existing tuning values
  - `detector_factory.py`: detector configuration and construction
  - `coordinates.py`: static camera projection and TWD97/WGS84 conversion
  - `events.py`: runtime-facing event and static georeference records

- `artillery/offline/src/pipelines/`
  - `pidnet_pipeline.py`: PIDNet-S algorithm entrypoint
  - `optical_flow_pipeline.py`: traditional optical-flow algorithm entrypoint
  - `optical_pidnet_pipeline.py`: optical onset followed by PIDNet confirmation
  - `main.py`, `cli.py`: common offline entrypoint and argument validation
  - `map_localization.py`: compatibility imports for the localization package
- `artillery/offline/src/runtime/`
  - `runner.py`: source → localization → detection → events → UI/output orchestration
  - `inputs.py`, `reference.py`: video input and legacy static-photo references
  - `events.py`: event localization and display lifetimes
  - `output.py`, `progress.py`: encoded video, coordinate logs and progress reporting
- `artillery/offline/src/ui/`
  - `drawing.py`, `panels.py`, `composition.py`: overlays, legacy panels and final frame composition
- `artillery/offline/src/localization/`
  - `regions.py`: GeoTIFF region bounds and GPS/estimated-view candidate selection
  - `photo.py`: GPS/XMP metadata, approximate camera intrinsics and image loading
  - `geo_map.py`: GeoTIFF mosaic, map cache and local geographic coordinates
  - `geometry.py`: ground-plane projection and validated camera pose estimation
  - `registration.py`: photo/map initialization, photo/video matching and map correction
  - `visual_odometry.py`: frame-to-frame feature tracking and pose history
  - `map_panel.py`: map UI, drone icon, trajectory and event markers
  - `localizer.py`: runner-facing session API, event projection and localization logging
- `artillery/offline/src/detectors/`
  - `pidnet_smoke_detector.py`: semantic smoke segmentation and new-smoke tracking
  - `optical_flow_smoke_detector.py`: frame differencing, motion compensation and optical-flow validation
  - `optical_pidnet_fusion_detector.py`: temporal/spatial event matching without merging detector internals
- `artillery/offline/src/pidnet_model/`
  - bundled PIDNet architecture and its MIT license
- `artillery/offline/model/`
  - detector checkpoints
- `artillery/online/`
  - Jetson/Spinnaker real-time runtime
  - `src/realtime_runner.py`: compatible launch entrypoint
  - `src/boom_online/`: separate CLI, configuration, pipeline, positioning, storage, logging and display
  - `src/spin_camera.py`, `src/spinnaker_camera.py`: camera acquisition adapters
- `wildfire-real-time-segmentation/`
  - optional upstream PIDNet research and training repository; not required at runtime
- `data/`
  - input media
  - `raw_video/`: raw input videos
  - `yolo_datasets/<dataset>/`: YOLO images, labels and data.yaml
  - `0603/`, `0604/`: local GPS-photo/video validation pairs
- `asset/`
  - `maps/`: local GeoTIFF region directories (data excluded from Git)
  - `pictures/`: drone UI icons
  - `ui/`: Taiwan overview geometry and attribution
- `output/offline/<video-name>_<method>/`
  - each pipeline marks its method (`pidnet` or `optical_flow`) on the result folder name

## Detector interface

Each detector should expose the same runtime-facing API:

```python
process_frame(frame_bgr, frame_idx) -> np.ndarray
consume_pending_confirmations() -> list
```

This keeps detection algorithms separate from video, UI and coordinate handling.

## Naming rules

- Python modules use lowercase `snake_case`.
- Pipeline entrypoints identify the algorithm: `pidnet_pipeline.py`,
  `optical_flow_pipeline.py`, and `optical_pidnet_pipeline.py`.
- Detector names identify the implementation rather than using generic names
  such as `smoke_dust_detector.py`.
- Pipeline entrypoints select/configure the detector and call a runtime runner.
- Reusable defaults, detector construction and static coordinates belong in `artillery/common/`.
- Runtime I/O, event presentation lifetimes and UI stay outside detector implementations.
- Each method-specific entrypoint fixes its detector; common runtime services are reused internally.

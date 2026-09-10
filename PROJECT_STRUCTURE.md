# Project Structure

Boom is organized first by runtime and then by implementation role.

## Current mapping

- `artillery/offline/src/pipelines/`
  - `pidnet_pipeline.py`: PIDNet-S algorithm entrypoint
  - `optical_flow_pipeline.py`: traditional optical-flow algorithm entrypoint
  - `optical_pidnet_pipeline.py`: optical onset followed by PIDNet confirmation
  - `shared_pipeline.py`: shared video I/O, UI, coordinates and event output
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
- `wildfire-real-time-segmentation/`
  - optional upstream PIDNet research and training repository; not required at runtime
- `data/`
  - input media
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
- Shared runtime logic belongs in `shared_pipeline.py`; it is not a third
  detection algorithm.

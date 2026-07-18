# Project Structure

This repository is now split by two dimensions:

- `methods/`: whose detection method it is
- `runtimes/`: where or how the method is executed

## Current mapping

- `methods/mine/`
  - Roger's smoke/dust detector entrypoint and exports
- `runtimes/offline/`
  - offline detector runner
  - impact-point pipeline runner
  - review and labeling helper tools
- `runtimes/online/`
  - reserved for on-device or live execution entrypoints
- `common/`
  - shared interfaces and future common utilities

## Recommended pattern for adding other methods

Add a new peer package under `methods/`:

```text
methods/
  mine/
  alice/
  bob/
```

Each method package should expose a detector with the same runtime-facing API:

```python
process_frame(frame_bgr, frame_idx) -> np.ndarray
consume_pending_confirmations() -> list
```

Then reuse the same runtime runner, for example:

```text
runtimes/offline/run_detector.py
runtimes/online/
```

## Compatibility note

Legacy source files are still kept under `mine/` so existing imports continue to work during the transition.

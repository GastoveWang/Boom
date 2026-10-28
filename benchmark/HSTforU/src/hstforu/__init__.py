"""HSTforU benchmark API; the official snapshot is preserved in origin/."""

UPSTREAM_COMMIT = "9548473306bbb89895b077275e8503de79653029"


def build_model(variant="ped2"):
    """Build the official four-history predictor without downloading weights.

    Inputs are a list of four [B,3,H,W] RGB tensors in [-1,1].
    Ped2 uses PVTv2-B1; Avenue uses PVTv2-B2. Both retain the
    official decoder, temporal module, parameter names and forward method.
    """
    if variant == "ped2":
        from .models.HSTforU_Ped2 import AnomalyDetection
    elif variant == "avenue":
        from .models.HSTforU_Avenue import AnomalyDetection
    else:
        raise ValueError("Supported package variants: ped2, avenue")
    return AnomalyDetection(config=None, logger=None, is_trained=False)


def load_checkpoint(model, path):
    """Load a trusted official checkpoint with exact parameter matching."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    return model.load_state_dict(state, strict=True)

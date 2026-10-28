"""Dependency diagnostics and a real, random-input forward check."""
import argparse
import importlib.metadata
import importlib.util
import json


def main():
    parser = argparse.ArgumentParser(description="HSTforU benchmark package")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check model dependencies without loading weights")
    forward = sub.add_parser("forward", help="Random-input forward check; not an accuracy test")
    forward.add_argument("--variant", choices=["ped2", "avenue"], default="ped2")
    forward.add_argument("--device", default="cpu")
    forward.add_argument("--size", type=int, default=256)
    forward.add_argument("--checkpoint", help="Optional trusted official checkpoint")
    args = parser.parse_args()
    if args.command == "doctor":
        versions = {}
        for name in ("torch", "torchvision", "numpy", "timm", "mmcv", "einops"):
            if importlib.util.find_spec(name) is None:
                versions[name] = None
            else:
                versions[name] = importlib.metadata.version(name)
        print(json.dumps(versions, indent=2))
        if any(value is None for value in versions.values()):
            raise SystemExit("Install the benchmark package dependencies; see README.md")
        if versions["mmcv"] != "1.7.2":
            raise SystemExit("This snapshot requires mmcv==1.7.2 (mmcv 2.x has no mmcv.runner)")
        return
    if args.size < 32 or args.size % 32:
        parser.error("--size must be at least 32 and divisible by 32")
    import torch
    from . import UPSTREAM_COMMIT, build_model, load_checkpoint
    torch.manual_seed(0)
    torch.set_num_threads(4)
    model = build_model(args.variant).to(args.device).eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    frames = [torch.rand(1, 3, args.size, args.size, device=args.device) * 2 - 1 for _ in range(4)]
    with torch.inference_mode():
        result = model(frames)
    if result.shape != frames[0].shape or not torch.isfinite(result).all():
        raise RuntimeError("Non-finite or unexpected output")
    print(json.dumps({"upstream_commit": UPSTREAM_COMMIT, "variant": args.variant,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "input_frames": 4, "output_shape": list(result.shape),
                      "finite": True, "checkpoint": args.checkpoint,
                      "validation": "forward only; no detection accuracy claim"}, indent=2))


if __name__ == "__main__":
    main()

"""Download the outdoor LoFTR weights to the project's ignored models directory."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile


def main(argv=None):
    import torch
    from st2.models.registration import ROOT, DEFAULT_WEIGHTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args(argv)
    destination = args.output.resolve()
    if destination.exists():
        parser.exit(0, f'Existing weights preserved: {destination}\n')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = ROOT / 'tmp'
    temporary_root.mkdir(exist_ok=True)
    # Kornia's documented outdoor checkpoint; no process-global torch cache changes.
    url = 'https://huggingface.co/kornia/loftr/resolve/main/loftr_outdoor.ckpt'
    created = False
    try:
        with tempfile.TemporaryDirectory(prefix='loftr_download_', dir=temporary_root) as directory:
            staging = Path(directory) / 'loftr_outdoor.ckpt'
            torch.hub.download_url_to_file(url, str(staging))
            from st2.models.network import ST2Config
            from st2.models.registration import LoFTRRegistration
            registration = LoFTRRegistration(ST2Config())
            registration.load_weights(staging)
            with destination.open('xb') as output, staging.open('rb') as source:
                created = True
                shutil.copyfileobj(source, output)
    except (OSError, RuntimeError, ValueError) as exc:
        if created:
            destination.unlink(missing_ok=True)
        parser.exit(1, f'LoFTR weights: {exc}\n')
    print(f'SHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}')
    print(f'LoFTR outdoor weights: {destination}')
    return 0

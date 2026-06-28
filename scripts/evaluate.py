"""
Evaluate a trained checkpoint on the test set.

    python scripts/evaluate.py --checkpoint results/S1_postfilter_fusion/best_model.pth

The checkpoint stores the resolved config, so dataloaders + model are rebuilt
from it. Outputs (slice_metrics.csv, summary.json, predictions/) are written
next to the checkpoint unless --output is given.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ffpa25d.utils import set_seed, resolve_device
from ffpa25d.data.dataset import create_dataloaders
from ffpa25d.models.ffpa_net_25d import FFPANet25D
from ffpa25d.engine.trainer import Evaluator


def run(checkpoint_path: str, output_dir: str = None):
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    config["training"]["device"] = resolve_device(config["training"].get("device", "cuda"))
    device = config["training"]["device"]
    set_seed(config.get("seed", 42))

    out_dir = Path(output_dir) if output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = create_dataloaders(config)
    model = FFPANet25D(config, base_channels=config["ffpa"].get("base_channels", 40)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', -1) + 1}")

    return Evaluator(model, config, out_dir, device).evaluate(test_loader)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a 2.5D FFPA-Net checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    run(args.checkpoint, args.output)
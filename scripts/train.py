"""
Train one variant of the 2.5D FFPA-Net.

    python scripts/train.py --config configs/variants/S1_postfilter_fusion.yaml
    python scripts/train.py --config configs/variants/S0_single_slice.yaml

Writes results/<experiment.name>/ : best_model.pth, training_log.csv,
slice_metrics.csv, summary.json, predictions/, config.yaml.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ffpa25d.utils import load_config, set_seed, resolve_device
from ffpa25d.data.dataset import create_dataloaders
from ffpa25d.models.ffpa_net_25d import FFPANet25D
from ffpa25d.engine.trainer import Trainer, Evaluator


def build_model(config, device):
    return FFPANet25D(config, base_channels=config["ffpa"].get("base_channels", 40)).to(device)


def run(config_path: str, output_base: str = "results", skip_training: bool = False):
    config = load_config(config_path)
    config["training"]["device"] = resolve_device(config["training"].get("device", "cuda"))
    device = config["training"]["device"]
    set_seed(config.get("seed", 42))

    name = config["experiment"]["name"]
    out_dir = Path(output_base)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir = out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(config)
    model = build_model(config, device)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    train_result = None
    if not skip_training:
        trainer = Trainer(model, config, out_dir, device)
        train_result = trainer.train(train_loader, val_loader)

    best = out_dir / "best_model.pth"
    if best.exists():
        ckpt = torch.load(best, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nLoaded best model from epoch {ckpt['epoch'] + 1}")

    summary = Evaluator(model, config, out_dir, device).evaluate(test_loader)
    if train_result:
        summary["training"] = train_result
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train one 2.5D FFPA-Net variant")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="results")
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()
    run(args.config, args.output, args.skip_training)
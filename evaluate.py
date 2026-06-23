
"""Evaluate a trained FFPA-Net baseline checkpoint on the test set."""
import argparse
from pathlib import Path

import torch
import yaml

from src.dataset import create_dataloaders
from src.model import FFPANet
from src.trainer import Trainer, set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate FFPA-Net baseline")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file (default: read from checkpoint)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for metrics and predictions",
    )
    args = parser.parse_args()

    kit_root = Path(__file__).parent
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = kit_root / checkpoint_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = kit_root / config_path
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint["config"]

    set_seed(config.get("seed", 42))

    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = kit_root / output_dir
    else:
        output_dir = checkpoint_path.parent / "eval"

    _, _, test_loader = create_dataloaders(config)
    model = FFPANet(config, base_channels=config.get("ffpa", {}).get("base_channels", 40))
    model.load_state_dict(checkpoint["model_state_dict"])

    trainer = Trainer(model, config, output_dir, device=device)
    save_predictions = config.get("evaluation", {}).get("save_predictions", True)
    summary = trainer.evaluate(test_loader, save_predictions=save_predictions)

    print("Evaluation summary:")
    print(f"  Dice:    {summary['dice']['mean']:.4f}")
    print(f"  Dice_FG: {summary['dice_fg']['mean']:.4f}")
    print(f"  HD95:    {summary['hd95']['mean']:.2f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

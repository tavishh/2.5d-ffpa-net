
"""Train FFPA-Net baseline model."""
import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.dataset import create_dataloaders
from src.model import FFPANet
from src.trainer import Trainer, set_seed


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train FFPA-Net baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: outputs/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run test-set evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for --eval-only",
    )
    args = parser.parse_args()

    kit_root = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = kit_root / config_path

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = kit_root / output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = kit_root / config.get("output", {}).get("dir", "outputs") / f"run_{timestamp}"

    device = config["training"]["device"] if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("FFPA-Net Baseline Kit")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(config)

    model = FFPANet(
        config,
        base_channels=config.get("ffpa", {}).get("base_channels", 40),
    )
    total_params, trainable_params = model.count_parameters()
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    trainer = Trainer(model, config, output_dir, device=device)

    if args.eval_only:
        checkpoint_path = Path(args.checkpoint or output_dir / "best_model.pth")
        if not checkpoint_path.is_absolute():
            checkpoint_path = kit_root / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        trainer.train(train_loader, val_loader)
        checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    save_predictions = config.get("evaluation", {}).get("save_predictions", True)
    summary = trainer.evaluate(test_loader, save_predictions=save_predictions)

    print("\nTest summary:")
    print(f"  Dice:    {summary['dice']['mean']:.4f} ± {summary['dice']['std']:.4f}")
    print(f"  Dice_FG: {summary['dice_fg']['mean']:.4f} ± {summary['dice_fg']['std']:.4f}")
    print(f"  HD95:    {summary['hd95']['mean']:.2f} ± {summary['hd95']['std']:.2f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

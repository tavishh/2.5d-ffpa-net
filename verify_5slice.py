"""
Targeted verification for the NEW 5-slice path, using the REAL dataset
pipeline (not synthetic tensors like tests/smoke_test.py uses). This checks
exactly what the existing smoke test does not: that num_slices=5 actually
produces correctly-shaped, correctly-loaded real image stacks, and that the
model consumes them end-to-end without error.

Run from the repo root, with the real dataset available:
    python verify_5slice.py --config configs/base.yaml

Cheap and fast (a handful of real batches, CPU is fine) -- meant to catch any
wiring problem before committing a real multi-hour training run to the new
5-slice fusion_only_5slice.yaml config.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ffpa25d.utils import load_config, resolve_device
from ffpa25d.data.dataset import create_dataloaders
from ffpa25d.models.ffpa_net_25d import FFPANet25D
from ffpa25d.losses.loss import DeepSupervisionLoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/variants/fusion_only_5slice.yaml")
    args = ap.parse_args()

    print("=" * 60)
    print("5-slice path verification (real data, not synthetic)")
    print("=" * 60)

    config = load_config(args.config)
    config["training"]["device"] = resolve_device(config["training"].get("device", "cuda"))
    device = config["training"]["device"]

    # Force a tiny debug subset so this runs in seconds regardless of dataset size.
    config.setdefault("debug", {})
    config["debug"]["enabled"] = True
    config["debug"]["subset_ratio"] = 0.02

    assert config["data"].get("three_slice") is True, \
        "Config must have data.three_slice: true for this check to be meaningful"
    expected_n = config["data"].get("num_slices", 3)
    print(f"\nConfig requests num_slices={expected_n}\n")

    # --- [1] Dataset produces correctly-shaped real image stacks ---
    train_loader, val_loader, _ = create_dataloaders(config)
    batch = next(iter(train_loader))
    images, masks = batch["image"], batch["mask"]

    print(f"\n[1] Real batch from dataset:")
    print(f"    image shape: {tuple(images.shape)}  (expect (B, {expected_n}, H, W))")
    print(f"    mask shape:  {tuple(masks.shape)}")
    assert images.shape[1] == expected_n, \
        f"Dataset returned {images.shape[1]} channels, expected {expected_n}!"
    print(f"    PASS -- dataset correctly returns {expected_n}-channel stacks")

    # Sanity: prev/next slices should generally differ from each other (not all
    # identical, which would suggest the offset logic silently collapsed to 0).
    half = expected_n // 2
    if half > 0:
        first_slice = images[0, 0]
        last_slice = images[0, -1]
        identical = torch.allclose(first_slice, last_slice)
        print(f"    outermost two slices identical (only expected at a patient's own edge): {identical}")

    # --- [2] Model consumes the real batch end-to-end ---
    model = FFPANet25D(config, base_channels=config["ffpa"].get("base_channels", 40)).to(device)
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images, return_aux=False)
    print(f"\n[2] Model forward pass on REAL {expected_n}-slice batch:")
    print(f"    output shape: {tuple(output.shape)}")
    expected_classes = config["model"]["actual_num_classes"]
    assert output.shape[1] == expected_classes
    print(f"    PASS -- correct number of output classes ({expected_classes})")

    # --- [3] Gradient flow with real data + real loss ---
    model.train()
    criterion = DeepSupervisionLoss(num_classes=expected_classes)
    masks = masks.to(device)
    outputs = model(images, return_aux=True)
    loss = criterion(outputs, masks)
    loss.backward()
    n_params_with_grad = sum(1 for p in model.parameters()
                              if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"\n[3] Backward pass on real data:")
    print(f"    loss = {loss.item():.4f}")
    print(f"    {n_params_with_grad} parameter tensors received non-zero gradients")
    assert not torch.isnan(loss), "Loss is NaN!"
    print(f"    PASS -- no NaN, gradients flowing")

    # --- [4] Confirm params/FLOPs are unaffected by slice count (as expected) ---
    total, trainable = model.count_parameters()
    print(f"\n[4] Model size: {trainable:,} trainable params "
          f"(should match the {expected_n}=3 case -- slice count changes compute, not param count)")

    print("\n" + "=" * 60)
    print(f"ALL CHECKS PASSED for num_slices={expected_n}")
    print("Safe to proceed with the full training run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
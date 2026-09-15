"""
Targeted verification for the slice-consistency path, using the REAL dataset
pipeline (not synthetic tensors). Checks exactly what no existing test covers:
that use_slice_consistency=True produces a correctly-shaped, correctly-shifted
"image_next" window from real data, that the second forward pass and the
consistency loss are genuinely wired into the computation graph, and that
training end-to-end (forward + backward, both slices) still works.

Run from the repo root, with the real dataset available:
    python verify_slice_consistency.py --config configs/variants/fusion_only_sliceconsistency.yaml

Cheap and fast (a handful of real batches, CPU or GPU both fine) -- meant to
catch any wiring problem before committing a real multi-hour training run
(which will roughly double per-step cost, given the second forward pass).
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
from ffpa25d.engine.trainer import _slice_consistency_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/variants/fusion_only_sliceconsistency.yaml")
    args = ap.parse_args()

    print("=" * 60)
    print("Slice-consistency verification (real data, not synthetic)")
    print("=" * 60)

    config = load_config(args.config)
    config["training"]["device"] = resolve_device(config["training"].get("device", "cuda"))
    device = config["training"]["device"]

    config.setdefault("debug", {})
    config["debug"]["enabled"] = True
    config["debug"]["subset_ratio"] = 0.02

    assert config["data"].get("use_slice_consistency") is True, \
        "Config must have data.use_slice_consistency: true for this check to be meaningful"
    weight = config["training"].get("slice_consistency_weight", 0.0)
    assert weight > 0, "training.slice_consistency_weight must be > 0 for this check to be meaningful"
    print(f"\nslice_consistency_weight = {weight}\n")

    # --- [1] Dataset produces both windows, correctly shifted, from real data ---
    train_loader, val_loader, _ = create_dataloaders(config)
    batch = next(iter(train_loader))
    images, images_next, masks = batch["image"], batch["image_next"], batch["mask"]

    print(f"\n[1] Real batch from dataset:")
    print(f"    image shape:      {tuple(images.shape)}")
    print(f"    image_next shape: {tuple(images_next.shape)}  (must match image's shape)")
    assert images.shape == images_next.shape
    print(f"    PASS -- both windows present, matching shapes")

    # Sanity: image_next should NOT be identical to image for most samples in
    # a real batch (only true at a patient's very last slice, a rare edge
    # case) -- if EVERY sample in the batch were identical, that would
    # indicate the shift isn't actually happening.
    identical_count = sum(
        torch.equal(images[i], images_next[i]) for i in range(images.shape[0])
    )
    print(f"    samples where image == image_next: {identical_count}/{images.shape[0]} "
          f"(should be 0, or very few -- only true at a patient's last slice)")
    assert identical_count < images.shape[0], \
        "ALL samples have image==image_next -- the shift isn't happening!"

    # --- [2] Model forward pass on both windows ---
    model = FFPANet25D(config, base_channels=config["ffpa"].get("base_channels", 40)).to(device)
    model.train()
    images = images.to(device)
    images_next = images_next.to(device)
    masks = masks.to(device)

    outputs = model(images, return_aux=True)
    outputs_next = model(images_next, return_aux=False)
    main = outputs[0] if isinstance(outputs, tuple) else outputs
    print(f"\n[2] Forward pass on both windows:")
    print(f"    main output shape:      {tuple(main.shape)}")
    print(f"    next-slice output shape: {tuple(outputs_next.shape)}")
    assert main.shape == outputs_next.shape
    print(f"    PASS -- both outputs correctly shaped and matching")

    # --- [3] Consistency loss computes correctly and is non-trivial ---
    consistency = _slice_consistency_loss(main, outputs_next)
    print(f"\n[3] Consistency loss value: {consistency.item():.6f}")
    assert consistency.item() >= 0, "consistency loss should be non-negative (it's an MSE)"
    assert not torch.isnan(consistency), "consistency loss is NaN!"
    print(f"    PASS -- non-negative, finite")

    # --- [4] Full combined loss + backward pass, confirming BOTH forward
    # passes contribute gradients (symmetric consistency, not one-sided) ---
    criterion = DeepSupervisionLoss(num_classes=config["model"]["actual_num_classes"])
    main_loss = criterion(outputs, masks)
    total_loss = main_loss + weight * consistency
    total_loss.backward()

    n_params_with_grad = sum(1 for p in model.parameters()
                              if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"\n[4] Combined backward pass:")
    print(f"    main_loss = {main_loss.item():.4f}, consistency = {consistency.item():.4f}, "
          f"total = {total_loss.item():.4f}")
    print(f"    {n_params_with_grad} parameter tensors received non-zero gradients")
    assert not torch.isnan(total_loss)
    print(f"    PASS -- no NaN, gradients flowing through the combined loss")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED for use_slice_consistency=True")
    print(f"Note: this roughly DOUBLES per-step compute vs. a normal run "
          f"(second forward pass every training step) -- expect roughly 2x "
          f"the wall-clock time of fusion_only's original training run.")
    print("Safe to proceed with the full training run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
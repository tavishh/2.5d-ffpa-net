"""
Targeted verification for the SC-UNet-style position-gate path, using the
REAL dataset pipeline (not synthetic tensors). Checks exactly what no
existing test covers: that use_position_encoding=True produces a correctly
ranged bucket index from real data, that the model's one-hot -> Linear ->
multiplicative gate is genuinely wired into the computation graph, and that
training end-to-end (forward + backward) still works with position passed
as its own argument (not smuggled through the image tensor).

Run from the repo root, with the real dataset available:
    python verify_position_encoding.py --config configs/variants/fusion_only_position.yaml

Cheap and fast (a handful of real batches, CPU is fine) -- meant to catch any
wiring problem before committing a real multi-hour training run. Requires
the trainer.py patch (trainer_patch.py) to already be applied, since this
script calls the model directly with position= rather than going through
the training loop -- but any real training run WILL need that patch applied
for position to reach the model at all.
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
    ap.add_argument("--config", default="configs/variants/fusion_only_position.yaml")
    args = ap.parse_args()

    print("=" * 60)
    print("SC-UNet position-gate verification (real data, not synthetic)")
    print("=" * 60)

    config = load_config(args.config)
    config["training"]["device"] = resolve_device(config["training"].get("device", "cuda"))
    device = config["training"]["device"]

    # Force a tiny debug subset so this runs in seconds regardless of dataset size.
    config.setdefault("debug", {})
    config["debug"]["enabled"] = True
    config["debug"]["subset_ratio"] = 0.02

    assert config["data"].get("use_position_encoding") is True, \
        "Config must have data.use_position_encoding: true for this check to be meaningful"

    # --- [1] Dataset produces a correctly-ranged position bucket from real data ---
    train_loader, val_loader, _ = create_dataloaders(config)
    batch = next(iter(train_loader))
    images, masks = batch["image"], batch["mask"]

    print(f"\n[1] Real batch from dataset:")
    print(f"    image shape: {tuple(images.shape)}  (position is NOT in here -- separate field)")
    assert "position" in batch, "Dataset did not return a 'position' field!"
    position = batch["position"]
    print(f"    position shape: {tuple(position.shape)}  (expect (B,))")
    print(f"    position values in this batch: {position.tolist()}")
    print(f"    dtype: {position.dtype}  (must be integer/long for one_hot)")
    assert position.dtype == torch.long
    assert position.min() >= 0 and position.max() <= 99, \
        f"position out of expected [0,99] range: min={position.min()}, max={position.max()}"
    print(f"    PASS -- position bucket correctly in [0, 99]")

    # --- [2] Model builds the position_gate layer with the right shape ---
    model = FFPANet25D(config, base_channels=config["ffpa"].get("base_channels", 40)).to(device)
    assert hasattr(model, "position_gate"), "Model did not build a position_gate layer!"
    print(f"\n[2] position_gate layer: {model.position_gate}")
    expected_classes = config["model"]["actual_num_classes"]
    assert model.position_gate.in_features == 100
    assert model.position_gate.out_features == expected_classes
    print(f"    PASS -- Linear(100, {expected_classes}), matching num_classes")

    # --- [3] Model consumes real image + real position end-to-end ---
    model.eval()
    images = images.to(device)
    position = position.to(device)
    with torch.no_grad():
        output = model(images, position=position, return_aux=False)
    print(f"\n[3] Model forward pass with REAL image + REAL position:")
    print(f"    output shape: {tuple(output.shape)}")
    assert output.shape[1] == expected_classes
    print(f"    PASS -- correct number of output classes ({expected_classes})")

    # --- [4] Confirm the gate actually changes the output (not a no-op) ---
    # The cleanest way to verify the gate is genuinely wired in: run the SAME
    # image through the model with two DIFFERENT position buckets. If the
    # gate were somehow disconnected from the computation graph, both would
    # produce identical output; if wired correctly, they must differ.
    same_image = images[0:1]
    pos_a = torch.tensor([3], device=device, dtype=torch.long)
    pos_b = torch.tensor([80], device=device, dtype=torch.long)
    with torch.no_grad():
        out_a = model(same_image, position=pos_a, return_aux=False)
        out_b = model(same_image, position=pos_b, return_aux=False)
    print(f"\n[4] Same image, two different position buckets (3 vs 80):")
    identical = torch.allclose(out_a, out_b)
    print(f"    outputs identical: {identical}  (must be False -- gate should change the output)")
    assert not identical, "Different position buckets produced identical output -- gate not wired in!"
    print(f"    PASS -- position genuinely affects the output")

    # --- [5] Gradient flow with real data + real loss, including position_gate ---
    model.train()
    criterion = DeepSupervisionLoss(num_classes=expected_classes)
    masks = masks.to(device)
    outputs = model(images, position=position, return_aux=True)
    loss = criterion(outputs, masks)
    loss.backward()
    print(f"\n[5] Backward pass on real data:")
    print(f"    loss = {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    pg_grad = model.position_gate.weight.grad
    assert pg_grad is not None and pg_grad.abs().sum() > 0, \
        "position_gate received no gradient -- not in the computation graph!"
    print(f"    PASS -- no NaN, and position_gate itself received a non-zero gradient")

    # --- [6] Param count sanity: should be only marginally higher than baseline ---
    total, trainable = model.count_parameters()
    print(f"\n[6] Model size: {trainable:,} trainable params "
          f"(expect a tiny increase over the no-position baseline -- one "
          f"Linear(100,{expected_classes}) layer, ~{100*expected_classes + expected_classes} params)")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED for use_position_encoding=True (SC-UNet style)")
    print("Reminder: the trainer_patch.py changes must be applied to trainer.py")
    print("for position to reach the model during a real training run.")
    print("Safe to proceed with the full training run.")
    print("=" * 60)


if __name__ == "__main__":
    main()

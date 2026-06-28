"""
Smoke test for the 2.5D FFPA-Net pipeline.

Runs WITHOUT the real dataset (synthetic tensors) and imports the REAL package
modules, so it catches import/path errors in the ffpa25d layout too.

Run from the project root:
    python tests/smoke_test.py
Expected: ALL TESTS PASSED. No GPU required (CPU only).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ffpa25d.models.slice_fusion import SliceCSA, MultiScaleSliceFusion
from ffpa25d.models.ffpa_net_25d import FFPANet25D, FUSION_VARIANTS

print("=" * 60)
print("2.5D FFPA-Net Smoke Test")
print("=" * 60)

DEVICE = "cpu"
B, H, W = 2, 64, 64           # small for speed
NUM_CLASSES = 11


def make_config(fusion_variant):
    return {
        "model": {"actual_num_classes": NUM_CLASSES, "num_classes": NUM_CLASSES,
                  "class_info": {"inverse_mapping": {i: i for i in range(1, 11)}}},
        "ffpa": {"fusion_variant": fusion_variant, "base_channels": 16,
                 "use_original_image": True, "use_deep_supervision": True},
        "training": {"device": DEVICE},
    }


# ── [1] SliceCSA unit: collapses slice axis, neighbour matters ────────────────
print("\n[1] SliceCSA unit")
x = torch.randn(B, 3, 10, H, W)
csa = SliceCSA(num_features=10, num_slices=3)
out = csa(x)
assert out.shape == (B, 10, H, W), out.shape
xb = x.clone(); xb[:, 0] += 5.0          # perturb prev slice only
assert not torch.allclose(csa(x), csa(xb), atol=1e-4), "neighbour had no effect!"
print("  PASS - fuses (B,3,10,H,W)->(B,10,H,W); neighbour affects output")

# ── [2] FFPANet25D forward for every fusion variant ───────────────────────────
print("\n[2] FFPANet25D forward (all fusion variants)")
expected = (B, NUM_CLASSES, H, W)
for variant, (three_slice, _, _) in FUSION_VARIANTS.items():
    model = FFPANet25D(make_config(variant), base_channels=16).to(DEVICE).eval()
    x_in = torch.randn(B, 3, H, W) if three_slice else torch.randn(B, 1, H, W)
    with torch.no_grad():
        y = model(x_in, return_aux=False)
    assert tuple(y.shape) == expected, f"{variant}: {tuple(y.shape)}"
    print(f"  PASS - {variant:<22} -> {tuple(y.shape)}")

# ── [3] THE 2.5D PROPERTY: neighbour slices change S1's output end-to-end ─────
print("\n[3] End-to-end neighbour sensitivity (the property decoder-CSA lacked)")
model_s1 = FFPANet25D(make_config("S1_postfilter_fusion"), base_channels=16).to(DEVICE).eval()
x_a = torch.randn(B, 3, H, W)
x_b = x_a.clone()
x_b[:, 0] += 3.0                          # change ONLY the prev slice
x_b[:, 2] -= 3.0                          # ... and the next slice; centre intact
with torch.no_grad():
    ya, yb = model_s1(x_a), model_s1(x_b)
assert not torch.allclose(ya, yb, atol=1e-4), \
    "S1 output unchanged when neighbours change -- 2.5D is NOT wired!"
print("  PASS - S1 genuinely uses prev/next slices")

# Control: S0 (single slice) must IGNORE the extra channels by construction
# (it only ever sees 1 channel), so we just confirm it runs on 1-channel input.
model_s0 = FFPANet25D(make_config("S0_single_slice"), base_channels=16).to(DEVICE).eval()
with torch.no_grad():
    _ = model_s0(torch.randn(B, 1, H, W))
print("  PASS - S0 runs on single-channel input")

# ── [4] Loss + backward: gradients flow, no NaN (S1) ─────────────────────────
print("\n[4] Loss + backward (S1, deep supervision)")
import torch.optim as optim
from ffpa25d.losses.loss import DeepSupervisionLoss

model_s1.train()
crit = DeepSupervisionLoss(num_classes=NUM_CLASSES)
opt = optim.AdamW(model_s1.parameters(), lr=1e-3)
target = torch.randint(0, NUM_CLASSES, (B, H, W))
opt.zero_grad()
outputs = model_s1(torch.randn(B, 3, H, W), return_aux=True)
assert isinstance(outputs, tuple) and len(outputs) == 3
loss = crit(outputs, target)
loss.backward()
opt.step()
nan = [n for n, p in model_s1.named_parameters()
       if p.grad is not None and torch.isnan(p.grad).any()]
assert not nan, f"NaN gradients in {nan[:3]}"
# SliceCSA must receive gradients (proves it's in the graph, not dead)
sf_grad = any(p.grad is not None and p.grad.abs().sum() > 0
              for n, p in model_s1.named_parameters() if "slice_fusion" in n)
assert sf_grad, "slice_fusion received no gradient!"
print(f"  PASS - loss={loss.item():.4f}, no NaN, slice_fusion has gradients")

# ── [5] Dataset neighbour index logic (pure, no files) ───────────────────────
print("\n[5] Patient-boundary neighbour clamping")
from collections import defaultdict


def build_bounds(samples):
    g = defaultdict(list)
    for i, s in enumerate(samples):
        g[s].append(i)
    bounds = [(0, 0)] * len(samples)
    for idxs in g.values():
        for i in idxs:
            bounds[i] = (idxs[0], idxs[-1])
    return bounds


def neighbor(i, off, b):
    first, last = b[i]
    return max(first, min(last, i + off))


samples = ["P01"] * 4 + ["P02"] * 3
b = build_bounds(samples)
assert neighbor(0, -1, b) == 0 and neighbor(3, +1, b) == 3      # P01 edges clamp
assert neighbor(4, -1, b) == 4 and neighbor(6, +1, b) == 6      # P02 edges clamp
for i in range(len(samples)):
    for off in (-1, +1):
        assert samples[neighbor(i, off, b)] == samples[i]        # never cross patient
print("  PASS - edges clamp, no cross-patient neighbours")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
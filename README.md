# 2.5D FFPA-Net

**2.5D extension of FFPA-Net for multi-class lower limb muscle segmentation from MRI.**

This repository extends the FFPA-Net baseline (Fixed-Filter Progressive Attention Network) with a Cross-Slice Attention (CSA) module adapted from XAG-Net. The core motivation is a documented ~18-point Dice gap between calf and knee performance in the 2D baseline, caused by processing each MRI slice independently with no inter-slice context. The 2.5D extension addresses this by stacking adjacent slices as a 3-channel input and inserting CSA into the decoder to reweight features across the cross-slice context.

**MIG Lab - Northeastern University Silicon Valley**
Supervised by Dr. Jeongkyu Lee.

---

## Motivation

| Region | FFPA-Net (2D baseline) |
|--------|----------------------|
| Overall Dice FG | 0.8604 |
| Calf Dice | 0.8927 |
| Thigh Dice | 0.8606 |
| Knee Dice | 0.7113 |

The knee is where muscles taper and converge fastest across slices. A 2D model attending to a single slice at a time lacks the inter-slice context needed to maintain consistent boundaries in this region. The 2.5D extension targets knee Dice > 0.80, overall Dice > 0.87, and trainable parameters < 3M.

---

## Method

**3-slice input:** Each slice is stacked with its immediate neighbours (prev, current, next) into a `(3, H, W)` tensor. Patient boundaries are handled by replicate padding - no cross-patient stacking.

**Cross-Slice Attention (CSA):** Adapted from XAG-Net (Ko et al., ACDSA 2025). Applies a pixel-wise channel reweighting via a learned 1x1 convolution and softmax, with a residual connection:

```
CSA(X) = X + (X * Softmax(W(X)))
```

**CSA insertion point (primary variant):** Applied to the finer-scale skip features (`x2`, `x1`) in the decoder just before each attention gate. This lets CSA suppress irrelevant feature channels before the gate decides what spatial regions to weight.

---

## Ablation Variants

Five variants are supported via a single config key (`ffpa.variant`):

| Variant | 3-slice input | CSA at input | CSA in decoder |
|---------|:---:|:---:|:---:|
| `baseline_2d` | - | - | - |
| `fusion_only` | yes | - | - |
| `csa_input_only` | yes | yes | - |
| `csa_decoder_only` | yes | - | yes |
| `csa_full` | yes | yes | yes |

`csa_decoder_only` is the primary ablation target.

---

## Repository Structure

```
2.5d-ffpa-net/
├── config.yaml          # All hyperparameters and variant selection
├── train.py             # Train + test evaluation
├── evaluate.py          # Evaluate an existing checkpoint
├── requirements.txt     # Python dependencies
├── smoke_test.py        # End-to-end pipeline verification (no data needed)
└── src/
    ├── csa.py           # CrossSliceAttention module
    ├── dataset.py       # PatientSliceDataset with 3-slice stacking
    ├── fixed_filters.py # Non-trainable handcrafted filter bank
    ├── model.py         # FFPANet with all 5 ablation variants
    ├── loss.py          # Dice+CE + deep supervision
    ├── metrics.py       # Dice, Dice_FG, HD95
    └── trainer.py       # Training loop, checkpointing, evaluation
```

---

## Data Layout

The dataset is not included in this repository. Point `data.root_dir` in `config.yaml` to your local copy:

```
{root_dir}/
├── train/
│   ├── images/{patient_id}/*.png
│   └── masks/{patient_id}/*.png
├── val/
│   ├── images/...
│   └── masks/...
└── test/
    ├── images/...
    └── masks/...
```

This repository was developed using the Sheffield Augmented Lower Limb MRI dataset (Henson et al., PLOS ONE 2024) with 10 muscle classes selected by prevalence.

---

## Quick Start

```bash
git clone https://github.com/<your-username>/2.5d-ffpa-net.git
cd 2.5d-ffpa-net
pip install -r requirements.txt
```

**Verify the pipeline (no data needed):**
```bash
python smoke_test.py
```

**Edit `config.yaml`** - set `data.root_dir` to your dataset path, then:

```bash
# Train primary variant
python train.py

# Train with explicit config and output directory
python train.py --config config.yaml --output outputs/csa_decoder_only

# Evaluate a saved checkpoint
python evaluate.py --checkpoint outputs/run_xxx/best_model.pth
```

**Switch variants** by editing one line in `config.yaml`:
```yaml
ffpa:
  variant: csa_decoder_only  # or: baseline_2d | fusion_only | csa_input_only | csa_full
data:
  three_slice: true          # set false for baseline_2d
```

---

## Output

Each run writes to `outputs/run_YYYYMMDD_HHMMSS/`:

| File | Description |
|------|-------------|
| `config.yaml` | Snapshot of the run configuration |
| `best_model.pth` | Best checkpoint, selected by Dice_FG |
| `training_log.csv` | Per-epoch train/val metrics |
| `slice_metrics.csv` | Per-slice test metrics |
| `summary.json` | Test-set summary (mean +/- std) |
| `predictions/` | Predicted masks (if enabled) |

---

## Smoke Test

`smoke_test.py` verifies the full pipeline without requiring any data. It uses synthetic tensors to check:

- Patient boundary clamping (no cross-patient slice lookups)
- Dataset output shapes for 2D and 2.5D modes
- Model initialization and forward pass for all 5 variants
- CSA parameter overhead (+8,120 params at base_channels=40)
- Gradient flow and deep supervision output shapes

```bash
python smoke_test.py
# Expected: ALL TESTS PASSED
```

---

## References

- FFPA-Net: Fixed-Filter Progressive Attention Network for Multi-Class Lower Limb Muscle Segmentation from MRI. MIG Lab, Northeastern University Silicon Valley, 2026 (unpublished).
- Ko, B., Tian, A., & Lee, J. XAG-Net: A Cross-Slice Attention and Skip Gating Network for 2.5D Femur MRI Segmentation. ACDSA 2025.
- Henson, R. et al. Sheffield Augmented Lower Limb MRI. PLOS ONE 2024.
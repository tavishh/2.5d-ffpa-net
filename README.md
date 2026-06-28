# 2.5D FFPA-Net

2.5D extension of FFPA-Net with Cross-Slice Attention (CSA) for multi-class
lower-limb muscle segmentation from MRI.

The 2D FFPA-Net processes each slice independently. This repo adds adjacent-slice
context by stacking `[prev, curr, next]` and fusing them with a Cross-Slice
Attention module adapted from XAG-Net.

## Design point

FFPA-Net's encoder is a fixed, single-channel filter bank: it processes one slice
at a time, so the slice axis disappears once a centre slice is chosen. XAG-Net's
CSA needs the channel axis to *be* the slice axis. The only window where features
are extracted *and* the slice axis still exists is immediately after per-slice
filtering. Fusion (`SliceCSA`) happens there, so the decoder is reused unchanged.

## Slice-fusion variants

| Variant | Input | Filtering | Fusion |
|---|---|---|---|
| `S0_single_slice` | 1ch | 1x | none (2D baseline) |
| `S1_postfilter_fusion` | 3ch | 3x | SliceCSA over slice axis (primary) |
| `S2_input_fusion` | 3ch | 1x | CSA on raw stack (early) |

S1 and S2 are not equal-compute (3x vs 1x filtering); report their FLOPs
separately.

## Layout

```
ffpa_csa_25d/
├── configs/
│   ├── base.yaml
│   └── variants/{S0_single_slice,S1_postfilter_fusion,S2_input_fusion}.yaml
├── ffpa25d/
│   ├── data/dataset.py
│   ├── engine/trainer.py
│   ├── losses/loss.py
│   ├── models/{ffpa_net_25d,slice_fusion,csa,decoder,fixed_filters}.py
│   ├── metrics.py
│   └── utils.py
├── scripts/{train.py, evaluate.py}
├── analysis/make_tables.py
├── tests/smoke_test.py
└── results/
```

## Data

Set `data.root_dir` (in `configs/base.yaml`) to the dataset root:

```
data/muscle_top10/
├── train/ images/{patient_id}/slice_XXXX.png
│          masks /{patient_id}/slice_XXXX.png
├── val/   ...
└── test/  ...
```

Masks store original class IDs `[21,34,76,90,96,200,214,241,248,255]`, remapped
to `1..10` at train time (background = 0, 11 channels) and mapped back on save.
Split is by patient, taken as-is (train 47 / val 11 / test 11).

## Quick start

Run from the project root.

```bash
pip install -r requirements.txt

# Verify the pipeline (no data needed)
python tests/smoke_test.py

# Train a variant
python scripts/train.py --config configs/variants/S1_postfilter_fusion.yaml
python scripts/train.py --config configs/variants/S0_single_slice.yaml

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint results/S1_postfilter_fusion/best_model.pth

# Aggregate results into a table
python analysis/make_tables.py
```

## Output

Each run writes to `results/<name>/`: `config.yaml`, `best_model.pth`,
`training_log.csv`, `slice_metrics.csv`, `summary.json`, `predictions/`.

## References

- FFPA-Net (MIG Lab, Northeastern University Silicon Valley).
- Ko, B., Tian, A., & Lee, J. XAG-Net: A Cross-Slice Attention and Skip Gating
  Network for 2.5D Femur MRI Segmentation. ACDSA 2025.

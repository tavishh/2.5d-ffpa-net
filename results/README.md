# Results

This directory holds training/evaluation outputs. It is gitignored (except this
file) because checkpoints and prediction masks are large.

Each variant writes to `results/<variant>/`:

| File | Description |
|------|-------------|
| `config.yaml` | Snapshot of the resolved run configuration |
| `best_model.pth` | Best checkpoint (selected by val Dice_FG) |
| `training_log.csv` | Per-epoch train/val metrics |
| `slice_metrics.csv` | Per-slice test metrics |
| `summary.json` | Test-set summary (mean / std) |
| `predictions/` | Predicted masks, remapped to original class IDs |

`ablation_table.{csv,md}` (written by `analysis/make_tables.py`) aggregate all
variants.

## Paper correspondence

| Paper table / row | Run directory | Seed |
|---|---|---|
| Table X, 2D baseline | `results/S0_single_slice/` | 42 |
| Table X, **ours (S1)** | `results/S1_postfilter_fusion/` | 42 |
| Table X, input fusion | `results/S2_input_fusion/` | 42 |
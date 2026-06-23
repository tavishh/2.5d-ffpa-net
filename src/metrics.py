"""Slice-level segmentation metrics."""
from typing import Dict, Tuple

import numpy as np
import torch


def _compute_dice_binary(pred, target, threshold=0.5, smooth=1e-5):
    pred = pred.squeeze()
    target = target.squeeze()
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()
    return ((2.0 * intersection + smooth) / (union + smooth)).item()


def _compute_dice_multiclass(
    pred, target, num_classes, smooth=1e-5, include_background=True, skip_empty_classes=False
):
    pred_classes = torch.argmax(pred, dim=0) if pred.dim() == 3 else pred
    target = target.squeeze()
    start_class = 0 if include_background else 1
    dice_scores = []

    for c in range(start_class, num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        intersection = (pred_c * target_c).sum()

        if target_sum == 0 and pred_sum == 0:
            if skip_empty_classes:
                continue
            dice_scores.append(1.0)
        elif target_sum == 0 or pred_sum == 0:
            dice_scores.append(0.0)
        else:
            dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
            dice_scores.append(dice.item())

    return float(np.mean(dice_scores)) if dice_scores else 1.0


def compute_dice_both(pred, target, num_classes=1, threshold=0.5):
    if num_classes == 1:
        dice = _compute_dice_binary(pred, target, threshold)
        return dice, dice
    dice_bg = _compute_dice_multiclass(
        pred, target, num_classes, include_background=True, skip_empty_classes=False
    )
    dice_fg = _compute_dice_multiclass(
        pred, target, num_classes, include_background=False, skip_empty_classes=True
    )
    return dice_bg, dice_fg


def _compute_hd95_from_masks(pred_mask, target_mask, percentile=95):
    pred_sum = pred_mask.sum().item()
    target_sum = target_mask.sum().item()
    if pred_sum == 0 and target_sum == 0:
        return 0.0
    if pred_sum == 0 or target_sum == 0:
        return float("inf")

    pred_points = torch.nonzero(pred_mask, as_tuple=False).float()
    target_points = torch.nonzero(target_mask, as_tuple=False).float()

    max_points = 10000
    if len(pred_points) > max_points:
        pred_points = pred_points[torch.randperm(len(pred_points))[:max_points]]
    if len(target_points) > max_points:
        target_points = target_points[torch.randperm(len(target_points))[:max_points]]

    dist = torch.cdist(pred_points, target_points, p=2)
    all_distances = torch.cat([dist.min(dim=1)[0], dist.min(dim=0)[0]])
    k = min(int(np.ceil(percentile / 100.0 * len(all_distances))), len(all_distances))
    return torch.kthvalue(all_distances, k)[0].item() if k > 0 else all_distances.max().item()


def compute_hd95(pred, target, num_classes=1, threshold=0.5):
    if num_classes == 1:
        pred = pred.squeeze()
        target = target.squeeze()
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        return _compute_hd95_from_masks((pred > threshold).bool(), (target > threshold).bool())

    pred_classes = torch.argmax(pred, dim=0) if pred.dim() == 3 else pred
    target = target.squeeze()
    hd95_scores = []
    for c in range(1, num_classes):
        pred_c = pred_classes == c
        target_c = target == c
        if pred_c.sum() > 0 or target_c.sum() > 0:
            hd95 = _compute_hd95_from_masks(pred_c, target_c)
            if np.isfinite(hd95):
                hd95_scores.append(hd95)
    return float(np.mean(hd95_scores)) if hd95_scores else 0.0


def compute_slice_metrics(pred, target, num_classes=1, threshold=0.5) -> Dict[str, float]:
    dice_bg, dice_fg = compute_dice_both(pred, target, num_classes, threshold)
    return {
        "dice": dice_bg,
        "dice_fg": dice_fg,
        "hd95": compute_hd95(pred, target, num_classes, threshold),
    }


class MetricsAccumulator:
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.slice_metrics = []

    def reset(self):
        self.slice_metrics = []

    def update(self, pred, target, patient_id, slice_name, threshold=0.5):
        metrics = compute_slice_metrics(pred, target, self.num_classes, threshold)
        metrics["patient_id"] = patient_id
        metrics["slice_name"] = slice_name
        self.slice_metrics.append(metrics)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        if not self.slice_metrics:
            return {}

        def summarize(key):
            values = [m[key] for m in self.slice_metrics]
            return {"mean": float(np.mean(values)), "std": float(np.std(values))}

        hd95_values = [m["hd95"] for m in self.slice_metrics if np.isfinite(m["hd95"])]
        return {
            "dice": summarize("dice"),
            "dice_fg": summarize("dice_fg"),
            "hd95": {
                "mean": float(np.mean(hd95_values)) if hd95_values else float("inf"),
                "std": float(np.std(hd95_values)) if hd95_values else 0.0,
            },
            "num_slices": len(self.slice_metrics),
        }

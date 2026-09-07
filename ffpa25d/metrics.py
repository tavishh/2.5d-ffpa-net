"""
Segmentation + efficiency metrics for the 2.5D FFPA-Net.

All segmentation metrics are computed at the 2D slice level:
  - dice / iou      : include background, empty classes score 1.0 (PFA-V2 style)
  - dice_fg / iou_fg: foreground only, SKIP empty classes (original FFPA-Net)
  - hd95            : foreground only
  - detect_*        : per-class presence/absence counts (NEW, additive -- see below)

The public entry points used by the pipeline are compute_dice_both /
compute_iou_both (training + accumulator), compute_slice_metrics,
MetricsAccumulator, and get_model_efficiency.

Detection-rate addition (2026-08): dice_fg scores a class 0.0 whenever it is
present in exactly one of (prediction, target) -- this conflates two very
different failure modes into one number:
  - a class present in target but NOT predicted at all (a genuine miss)
  - a class predicted but NOT present in target at all (a hallucination)
_compute_detection_counts distinguishes these directly. This is purely
additive: every existing key/column/metric is unchanged, this only adds new
ones alongside them.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Union


# ============================================================================
# Segmentation Metrics (2D Slice Level) - Unified Binary/Multi-class
# ============================================================================

def _compute_dice_binary(pred: torch.Tensor, target: torch.Tensor,
                         threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """Dice for binary segmentation."""
    pred = pred.squeeze()
    target = target.squeeze()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def _compute_dice_multiclass(pred: torch.Tensor, target: torch.Tensor,
                             num_classes: int, smooth: float = 1e-5,
                             include_background: bool = True,
                             skip_empty_classes: bool = False) -> float:
    """
    Dice for multi-class segmentation (mean over classes).

    Args:
        pred: [C, H, W] logits
        target: [H, W] class indices
        include_background: if True, include class 0 in the mean.
        skip_empty_classes: if True, skip classes absent from both pred and
            target (original FFPA-Net foreground behaviour); otherwise such
            classes score 1.0 (PFA-V2 behaviour).
    """
    if pred.dim() == 3:
        pred_classes = torch.argmax(pred, dim=0)  # [H, W]
    else:
        pred_classes = pred

    target = target.squeeze()

    dice_scores = []
    start_class = 0 if include_background else 1

    for c in range(start_class, num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()

        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        intersection = (pred_c * target_c).sum()

        if target_sum == 0 and pred_sum == 0:
            if skip_empty_classes:
                continue
            else:
                dice_scores.append(1.0)
        elif target_sum == 0 or pred_sum == 0:
            dice_scores.append(0.0)
        else:
            dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
            dice_scores.append(dice.item())

    if len(dice_scores) > 0:
        return np.mean(dice_scores)
    else:
        return 1.0  # No classes to evaluate


def compute_dice_both(pred: torch.Tensor, target: torch.Tensor,
                      num_classes: int = 1, threshold: float = 0.5,
                      smooth: float = 1e-5) -> Tuple[float, float]:
    """
    Compute both Dice scores: with-background and foreground-only.

    Returns:
        (dice_with_background, dice_foreground_only)
        For binary segmentation both values are identical.
    """
    if num_classes == 1:
        dice = _compute_dice_binary(pred, target, threshold, smooth)
        return dice, dice
    else:
        dice_with_bg = _compute_dice_multiclass(
            pred, target, num_classes, smooth,
            include_background=True, skip_empty_classes=False
        )
        dice_fg_only = _compute_dice_multiclass(
            pred, target, num_classes, smooth,
            include_background=False, skip_empty_classes=True
        )
        return dice_with_bg, dice_fg_only


def _compute_iou_binary(pred: torch.Tensor, target: torch.Tensor,
                        threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """IoU for binary segmentation."""
    pred = pred.squeeze()
    target = target.squeeze()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def _compute_iou_multiclass(pred: torch.Tensor, target: torch.Tensor,
                            num_classes: int, smooth: float = 1e-5,
                            include_background: bool = True,
                            skip_empty_classes: bool = False) -> float:
    """IoU for multi-class segmentation (mean over classes)."""
    if pred.dim() == 3:
        pred_classes = torch.argmax(pred, dim=0)
    else:
        pred_classes = pred

    target = target.squeeze()

    iou_scores = []
    start_class = 0 if include_background else 1

    for c in range(start_class, num_classes):
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()

        pred_sum = pred_c.sum()
        target_sum = target_c.sum()
        intersection = (pred_c * target_c).sum()
        union = pred_sum + target_sum - intersection

        if target_sum == 0 and pred_sum == 0:
            if skip_empty_classes:
                continue
            else:
                iou_scores.append(1.0)
        elif union == 0:
            iou_scores.append(1.0)
        else:
            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou.item())

    if len(iou_scores) > 0:
        return np.mean(iou_scores)
    else:
        return 1.0


def compute_iou_both(pred: torch.Tensor, target: torch.Tensor,
                     num_classes: int = 1, threshold: float = 0.5,
                     smooth: float = 1e-5) -> Tuple[float, float]:
    """
    Compute both IoU scores: with-background and foreground-only.

    Returns:
        (iou_with_background, iou_foreground_only)
        For binary segmentation both values are identical.
    """
    if num_classes == 1:
        iou = _compute_iou_binary(pred, target, threshold, smooth)
        return iou, iou
    else:
        iou_with_bg = _compute_iou_multiclass(
            pred, target, num_classes, smooth,
            include_background=True, skip_empty_classes=False
        )
        iou_fg_only = _compute_iou_multiclass(
            pred, target, num_classes, smooth,
            include_background=False, skip_empty_classes=True
        )
        return iou_with_bg, iou_fg_only


def compute_hd95(pred: torch.Tensor, target: torch.Tensor,
                 num_classes: int = 1, threshold: float = 0.5,
                 percentile: float = 95, include_background: bool = False) -> float:
    """Compute HD95 for a single 2D slice."""
    if num_classes == 1:
        return _compute_hd95_binary(pred, target, threshold, percentile)
    else:
        return _compute_hd95_multiclass(pred, target, num_classes, percentile, include_background)


def _compute_hd95_binary(pred: torch.Tensor, target: torch.Tensor,
                         threshold: float = 0.5, percentile: float = 95) -> float:
    """HD95 for binary segmentation."""
    pred = pred.squeeze()
    target = target.squeeze()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).bool()
    target_binary = (target > threshold).bool()

    return _compute_hd95_from_masks(pred_binary, target_binary, percentile)


def _compute_hd95_multiclass(pred: torch.Tensor, target: torch.Tensor,
                             num_classes: int, percentile: float = 95,
                             include_background: bool = False) -> float:
    """HD95 for multi-class segmentation (mean over classes)."""
    if pred.dim() == 3:
        pred_classes = torch.argmax(pred, dim=0)
    else:
        pred_classes = pred

    target = target.squeeze()

    hd95_scores = []
    start_class = 0 if include_background else 1

    for c in range(start_class, num_classes):
        pred_c = (pred_classes == c)
        target_c = (target == c)

        if pred_c.sum() > 0 or target_c.sum() > 0:
            hd95 = _compute_hd95_from_masks(pred_c, target_c, percentile)
            if np.isfinite(hd95):
                hd95_scores.append(hd95)

    if len(hd95_scores) > 0:
        return np.mean(hd95_scores)
    else:
        return 0.0


def _compute_hd95_from_masks(pred_mask: torch.Tensor, target_mask: torch.Tensor,
                             percentile: float = 95) -> float:
    """Compute HD95 from two binary masks."""
    pred_sum = pred_mask.sum().item()
    target_sum = target_mask.sum().item()

    if pred_sum == 0 and target_sum == 0:
        return 0.0
    if pred_sum == 0 or target_sum == 0:
        return float('inf')

    pred_points = torch.nonzero(pred_mask, as_tuple=False).float()
    target_points = torch.nonzero(target_mask, as_tuple=False).float()

    # Sample if too many points (keeps the pairwise distance tractable)
    max_points = 10000
    if len(pred_points) > max_points:
        indices = torch.randperm(len(pred_points))[:max_points]
        pred_points = pred_points[indices]
    if len(target_points) > max_points:
        indices = torch.randperm(len(target_points))[:max_points]
        target_points = target_points[indices]

    dist_pred_to_target = torch.cdist(pred_points, target_points, p=2)
    min_dist_pred = dist_pred_to_target.min(dim=1)[0]
    min_dist_target = dist_pred_to_target.min(dim=0)[0]

    all_distances = torch.cat([min_dist_pred, min_dist_target])

    k = int(np.ceil(percentile / 100.0 * len(all_distances)))
    k = min(k, len(all_distances))

    if k > 0:
        hd95 = torch.kthvalue(all_distances, k)[0].item()
    else:
        hd95 = all_distances.max().item()

    return hd95


# ============================================================================
# Detection-rate metric (NEW, additive)
# ============================================================================

def _compute_detection_counts(pred: torch.Tensor, target: torch.Tensor,
                              num_classes: int) -> Dict[str, int]:
    """
    Per-slice, per-foreground-class presence/absence counts.

    dice_fg scores a class 0.0 whenever it is present in exactly one of
    (prediction, target) -- this conflates two different failure modes into
    one number. This function separates them:
      - detect_tp: class present in BOTH target and prediction (a real
        detection, regardless of how good the overlap is)
      - detect_fn: class present in target, ABSENT from prediction (a
        genuine miss -- e.g. a thin/tapering muscle the model failed to
        find at all)
      - detect_fp: class ABSENT from target, present in prediction (a
        hallucination)
      - detect_tn: class absent from both (correctly predicted nothing)

    Only foreground classes (1..num_classes-1) are counted, matching the
    dice_fg / iou_fg convention elsewhere in this file. Counts, not rates,
    are returned so they can be summed correctly across slices before a
    rate is computed at the accumulator level (summing rates directly would
    incorrectly weight slices with fewer evaluated classes).

    Args:
        pred: [C, H, W] logits (or already-argmaxed [H, W] class map)
        target: [H, W] class indices
        num_classes: total classes including background (class 0)
    """
    if pred.dim() == 3:
        pred_classes = torch.argmax(pred, dim=0)
    else:
        pred_classes = pred
    target = target.squeeze()

    tp = fn = fp = tn = 0
    for c in range(1, num_classes):
        pred_present = bool((pred_classes == c).any())
        target_present = bool((target == c).any())

        if target_present and pred_present:
            tp += 1
        elif target_present and not pred_present:
            fn += 1
        elif not target_present and pred_present:
            fp += 1
        else:
            tn += 1

    return {'detect_tp': tp, 'detect_fn': fn, 'detect_fp': fp, 'detect_tn': tn}


def compute_slice_metrics(pred: torch.Tensor, target: torch.Tensor,
                          num_classes: int = 1, threshold: float = 0.5) -> Dict[str, float]:
    """Compute all metrics (dice, dice_fg, iou, iou_fg, hd95, detection) for one 2D slice."""
    dice_bg, dice_fg = compute_dice_both(pred, target, num_classes, threshold)
    iou_bg, iou_fg = compute_iou_both(pred, target, num_classes, threshold)
    hd95 = compute_hd95(pred, target, num_classes, threshold, include_background=False)

    metrics = {
        'dice': dice_bg,
        'dice_fg': dice_fg,
        'iou': iou_bg,
        'iou_fg': iou_fg,
        'hd95': hd95
    }

    if num_classes > 1:
        metrics.update(_compute_detection_counts(pred, target, num_classes))
    else:
        # Binary segmentation collapses to a single foreground class -- not
        # the case this metric was built for (it exists to distinguish
        # WHICH of several tracked classes was missed). Report zeros rather
        # than guessing at a definition nobody asked for.
        metrics.update({'detect_tp': 0, 'detect_fn': 0, 'detect_fp': 0, 'detect_tn': 0})

    return metrics


# ============================================================================
# Model Efficiency Metrics
# ============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_params(num_params: int) -> str:
    """Format parameter count as a readable string."""
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def compute_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 256, 256),
                  device: str = 'cpu') -> int:
    """
    Estimate FLOPs for a model.

    NOTE: both backends only count nn.Conv2d / nn.Linear modules. The fixed
    filter bank uses functional F.conv2d under no_grad, so its (3x in S1)
    filtering cost is NOT reflected here -- use measure_inference_time for the
    real wall-clock difference between S0/S1/S2.
    """
    try:
        from thop import profile
        model = model.to(device)
        model.eval()
        dummy_input = torch.randn(input_size).to(device)

        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

        return int(flops)

    except ImportError:
        return _estimate_flops_simple(model, input_size)


def _estimate_flops_simple(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """Simple FLOPs estimation without external dependencies."""
    total_flops = 0

    def hook_fn(module, input, output):
        nonlocal total_flops

        if isinstance(module, nn.Conv2d):
            batch_size = input[0].size(0)
            out_channels = module.out_channels
            out_h, out_w = output.size(2), output.size(3)
            in_channels = module.in_channels
            kh, kw = module.kernel_size
            groups = module.groups

            flops = 2 * out_channels * out_h * out_w * (in_channels // groups) * kh * kw
            total_flops += flops * batch_size

        elif isinstance(module, nn.Linear):
            batch_size = input[0].size(0)
            flops = 2 * module.in_features * module.out_features
            total_flops += flops * batch_size

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        model(dummy_input)

    for hook in hooks:
        hook.remove()

    return total_flops


def format_flops(flops: int) -> str:
    """Format FLOPs as a readable string."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f}K"
    else:
        return str(flops)


def measure_inference_time(model: nn.Module,
                           input_size: Tuple[int, ...] = (1, 1, 256, 256),
                           device: str = 'cuda',
                           num_runs: int = 100,
                           warmup_runs: int = 10) -> float:
    """Measure average inference time in milliseconds."""
    import time

    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)

    return np.mean(times)


# ============================================================================
# Batch Metrics Computation
# ============================================================================

class MetricsAccumulator:
    """Accumulator for computing metrics over multiple slices."""

    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.slice_metrics = []
        self.patient_slice_map = {}

    def update(self, pred: torch.Tensor, target: torch.Tensor,
               patient_id: str, slice_name: str, threshold: float = 0.5):
        """Update metrics with a new slice."""
        metrics = compute_slice_metrics(pred, target, self.num_classes, threshold)
        metrics['patient_id'] = patient_id
        metrics['slice_name'] = slice_name

        self.slice_metrics.append(metrics)

        if patient_id not in self.patient_slice_map:
            self.patient_slice_map[patient_id] = []
        self.patient_slice_map[patient_id].append(len(self.slice_metrics) - 1)

    def get_slice_metrics(self) -> list:
        return self.slice_metrics

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics (mean, std) for all metrics."""
        if not self.slice_metrics:
            return {}

        dice_values = [m['dice'] for m in self.slice_metrics]
        dice_fg_values = [m['dice_fg'] for m in self.slice_metrics]
        iou_values = [m['iou'] for m in self.slice_metrics]
        iou_fg_values = [m['iou_fg'] for m in self.slice_metrics]
        hd95_values = [m['hd95'] for m in self.slice_metrics if np.isfinite(m['hd95'])]

        summary = {
            'dice': {
                'mean': float(np.mean(dice_values)),
                'std': float(np.std(dice_values))
            },
            'dice_fg': {
                'mean': float(np.mean(dice_fg_values)),
                'std': float(np.std(dice_fg_values))
            },
            'iou': {
                'mean': float(np.mean(iou_values)),
                'std': float(np.std(iou_values))
            },
            'iou_fg': {
                'mean': float(np.mean(iou_fg_values)),
                'std': float(np.std(iou_fg_values))
            },
            'hd95': {
                'mean': float(np.mean(hd95_values)) if hd95_values else float('inf'),
                'std': float(np.std(hd95_values)) if hd95_values else 0.0,
                'valid_count': len(hd95_values),
                'total_count': len(self.slice_metrics)
            },
            'num_slices': len(self.slice_metrics),
            'num_patients': len(self.patient_slice_map)
        }

        # ---- NEW, additive: detection-rate summary ----
        # Sum raw counts across slices first, THEN take a rate -- summing
        # per-slice rates directly would incorrectly over-weight slices that
        # had fewer evaluated classes.
        total_tp = sum(m.get('detect_tp', 0) for m in self.slice_metrics)
        total_fn = sum(m.get('detect_fn', 0) for m in self.slice_metrics)
        total_fp = sum(m.get('detect_fp', 0) for m in self.slice_metrics)
        total_tn = sum(m.get('detect_tn', 0) for m in self.slice_metrics)

        denom_rate = total_tp + total_fn
        denom_fpr = total_fp + total_tn
        summary['detection'] = {
            # Of classes truly present, fraction the model actually predicted
            # (regardless of overlap quality) -- distinguishes real misses
            # from the harsh dice_fg=0.0 floor, which conflates misses with
            # hallucinations.
            'detection_rate': (total_tp / denom_rate) if denom_rate > 0 else float('nan'),
            # Of classes truly absent, fraction the model hallucinated.
            'false_positive_rate': (total_fp / denom_fpr) if denom_fpr > 0 else float('nan'),
            'true_positive': total_tp,
            'false_negative': total_fn,
            'false_positive': total_fp,
            'true_negative': total_tn,
        }

        return summary


# ============================================================================
# Convenience Functions
# ============================================================================

def get_model_efficiency(model: nn.Module,
                         input_size: Tuple[int, ...] = (1, 1, 256, 256),
                         device: str = 'cuda') -> Dict[str, Union[int, float, str]]:
    """Get all efficiency metrics for a model."""
    params = count_parameters(model)
    flops = compute_flops(model, input_size, device='cpu')

    if device == 'cuda' and torch.cuda.is_available():
        inference_time = measure_inference_time(model, input_size, device)
    else:
        inference_time = measure_inference_time(model, input_size, 'cpu')

    return {
        'params': params,
        'params_readable': format_params(params),
        'flops': flops,
        'flops_readable': format_flops(flops),
        'inference_time_ms': inference_time
    }
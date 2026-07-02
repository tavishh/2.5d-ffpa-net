"""
No-attention slice fusion control for the 2.5D FFPA-Net (variant: fusion_only).

This is the ablation control for S1_postfilter_fusion. It has the exact same
interface, input/output shapes, and residual structure as SliceCSA /
MultiScaleSliceFusion, but replaces the LEARNED softmax attention over the
slice axis with FIXED uniform (1/S) weighting. Zero learnable parameters.

Purpose: isolate whether SliceCSA's learned attention is doing anything, vs.
whether any gain from S1_postfilter_fusion is coming purely from running the
filter bank on all 3 slices (prev/curr/next) rather than just the centre slice.

Everything else in the pipeline (3x filter bank passes, decoder, deep
supervision, training config) must stay identical to S1_postfilter_fusion for
this to be a valid control -- only the fusion module differs.

Shapes (matching SliceCSA / MultiScaleSliceFusion exactly):
    MeanSliceFusion:        (B, S, F, H, W) -> (B, F, H, W)
    MultiScaleMeanFusion:   list over S slices (each a list over scales) ->
                            list over scales of (B, F, H, W)
"""
from typing import List

import torch
import torch.nn as nn


class MeanSliceFusion(nn.Module):
    """No-attention control for SliceCSA: uniform (1/S) weighting instead of
    a learned softmax, with the same residual-center structure. Zero
    learnable parameters.

    Mirrors SliceCSA's forward exactly:
        SliceCSA:         fused = sum_s(softmax(score(x))_s * x_s) + x_center
        MeanSliceFusion:  fused = sum_s((1/S)            * x_s) + x_center
                                = mean_s(x_s) + x_center

    Args:
        num_features: feature channels F (10 for the fixed filter bank).
        num_slices:   stacked slices S (3 = prev/curr/next).
        residual_center: add the centre slice back as a residual. Must match
            whatever S1_postfilter_fusion used (default True in SliceCSA) for
            this to be a valid control.
    """

    def __init__(self, num_features: int, num_slices: int = 3,
                 residual_center: bool = True):
        super().__init__()
        self.F = num_features
        self.S = num_slices
        self.center = num_slices // 2          # prev=0, curr=1, next=2 -> 1
        self.residual_center = residual_center
        # No nn.Conv2d, no learnable weights -- deliberately zero parameters.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F, H, W)
        assert x.dim() == 5 and x.size(1) == self.S and x.size(2) == self.F, \
            f"MeanSliceFusion expects (B,{self.S},{self.F},H,W), got {tuple(x.shape)}"

        fused = x.mean(dim=1)                                    # uniform 1/S weights, (B,F,H,W)

        if self.residual_center:
            fused = fused + x[:, self.center]                    # same residual as SliceCSA
        return fused

    def extra_repr(self) -> str:
        return f"num_features={self.F}, num_slices={self.S}, residual_center={self.residual_center}, params=0"


class MultiScaleMeanFusion(nn.Module):
    """Drop-in replacement for MultiScaleSliceFusion using MeanSliceFusion at
    every scale. Same call signature, same shapes, zero learnable parameters
    across the whole fusion stage.

    forward input:  per_slice_features -- list of length S (slices); each entry
                    a list of length num_scales: [(B,F,H,W), (B,F,H/2,W/2), ...]
    forward output: list of length num_scales of fused (B,F,H,W) maps.
    """

    def __init__(self, feature_dims: List[int], num_slices: int = 3,
                 residual_center: bool = True):
        super().__init__()
        self.num_scales = len(feature_dims)
        self.num_slices = num_slices
        self.fusers = nn.ModuleList([
            MeanSliceFusion(num_features=f, num_slices=num_slices,
                             residual_center=residual_center)
            for f in feature_dims
        ])

    def forward(self, per_slice_features: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        assert len(per_slice_features) == self.num_slices, \
            f"expected {self.num_slices} slices, got {len(per_slice_features)}"
        fused_scales = []
        for scale in range(self.num_scales):
            stack = torch.stack([per_slice_features[s][scale]
                                 for s in range(self.num_slices)], dim=1)   # (B,S,F,H,W)
            fused_scales.append(self.fusers[scale](stack))
        return fused_scales
"""
Relational cross-slice fusion for the 2.5D FFPA-Net (variant: relational_fusion).

Motivation: offline inspection of the trained S1_postfilter_fusion checkpoint
showed SliceCSA's scoring conv learned a highly anisotropic transform (large
singular values along 2-3 directions, near-zero along the rest) -- yet the
ablation control fusion_only (fixed uniform 1/S weighting) matched or slightly
beat S1 across every region, including the knee. SliceCSA scores each slice's
features INDEPENDENTLY (the same shared conv applied to x_s alone) -- it never
directly compares a slice to the centre, so it cannot express "does this
neighbour agree with the centre" or "is this neighbour drifting away", which
is the actual anatomical question a cross-slice fusion module should answer.

This module makes the scorer explicitly RELATIONAL: each slice's attention
logit is computed from a descriptor built from its relationship to the centre
slice, not from the slice alone:

    r_s = concat([x_s, x_centre, x_s - x_centre, |x_s - x_centre|])   # 4F channels

For the centre slice itself, x_centre - x_centre = 0 and |x_centre - x_centre| = 0
identically -- a unique, exactly-reproducible input signature no neighbour can
produce. The network can learn to map that signature to a high logit FROM
DATA, rather than the centre's dominance being hardcoded via a residual
addition (as both SliceCSA and MeanSliceFusion do). Accordingly this module
has NO residual term -- the centre competes for its weight inside the softmax
like every other slice.

Deliberately much lighter than published cross-slice attention designs like
CSA-Net (Kumar et al., Computers in Biology and Medicine, 2024), which uses
full query/key/value cross-attention between ALL spatial position pairs
across slices -- a mechanism built for the general case where two feature
maps may not be spatially aligned, and which CSA-Net only applies after 16x
downsampling to a ~20x20 grid for exactly this cost reason. Our 3 slices ARE
already spatially aligned (pixel (x,y) in prev/curr/next all refer to the
same in-plane anatomical location, one z-step apart), so a per-position
O(HW) relational comparison -- not O(HW^2) attention -- captures the same
"does this neighbour agree with the centre" question at a small fraction of
the cost, without needing to search for correspondences we already know.

Shapes (matching SliceCSA / MeanSliceFusion / MultiScaleSliceFusion exactly):
    RelationalSliceFusion:      (B, S, F, H, W) -> (B, F, H, W)
    MultiScaleRelationalFusion: list over S slices (each a list over scales) ->
                                list over scales of (B, F, H, W)
"""
from typing import List

import torch
import torch.nn as nn


class RelationalSliceFusion(nn.Module):
    """Cross-slice fusion via a relational (centre-referenced) attention score.

    Mechanism: for each slice s (including the centre), build a descriptor
    r_s = [x_s, x_centre, x_s - x_centre, |x_s - x_centre|] (4F channels), pass
    it through a SHARED 3x3 conv (4F -> F) to get per-slice logits, softmax
    over the slice axis, then take the weighted sum -- NO residual added. The
    centre's influence, if warranted, must be learned via its unique
    zero-difference signature rather than hardcoded via a residual.

    Args:
        num_features: feature channels F (10 for the fixed filter bank).
        num_slices:   stacked slices S (3 = prev/curr/next).
    """

    def __init__(self, num_features: int, num_slices: int = 3):
        super().__init__()
        self.F = num_features
        self.S = num_slices
        self.center = num_slices // 2                 # prev=0, curr=1, next=2 -> 1

        # Shared 3x3 conv: [x_s, x_centre, x_s - x_centre, |x_s - x_centre|]
        # (4F channels in) -> F channels out (per-slice logit map). Shared
        # across all slices (including centre) -- same convention as
        # SliceCSA's shared_score=True, for a clean single-variable ablation.
        self.score = nn.Conv2d(4 * num_features, num_features,
                               kernel_size=3, padding=1)

    def _descriptor(self, x_s: torch.Tensor, x_center: torch.Tensor) -> torch.Tensor:
        diff = x_s - x_center
        return torch.cat([x_s, x_center, diff, diff.abs()], dim=1)  # (B, 4F, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F, H, W)
        assert x.dim() == 5 and x.size(1) == self.S and x.size(2) == self.F, \
            f"RelationalSliceFusion expects (B,{self.S},{self.F},H,W), got {tuple(x.shape)}"

        x_center = x[:, self.center]  # (B, F, H, W)

        logits = torch.stack(
            [self.score(self._descriptor(x[:, s], x_center)) for s in range(self.S)],
            dim=1
        )  # (B, S, F, H, W)

        attn = torch.softmax(logits, dim=1)          # (B, S, F, H, W)
        fused = (attn * x).sum(dim=1)                 # (B, F, H, W) -- NO residual

        return fused

    def extra_repr(self) -> str:
        return f"num_features={self.F}, num_slices={self.S}, residual=False"


class MultiScaleRelationalFusion(nn.Module):
    """Apply a RelationalSliceFusion independently at each feature scale.

    Same call signature as MultiScaleSliceFusion / MultiScaleMeanFusion:
    forward input:  per_slice_features -- list of length S (slices); each entry
                    a list of length num_scales: [(B,F,H,W), (B,F,H/2,W/2), ...]
    forward output: list of length num_scales of fused (B,F,H,W) maps.
    """

    def __init__(self, feature_dims: List[int], num_slices: int = 3):
        super().__init__()
        self.num_scales = len(feature_dims)
        self.num_slices = num_slices
        self.fusers = nn.ModuleList([
            RelationalSliceFusion(num_features=f, num_slices=num_slices)
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
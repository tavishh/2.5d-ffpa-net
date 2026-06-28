"""
Post-filter slice fusion for the 2.5D FFPA-Net (variant S1).

After prev/curr/next are each filtered, each scale holds a stack of `num_slices`
feature groups sharing one feature axis: (B, S, F, H, W). SliceCSA fuses that
stack down the slice axis (S -> 1) while keeping the feature axis, so the decoder
is unchanged.

It reuses XAG-Net's CSA mechanism (1x1 conv logits -> softmax -> gating multiply
-> residual) with one adaptation: the softmax is taken over the SLICE axis and
summed to collapse S -> 1, instead of XAG's softmax over channels. With
shared_score=True the only learnable part is one 1x1 conv F->F (110 params at F=10).

Shapes:
    SliceCSA:              (B, S, F, H, W) -> (B, F, H, W)
    MultiScaleSliceFusion: list over S slices (each a list over scales) ->
                           list over scales of (B, F, H, W)
"""
from typing import List

import torch
import torch.nn as nn


class SliceCSA(nn.Module):
    """Cross-Slice Attention over the slice axis of post-filter features.

    Mechanism (XAG-style): a shared 1x1 conv F->F produces per-slice attention
    logits; softmax over the slice axis; gate the original features; sum over
    slices to collapse; add the centre slice as the residual.

    Args:
        num_features: feature channels F (10 for the fixed filter bank).
        num_slices:   stacked slices S (3 = prev/curr/next).
        residual_center: add the centre slice back as a residual (XAG's Add).
        shared_score: share one scoring conv across all slices (True) or use one
            conv per slice (False).
    """

    def __init__(self, num_features: int, num_slices: int = 3,
                 residual_center: bool = True, shared_score: bool = True):
        super().__init__()
        self.F = num_features
        self.S = num_slices
        self.center = num_slices // 2                 # prev=0, curr=1, next=2 -> 1
        self.residual_center = residual_center
        self.shared_score = shared_score

        if shared_score:
            # one Conv2d(F, F, 1x1) reused for every slice -> 110 params at F=10
            self.score = nn.Conv2d(num_features, num_features, kernel_size=1, padding=0)
        else:
            self.score = nn.ModuleList(
                [nn.Conv2d(num_features, num_features, 1, padding=0) for _ in range(num_slices)])

    def _logit(self, x_s: torch.Tensor, s: int) -> torch.Tensor:
        return self.score(x_s) if self.shared_score else self.score[s](x_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F, H, W)
        assert x.dim() == 5 and x.size(1) == self.S and x.size(2) == self.F, \
            f"SliceCSA expects (B,{self.S},{self.F},H,W), got {tuple(x.shape)}"

        # Per-slice attention logits via the (shared) 1x1 conv: (B,F,H,W) each.
        logits = torch.stack([self._logit(x[:, s], s) for s in range(self.S)], dim=1)  # (B,S,F,H,W)
        # Cross-slice softmax: for each (feature, pixel), distribute weight over slices.
        attn = torch.softmax(logits, dim=1)                                            # (B,S,F,H,W)

        # Gate the ORIGINAL features and collapse the slice axis.
        fused = (attn * x).sum(dim=1)                                                   # (B,F,H,W)

        if self.residual_center:
            fused = fused + x[:, self.center]                                           # XAG-style Add
        return fused


class MultiScaleSliceFusion(nn.Module):
    """Apply a SliceCSA independently at each feature scale.

    forward input:  per_slice_features -- list of length S (slices); each entry
                    a list of length num_scales: [(B,F,H,W), (B,F,H/2,W/2), ...]
    forward output: list of length num_scales of fused (B,F,H,W) maps.
    """

    def __init__(self, feature_dims: List[int], num_slices: int = 3,
                 residual_center: bool = True, shared_score: bool = True):
        super().__init__()
        self.num_scales = len(feature_dims)
        self.num_slices = num_slices
        self.fusers = nn.ModuleList([
            SliceCSA(num_features=f, num_slices=num_slices,
                     residual_center=residual_center, shared_score=shared_score)
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
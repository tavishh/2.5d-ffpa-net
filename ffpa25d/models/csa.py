"""
Cross-Slice Attention on the raw slice stack (variant S2: early fusion).

PyTorch port of XAG-Net's cross_slice_attention: a 1x1 conv (C->C) produces one
attention map per channel, softmax over channels, channel-wise gating, plus a
residual; the channel count is preserved. In S2 this runs on the raw
(B, 3, H, W) [prev, curr, next] stack before a single filter-bank pass on the
centre channel.
"""
import torch
import torch.nn as nn


class CrossSliceAttention(nn.Module):
    """XAG-Net cross-slice attention (channel softmax + residual).

    Args:
        in_channels: number of channels = number of slices at the input stage.
    Input/Output: (B, in_channels, H, W) -> (B, in_channels, H, W).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        # Conv2D(num_slices, 1x1, padding="same") in the original.
        self.att = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(1) == self.in_channels, \
            f"CrossSliceAttention expects (B,{self.in_channels},H,W), got {tuple(x.shape)}"
        att = torch.softmax(self.att(x), dim=1)   # softmax over the channel/slice axis
        weighted = x * att
        return x + weighted                       # residual (Add)
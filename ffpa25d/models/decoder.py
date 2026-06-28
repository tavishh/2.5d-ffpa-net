"""
Shared decoder components for FFPA-Net.

Reused unchanged from the original 2D FFPA-Net; only the upstream slice-feature
fusion differs. Deep-supervision output is controlled by an explicit `return_aux`
argument (True during training, False during validation / inference) rather than
a stored flag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 conv layers (original FFPA-Net)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class AttentionGate(nn.Module):
    """Attention gate for feature refinement (original FFPA-Net)."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ProgressiveAttentionDecoder(nn.Module):
    """Progressive coarse-to-fine decoder with attention gates and deep supervision.

    Identical in structure to the original FFPA-Net decoder. Takes a list of
    three feature tensors [scale1(full), scale2(1/2), scale3(1/4)], each with
    `in_channels[i]` channels, and produces a `num_classes`-channel logit map at
    full resolution.

    Args:
        in_channels: list of channel counts per scale, e.g. [10, 10, 10].
        base_channels: base width c1 (doubles at each coarser scale).
        num_classes: number of output segmentation classes.
        use_attention: if False, skip attention gates.
        use_deep_supervision: if True, build the auxiliary heads. Whether they
            are actually returned is controlled per-call by `return_aux`.
    """

    def __init__(
        self,
        in_channels=(10, 10, 10),
        base_channels: int = 64,
        num_classes: int = 1,
        use_attention: bool = True,
        use_deep_supervision: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_deep_supervision = use_deep_supervision

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 2
        c5 = base_channels
        c6 = base_channels // 2

        self.adapt3 = self._adapt_block(in_channels[2], c3)
        self.adapt2 = self._adapt_block(in_channels[1], c2)
        self.adapt1 = self._adapt_block(in_channels[0], c1)

        self.res3 = nn.Sequential(ResidualBlock(c3), ResidualBlock(c3))
        self.res2 = nn.Sequential(ResidualBlock(c2), ResidualBlock(c2))
        self.res1 = nn.Sequential(ResidualBlock(c1), ResidualBlock(c1))

        if use_attention:
            self.att2 = AttentionGate(c3, c2, c2 // 2)
            self.att1 = AttentionGate(c4, c1, c1 // 2)

        self.aggregate2 = nn.Sequential(
            nn.Conv2d(c3 + c2, c4, 3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            ResidualBlock(c4),
        )
        self.aggregate1 = nn.Sequential(
            nn.Conv2d(c4 + c1, c5, 3, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
            ResidualBlock(c5),
        )

        self.refine1 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5, c6, 3, padding=1),
            nn.BatchNorm2d(c6),
            nn.ReLU(inplace=True),
        )
        self.refine2 = nn.Sequential(
            ResidualBlock(c6),
            nn.Conv2d(c6, c6, 3, padding=1),
            nn.BatchNorm2d(c6),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(c6, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1),
        )

        if use_deep_supervision:
            self.deep_sup2 = nn.Conv2d(c4, num_classes, 1)
            self.deep_sup1 = nn.Conv2d(c5, num_classes, 1)

    @staticmethod
    def _adapt_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, features, return_aux: bool = False):
        """
        Args:
            features: list [scale1, scale2, scale3] of (B, C, H', W') tensors.
            return_aux: if True and deep supervision is built, also return the
                        two auxiliary outputs (output, ds1, ds2).

        Returns:
            output tensor, or (output, ds1, ds2) when return_aux and DS enabled.
        """
        x3 = self.res3(self.adapt3(features[2]))
        x2 = self.res2(self.adapt2(features[1]))
        x1 = self.res1(self.adapt1(features[0]))

        # Stage 1: aggregate scale 3 (up) with scale 2
        x3_up = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x2_att = self.att2(x3_up, x2) if self.use_attention else x2
        agg2 = self.aggregate2(torch.cat([x3_up, x2_att], dim=1))

        # Stage 2: aggregate agg2 (up) with scale 1
        agg2_up = F.interpolate(agg2, scale_factor=2, mode="bilinear", align_corners=False)
        x1_att = self.att1(agg2_up, x1) if self.use_attention else x1
        agg1 = self.aggregate1(torch.cat([agg2_up, x1_att], dim=1))

        output = self.output(self.refine2(self.refine1(agg1)))

        if self.use_deep_supervision and return_aux:
            ds2 = self.deep_sup2(agg2)
            ds2 = F.interpolate(ds2, scale_factor=2, mode="bilinear", align_corners=False)
            ds1 = self.deep_sup1(agg1)
            return output, ds1, ds2

        return output
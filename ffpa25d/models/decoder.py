"""
Shared decoder components for FFPA-Net.

Reused unchanged from the original 2D FFPA-Net; only the upstream slice-feature
fusion differs. Deep-supervision output is controlled by an explicit `return_aux`
argument (True during training, False during validation / inference) rather than
a stored flag.

Decoder-level modifications (2026-09, branch: decoder-experiments):
Motivated by the failure-zone diagnosis -- near-total segmentation failures
shared across every independently-designed fusion mechanism tested so far
(S1, fusion_only, relational_fusion, the position gate), pointing to a
bottleneck downstream of the fusion stage entirely, i.e. in this shared
decoder. Tracing the actual attention-gate mechanism: the gate at the FINEST
scale (att1) is conditioned on agg2_up, which is derived from x3 -- the
COARSEST (64x64) representation. A thin, tapering structure that is lost
during that coarse downsampling has no way to "tell" the gate it is still
present in the finer scale's features, so the gate may multiplicatively
suppress genuine fine-detail signal it never had visibility into losing.

Two independent, opt-in fixes are added, each defaulting to EXACTLY the
original behaviour when disabled, so every existing config/checkpoint is
completely unaffected:

  1. use_gate_floor / gate_floor: puts a floor under the attention gate's
     output so it can attenuate but never fully zero out a location
     (psi = floor + (1-floor)*psi). floor=0.0 (default) reproduces the
     original gate exactly.

  2. use_fine_skip: adds a direct path from the RAW finest-scale features
     (x1, captured BEFORE attention gating) to just before final refinement,
     via a small learned 1x1 projection -- so survival of fine detail does
     not depend entirely on passing through the gated coarse-to-fine chain.
     Disabled by default; adds one small learnable conv only when enabled.

These are two SEPARATE, independently-testable changes -- per this project's
"isolate one variable" methodology, they are never combined with each other
or with any fusion-mechanism/slice-consistency experiment in a single run
without first understanding each in isolation.
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
    """Attention gate for feature refinement (original FFPA-Net).

    Args:
        F_g, F_l, F_int: as original.
        gate_floor: minimum pass-through fraction, in [0, 1). 0.0 (default)
            reproduces the ORIGINAL gate exactly -- psi can range fully
            [0, 1], including zeroing out a location entirely. A value like
            0.1 guarantees at least 10% of x's signal always survives the
            gate, regardless of how confidently the gating signal suppresses
            that location. This is the ONLY change from the original
            AttentionGate; everything else is untouched.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int, gate_floor: float = 0.0):
        super().__init__()
        assert 0.0 <= gate_floor < 1.0, f"gate_floor must be in [0,1), got {gate_floor}"
        self.gate_floor = gate_floor
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
        if self.gate_floor > 0.0:
            # Floors psi so it can attenuate (down to gate_floor) but never
            # fully zero out a location -- e.g. floor=0.1: psi in [0,1] maps
            # to [0.1, 1.0]. When gate_floor=0.0 this line is skipped
            # entirely, so the original behaviour is bit-for-bit unchanged.
            psi = self.gate_floor + (1.0 - self.gate_floor) * psi
        return x * psi


class ProgressiveAttentionDecoder(nn.Module):
    """Progressive coarse-to-fine decoder with attention gates and deep supervision.

    Identical in structure to the original FFPA-Net decoder, plus two
    independent, opt-in decoder-level modifications (see module docstring).
    Both default to exactly the original behaviour when disabled.

    Args:
        in_channels: list of channel counts per scale, e.g. [10, 10, 10].
        base_channels: base width c1 (doubles at each coarser scale).
        num_classes: number of output segmentation classes.
        use_attention: if False, skip attention gates.
        use_deep_supervision: if True, build the auxiliary heads. Whether they
            are actually returned is controlled per-call by `return_aux`.
        gate_floor: see AttentionGate. 0.0 (default) = original behaviour.
        use_fine_skip: if True, add a direct raw-finest-scale skip to the
            output path (see module docstring). False (default) = original
            behaviour, no extra parameters.
    """

    def __init__(
        self,
        in_channels=(10, 10, 10),
        base_channels: int = 64,
        num_classes: int = 1,
        use_attention: bool = True,
        use_deep_supervision: bool = True,
        gate_floor: float = 0.0,
        use_fine_skip: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_deep_supervision = use_deep_supervision
        self.use_fine_skip = use_fine_skip

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
            self.att2 = AttentionGate(c3, c2, c2 // 2, gate_floor=gate_floor)
            self.att1 = AttentionGate(c4, c1, c1 // 2, gate_floor=gate_floor)

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

        # Fine-detail skip: a single 1x1 conv projecting x1 (the RAW finest-
        # scale features, c1 channels, captured BEFORE attention gating) to
        # c5 channels, so it can be added directly into agg1. Only allocated
        # when enabled -- adds zero parameters otherwise.
        if use_fine_skip:
            self.fine_skip_proj = nn.Conv2d(c1, c5, 1)

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

        # Fine-detail skip (opt-in): adds the RAW x1 (pre-gating, so it was
        # never at risk of being zeroed by att1) directly into agg1, via a
        # small learned projection. When disabled, this line never executes,
        # so agg1 is bit-for-bit identical to the original decoder's value.
        if self.use_fine_skip:
            agg1 = agg1 + self.fine_skip_proj(x1)

        output = self.output(self.refine2(self.refine1(agg1)))

        if self.use_deep_supervision and return_aux:
            ds2 = self.deep_sup2(agg2)
            ds2 = F.interpolate(ds2, scale_factor=2, mode="bilinear", align_corners=False)
            ds1 = self.deep_sup1(agg1)
            return output, ds1, ds2

        return output
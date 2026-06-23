"""FFPA-Net: Fixed-Filter Progressive Attention Network.

2.5D extension: FFPANet now accepts a `variant` config key that controls which
CSA insertions are active. This single file covers all 5 ablation variants:

    variant 1: baseline_2d        - original FFPA-Net, no changes
    variant 2: fusion_only        - 3-slice input, NO CSA anywhere
    variant 3: csa_input_only     - 3-slice input + CSA at input stage only
    variant 4: csa_decoder_only   - 3-slice input + CSA in decoder only  [PRIMARY]
    variant 5: csa_full           - 3-slice input + CSA at input AND decoder

The two CSA insertion points are:
  - Input stage (FFPANet.forward): applied to the raw 3-channel slice stack
    before it enters the fixed filter bank.
  - Decoder stage (ProgressiveAttentionDecoder): applied to x2 and x1 (the
    finer-scale adapted features) just before each attention gate. This is
    the primary ablation target.

Channel counts at each decoder stage are UNCHANGED by CSA insertion because
CrossSliceAttention is an in-place residual operation (output shape = input
shape). No changes to aggregate2 or aggregate1 input dimensions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .csa import CrossSliceAttention
from .fixed_filters import FixedFilterBank

# ── Variant routing table ─────────────────────────────────────────────────────
# Maps the variant name from config to (use_csa_input, use_csa_decoder) flags.
VARIANT_FLAGS = {
    "baseline_2d":      (False, False),
    "fusion_only":      (False, False),
    "csa_input_only":   (True,  False),
    "csa_decoder_only": (False, True),
    "csa_full":         (True,  True),
}


class ResidualBlock(nn.Module):
    def __init__(self, channels):
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
        return self.relu(out + residual)


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.w_g(g) + self.w_x(x))
        return x * self.psi(psi)


class ProgressiveAttentionDecoder(nn.Module):
    """Progressive coarse-to-fine decoder with optional CSA at aggregation points.

    Args:
        in_channels: tuple of (scale1_ch, scale2_ch, scale3_ch) from filter bank
        base_channels: base channel width (c1). Doubles at each coarser scale.
        num_classes: number of output segmentation classes
        use_csa_decoder (bool): if True, insert CrossSliceAttention on x2 and x1
            before the attention gate at each aggregation step. Default False.

    CSA insertion points (when use_csa_decoder=True):
        Before att2(x3_up, x2):  x2 = self.csa2(x2)
        Before att1(agg2_up, x1): x1 = self.csa1(x1)

    This placement applies inter-channel reweighting to the finer-scale skip
    features before the attention gate decides what to pass forward, giving
    CSA a chance to suppress irrelevant channels prior to gating.
    """

    def __init__(
        self,
        in_channels=(10, 10, 10),
        base_channels=64,
        num_classes=1,
        use_csa_decoder=False,
    ):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        c4, c5, c6 = base_channels * 2, base_channels, base_channels // 2

        self.use_csa_decoder = use_csa_decoder

        self.adapt3 = self._adapt_block(in_channels[2], c3)
        self.adapt2 = self._adapt_block(in_channels[1], c2)
        self.adapt1 = self._adapt_block(in_channels[0], c1)

        self.res3 = nn.Sequential(ResidualBlock(c3), ResidualBlock(c3))
        self.res2 = nn.Sequential(ResidualBlock(c2), ResidualBlock(c2))
        self.res1 = nn.Sequential(ResidualBlock(c1), ResidualBlock(c1))

        self.att2 = AttentionGate(c3, c2, c2 // 2)
        self.att1 = AttentionGate(c2, c1, c1 // 2)

        # CSA modules for decoder insertion points.
        # Only instantiated when use_csa_decoder=True to keep the baseline's
        # parameter count exactly unchanged.
        if use_csa_decoder:
            self.csa2 = CrossSliceAttention(in_channels=c2)  # (B, 128, 128, 128)
            self.csa1 = CrossSliceAttention(in_channels=c1)  # (B,  64, 256, 256)

        # Aggregate blocks - input dims are UNCHANGED by CSA (residual, same shape)
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

    def forward(self, features, use_deep_supervision=False):
        x3 = self.res3(self.adapt3(features[2]))
        x2 = self.res2(self.adapt2(features[1]))
        x1 = self.res1(self.adapt1(features[0]))

        # ── Optional CSA on finer-scale features before each attention gate ──
        if self.use_csa_decoder:
            x2 = self.csa2(x2)
            x1 = self.csa1(x1)

        x3_up = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        agg2 = self.aggregate2(torch.cat([x3_up, self.att2(x3_up, x2)], dim=1))

        agg2_up = F.interpolate(agg2, scale_factor=2, mode="bilinear", align_corners=False)
        agg1 = self.aggregate1(torch.cat([agg2_up, self.att1(agg2_up, x1)], dim=1))

        output = self.output(self.refine2(self.refine1(agg1)))

        if use_deep_supervision:
            ds2 = F.interpolate(
                self.deep_sup2(agg2), scale_factor=2, mode="bilinear", align_corners=False
            )
            ds1 = self.deep_sup1(agg1)
            return output, ds1, ds2
        return output


class FFPANet(nn.Module):
    """FFPA-Net with optional 2.5D extensions.

    Variant is controlled via config["ffpa"]["variant"], which must be one of:
        "baseline_2d"      - original 2D FFPA-Net (single-channel input, no CSA)
        "fusion_only"      - 3-slice input, no CSA
        "csa_input_only"   - 3-slice input, CSA at input stage only
        "csa_decoder_only" - 3-slice input, CSA in decoder only  [PRIMARY]
        "csa_full"         - 3-slice input, CSA at both stages

    Input channel handling:
        baseline_2d:  expects (B, 1, H, W) - single slice
        all others:   expects (B, 3, H, W) - 3-slice stack from dataset

    The fixed filter bank (FixedFilterBank) accepts any single-channel input.
    For 3-slice variants, we extract filter features from the CENTRE slice only
    (channel index 1) so that the filter bank behaviour is identical to the
    baseline. The CSA at input (when active) operates on the full 3-channel
    stack BEFORE centre-slice extraction.

    original_fusion layer:
        baseline_2d:  concatenates 1-channel raw image with 10 filter features
                      -> 11 channels -> fused to 10  (unchanged from baseline)
        3-slice variants: concatenates 1-channel centre slice with 10 filter
                      features -> 11 channels -> fused to 10  (same fusion layer)
    """

    def __init__(self, config, base_channels=64):
        super().__init__()
        self.num_classes = config["model"].get(
            "actual_num_classes", config["model"].get("num_classes", 1)
        )
        self.class_info = config["model"].get("class_info")

        ffpa_cfg = config.get("ffpa", {})
        self.use_original = ffpa_cfg.get("use_original_image", True)
        self.use_deep_supervision = ffpa_cfg.get("use_deep_supervision", True)
        variant = ffpa_cfg.get("variant", "baseline_2d")

        if variant not in VARIANT_FLAGS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from: {list(VARIANT_FLAGS.keys())}"
            )

        use_csa_input, use_csa_decoder = VARIANT_FLAGS[variant]
        self.use_csa_input = use_csa_input
        self.variant = variant
        # 3-slice input is needed for all non-baseline variants
        self.three_slice = variant != "baseline_2d"

        print(f"  Initializing FFPANet variant='{variant}' with {self.num_classes} output classes")
        print(f"    3-slice input: {self.three_slice}")
        print(f"    CSA at input:  {use_csa_input}")
        print(f"    CSA in decoder: {use_csa_decoder}")

        # ── Optional CSA at input stage ───────────────────────────────────────
        # Operates on the raw 3-channel stack (B, 3, H, W) before the filter
        # bank. Only instantiated for csa_input_only and csa_full variants.
        if use_csa_input:
            self.csa_input = CrossSliceAttention(in_channels=3)

        self.feature_extractor = FixedFilterBank(
            num_scales=3,
            device=config["training"]["device"],
        )
        self.decoder = ProgressiveAttentionDecoder(
            in_channels=[10, 10, 10],
            base_channels=base_channels,
            num_classes=self.num_classes,
            use_csa_decoder=use_csa_decoder,
        )

        if self.use_original:
            # Always fuses centre slice (1 channel) + filter features (10 channels)
            self.original_fusion = nn.Sequential(
                nn.Conv2d(11, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 10, 1),
            )

    def forward(self, x, return_aux=False):
        """
        Args:
            x: (B, 1, H, W) for baseline_2d, or (B, 3, H, W) for all other variants
            return_aux: if True and deep supervision is enabled, return tuple of outputs
        """
        if self.three_slice:
            # ── Optional CSA on the 3-channel stack ──────────────────────────
            if self.use_csa_input:
                x = self.csa_input(x)  # (B, 3, H, W) -> (B, 3, H, W)

            # Extract filter features from the CENTRE slice (index 1).
            # The filter bank expects (B, 1, H, W) or (B, H, W).
            centre = x[:, 1:2, :, :]  # (B, 1, H, W)
        else:
            centre = x  # (B, 1, H, W) - baseline passthrough

        with torch.no_grad():
            filter_features = self.feature_extractor.extract_features(centre)

        if self.use_original:
            combined = torch.cat([filter_features[0], centre], dim=1)  # (B, 11, H, W)
            filter_features[0] = self.original_fusion(combined)

        use_ds = return_aux and self.use_deep_supervision
        return self.decoder(filter_features, use_deep_supervision=use_ds)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
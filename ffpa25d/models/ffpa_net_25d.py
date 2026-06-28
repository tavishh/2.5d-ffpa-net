"""
2.5D FFPA-Net (FFPANet25D).

The only change relative to the 2D FFPA-Net is WHERE adjacent-slice information
is fused; everything downstream (filter bank, decoder, deep supervision,
original-image fusion) is identical. The fixed single-channel filter bank
destroys the slice axis once a centre slice is chosen, while XAG-Net's CSA needs
the channel axis to be the slice axis. The one window where features exist and
the slice axis still exists is immediately after per-slice filtering, so fusion
(SliceCSA) happens there and the decoder needs no change.

Variants (config["ffpa"]["fusion_variant"]):
    S0_single_slice      : 2D baseline. Input (B,1,H,W), no neighbours.
    S1_postfilter_fusion : primary. Input (B,3,H,W); each slice filtered, then a
                           per-scale SliceCSA fuses 3 -> 1 along the slice axis.
    S2_input_fusion      : cheap reference. CSA over the raw 3-channel stack
                           before a single filter-bank pass (early fusion).

Cost: S1 runs the zero-parameter filter bank 3x per forward; S0/S2 run it once.
SliceCSA is ~110 params per scale at F=10.
"""

import torch
import torch.nn as nn

from .fixed_filters import FixedFilterBank
from .slice_fusion import MultiScaleSliceFusion
from .csa import CrossSliceAttention
from .decoder import ProgressiveAttentionDecoder


# Maps fusion_variant -> (three_slice_input, use_postfilter_fusion, use_input_fusion)
FUSION_VARIANTS = {
    "S0_single_slice":      (False, False, False),
    "S1_postfilter_fusion": (True,  True,  False),
    "S2_input_fusion":      (True,  False, True),
}


class FFPANet25D(nn.Module):
    """2.5D FFPA-Net with selectable slice-fusion strategy.

    Args:
        config: experiment config dict. Reads:
            config["model"]["actual_num_classes"] (or ["num_classes"])
            config["model"]["class_info"]                (optional)
            config["ffpa"]["fusion_variant"]             (default S0_single_slice)
            config["ffpa"]["use_original_image"]         (default True)
            config["ffpa"]["use_deep_supervision"]       (default True)
            config["training"]["device"]
        base_channels: decoder base width (config typically sets 40).
    """

    def __init__(self, config, base_channels: int = 64):
        super().__init__()

        self.num_classes = config["model"].get(
            "actual_num_classes", config["model"].get("num_classes", 1)
        )
        self.class_info = config["model"].get("class_info", None)

        ffpa_cfg = config.get("ffpa", {})
        self.fusion_variant = ffpa_cfg.get("fusion_variant", "S0_single_slice")
        self.use_original = ffpa_cfg.get("use_original_image", True)
        self.use_deep_supervision = ffpa_cfg.get("use_deep_supervision", True)

        if self.fusion_variant not in FUSION_VARIANTS:
            raise ValueError(
                f"Unknown fusion_variant '{self.fusion_variant}'. "
                f"Choose from: {list(FUSION_VARIANTS.keys())}"
            )
        three_slice, use_postfilter, use_input = FUSION_VARIANTS[self.fusion_variant]
        self.three_slice = three_slice
        self.use_postfilter_fusion = use_postfilter
        self.use_input_fusion = use_input
        self.num_slices = 3 if three_slice else 1

        print(f"  Initializing FFPANet25D variant='{self.fusion_variant}' "
              f"with {self.num_classes} output classes")
        print(f"    3-slice input:        {self.three_slice}")
        print(f"    post-filter fusion:   {self.use_postfilter_fusion}  (SliceCSA)")
        print(f"    input fusion:         {self.use_input_fusion}  (raw-stack CSA)")

        # ── Fixed filter bank (single-channel, unchanged from 2D FFPA-Net) ──
        self.feature_extractor = FixedFilterBank(
            num_scales=3,
            device=config["training"]["device"],
        )
        feature_dims = [10, 10, 10]

        # ── S1: per-scale SliceCSA that fuses 3 slice-feature groups -> 1 ───
        if self.use_postfilter_fusion:
            self.slice_fusion = MultiScaleSliceFusion(
                feature_dims=feature_dims,
                num_slices=3,
                residual_center=True,
                shared_score=True,
            )

        # ── S2: CSA over the raw 3-channel slice stack (early fusion) ───────
        if self.use_input_fusion:
            self.csa_input = CrossSliceAttention(in_channels=3)

        # ── Decoder (identical to 2D FFPA-Net) ─────────────────────────────
        self.decoder = ProgressiveAttentionDecoder(
            in_channels=feature_dims,
            base_channels=base_channels,
            num_classes=self.num_classes,
            use_attention=True,
            use_deep_supervision=self.use_deep_supervision,
        )

        # ── Original-image fusion: centre slice (1ch) + scale-0 feats (10ch) ─
        if self.use_original:
            self.original_fusion = nn.Sequential(
                nn.Conv2d(11, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 10, 1),
            )

    # ── helpers ────────────────────────────────────────────────────────────
    def _filter_one(self, single_channel):
        """Run the fixed (no-grad) filter bank on one (B,1,H,W) slice."""
        with torch.no_grad():
            return self.feature_extractor.extract_features(single_channel)

    def _maybe_fuse_original(self, features, centre):
        """Fuse the centre slice into scale-0 features (in place on the list)."""
        if self.use_original:
            combined = torch.cat([features[0], centre], dim=1)  # (B, 11, H, W)
            features[0] = self.original_fusion(combined)
        return features

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x, return_aux: bool = False):
        """
        Args:
            x: (B, 1, H, W) for S0; (B, 3, H, W) for S1/S2 (prev, curr, next).
            return_aux: if True and deep supervision enabled, return tuple.
        """
        if self.three_slice:
            self._check_3ch(x)

            if self.use_input_fusion:
                # ── S2: mix neighbours into channels, then ONE filter pass ──
                x = self.csa_input(x)                # (B, 3, H, W)
                centre = x[:, 1:2, :, :]             # (B, 1, H, W)
                features = self._filter_one(centre)
                features = self._maybe_fuse_original(features, centre)

            else:
                # ── S1: filter EACH slice, then SliceCSA fuses 3 -> 1 ───────
                per_slice_features = [
                    self._filter_one(x[:, s:s + 1, :, :]) for s in range(3)
                ]
                features = self.slice_fusion(per_slice_features)  # list of 3 scales
                centre = x[:, 1:2, :, :]
                features = self._maybe_fuse_original(features, centre)
        else:
            # ── S0: original 2D path ───────────────────────────────────────
            centre = x                               # (B, 1, H, W)
            features = self._filter_one(centre)
            features = self._maybe_fuse_original(features, centre)

        use_ds = return_aux and self.use_deep_supervision
        return self.decoder(features, return_aux=use_ds)

    def _check_3ch(self, x):
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(
                f"fusion_variant='{self.fusion_variant}' expects a 3-channel "
                f"(prev,curr,next) input of shape (B,3,H,W), but got "
                f"{tuple(x.shape)}. Set data.three_slice=True so the dataset "
                f"returns stacked slices."
            )

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_class_mapping(self):
        return self.class_info
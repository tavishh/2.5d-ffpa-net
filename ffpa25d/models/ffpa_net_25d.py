"""
2.5D FFPA-Net (FFPANet25D).

The only change relative to the 2D FFPA-Net is WHERE adjacent-slice information
is fused; everything downstream (filter bank, decoder, deep supervision,
original-image fusion) is identical. The fixed single-channel filter bank
destroys the slice axis once a centre slice is chosen, while XAG-Net's CSA needs
the channel axis to be the slice axis. The one window where features exist and
the slice axis still exists is immediately after per-slice filtering, so fusion
happens there and the decoder needs no change.

Variants (config["ffpa"]["fusion_variant"]):
    S0_single_slice      : 2D baseline. Input (B,1,H,W), no neighbours.
    S1_postfilter_fusion : each slice filtered, then a per-scale SliceCSA fuses
                           N -> 1 along the slice axis via INDEPENDENT per-slice
                           softmax attention (no direct slice-to-slice comparison).
    S2_input_fusion      : cheap reference. CSA over the raw N-channel stack
                           before a single filter-bank pass (early fusion).
    fusion_only          : ABLATION CONTROL for S1. Identical forward path to
                           S1 but slice-axis fusion uses FIXED uniform (1/N)
                           weights (MeanSliceFusion) instead of SliceCSA's
                           learned softmax. Zero extra parameters.
    relational_fusion    : REDESIGN. Same forward path as S1/fusion_only, but
                           each slice's attention logit is computed from an
                           explicit RELATIONAL descriptor referenced against the
                           centre slice (x_s, x_centre, x_s-x_centre, |x_s-x_centre|)
                           via a shared 3x3 conv, with NO residual.

Number of slices (config["data"]["num_slices"], default 3): controls how many
neighbouring slices get stacked and fused for any of the multi-slice variants
above (S1, S2, fusion_only, relational_fusion). Must be odd. This is decoupled
from fusion_variant so any slice-count can be combined with any fusion
mechanism without adding new variant names.

Positional encoding (config["data"]["use_position_encoding"], default False,
added 2026-09, corrected after reading the source paper): reproduces Henson et
al.'s SC-UNet mechanism (PLOS ONE, 2024) precisely, NOT an approximation:
  - the dataset computes a percentage-along-the-limb bucket (0-99) for each
    slice (see dataset.py's _get_position_bucket)
  - this model converts that bucket to a 100-element one-hot vector, passes it
    through a SEPARATE, small fully-connected layer (100 -> num_classes) --
    running in PARALLEL to the segmentation backbone, not mixed into the image
    input at any point
  - the FC layer's output (one scalar per class) multiplicatively GATES the
    decoder's final output (and every deep-supervision auxiliary output, if
    enabled), channel-wise -- exactly as described in the paper ("multiplied
    with the result of the final convolutional layer")
An earlier version of this feature concatenated a constant-valued position
channel into the image input instead; that was a plausible-sounding guess
made before the source paper was available, not a reproduction of the actual
method, and has been replaced by this version.

Cost: S1, fusion_only, and relational_fusion all run the zero-parameter filter
bank N times per forward (N = num_slices); S0/S2 run it once. SliceCSA is
~110 params/scale regardless of N (shared_score). MeanSliceFusion: 0 params.
RelationalSliceFusion: ~3610 params/scale (3x3 conv, 4F->F channels),
independent of N. Positional encoding adds a single Linear(100, num_classes)
layer -- negligible parameter/compute cost regardless of num_classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fixed_filters import FixedFilterBank
from .slice_fusion import MultiScaleSliceFusion
from .mean_fusion import MultiScaleMeanFusion
from .relational_fusion import MultiScaleRelationalFusion
from .csa import CrossSliceAttention
from .decoder import ProgressiveAttentionDecoder


# Maps fusion_variant -> (three_slice_input, use_postfilter_fusion, use_input_fusion)
# fusion_only and relational_fusion share S1's forward path (three_slice=True,
# use_postfilter=True) -- the only difference is which class gets instantiated
# for self.slice_fusion, handled below in __init__, not here. The actual
# NUMBER of slices (3, 5, ...) and positional encoding are separate, orthogonal
# config keys read directly in __init__, not part of this dict.
FUSION_VARIANTS = {
    "S0_single_slice":      (False, False, False),
    "S1_postfilter_fusion": (True,  True,  False),
    "S2_input_fusion":      (True,  False, True),
    "fusion_only":          (True,  True,  False),
    "relational_fusion":    (True,  True,  False),
}

# Variants that use the fixed-weight MeanSliceFusion instead of learned SliceCSA
# at the post-filter fusion stage.
_MEAN_FUSION_VARIANTS = {"fusion_only"}

# Variants that use the relational (centre-referenced) fusion instead of
# SliceCSA's independent-per-slice scoring.
_RELATIONAL_FUSION_VARIANTS = {"relational_fusion"}

# Number of position buckets for the SC-UNet-style gate. Matches the paper's
# "100 input neurons representing each percentage along the lower limb".
_NUM_POSITION_BUCKETS = 100


class FFPANet25D(nn.Module):
    """2.5D FFPA-Net with selectable slice-fusion strategy, slice count, and
    optional SC-UNet-style positional gating.

    Args:
        config: experiment config dict. Reads:
            config["model"]["actual_num_classes"] (or ["num_classes"])
            config["model"]["class_info"]                (optional)
            config["ffpa"]["fusion_variant"]             (default S0_single_slice)
            config["ffpa"]["use_original_image"]         (default True)
            config["ffpa"]["use_deep_supervision"]       (default True)
            config["data"]["num_slices"]                 (default 3, must be odd)
            config["data"]["use_position_encoding"]      (default False)
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

        # Number of slices is a separate, orthogonal config key (defaults to 3
        # so every existing config that doesn't set it behaves exactly as
        # before). Only meaningful when three_slice=True.
        requested_num_slices = config.get("data", {}).get("num_slices", 3)
        if self.three_slice:
            if requested_num_slices < 1 or requested_num_slices % 2 == 0:
                raise ValueError(
                    f"data.num_slices must be odd and >= 1, got {requested_num_slices}"
                )
            self.num_slices = requested_num_slices
        else:
            self.num_slices = 1
        self.center_idx = self.num_slices // 2  # e.g. 3->1, 5->2, 7->3

        # Positional encoding (SC-UNet style): another separate, orthogonal
        # config key (defaults to False, so every existing config is
        # unaffected). Unlike num_slices, this does NOT touch the image input
        # at all -- it is a fully parallel pathway (see forward()).
        self.use_position = config.get("data", {}).get("use_position_encoding", False)
        self._warned_missing_position = False  # see forward(): one-time warning, not a hard error

        # Which class implements post-filter fusion for this variant.
        self._uses_mean_fusion = self.fusion_variant in _MEAN_FUSION_VARIANTS
        self._uses_relational_fusion = self.fusion_variant in _RELATIONAL_FUSION_VARIANTS
        if self._uses_mean_fusion:
            fusion_label = "MeanSliceFusion (fixed 1/N, 0 params, ablation control)"
        elif self._uses_relational_fusion:
            fusion_label = "RelationalSliceFusion (centre-referenced, learned, no residual)"
        else:
            fusion_label = "SliceCSA (learned, independent per-slice scoring)"

        print(f"  Initializing FFPANet25D variant='{self.fusion_variant}' "
              f"with {self.num_classes} output classes")
        print(f"    3-slice input:        {self.three_slice}"
              + (f"  (num_slices={self.num_slices})" if self.three_slice else ""))
        print(f"    post-filter fusion:   {self.use_postfilter_fusion}  ({fusion_label})")
        print(f"    input fusion:         {self.use_input_fusion}  (raw-stack CSA)")
        print(f"    position gate:        {self.use_position}"
              + ("  (SC-UNet style, 100-bucket -> Linear -> multiplicative gate)"
                 if self.use_position else ""))

        # ── Fixed filter bank (single-channel, unchanged from 2D FFPA-Net) ──
        self.feature_extractor = FixedFilterBank(
            num_scales=3,
            device=config["training"]["device"],
        )
        feature_dims = [10, 10, 10]

        # ── S1 / fusion_only / relational_fusion: fuse N slice-feature groups
        # -> 1 per scale. Same call signature for all three classes:
        # (per_slice_features) -> list of 3 fused (B,F,H,W) maps. forward()
        # below does not need to know which class this is. All three classes
        # are already parameterized by num_slices, so no change needed there.
        if self.use_postfilter_fusion:
            if self._uses_mean_fusion:
                self.slice_fusion = MultiScaleMeanFusion(
                    feature_dims=feature_dims,
                    num_slices=self.num_slices,
                    residual_center=True,   # must match S1 for a valid control
                )
            elif self._uses_relational_fusion:
                self.slice_fusion = MultiScaleRelationalFusion(
                    feature_dims=feature_dims,
                    num_slices=self.num_slices,
                )
            else:
                self.slice_fusion = MultiScaleSliceFusion(
                    feature_dims=feature_dims,
                    num_slices=self.num_slices,
                    residual_center=True,
                    shared_score=True,
                )

        # ── S2: CSA over the raw N-channel slice stack (early fusion) ───────
        if self.use_input_fusion:
            self.csa_input = CrossSliceAttention(in_channels=self.num_slices)

        # ── Decoder (identical to 2D FFPA-Net, untouched by position gating) ─
        self.decoder = ProgressiveAttentionDecoder(
            in_channels=feature_dims,
            base_channels=base_channels,
            num_classes=self.num_classes,
            use_attention=True,
            use_deep_supervision=self.use_deep_supervision,
        )

        # ── Original-image fusion: centre slice (1ch) + scale-0 feats (10ch) ─
        # Unaffected by positional encoding -- always 11 channels, since
        # position no longer touches the image/feature pathway at all.
        if self.use_original:
            self.original_fusion = nn.Sequential(
                nn.Conv2d(11, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 10, 1),
            )

        # ── SC-UNet-style position gate: 100-bucket one-hot -> Linear ->
        # one scalar per output class, later multiplied into the decoder's
        # output(s). This is the ENTIRE extra parameter cost of this feature.
        if self.use_position:
            self.position_gate = nn.Linear(_NUM_POSITION_BUCKETS, self.num_classes)

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

    def _apply_position_gate(self, output, position):
        """Multiplicatively gate a (B, num_classes, H, W) tensor by the
        SC-UNet-style per-class position score. Broadcasts over H, W.

        Args:
            output: (B, num_classes, H, W) -- decoder output or an aux head.
            position: (B,) long tensor, bucket indices in [0, 99].
        """
        position_onehot = F.one_hot(position, num_classes=_NUM_POSITION_BUCKETS).float()  # (B,100)
        gate = self.position_gate(position_onehot)          # (B, num_classes)
        gate = gate.view(gate.size(0), gate.size(1), 1, 1)   # (B, num_classes, 1, 1)
        return output * gate

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x, position=None, return_aux: bool = False):
        """
        Args:
            x: (B, num_slices, H, W). NOT affected by positional encoding --
               position travels as a fully separate argument (matching
               SC-UNet's genuinely parallel pathway), not packed into x.
            position: (B,) long tensor of bucket indices in [0, 99], required
                if use_position_encoding=True (ignored otherwise, with a
                warning if the config expected it but it was not provided).
            return_aux: if True and deep supervision enabled, return tuple.
        """
        if self.three_slice:
            self._check_slice_count(x)

            if self.use_input_fusion:
                # ── S2: mix neighbours into channels, then ONE filter pass ──
                x = self.csa_input(x)                                       # (B, N, H, W)
                centre = x[:, self.center_idx:self.center_idx + 1, :, :]    # (B, 1, H, W)
                features = self._filter_one(centre)
                features = self._maybe_fuse_original(features, centre)

            else:
                # ── S1 / fusion_only / relational_fusion: filter EACH slice,
                # then fuse N -> 1. self.slice_fusion is SliceCSA-based (S1),
                # MeanSliceFusion-based (fusion_only), or RelationalSliceFusion
                # -based (relational_fusion) -- identical call here.
                per_slice_features = [
                    self._filter_one(x[:, s:s + 1, :, :]) for s in range(self.num_slices)
                ]
                features = self.slice_fusion(per_slice_features)  # list of 3 scales
                centre = x[:, self.center_idx:self.center_idx + 1, :, :]
                features = self._maybe_fuse_original(features, centre)
        else:
            # ── S0: original 2D path ───────────────────────────────────────
            centre = x                               # (B, 1, H, W)
            features = self._filter_one(centre)
            features = self._maybe_fuse_original(features, centre)

        use_ds = return_aux and self.use_deep_supervision
        decoder_out = self.decoder(features, return_aux=use_ds)

        # ── SC-UNet-style position gate: applied AFTER the decoder, to its
        # final output and every deep-supervision auxiliary output, exactly
        # matching "multiplied with the result of the final convolutional
        # layer in the UNet structure" from the source paper. The decoder
        # itself (shared across every variant) is completely untouched.
        if self.use_position:
            if position is None:
                # metrics.py's get_model_efficiency (FLOPs / inference-time
                # measurement, called at the end of every Evaluator.evaluate())
                # calls this model with a dummy image tensor and NO position
                # argument -- that is a legitimate, expected caller, not a bug.
                # A real wiring problem (e.g. the trainer.py patch not applied)
                # would ALSO hit this path during actual training/validation,
                # so we still warn -- just once, not 100+ times inside the
                # FLOPs/timing loops -- rather than crash either caller.
                if not self._warned_missing_position:
                    print("  WARNING: use_position_encoding=True but no position "
                          "tensor was passed to forward() -- skipping the position "
                          "gate for this call. Expected during FLOPs/inference-time "
                          "measurement; if this appears during real training or "
                          "evaluation instead, the trainer.py position-passing patch "
                          "may not be applied correctly.")
                    self._warned_missing_position = True
                # Skip gating entirely this call -- decoder_out passed through unchanged.
            else:
                if isinstance(decoder_out, tuple):
                    decoder_out = tuple(self._apply_position_gate(o, position) for o in decoder_out)
                else:
                    decoder_out = self._apply_position_gate(decoder_out, position)

        return decoder_out

    def _check_slice_count(self, x):
        if x.dim() != 4 or x.size(1) != self.num_slices:
            raise ValueError(
                f"fusion_variant='{self.fusion_variant}' with num_slices="
                f"{self.num_slices} expects input shape (B,{self.num_slices},H,W), "
                f"but got {tuple(x.shape)}. Set data.three_slice=True and "
                f"data.num_slices={self.num_slices} so the dataset returns "
                f"correctly stacked slices."
            )

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_class_mapping(self):
        return self.class_info
"""Cross-Slice Attention (CSA) module adapted from XAG-Net (Ko et al., 2025).

The CSA module applies a pixel-wise softmax across the channel (slice) dimension,
allowing the model to learn which of the three input slices is most informative
at each spatial location. A residual connection preserves the original features.

Reference:
    Ko, B., Tian, A., & Lee, J. (2025). XAG-Net: A Cross-Slice Attention and
    Skip Gating Network for 2.5D Femur MRI Segmentation. ACDSA 2025.

Equation (from paper):
    CSA(X) = X + (X ⊙ Softmax(W(X)))

    where W(·) is a 1×1 convolution and ⊙ is element-wise multiplication.
    Softmax is applied along the channel dimension (dim=1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSliceAttention(nn.Module):
    """
    Pixel-wise cross-slice attention module.

    Takes a feature tensor of shape (B, C, H, W) where C spans the
    contributions from adjacent slices. At each spatial location (h, w),
    a 1x1 convolution predicts attention scores across the C channels,
    which are then normalised via softmax so they sum to 1 per pixel.
    The attended features are added back via a residual connection.

    Args:
        in_channels (int): Number of input channels C (typically 10 for
                           FFPA-Net filter features, or 3 for raw slices).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # 1x1 conv predicts one attention score per channel per pixel
        self.attn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, C, H, W) - same shape as input
        """
        # Predict attention logits: (B, C, H, W)
        attn_logits = self.attn_conv(x)

        # Normalise across channel dim so scores sum to 1 at each pixel
        attn_weights = F.softmax(attn_logits, dim=1)

        # Element-wise weighting + residual connection
        return x + (x * attn_weights)


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running CSA sanity checks...\n")

    # Test 1: output shape matches input shape
    B, C, H, W = 4, 10, 64, 64
    x = torch.randn(B, C, H, W)
    csa = CrossSliceAttention(in_channels=C)
    out = csa(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"Test 1 PASSED - output shape: {out.shape}")

    # Test 2: attention weights sum to 1 across channel dim at every pixel
    with torch.no_grad():
        logits = csa.attn_conv(x)
        weights = F.softmax(logits, dim=1)
        channel_sums = weights.sum(dim=1)  # should be all 1s: (B, H, W)
        assert torch.allclose(channel_sums, torch.ones_like(channel_sums), atol=1e-5), \
            "Attention weights do not sum to 1 across channels"
    print(f"Test 2 PASSED - attention weights sum to 1.0 at every pixel")

    # Test 3: residual connection preserved (output != pure attention output)
    with torch.no_grad():
        attended = x * F.softmax(csa.attn_conv(x), dim=1)
        residual_output = x + attended
        assert torch.allclose(out, residual_output, atol=1e-6), \
            "Residual connection not applied correctly"
    print(f"Test 3 PASSED - residual connection verified")

    # Test 4: works with raw 3-channel slice input (B, 3, H, W)
    x_raw = torch.randn(2, 3, 256, 256)
    csa_raw = CrossSliceAttention(in_channels=3)
    out_raw = csa_raw(x_raw)
    assert out_raw.shape == x_raw.shape
    print(f"Test 4 PASSED - works with 3-channel raw slice input: {out_raw.shape}")

    # Test 5: parameter count (should be tiny - just one 1x1 conv)
    total_params = sum(p.numel() for p in csa.parameters())
    print(f"Test 5 PASSED - total parameters: {total_params} "
          f"(expected {C * C + C} = {C*C + C})")

    print("\nAll tests passed.")
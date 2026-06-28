"""
Fixed-Filter Feature Extractor.

Multi-scale, non-trainable handcrafted filter bank. Takes a single-channel
slice (B, 1, H, W) and returns a list of [B, 10, H', W'] feature tensors at
three spatial scales. Used identically by every slice-fusion variant; the
fusion strategy decides how many times it is run per forward.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedFilterBank(nn.Module):
    """Fixed filter bank: edge, texture, Gaussian, statistical, morphological
    features at multiple scales. Zero trainable parameters (all buffers)."""

    def __init__(self, num_scales=3, device='cuda'):
        super().__init__()
        self.num_scales = num_scales
        self.device = device
        self._create_filters()

    def _create_filters(self):
        """Create filter kernels and register them as (non-trainable) buffers."""
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [0,  0,  0],
                                [1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        laplacian = torch.tensor([[0,  1, 0],
                                  [1, -4, 1],
                                  [0,  1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        gaussian_3 = self._gaussian_kernel(3, 1.0)
        gaussian_5 = self._gaussian_kernel(5, 1.5)
        gaussian_7 = self._gaussian_kernel(7, 2.0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)
        self.register_buffer('gaussian_3', gaussian_3)
        self.register_buffer('gaussian_5', gaussian_5)
        self.register_buffer('gaussian_7', gaussian_7)

    def _gaussian_kernel(self, size, sigma):
        """Create a normalized 2D Gaussian kernel."""
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
            np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        kernel = kernel / np.sum(kernel)
        return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def extract_features(self, x):
        """
        Extract multi-scale features from input images.

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            List of feature tensors at different scales, each [B, 10, H', W'].
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B, C, H, W = x.shape
        device = x.device
        self.to(device)

        all_features = []
        current_x = x

        for scale in range(self.num_scales):
            scale_features = []

            # 1. Edge features (Sobel)
            edge_x = F.conv2d(current_x, self.sobel_x, padding=1)
            edge_y = F.conv2d(current_x, self.sobel_y, padding=1)
            edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            edge_direction = torch.atan2(edge_y, edge_x)
            scale_features.append(edge_magnitude)
            scale_features.append(torch.cos(edge_direction))
            scale_features.append(torch.sin(edge_direction))

            # 2. Texture features (Laplacian)
            texture = F.conv2d(current_x, self.laplacian, padding=1)
            scale_features.append(torch.abs(texture))

            # 3. Gaussian smoothing at different scales
            smooth_3 = F.conv2d(current_x, self.gaussian_3, padding=1)
            smooth_5 = F.conv2d(current_x, self.gaussian_5, padding=2)
            smooth_7 = F.conv2d(current_x, self.gaussian_7, padding=3)
            scale_features.append(smooth_3)
            scale_features.append(smooth_5)
            scale_features.append(smooth_7)

            # 4. Local statistics
            local_mean = F.avg_pool2d(current_x, kernel_size=5, stride=1, padding=2)
            scale_features.append(local_mean)

            local_var = F.avg_pool2d(current_x ** 2, kernel_size=5, stride=1, padding=2) - local_mean ** 2
            scale_features.append(torch.sqrt(torch.clamp(local_var, min=1e-6)))

            # 5. Morphological gradient approximation
            max_pool = F.max_pool2d(current_x, kernel_size=3, stride=1, padding=1)
            min_pool = -F.max_pool2d(-current_x, kernel_size=3, stride=1, padding=1)
            morph_gradient = max_pool - min_pool
            scale_features.append(morph_gradient)

            scale_features = torch.cat(scale_features, dim=1)

            # Resize to the target resolution for each scale
            if scale == 0:
                target_size = (H, W)
            elif scale == 1:
                target_size = (H // 2, W // 2)
                scale_features = F.interpolate(scale_features, size=target_size,
                                               mode='bilinear', align_corners=False)
            else:
                target_size = (H // 4, W // 4)
                scale_features = F.interpolate(scale_features, size=target_size,
                                               mode='bilinear', align_corners=False)

            all_features.append(scale_features)

            # Downsample for next scale
            if scale < self.num_scales - 1:
                current_x = F.avg_pool2d(current_x, kernel_size=2, stride=2)

        return all_features

    def get_feature_dims(self):
        """Return the number of features at each scale (10 per scale)."""
        return [10, 10, 10]
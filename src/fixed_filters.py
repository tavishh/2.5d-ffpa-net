"""Fixed handcrafted filter bank for multi-scale feature extraction."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedFilterBank(nn.Module):
    """Extract edge, texture, and statistical features at multiple scales."""

    def __init__(self, num_scales=3, device="cuda"):
        super().__init__()
        self.num_scales = num_scales
        self.device = device
        self._create_filters()

    def _create_filters(self):
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)
        self.register_buffer("gaussian_3", self._gaussian_kernel(3, 1.0))
        self.register_buffer("gaussian_5", self._gaussian_kernel(5, 1.5))
        self.register_buffer("gaussian_7", self._gaussian_kernel(7, 2.0))

    def _gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma**2)),
            (size, size),
        )
        kernel = kernel / np.sum(kernel)
        return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def extract_features(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        _, _, height, width = x.shape
        self.to(x.device)

        all_features = []
        current_x = x

        for scale in range(self.num_scales):
            scale_features = []

            edge_x = F.conv2d(current_x, self.sobel_x, padding=1)
            edge_y = F.conv2d(current_x, self.sobel_y, padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            edge_direction = torch.atan2(edge_y, edge_x)
            scale_features.extend(
                [edge_magnitude, torch.cos(edge_direction), torch.sin(edge_direction)]
            )

            texture = F.conv2d(current_x, self.laplacian, padding=1)
            scale_features.append(torch.abs(texture))

            scale_features.extend(
                [
                    F.conv2d(current_x, self.gaussian_3, padding=1),
                    F.conv2d(current_x, self.gaussian_5, padding=2),
                    F.conv2d(current_x, self.gaussian_7, padding=3),
                ]
            )

            local_mean = F.avg_pool2d(current_x, kernel_size=5, stride=1, padding=2)
            local_var = F.avg_pool2d(current_x**2, kernel_size=5, stride=1, padding=2) - local_mean**2
            scale_features.extend([local_mean, torch.sqrt(torch.clamp(local_var, min=1e-6))])

            max_pool = F.max_pool2d(current_x, kernel_size=3, stride=1, padding=1)
            min_pool = -F.max_pool2d(-current_x, kernel_size=3, stride=1, padding=1)
            scale_features.append(max_pool - min_pool)

            scale_features = torch.cat(scale_features, dim=1)

            if scale == 0:
                target_size = (height, width)
            elif scale == 1:
                target_size = (height // 2, width // 2)
                scale_features = F.interpolate(
                    scale_features, size=target_size, mode="bilinear", align_corners=False
                )
            else:
                target_size = (height // 4, width // 4)
                scale_features = F.interpolate(
                    scale_features, size=target_size, mode="bilinear", align_corners=False
                )

            all_features.append(scale_features)

            if scale < self.num_scales - 1:
                current_x = F.avg_pool2d(current_x, kernel_size=2, stride=2)

        return all_features

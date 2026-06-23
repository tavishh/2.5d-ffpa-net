"""Loss functions for binary and multi-class segmentation."""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice: float = 0.5, weight_bce: float = 0.5):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 3:
            target = target.unsqueeze(1)

        bce_loss = self.bce(pred, target.float())
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice_loss = 1 - (2 * intersection + 1e-5) / (union + 1e-5)
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        weight_dice: float = 0.5,
        weight_ce: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.long()

        ce_loss = self.ce(pred, target)
        pred_soft = F.softmax(pred, dim=1)
        dice_loss = 0.0
        num_valid_classes = 0

        for c in range(1, self.num_classes):
            pred_c = pred_soft[:, c, :, :]
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            if union > 0:
                dice_loss += 1 - (2 * intersection + 1e-5) / (union + 1e-5)
                num_valid_classes += 1

        if num_valid_classes > 0:
            dice_loss = dice_loss / num_valid_classes

        return self.weight_dice * dice_loss + self.weight_ce * ce_loss


class DeepSupervisionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        weight_dice: float = 0.5,
        weight_ce: float = 0.5,
        ds_weights: List[float] = None,
    ):
        super().__init__()
        if ds_weights is None:
            ds_weights = [1.0, 0.5, 0.25]
        self.ds_weights = ds_weights

        if num_classes == 1:
            self.base_loss = DiceBCELoss(weight_dice, weight_ce)
        else:
            self.base_loss = DiceCELoss(num_classes, weight_dice, weight_ce)

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, tuple):
            total_loss = 0.0
            for output, weight in zip(outputs, self.ds_weights):
                if output.shape[2:] != target.shape[-2:]:
                    output = F.interpolate(
                        output,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                total_loss += weight * self.base_loss(output, target)
            return total_loss
        return self.base_loss(outputs, target)

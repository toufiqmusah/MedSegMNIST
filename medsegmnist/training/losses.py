import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss, differentiable.

    Accepts raw logits (``B × C × ...``) and integer masks
    (``B × ...``).  Works with both 2D and 3D inputs.

    Parameters
    ----------
    smooth : float
        Smoothing factor to avoid division by zero.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        targets = targets.long()
        probs = F.softmax(logits, dim=1)
        targets_one_hot = (
            F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
            if logits.dim() == 5
            else F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        )
        dims = tuple(range(2, logits.dim()))
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss.

    Parameters
    ----------
    smooth : float
        Smoothing factor for Dice loss.
    dice_weight : float
        Weight of the Dice component.
    ce_weight : float
        Weight of the Cross-Entropy component.
    """

    def __init__(self, smooth=1e-6, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * self.ce(
            logits, targets.long()
        )

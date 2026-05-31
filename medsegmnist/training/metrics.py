import torch
import torchmetrics


class DiceScore(torchmetrics.Metric):
    """Dice coefficient per class, with macro or per-class averaging.

    Parameters
    ----------
    num_classes : int
    average : str
        ``"macro"`` (average over classes with ground-truth) or ``"none"``
        (per-class scores).
    """

    def __init__(self, num_classes, average="macro", **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.average = average
        self.add_state(
            "intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "cardinality", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )

    def update(self, preds, targets):
        for c in range(self.num_classes):
            pred_c = preds == c
            target_c = targets == c
            self.intersection[c] += (pred_c & target_c).sum()
            self.cardinality[c] += pred_c.sum() + target_c.sum()

    def compute(self):
        numerator = 2.0 * self.intersection
        denominator = self.cardinality
        has_gt = self.cardinality > 0
        dice = torch.where(
            has_gt, numerator / (denominator + 1e-6), torch.ones_like(numerator)
        )
        if self.average == "macro":
            if has_gt.any():
                return dice[has_gt].mean()
            return dice.mean()
        elif self.average == "none":
            return dice
        return dice


class IoUScore(torchmetrics.Metric):
    """Intersection-over-Union (Jaccard index), macro-averaged.

    Parameters
    ----------
    num_classes : int
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state(
            "intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds, targets):
        for c in range(self.num_classes):
            pred_c = preds == c
            target_c = targets == c
            self.intersection[c] += (pred_c & target_c).sum()
            self.union[c] += (pred_c | target_c).sum()

    def compute(self):
        has_gt = self.union > 0
        iou = torch.where(
            has_gt,
            self.intersection / (self.union + 1e-6),
            torch.ones_like(self.intersection),
        )
        if has_gt.any():
            return iou[has_gt].mean()
        return iou.mean()

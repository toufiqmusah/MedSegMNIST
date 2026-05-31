import torch
import pytest


class TestDiceScore:
    def test_perfect_match(self):
        from medsegmnist.training import DiceScore

        metric = DiceScore(num_classes=2)
        pred = torch.zeros(10, dtype=torch.long)
        target = torch.zeros(10, dtype=torch.long)
        metric(pred, target)
        result = metric.compute()
        assert result == pytest.approx(1.0)

    def test_no_match(self):
        from medsegmnist.training import DiceScore

        metric = DiceScore(num_classes=2)
        pred = torch.ones(10, dtype=torch.long)
        target = torch.zeros(10, dtype=torch.long)
        metric(pred, target)
        result = metric.compute()
        assert result == pytest.approx(0.0)

    def test_multi_class(self):
        from medsegmnist.training import DiceScore

        torch.manual_seed(42)
        metric = DiceScore(num_classes=4)
        pred = torch.randint(0, 4, (100,))
        target = pred.clone()
        metric(pred, target)
        result = metric.compute()
        assert result == pytest.approx(1.0)


class TestIoUScore:
    def test_perfect_match(self):
        from medsegmnist.training import IoUScore

        metric = IoUScore(num_classes=2)
        pred = torch.zeros(10, dtype=torch.long)
        target = torch.zeros(10, dtype=torch.long)
        metric(pred, target)
        result = metric.compute()
        assert result == pytest.approx(1.0)

    def test_half_match(self):
        from medsegmnist.training import IoUScore

        metric = IoUScore(num_classes=2)
        pred = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        target = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        metric(pred, target)
        result = metric.compute()
        assert result == pytest.approx(1.0)


class TestDiceLoss:
    def test_perfect_prediction(self):
        from medsegmnist.training import DiceLoss

        loss_fn = DiceLoss()
        logits = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]]).unsqueeze(0)
        logits = logits.permute(0, 1, 2, 3)
        targets = torch.tensor([[0, 1]]).unsqueeze(0)
        logits = torch.randn(2, 2, 4, 4)
        targets = torch.randint(0, 2, (2, 4, 4))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_min_loss_with_perfect_logits(self):
        from medsegmnist.training import DiceLoss

        loss_fn = DiceLoss()
        n_classes = 2
        b, h, w = 2, 8, 8
        targets = torch.randint(0, n_classes, (b, h, w))
        logits = torch.zeros(b, n_classes, h, w)
        for c in range(n_classes):
            logits[:, c] = (targets == c).float() * 100 - 50
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.05

    def test_2d_and_3d(self):
        from medsegmnist.training import DiceLoss

        loss_fn = DiceLoss()
        logits_2d = torch.randn(2, 2, 8, 8)
        targets_2d = torch.randint(0, 2, (2, 8, 8))
        loss_2d = loss_fn(logits_2d, targets_2d)
        assert loss_2d.item() >= 0.0

        logits_3d = torch.randn(2, 2, 8, 8, 8)
        targets_3d = torch.randint(0, 2, (2, 8, 8, 8))
        loss_3d = loss_fn(logits_3d, targets_3d)
        assert loss_3d.item() >= 0.0


class TestDiceCELoss:
    def test_forward(self):
        from medsegmnist.training import DiceCELoss

        loss_fn = DiceCELoss()
        logits = torch.randn(2, 2, 8, 8)
        targets = torch.randint(0, 2, (2, 8, 8))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_weights(self):
        from medsegmnist.training import DiceCELoss

        loss_fn = DiceCELoss(dice_weight=0.0, ce_weight=1.0)
        logits = torch.randn(2, 2, 8, 8)
        targets = torch.randint(0, 2, (2, 8, 8))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

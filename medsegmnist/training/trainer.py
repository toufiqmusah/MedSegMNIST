import lightning as L
import torch
import torchmetrics

from .losses import DiceCELoss
from .metrics import DiceScore


class MedSegModule(L.LightningModule):
    """LightningModule wrapping a segmentation model for training.

    Handles the training and validation loops, logging of Dice and IoU,
    optimiser (AdamW), and learning-rate scheduler (CosineAnnealingLR).

    Parameters
    ----------
    model : nn.Module
        A segmentation model (e.g. ``UNet2D`` or ``UNet3D``).
    num_classes : int
        Number of segmentation classes.
    learning_rate : float
        Initial learning rate for AdamW.
    loss_fn : callable, optional
        Loss function.  Defaults to ``DiceCELoss()``.
    weight_decay : float
        Weight decay for AdamW.
    """

    def __init__(
        self, model, num_classes, learning_rate=1e-3, loss_fn=None, weight_decay=0.0
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn or DiceCELoss()

        self.train_dice = DiceScore(num_classes=num_classes, average="macro")
        self.val_dice = DiceScore(num_classes=num_classes, average="macro")
        self.train_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_dice_per_class = DiceScore(num_classes=num_classes, average="none")

        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        images, masks = batch
        logits = self.model(images)
        loss = self.loss_fn(logits, masks)
        preds = logits.argmax(dim=1)
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        loss, preds, masks = self._shared_step(batch)
        self.train_dice(preds, masks)
        self.train_iou(preds, masks)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train_iou", self.train_iou, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self._shared_step(batch)
        self.val_dice(preds, masks)
        self.val_iou(preds, masks)
        self.val_dice_per_class(preds, masks)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        per_class = self.val_dice_per_class.compute()
        for i in range(self.num_classes):
            self.log(f"val_dice_class_{i}", per_class[i])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

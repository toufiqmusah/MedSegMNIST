"""Eval subcommand — evaluates a checkpoint on a MedSegMNIST dataset."""

import torch
from torch.utils.data import DataLoader

from medsegmnist import info
from medsegmnist.training import DiceScore


def get_dataset_class(flag):
    import medsegmnist

    entry = info(flag)
    return getattr(medsegmnist, entry["class"])


@torch.no_grad()
def _evaluate(model, dataloader, num_classes, device):
    model.eval()
    model.to(device)
    dice = DiceScore(num_classes=num_classes, average="none")
    iou = DiceScore(num_classes=num_classes, average="macro")
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        dice.update(preds, masks)
        iou.update(preds, masks)
    return dice.compute(), iou.compute()


def run(args):
    import lightning as L

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = L.LightningModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model = module.model
    hparams = module.hparams
    num_classes = hparams.get("num_classes", 2)
    flag = args.flag or getattr(model, "flag", None)
    size = args.size

    if flag is None:
        import re

        ckpt_name = args.checkpoint
        match = re.search(r"(brain3d|lung2d|nuclei2d)", ckpt_name)
        if match:
            flag = match.group(1)
        else:
            print("Could not detect dataset flag from checkpoint name.")
            print("Please provide --flag")
            return

    if size is None:
        import re

        match = re.search(r"-(\d+|native)-", args.checkpoint)
        if match:
            size = match.group(1)
            size = int(size) if size.isdigit() else size
        else:
            size = None

    dataset_cls = get_dataset_class(flag)
    if size is None:
        size = dataset_cls.available_sizes[0]
    if isinstance(size, str) and size.isdigit():
        size = int(size)

    test_dataset = dataset_cls(split="test", size=size, root=args.root)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    print(f"Test samples: {len(test_dataset)}")

    per_class_dice, macro_iou = _evaluate(model, test_loader, num_classes, device)

    label_names = test_dataset.get_label_names()
    print(f"\n{'Class':<20} {'Dice':<10} {'IoU':<10}")
    print("-" * 40)
    for c in range(num_classes):
        name = label_names.get(str(c), f"class_{c}")
        print(f"{name:<20} {per_class_dice[c]:<10.4f} {'':>10}")
    macro_dice = per_class_dice.mean().item()
    print(f"\n{'Macro average':<20} {macro_dice:<10.4f} {macro_iou.item():<10.4f}")

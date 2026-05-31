"""Train subcommand — trains a user-provided model on a MedSegMNIST dataset."""

import importlib
import json
import os

import lightning as L
from torch.utils.data import DataLoader

from medsegmnist import info
from medsegmnist.training import MedSegModule


def _import_model(model_path):
    module_path, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_dataset_class(flag):
    import medsegmnist

    entry = info(flag)
    return getattr(medsegmnist, entry["class"])


def run(args):
    L.seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_cls = get_dataset_class(args.flag)
    meta = info(args.flag)
    n_classes = meta["n_classes"]
    n_channels = dataset_cls.n_channels

    size = args.size or dataset_cls.available_sizes[0]
    if isinstance(size, str) and size.isdigit():
        size = int(size)

    train_dataset = dataset_cls(split="train", size=size, root=args.root)
    test_dataset = dataset_cls(split="test", size=size, root=args.root)

    if hasattr(train_dataset, "get_fold"):
        train_subset, val_subset = train_dataset.get_fold(args.fold)
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
        print(f"Fold {args.fold}: train={len(train_subset)}, val={len(val_subset)}")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
        print(f"Train={len(train_dataset)}, Val={len(test_dataset)} (no CV folds)")

    model_cls = _import_model(args.model)
    model_kwargs = json.loads(args.model_kwargs)
    model = model_cls(in_channels=n_channels, n_classes=n_classes, **model_kwargs)

    module = MedSegModule(
        model=model,
        num_classes=n_classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename=f"{args.flag}-{size}-" + "{epoch:02d}-{val_dice:.4f}",
            monitor="val_dice",
            mode="max",
            save_top_k=3,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_dice",
            mode="max",
            patience=20,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=1 if args.fast_dev_run else args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(module, train_loader, val_loader)
    print(f"Training complete. Best checkpoint: {callbacks[0].best_model_path}")

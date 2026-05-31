"""Command-line interface for MedSegMNIST.

Usage::

    medsegmnist train --model mypackage.MyModel --flag lung2d --size 128
    medsegmnist eval --checkpoint checkpoints/lung2d-128.ckpt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="medsegmnist",
        description="MedSegMNIST: biomedical image segmentation datasets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train a segmentation model")
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Dotted import path to the model class, e.g. 'mypackage.models.MyModel'",
    )
    train_parser.add_argument(
        "--model-kwargs",
        type=str,
        default="{}",
        help="JSON string of keyword arguments for the model constructor",
    )
    train_parser.add_argument("--flag", type=str, default="lung2d", help="Dataset flag")
    train_parser.add_argument(
        "--size", type=str, default=None, help="Dataset size (128, 256, native, …)"
    )
    train_parser.add_argument(
        "--root",
        type=str,
        default="/teamspace/studios/this_studio/datasets",
        help="Dataset root directory",
    )
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay"
    )
    train_parser.add_argument(
        "--fold", type=int, default=0, help="Cross-validation fold (0–4)"
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices (GPUs)"
    )
    train_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Accelerator type",
    )
    train_parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a single batch for smoke-testing",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    train_parser.add_argument(
        "--default-root",
        type=str,
        default=None,
        help="Default MedSegMNIST data root",
    )

    # --- eval subcommand ---
    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint")
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .ckpt file"
    )
    eval_parser.add_argument(
        "--flag", type=str, default=None, help="Dataset flag (auto-detected if omitted)"
    )
    eval_parser.add_argument(
        "--size", type=str, default=None, help="Dataset size (auto-detected if omitted)"
    )
    eval_parser.add_argument(
        "--root",
        type=str,
        default="/teamspace/studios/this_studio/datasets",
        help="Dataset root directory",
    )
    eval_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    eval_parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices (GPUs)"
    )
    eval_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Accelerator type",
    )

    args = parser.parse_args()

    if args.command == "train":
        from medsegmnist.cli.train import run

        run(args)
    elif args.command == "eval":
        from medsegmnist.cli.eval import run

        run(args)


if __name__ == "__main__":
    main()

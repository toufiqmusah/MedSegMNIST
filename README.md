# MedSegMNIST

[![CI](https://github.com/MedSegMNIST/MedSegMNIST/actions/workflows/ci.yml/badge.svg)](https://github.com/MedSegMNIST/MedSegMNIST/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-CC%20BY%204.0-green)]()

A collection of **standardised biomedical image segmentation datasets** in NPZ format, with a unified PyTorch API, pre-configured U-Net models, and a Lightning-based training pipeline.

> Inspired by [MedMNIST](https://medmnist.com/) — but for **segmentation** instead of classification.

---

## Features

- **Unified API**: Load any dataset in 3 lines — same interface for 2D and 3D
- **Multiple sizes**: Each dataset is available in several resolutions (64–512 px, + native)
- **Built-in CV folds**: 5-fold cross-validation splits included out of the box
- **Ready-to-train**: U-Net (2D/3D) + PyTorch Lightning trainer + Dice/IoU metrics
- **Extensible**: Register new datasets with a single decorator

---

## Supported Datasets

| Flag | Dataset | Modality | Anatomy | Dim | Classes | Train | Test | Sizes |
|------|---------|----------|---------|-----|---------|-------|------|-------|
| `brain3d` | BrainSegMNIST3D | MRI | Brain (gliomas) | 3D | 4 | 116 | 30 | 96, 128, 224, native |
| `lung2d` | LungSegMNIST2D | X-ray | Chest / Lungs | 2D | 2 | 5,448 | 1,362 | 128, 256, 512 |
| `nuclei2d` | NucleiSegMNIST2D | Pathology | Multi-organ (nuclei) | 2D | 2 | 112 | 39 | 256, 512, native |

**BrainSegMNIST3D** — Brain tumour sub-region segmentation from BraTS-Africa (T2-FLAIR). Labels: background (0), necrotic core (1), oedema (2), enhancing tumour (3). Native resolution: 240×240×155 at 1.0 mm isotropic.

**LungSegMNIST2D** — Lung field segmentation from chest X-rays (Darwin + Montgomery + Shenzhen). Binary: background (0), lung (1). Native resolution: 512×512, converted from RGB to grayscale.

**NucleiSegMNIST2D** — Nuclei segmentation from NuSeC + MoNuSeg 2018. RGB input (3 channels). Binary: background (0), nuclei (1). Native resolution: 1024×1024 (MoNuSeg centre-padded to match).

---

## Installation

```bash
pip install medsegmnist
```

To also run preprocessing scripts (for building datasets from raw sources):

```bash
pip install "medsegmnist[preprocess]"
```

For development (testing, linting):

```bash
pip install "medsegmnist[dev]"
```

---

## Quick Start

```python
from medsegmnist import LungSegMNIST2D, list_datasets

# List all available datasets
print(list_datasets())

# Load a dataset
ds = LungSegMNIST2D(split="train", size=128, root="/path/to/datasets")
print(len(ds))  # 5448

# Access a sample
image, mask = ds[0]
print(image.shape)  # (1, 128, 128) — channel-first float32
print(mask.shape)   # (128, 128) — uint8
print(mask.unique())  # [0, 1]

# Get metadata for any dataset
from medsegmnist import info
info("brain3d")
```

### Data shape convention

| Dimensionality | Image shape | Mask shape |
|----------------|-------------|------------|
| 2D (1-channel) | `(1, H, W)` float32 | `(H, W)` uint8 |
| 2D (3-channel) | `(3, H, W)` float32 | `(H, W)` uint8 |
| 3D | `(1, D, H, W)` float32 | `(D, H, W)` uint8 |

---

## Training

```python
from medsegmnist import LungSegMNIST2D
from medsegmnist.training import MedSegModule
import lightning as L
from torch.utils.data import DataLoader

# Dataset
ds = LungSegMNIST2D(split="train", size=128, root="/path/to/datasets")
train_subset, val_subset = ds.get_fold(0)

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16)

# Your model — any PyTorch segmentation model works
model = ...  # e.g. UNet2D(in_channels=1, n_classes=2)
module = MedSegModule(model=model, num_classes=2)

# Train
trainer = L.Trainer(max_epochs=50, accelerator="auto")
trainer.fit(module, train_loader, val_loader)
```

Or use the CLI:

```bash
medsegmnist train --model "mymodel.MyModel" --flag lung2d --size 128 --epochs 50
```

A reference ``UNet2D``/``UNet3D`` implementation is provided in ``examples/`` for convenience.

### Training options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Dotted import path to model class |
| `--model-kwargs` | `{}` | JSON kwargs for model constructor |
| `--flag` | `lung2d` | Dataset flag |
| `--size` | first available | Image size |
| `--epochs` | 50 | Number of epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--fold` | 0 | Cross-validation fold (0–4) |
| `--accelerator` | auto | `auto`, `cpu`, or `gpu` |
| `--fast-dev-run` | — | Run one batch for smoke-testing |

---

## Evaluation

```bash
medsegmnist eval --checkpoint checkpoints/lung2d-128-epoch=42-val_dice=0.97.ckpt
```

Output:
```
Class           Dice      IoU
─────────────────────────────
background      0.9923    0.9847
lung            0.9718    0.9452
─────────────────────────────
Macro average   0.9820    0.9649
```

---

## Visualization

```python
from medsegmnist.utils import plot_sample, plot_grid

ds = LungSegMNIST2D(split="test", size=128, root="/path/to/datasets")
img, mask = ds[0]

fig, ax = plt.subplots(1, 2)
plot_sample(img, mask, ax=ax)
plt.show()
```

---

## API Reference

### Dataset classes

| Class | Flag | Base class | Channels | Classes | Sizes |
|-------|------|------------|----------|---------|-------|
| `BrainSegMNIST3D` | `brain3d` | `MedSegMNIST3D` | 1 | 4 | 96, 128, 224, native |
| `LungSegMNIST2D` | `lung2d` | `MedSegMNIST2D` | 1 | 2 | 128, 256, 512 |
| `NucleiSegMNIST2D` | `nuclei2d` | `MedSegMNIST2D` | 3 | 2 | 256, 512, native |

All dataset classes share the same interface (inherited from `MedSegMNIST2D` or `MedSegMNIST3D`):

| Method | Description |
|--------|-------------|
| `__init__(split, size, root, transform, target_transform, mmap_mode, download)` | Load a dataset. `split` must be `"train"` or `"test"`. |
| `__len__()` | Number of samples |
| `__getitem__(index)` | `(image_tensor, mask_tensor)` |
| `get_data()` | Raw numpy arrays `(images, masks)` |
| `get_fold(k)` | `(train_subset, val_subset)` for fold `k` (0–4) |
| `get_label_names()` | `dict` mapping class IDs to name strings |
| `info()` | Print dataset metadata |

### Registry

```python
from medsegmnist import info, list_datasets

# List all datasets
print(list_datasets())
# → [("brain3d", "BrainSegMNIST3D", "MRI"), ...]

# Filter by dimensionality
print(list_datasets(dimensionality="2D"))
# → [("lung2d", "LungSegMNIST2D", "X-ray"), ("nuclei2d", "NucleiSegMNIST2D", "Pathology")]

# Get metadata
info("brain3d")
```

### Reference model implementations

Reference implementations are provided in ``examples/`` for convenience:

| File | Model | Description |
|------|-------|-------------|
| `examples/unet.py` | `UNet2D(in_channels, n_classes, base_filters, depth, bilinear)` | Standard 2D U-Net |
| `examples/unet3d.py` | `UNet3D(in_channels, n_classes, base_filters, depth, trilinear)` | 3D U-Net counterpart |

These are **not** part of the ``medsegmnist`` package — copy or adapt them as needed.

### Training (`medsegmnist.training`)

| Component | Description |
|-----------|-------------|
| `DiceScore(num_classes, average="macro")` | Dice coefficient metric |
| `IoUScore(num_classes, average="macro")` | IoU / Jaccard index metric |
| `DiceLoss(smooth=1e-6)` | Differentiable Dice loss |
| `DiceCELoss(smooth=1e-6, dice_weight=0.5, ce_weight=0.5)` | Combined Dice + Cross-Entropy |
| `MedSegModule(model, num_classes, learning_rate, loss_fn, weight_decay)` | LightningModule with training/val steps, AdamW, cosine annealing |

### Visualization (`medsegmnist.utils`)

| Function | Description |
|----------|-------------|
| `plot_sample(image, mask, slice_idx, label_names, ax)` | Image + mask side-by-side |
| `plot_overlay(image, mask, alpha, slice_idx, ax)` | Mask overlaid on image |
| `plot_grid(images, masks, n_cols, slice_idx)` | Grid of sample plots |

---

## Adding a New Dataset

1. Create a dataset class inheriting from `MedSegMNIST2D` or `MedSegMNIST3D`
2. Decorate with `@register`
3. Add preprocessing to `scripts/preprocess/`
4. Generate NPZ files and JSON metadata using the preprocessing script

```python
from medsegmnist.datasets.base import MedSegMNIST2D
from medsegmnist.registry import register

@register
class MyDataset(MedSegMNIST2D):
    flag = "my2d"
    class_name = "MyDataset"
    available_sizes = [128, 256]
    n_classes = 3
    modality = "CT"
    n_channels = 1
```

The rest — data loading, folds, ``info()``, ``list_datasets()`` — works automatically.

---
medsegmnist/
├── __init__.py              # Public API (dataset classes, info, list_datasets)
├── registry.py              # Dataset registry (@register, info, list_datasets)
├── datasets/
│   ├── base.py              # MedSegMNIST2D / MedSegMNIST3D base classes
│   ├── mri/brain.py         # BrainSegMNIST3D
│   ├── xray/lung.py         # LungSegMNIST2D
│   └── pathology/nuclei.py  # NucleiSegMNIST2D
├── cli/
│   ├── __init__.py          # medsegmnist CLI entry point
│   ├── train.py             # train subcommand
│   └── eval.py              # eval subcommand
├── training/
│   ├── metrics.py           # DiceScore, IoUScore
│   ├── losses.py            # DiceLoss, DiceCELoss
│   └── trainer.py           # MedSegModule (LightningModule)
└── utils/
    └── visualize.py         # Plotting utilities
scripts/preprocess/          # Preprocessing scripts (build NPZ from raw data)
├── common.py
├── brain.py
├── lung.py
└── nuclei.py
examples/                    # Reference model implementations
├── unet.py
└── unet3d.py
docs/                        # Sphinx documentation
├── source/
└── Makefile
```

---

## Citation

If you use MedSegMNIST in your research, please cite:

```bibtex
@software{medsegmnist,
  title = {MedSegMNIST: Standardised Biomedical Image Segmentation Datasets},
  url = {https://github.com/MedSegMNIST/MedSegMNIST},
  year = {2026}
}
```

Please also cite the original source papers of the constituent datasets:

- **BrainSegMNIST3D**: BraTS-Africa (IEEE ISBI 2022)
- **LungSegMNIST2D**: chest-xray-lungs (multiple sources)
- **NucleiSegMNIST2D**: NuSeC + MoNuSeg (IEEE ISBI 2019)

---

## License

The MedSegMNIST code is distributed under the [Apache 2.0 License](LICENSE).

The constituent datasets retain their original licenses:

| Dataset | License |
|---------|---------|
| BrainSegMNIST3D | BraTS — CC BY 4.0 |
| LungSegMNIST2D | Varies by source subset |
| NucleiSegMNIST2D | Research purposes (original terms apply) |

These datasets are **not** intended for clinical use.

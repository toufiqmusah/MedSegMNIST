# MedSegMNIST

[![CI](https://github.com/toufiqmusah/MedSegMNIST/actions/workflows/ci.yml/badge.svg)](https://github.com/toufiqmusah/MedSegMNIST/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green)]()

A collection of **10 standardised biomedical image segmentation datasets** across 8 modalities, with a unified PyTorch API, pre-configured U-Net models, and a Lightning-based training pipeline.

> Inspired by [MedMNIST](https://medmnist.com/) — but for **segmentation** instead of classification.

Data hosted on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20694762.svg)](https://doi.org/10.5281/zenodo.20694762)

---

## Features

- **Unified API**: Load any dataset in 3 lines — same interface for 2D and 3D
- **Multiple resolutions**: Each dataset at 2–4 standardised sizes, plus native
- **Built-in 5-fold CV**: Cross-validation splits included out of the box
- **Auto-download**: Data fetches from Zenodo with MD5 verification (`download=True`)
- **Ready-to-train**: U-Net (2D/3D) + PyTorch Lightning trainer + Dice/IoU metrics
- **Extensible**: Register new datasets with a single decorator

---

## Supported Datasets

| Flag | Class | Modality | Dim | Classes | Channels | Sizes |
|------|-------|----------|-----|---------|----------|-------|
| `abdomen3d` | AbdomenSegMNIST3D | CT | 3D | 6 | 1 | 64, 96, 128, 192, native |
| `brain3d` | BrainSegMNIST3D | MRI | 3D | 4 | 1 | 96, 128, 224, native |
| `spine3d` | SpineSegMNIST3D | MR | 3D | 3 | 1 | 64, 96, 128, 192, native |
| `knee3d` | KneeSegMNIST3D | MR | 3D | 6 | 1 | 64, 96, 128, 192, native |
| `lung2d` | LungSegMNIST2D | X-ray | 2D | 2 | 1 | 128, 256, 512 |
| `nuclei2d` | NucleiSegMNIST2D | Pathology | 2D | 2 | 3 | 256, 512, native |
| `polyp2d` | PolypSegMNIST2D | Endoscopy | 2D | 2 | 3 | 128, 256, 512, native |
| `breast2d` | BreastSegMNIST2D | Mammography | 2D | 2 | 1 | 128, 256, 512, native |
| `fives2d` | FundusSegMNIST2D | Fundus photography | 2D | 2 | 3 | 256, 512, 1024, native |
| `derm2d` | SkinSegMNIST2D | Dermoscopy | 2D | 2 | 3 | 128, 256, 512, native |

---

## Installation

```bash
# Core package (dataset loading + CLI)
pip install git+https://github.com/toufiqmusah/MedSegMNIST.git

# With PyTorch training support
pip install "medsegmnist[torch]"

# For preprocessing scripts (building NPZs from raw sources)
pip install "medsegmnist[preprocess]"

# Development (testing, linting)
pip install "medsegmnist[dev]"
```

---

## Quick Start

```python
from medsegmnist import LungSegMNIST2D, list_datasets

# List all available datasets
print(list_datasets())

# Load a dataset (auto-downloads from Zenodo on first use)
ds = LungSegMNIST2D(split="train", size=128, download=True)
print(len(ds))  # 5448

# Access a sample
image, mask = ds[0]
print(image.shape)  # (1, 128, 128) — channel-first float32
print(mask.shape)   # (128, 128) — uint8
print(mask.unique())  # [0, 1]

# Get metadata
from medsegmnist import info
info("brain3d")
```

Data is stored in `~/.medsegmnist/data/` by default. Override with `root="/your/path"`.

### Data shape convention

| Dimensionality | Image shape | Mask shape |
|----------------|-------------|------------|
| 2D (1-channel) | `(1, H, W)` float32 | `(H, W)` uint8 |
| 2D (3-channel) | `(3, H, W)` float32 | `(H, W)` uint8 |
| 3D | `(1, D, H, W)` float32 | `(D, H, W)` uint8 |

---

## Training

Requires `pip install "medsegmnist[torch]"`.

```python
from medsegmnist import LungSegMNIST2D
from medsegmnist.training import MedSegModule
import lightning as L
from torch.utils.data import DataLoader

ds = LungSegMNIST2D(split="train", size=128, download=True)
train_subset, val_subset = ds.get_fold(0)

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16)

model = ...  # any segmentation model, e.g. UNet2D(in_channels=1, n_classes=2)
module = MedSegModule(model=model, num_classes=2)

trainer = L.Trainer(max_epochs=50, accelerator="auto")
trainer.fit(module, train_loader, val_loader)
```

Or use the CLI:

```bash
medsegmnist train --model "examples.unet.UNet2D" --flag lung2d --size 128 --epochs 50
```

Reference U-Net implementations are in `examples/unet.py` (2D) and `examples/unet3d.py` (3D).

### CLI options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Dotted import path to model class |
| `--model-kwargs` | `{}` | JSON kwargs for model constructor |
| `--flag` | `lung2d` | Dataset flag |
| `--size` | first available | Image size |
| `--root` | `~/.medsegmnist/data` | Dataset root directory |
| `--epochs` | 50 | Number of epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight-decay` | 1e-4 | Weight decay |
| `--fold` | 0 | Cross-validation fold (0–4) |
| `--accelerator` | `auto` | `auto`, `cpu`, or `gpu` |
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

ds = LungSegMNIST2D(split="test", size=128)
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
| `AbdomenSegMNIST3D` | `abdomen3d` | `MedSegMNIST3D` | 1 | 6 | 64, 96, 128, 192, native |
| `BrainSegMNIST3D` | `brain3d` | `MedSegMNIST3D` | 1 | 4 | 96, 128, 224, native |
| `SpineSegMNIST3D` | `spine3d` | `MedSegMNIST3D` | 1 | 3 | 64, 96, 128, 192, native |
| `KneeSegMNIST3D` | `knee3d` | `MedSegMNIST3D` | 1 | 6 | 64, 96, 128, 192, native |
| `LungSegMNIST2D` | `lung2d` | `MedSegMNIST2D` | 1 | 2 | 128, 256, 512 |
| `NucleiSegMNIST2D` | `nuclei2d` | `MedSegMNIST2D` | 3 | 2 | 256, 512, native |
| `PolypSegMNIST2D` | `polyp2d` | `MedSegMNIST2D` | 3 | 2 | 128, 256, 512, native |
| `BreastSegMNIST2D` | `breast2d` | `MedSegMNIST2D` | 1 | 2 | 128, 256, 512, native |
| `FundusSegMNIST2D` | `fives2d` | `MedSegMNIST2D` | 3 | 2 | 256, 512, 1024, native |
| `SkinSegMNIST2D` | `derm2d` | `MedSegMNIST2D` | 3 | 2 | 128, 256, 512, native |

All dataset classes share the same interface:

| Method | Description |
|--------|-------------|
| `__init__(split, size, root, transform, target_transform, mmap_mode, download)` | Load a dataset. `split`: `"train"`, `"test"`, or `"all"`. |
| `__len__()` | Number of samples |
| `__getitem__(index)` | `(image_tensor, mask_tensor)` |
| `get_data()` | Raw numpy arrays `(images, masks)` |
| `get_fold(k)` | `(train_subset, val_subset)` for fold `k` (0–4) |
| `get_label_names()` | `dict` mapping class IDs to name strings |
| `info()` | Print dataset metadata |
| `get_voxel_spacing()` | Voxel spacing in mm (3D datasets only) |

### Registry

```python
from medsegmnist import info, list_datasets

print(list_datasets())
# → [("abdomen3d", "AbdomenSegMNIST3D", "CT"), ("brain3d", "BrainSegMNIST3D", "MRI"), ...]

print(list_datasets(dimensionality="2D"))
# → [("lung2d", "LungSegMNIST2D", "X-ray"), ("nuclei2d", "NucleiSegMNIST2D", "Pathology"), ...]

info("brain3d")
```

### Training (`medsegmnist.training`)

| Component | Description |
|-----------|-------------|
| `DiceScore(num_classes, average="macro")` | Dice coefficient metric |
| `IoUScore(num_classes, average="macro")` | IoU / Jaccard index metric |
| `DiceLoss(smooth=1e-6)` | Differentiable Dice loss |
| `DiceCELoss(smooth=1e-6, dice_weight=0.5, ce_weight=0.5)` | Combined Dice + Cross-Entropy |
| `MedSegModule(model, num_classes, learning_rate, loss_fn, weight_decay)` | LightningModule with AdamW + cosine annealing |

### Visualization (`medsegmnist.utils`)

| Function | Description |
|----------|-------------|
| `plot_sample(image, mask, slice_idx, label_names, ax)` | Image + mask side-by-side |
| `plot_overlay(image, mask, alpha, slice_idx, ax)` | Mask overlaid on image |
| `plot_grid(images, masks, n_cols, slice_idx)` | Grid of sample plots |

---

## Adding a New Dataset

```python
from medsegmnist.datasets.base import MedSegMNIST2D
from medsegmnist.registry import register

@register
class MyDataset(MedSegMNIST2D):
    flag = "my2d"
    class_name = "MyDataset"
    organ = "my_organ"
    available_sizes = [128, 256]
    n_classes = 3
    modality = "CT"
    n_channels = 1
    citation = "Author et al. (2026)"
    zenodo_record_id = "1234567"
```

1. Create the dataset class with `@register`, extending `MedSegMNIST2D` or `MedSegMNIST3D`
2. Add import to `medsegmnist/datasets/__init__.py` and `medsegmnist/__init__.py`
3. Generate NPZ files and JSON metadata (see `scripts/preprocess/` for reference)
4. Upload data and update `zenodo_record_id`

The rest — data loading, CV folds, `info()`, `list_datasets()` — works automatically.

---

## Package Structure

```
medsegmnist/
├── __init__.py              # Public API (all 10 dataset classes, info, list_datasets)
├── registry.py              # @register decorator + registry queries
├── datasets/
│   ├── base.py              # MedSegMNIST2D / MedSegMNIST3D base classes
│   ├── ct/abdomen.py        # AbdomenSegMNIST3D
│   ├── mri/brain.py         # BrainSegMNIST3D
│   ├── mri/spine.py         # SpineSegMNIST3D
│   ├── mri/knee.py          # KneeSegMNIST3D
│   ├── xray/lung.py         # LungSegMNIST2D
│   ├── pathology/nuclei.py  # NucleiSegMNIST2D
│   ├── endoscopy/polyp.py   # PolypSegMNIST2D
│   ├── ultrasound/breast.py # BreastSegMNIST2D
│   ├── fundus/fives.py      # FundusSegMNIST2D
│   └── dermoscopy/derm.py   # SkinSegMNIST2D
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
examples/                    # Reference model implementations
├── unet.py                  # UNet2D
└── unet3d.py                # UNet3D
tests/                       # pytest suite
```

---

## Citation

If you use MedSegMNIST in your research, please cite:

```bibtex
@software{medsegmnist,
  title = {MedSegMNIST: Standardised Biomedical Image Segmentation Datasets},
  url = {https://github.com/toufiqmusah/MedSegMNIST},
  doi = {10.5281/zenodo.20694762},
  year = {2026}
}
```

Please also cite the original source papers of the constituent datasets:

| Dataset | Citation |
|---------|----------|
| AbdomenSegMNIST3D | Rister, Blaine, et al. "CT-ORG, a new dataset for multiple organ segmentation in computed tomography." *Scientific Data* 7.1 (2020): 381. |
| BrainSegMNIST3D | Adewole, Maruf, et al. "The BraTS-Africa dataset: expanding the brain tumor segmentation data to capture African populations." *Radiology: Artificial Intelligence* 7.4 (2025): e240528. |
| SpineSegMNIST3D | Zhou, Longfei, et al. "The Duke University Cervical Spine MRI Segmentation Dataset (CSpineSeg)." *Scientific Data* 12.1 (2025): 1695. |
| KneeSegMNIST3D | Ambellan, Felix, et al. "Automated segmentation of knee bone and cartilage combining statistical shape knowledge and convolutional neural networks: Data from the Osteoarthritis Initiative." *Medical Image Analysis* 52 (2019): 109-118. |
| LungSegMNIST2D | Danilov, Viacheslav, et al. "Chest X-ray dataset for lung segmentation." *Mendeley Data*, V2, doi: 10.17632/8gf9vpkhgy.2 |
| NucleiSegMNIST2D | (1) Kumar, Neeraj, et al. "A multi-organ nucleus segmentation challenge." *IEEE TMI* 39.5 (2019): 1380–1391. (2) Samet, Refik, et al. "NuSeC." arXiv:2507.14272 (2025). |
| PolypSegMNIST2D | Jha, Debesh, et al. "Kvasir-SEG: A Segmented Polyp Dataset." *MMM 2020*. |
| BreastSegMNIST2D | Oza, Parita, et al. "Digital mammography dataset for breast cancer diagnosis research (DMID) with breast mass segmentation analysis." *Biomedical Engineering Letters* 14.2 (2024): 317-330. |
| FundusSegMNIST2D | Jin, Kai, et al. "FIVES: A Fundus Image Dataset for AI based Vessel Segmentation." *Scientific Data* 11.1 (2024): 1064. |
| SkinSegMNIST2D | Codella, Noel, et al. "Skin Lesion Analysis Toward Melanoma Detection 2018." arXiv:1902.03368 (2019). |

---

## License

The MedSegMNIST code is distributed under the [Apache 2.0 License](LICENSE).

The constituent datasets retain their original licenses. They are **not** intended for clinical use.

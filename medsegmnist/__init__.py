"""MedSegMNIST — standardised biomedical image segmentation datasets.

A unified PyTorch API for loading, training, and evaluating segmentation
models across multiple medical imaging modalities and resolutions.

Typical usage::

    from medsegmnist import LungSegMNIST, list_datasets

    print(list_datasets())
    ds = LungSegMNIST(split="train", size=128)
    image, mask = ds[0]
"""

from .datasets.mri.brain import BrainSegMNIST3D
from .datasets.xray.lung import LungSegMNIST
from .datasets.pathology.nuclei import NucleiSegMNIST

from .registry import info, list_datasets

__version__ = "0.1.0"

__all__ = [
    "BrainSegMNIST3D",
    "LungSegMNIST",
    "NucleiSegMNIST",
    "info",
    "list_datasets",
]

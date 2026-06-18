"""MedSegMNIST — standardised biomedical image segmentation datasets.

A unified PyTorch API for loading, training, and evaluating segmentation
models across multiple medical imaging modalities and resolutions.

Typical usage::

    from medsegmnist import LungSegMNIST2D, list_datasets

    print(list_datasets())
    ds = LungSegMNIST2D(split="train", size=128)
    image, mask = ds[0]
"""

from .datasets.ct.abdomen import AbdomenSegMNIST3D
from .datasets.mri.brain import BrainSegMNIST3D
from .datasets.mri.spine import SpineSegMNIST3D
from .datasets.mri.knee import KneeSegMNIST3D
from .datasets.xray.lung import LungSegMNIST2D
from .datasets.pathology.nuclei import NucleiSegMNIST2D
from .datasets.endoscopy.polyp import PolypSegMNIST2D
from .datasets.ultrasound.breast import BreastSegMNIST
from .datasets.fundus.fives import FundusSegMNIST2D
from .datasets.dermoscopy.derm import SkinSegMNIST2D

from .registry import info, list_datasets
from . import utils

__version__ = "0.1.0"

__all__ = [
    "AbdomenSegMNIST3D",
    "BrainSegMNIST3D",
    "SpineSegMNIST3D",
    "KneeSegMNIST3D",
    "LungSegMNIST2D",
    "NucleiSegMNIST2D",
    "PolypSegMNIST2D",
    "BreastSegMNIST",
    "FundusSegMNIST2D",
    "SkinSegMNIST2D",
    "info",
    "list_datasets",
    "utils",
]

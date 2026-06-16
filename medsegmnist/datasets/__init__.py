from .ct.abdomen import AbdomenSegMNIST3D
from .mri.brain import BrainSegMNIST3D
from .xray.lung import LungSegMNIST2D
from .pathology.nuclei import NucleiSegMNIST2D

__all__ = [
    "AbdomenSegMNIST3D",
    "BrainSegMNIST3D",
    "LungSegMNIST2D",
    "NucleiSegMNIST2D",
]

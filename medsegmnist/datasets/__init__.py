from .mri.brain import BrainSegMNIST3D
from .xray.lung import LungSegMNIST
from .pathology.nuclei import NucleiSegMNIST

__all__ = [
    "BrainSegMNIST3D",
    "LungSegMNIST",
    "NucleiSegMNIST",
]

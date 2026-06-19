from .ct.abdomen import AbdomenSegMNIST3D
from .mri.brain import BrainSegMNIST3D
from .mri.spine import SpineSegMNIST3D
from .mri.knee import KneeSegMNIST3D
from .xray.lung import LungSegMNIST2D
from .pathology.nuclei import NucleiSegMNIST2D
from .endoscopy.polyp import PolypSegMNIST2D
from .dermoscopy.derm import SkinSegMNIST2D
from .ultrasound.breast import BreastSegMNIST2D

__all__ = [
    "AbdomenSegMNIST3D",
    "BrainSegMNIST3D",
    "SpineSegMNIST3D",
    "KneeSegMNIST3D",
    "LungSegMNIST2D",
    "NucleiSegMNIST2D",
    "PolypSegMNIST2D",
    "SkinSegMNIST2D",
    "BreastSegMNIST2D",
]

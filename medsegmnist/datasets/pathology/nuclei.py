from medsegmnist.datasets.base import MedSegMNIST2D
from medsegmnist.registry import register


@register
class NucleiSegMNIST(MedSegMNIST2D):
    flag = "nuclei2d"
    class_name = "NucleiSegMNIST"
    available_sizes = [256, 512, "native"]
    n_classes = 2
    modality = "Pathology"
    n_channels = 3

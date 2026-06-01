from medsegmnist.datasets.base import MedSegMNIST2D
from medsegmnist.registry import register


@register
class NucleiSegMNIST2D(MedSegMNIST2D):
    flag = "nuclei2d"
    class_name = "NucleiSegMNIST2D"
    available_sizes = [256, 512, "native"]
    n_classes = 2
    modality = "Pathology"
    n_channels = 3

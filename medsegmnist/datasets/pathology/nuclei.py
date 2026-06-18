from medsegmnist.datasets.base import MedSegMNIST2D
from medsegmnist.registry import register


@register
class NucleiSegMNIST2D(MedSegMNIST2D):
    flag = "nuclei2d"
    class_name = "NucleiSegMNIST2D"
    organ = "nuclei"
    available_sizes = [256, 512, "native"]
    n_classes = 2
    modality = "Pathology"
    n_channels = 3
    citation = (
        '(1) Kumar, Neeraj, et al. "A dataset and a technique for generalized nuclear '
        'segmentation for computational pathology." '
        "IEEE Transactions on Medical Imaging 36.7 (2017): 1550-1560. "
        '(2) Samet, Refik, et al. "NuSeC: A Dataset for Nuclei Segmentation in Breast '
        'Cancer Histopathology Images." '
        "arXiv preprint arXiv:2507.14272 (2025)."
    )

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
        '(1) Kumar, Neeraj, et al. "A multi-organ nucleus segmentation challenge." '
        'IEEE Transactions on Medical Imaging 39.5 (2019): 1380-1391. '
        '(2) Samet, Refik, et al. "NuSeC: A Dataset for Nuclei Segmentation in Breast '
        'Cancer Histopathology Images." '
        "arXiv preprint arXiv:2507.14272 (2025)."
    )

    zenodo_record_id = "20694762"
    zenodo_file_ids = {}
    hf_repo_id = None

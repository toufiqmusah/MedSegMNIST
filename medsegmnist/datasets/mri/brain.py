from ..base import MedSegMNIST3D
from ...registry import register


@register
class BrainSegMNIST3D(MedSegMNIST3D):
    flag = "brain3d"
    class_name = "BrainSegMNIST3D"
    available_sizes = [96, 128, 224, "native"]
    n_classes = 4
    modality = "MRI"
    n_channels = 1

    citation = (
        'Adewole, Maruf, et al. "The BraTS-Africa dataset: expanding the brain tumor '
        'segmentation data to capture African populations." '
        "Radiology: Artificial Intelligence 7.4 (2025): e240528."
    )

    zenodo_record_id = None
    zenodo_file_ids = {}
    hf_repo_id = None

from ..base import MedSegMNIST2D
from ...registry import register


@register
class LungSegMNIST(MedSegMNIST2D):
    flag = "lung2d"
    class_name = "LungSegMNIST"
    available_sizes = [128, 256, 512]
    n_classes = 2
    modality = "X-ray"
    n_channels = 1

    zenodo_record_id = None
    zenodo_file_ids = {}
    hf_repo_id = None

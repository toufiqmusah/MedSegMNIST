from ..base import MedSegMNIST2D
from ...registry import register


@register
class FundusSegMNIST2D(MedSegMNIST2D):
    flag = "fives2d"
    class_name = "FundusSegMNIST2D"
    organ = "fundus"
    available_sizes = [256, 512, 1024, "native"]
    n_classes = 2
    modality = "Fundus photography"
    n_channels = 3

    citation = (
        'Jin, Kai, et al. "FIVES: A Fundus Image Dataset for Artificial Intelligence '
        'based Vessel Segmentation." Scientific Data 11.1 (2024): 1064.'
    )

    zenodo_record_id = "20694762"
    zenodo_file_ids = {}
    hf_repo_id = None

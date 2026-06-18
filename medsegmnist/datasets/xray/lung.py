from ..base import MedSegMNIST2D
from ...registry import register


@register
class LungSegMNIST2D(MedSegMNIST2D):
    flag = "lung2d"
    class_name = "LungSegMNIST2D"
    organ = "lung"
    available_sizes = [128, 256, 512]
    n_classes = 2
    modality = "X-ray"
    n_channels = 1

    citation = (
        "Danilov, Viacheslav; Proutski, Alex; Kirpich, Alexander; Litmanovich, Diana; "
        'Gankin, Yuriy (2022), "Chest X-ray dataset for lung segmentation", '
        "Mendeley Data, V2, doi: 10.17632/8gf9vpkhgy.2"
    )

    zenodo_record_id = None
    zenodo_file_ids = {}
    hf_repo_id = None

import pytest


def test_registry_contents():
    from medsegmnist.registry import DATASET_REGISTRY

    assert "brain3d" in DATASET_REGISTRY
    assert "lung2d" in DATASET_REGISTRY
    assert "nuclei2d" in DATASET_REGISTRY

    brain = DATASET_REGISTRY["brain3d"]
    assert brain["class"] == "BrainSegMNIST3D"
    assert brain["dimensionality"] == "3D"
    assert brain["modality"] == "MRI"
    assert brain["n_classes"] == 4
    assert brain["available_sizes"] == [96, 128, 224, "native"]

    lung = DATASET_REGISTRY["lung2d"]
    assert lung["class"] == "LungSegMNIST2D"
    assert lung["dimensionality"] == "2D"
    assert lung["modality"] == "X-ray"
    assert lung["n_classes"] == 2
    assert lung["available_sizes"] == [128, 256, 512]

    nuclei = DATASET_REGISTRY["nuclei2d"]
    assert nuclei["class"] == "NucleiSegMNIST2D"
    assert nuclei["dimensionality"] == "2D"
    assert nuclei["modality"] == "Pathology"
    assert nuclei["n_classes"] == 2
    assert nuclei["available_sizes"] == [256, 512, "native"]


def test_list_datasets():
    from medsegmnist.registry import list_datasets

    all_ds = list_datasets()
    assert len(all_ds) == 10

    threed = list_datasets(dimensionality="3D")
    assert len(threed) == 4
    flags_3d = [e[0] for e in threed]
    assert "abdomen3d" in flags_3d
    assert "brain3d" in flags_3d
    assert "spine3d" in flags_3d
    assert "knee3d" in flags_3d

    twod = list_datasets(dimensionality="2D")
    assert len(twod) == 6
    flags = [e[0] for e in twod]
    assert "lung2d" in flags
    assert "nuclei2d" in flags
    assert "polyp2d" in flags


def test_info():
    from medsegmnist.registry import info

    result = info("brain3d")
    assert result["flag"] == "brain3d"

    with pytest.raises(KeyError):
        info("nonexistent")


def test_info_via_module():
    import medsegmnist

    result = medsegmnist.info("lung2d")
    assert result["flag"] == "lung2d"

def test_top_level_imports():
    import medsegmnist

    assert hasattr(medsegmnist, "BrainSegMNIST3D")
    assert hasattr(medsegmnist, "LungSegMNIST")
    assert hasattr(medsegmnist, "NucleiSegMNIST")
    assert hasattr(medsegmnist, "info")
    assert hasattr(medsegmnist, "list_datasets")
    assert hasattr(medsegmnist, "__version__")


def test_example_models_import():
    import sys
    import os

    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    sys.path.insert(0, os.path.abspath(examples_dir))
    from unet import UNet2D
    from unet3d import UNet3D

    assert UNet2D is not None
    assert UNet3D is not None


def test_training_import():
    from medsegmnist.training import (
        DiceScore,
        IoUScore,
        DiceLoss,
        DiceCELoss,
        MedSegModule,
    )

    assert DiceScore is not None
    assert IoUScore is not None
    assert DiceLoss is not None
    assert DiceCELoss is not None
    assert MedSegModule is not None


def test_registry():
    from medsegmnist.registry import DATASET_REGISTRY, list_datasets

    assert "brain3d" in DATASET_REGISTRY
    assert "lung2d" in DATASET_REGISTRY
    assert "nuclei2d" in DATASET_REGISTRY
    entries = list_datasets()
    flags = [e[0] for e in entries]
    assert "brain3d" in flags
    assert "lung2d" in flags
    assert "nuclei2d" in flags

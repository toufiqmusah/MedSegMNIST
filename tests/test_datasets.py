import pytest

DATASET_DIR = "/teamspace/studios/this_studio/datasets"


class TestBrainSegMNIST3D:
    @pytest.mark.requires_data
    def test_init_defaults(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="train", root=DATASET_DIR)
        assert ds.flag == "brain3d"
        assert ds.dimensionality == "3D"
        assert ds.n_classes == 4

    @pytest.mark.requires_data
    def test_len_train(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="train", size=96, root=DATASET_DIR)
        assert len(ds) == 116

    @pytest.mark.requires_data
    def test_len_test(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="test", size=96, root=DATASET_DIR)
        assert len(ds) == 30

    @pytest.mark.requires_data
    def test_getitem_shape(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="train", size=96, root=DATASET_DIR)
        img, msk = ds[0]
        assert img.shape == (1, 96, 96, 64)
        assert msk.shape == (96, 96, 64)
        assert img.dtype == "float32"
        assert msk.dtype == "uint8"

    @pytest.mark.requires_data
    def test_all_sizes(self):
        from medsegmnist import BrainSegMNIST3D

        for size in [96, 128, 224, "native"]:
            ds = BrainSegMNIST3D(
                split="train", size=size, root=DATASET_DIR, mmap_mode="r"
            )
            img, _ = ds[0]
            assert img.shape[0] == 1

    @pytest.mark.requires_data
    def test_mmap_mode(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="train", size=96, root=DATASET_DIR, mmap_mode="r")
        img, msk = ds[0]
        assert img.shape == (1, 96, 96, 64)

    @pytest.mark.requires_data
    def test_get_data(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="train", size=96, root=DATASET_DIR)
        images, masks = ds.get_data()
        assert images.shape[0] == len(ds)
        assert masks.shape[0] == len(ds)

    @pytest.mark.requires_data
    def test_available_sizes(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="train", root=DATASET_DIR)
        assert ds.available_sizes == [96, 128, 224, "native"]

    @pytest.mark.requires_data
    def test_voxel_spacing(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="train", size=96, root=DATASET_DIR)
        spacing = ds.get_voxel_spacing()
        assert len(spacing) == 3
        assert spacing[0] == pytest.approx(2.5, abs=0.01)

    @pytest.mark.requires_data
    def test_fold(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="train", size=96, root=DATASET_DIR)
        train_subset, val_subset = ds.get_fold(0)
        assert len(train_subset) + len(val_subset) == len(ds)
        assert len(val_subset) > 0

    @pytest.mark.requires_data
    def test_transform(self):
        from medsegmnist import BrainSegMNIST3D

        transform = type("Identity", (), {"__call__": lambda self, x: x})()
        ds = BrainSegMNIST3D(
            split="train", size=96, root=DATASET_DIR, transform=transform
        )
        img, msk = ds[0]
        assert img is not None

    @pytest.mark.requires_data
    def test_get_label_names(self):
        from medsegmnist import BrainSegMNIST3D

        ds = BrainSegMNIST3D(split="train", root=DATASET_DIR)
        names = ds.get_label_names()
        assert names["0"] == "background"
        assert "1" in names
        assert "2" in names
        assert "3" in names

    @pytest.mark.requires_data
    def test_info(self):
        from medsegmnist import info

        info("brain3d")


class TestLungSegMNIST:
    @pytest.mark.requires_data
    def test_init_defaults(self):
        from medsegmnist import LungSegMNIST

        ds = LungSegMNIST(split="train", root=DATASET_DIR)
        assert ds.flag == "lung2d"
        assert ds.dimensionality == "2D"
        assert ds.n_classes == 2

    @pytest.mark.requires_data
    def test_len_train(self):
        from medsegmnist import LungSegMNIST

        ds = LungSegMNIST(split="train", size=128, root=DATASET_DIR)
        assert len(ds) == 5448

    @pytest.mark.requires_data
    def test_len_test(self):
        from medsegmnist import LungSegMNIST

        ds = LungSegMNIST(split="test", size=128, root=DATASET_DIR)
        assert len(ds) == 1362

    @pytest.mark.requires_data
    def test_getitem_shape(self):
        from medsegmnist import LungSegMNIST

        ds = LungSegMNIST(split="train", size=128, root=DATASET_DIR)
        img, msk = ds[0]
        assert img.shape == (1, 128, 128)
        assert msk.shape == (128, 128)
        assert img.dtype == "float32"
        assert msk.dtype == "uint8"

    @pytest.mark.requires_data
    def test_all_sizes(self):
        from medsegmnist import LungSegMNIST

        for size in [128, 256, 512]:
            ds = LungSegMNIST(split="train", size=size, root=DATASET_DIR, mmap_mode="r")
            img, _ = ds[0]
            assert img.ndim == 3

    @pytest.mark.requires_data
    def test_fold(self):
        from medsegmnist import LungSegMNIST

        ds = LungSegMNIST(split="train", size=128, root=DATASET_DIR)
        train_subset, val_subset = ds.get_fold(0)
        assert len(train_subset) + len(val_subset) == len(ds)

    @pytest.mark.requires_data
    def test_get_label_names(self):
        from medsegmnist import LungSegMNIST

        ds = LungSegMNIST(split="train", root=DATASET_DIR)
        names = ds.get_label_names()
        assert names["0"] == "background"
        assert "1" in names

    @pytest.mark.requires_data
    def test_init_failures(self):
        from medsegmnist import LungSegMNIST

        with pytest.raises((ValueError, AssertionError)):
            LungSegMNIST(split="invalid", root=DATASET_DIR)

    @pytest.mark.requires_data
    def test_info(self):
        from medsegmnist import info

        info("lung2d")


class TestNucleiSegMNIST:
    @pytest.mark.requires_data
    def test_init_defaults(self):
        from medsegmnist import NucleiSegMNIST

        ds = NucleiSegMNIST(split="train", root=DATASET_DIR)
        assert ds.flag == "nuclei2d"
        assert ds.dimensionality == "2D"
        assert ds.n_classes == 2
        assert ds.n_channels == 3

    @pytest.mark.requires_data
    def test_len_train(self):
        from medsegmnist import NucleiSegMNIST

        ds = NucleiSegMNIST(split="train", size=256, root=DATASET_DIR)
        assert len(ds) == 112

    @pytest.mark.requires_data
    def test_len_test(self):
        from medsegmnist import NucleiSegMNIST

        ds = NucleiSegMNIST(split="test", size=256, root=DATASET_DIR)
        assert len(ds) == 39

    @pytest.mark.requires_data
    def test_getitem_shape(self):
        from medsegmnist import NucleiSegMNIST

        ds = NucleiSegMNIST(split="train", size=256, root=DATASET_DIR)
        img, msk = ds[0]
        assert img.shape == (3, 256, 256)
        assert msk.shape == (256, 256)
        assert img.dtype == "float32"
        assert msk.dtype == "uint8"

    @pytest.mark.requires_data
    def test_all_sizes(self):
        from medsegmnist import NucleiSegMNIST

        for size in [256, 512, "native"]:
            ds = NucleiSegMNIST(
                split="train", size=size, root=DATASET_DIR, mmap_mode="r"
            )
            img, _ = ds[0]
            assert img.shape[0] == 3

    @pytest.mark.requires_data
    def test_get_label_names(self):
        from medsegmnist import NucleiSegMNIST

        ds = NucleiSegMNIST(split="train", root=DATASET_DIR)
        names = ds.get_label_names()
        assert names["0"] == "background"
        assert names["1"] == "nuclei"

    @pytest.mark.requires_data
    def test_fold(self):
        from medsegmnist import NucleiSegMNIST

        ds = NucleiSegMNIST(split="train", size=256, root=DATASET_DIR)
        train_subset, val_subset = ds.get_fold(0)
        assert len(train_subset) + len(val_subset) == len(ds)

    @pytest.mark.requires_data
    def test_info(self):
        from medsegmnist import info

        info("nuclei2d")

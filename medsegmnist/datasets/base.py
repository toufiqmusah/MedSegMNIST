import os
import json
import numpy as np

try:
    from torch.utils.data import Dataset, Subset
except ImportError:
    Dataset = object
    Subset = None

DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".medsegmnist")
_citations_shown = set()


def _show_citation(cls):
    citation = getattr(cls, "citation", None)
    if citation and cls.class_name not in _citations_shown:
        _citations_shown.add(cls.class_name)
        print(
            f"\n[MedSegMNIST] When using {cls.class_name}, please cite:\n"
            f"  {citation}\n",
            flush=True,
        )


class _MedSegMNISTBase(Dataset):
    flag: str = None
    class_name: str = None
    available_sizes: list = []
    n_classes: int = None
    modality: str = None
    dimensionality: str = None
    n_channels: int = 1

    def __init__(
        self,
        split="all",
        transform=None,
        target_transform=None,
        download=False,
        download_native=False,
        root=DEFAULT_ROOT,
        size=None,
        mmap_mode=None,
        verify=False,
    ):
        assert split in ("all", "train", "test"), (
            f"split must be 'all', 'train', or 'test', got {split!r}"
        )
        self._split = split
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.mmap_mode = mmap_mode

        if size is None:
            size = self.available_sizes[0]
        self.size = size
        self._validate_size()

        npz_path = self._resolve_npz_path()

        if not os.path.isfile(npz_path):
            if download:
                self.download()
            else:
                raise FileNotFoundError(
                    f"NPZ not found at {npz_path}. "
                    f"Set download=True or place the file at this path."
                )

        loader = np.load(npz_path, mmap_mode=mmap_mode)
        self._all_images = loader["train_images"]
        self._all_masks = loader["train_masks"]

        json_path = os.path.join(root, f"{self.flag}.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        self._resolve_indices()

        _show_citation(type(self))

    def _validate_size(self):
        if self.size not in self.available_sizes:
            raise ValueError(
                f"Invalid size {self.size!r} for {self.class_name}. "
                f"Available sizes: {self.available_sizes}"
            )

    def _resolve_npz_path(self):
        filename = (
            f"{self.flag}_native.npz"
            if self.size == "native"
            else f"{self.flag}_{self.size}.npz"
        )
        return os.path.join(self.root, filename)

    def _resolve_indices(self):
        folds = self.meta.get("cv_folds", {})
        fold0 = folds.get("fold_0", {})
        if self._split == "test":
            self._indices = np.asarray(fold0.get("test", []), dtype=int)
            if len(self._indices) == 0:
                self._indices = np.arange(len(self._all_images))
        elif self._split == "train":
            self._indices = np.asarray(fold0.get("train", []), dtype=int)
            if len(self._indices) == 0:
                self._indices = np.arange(len(self._all_images))
        else:
            self._indices = np.arange(len(self._all_images))

    def __len__(self):
        return len(self._indices)

    def _make_full_view(self):
        ds = object.__new__(type(self))
        ds.__dict__.update(self.__dict__)
        ds._indices = np.arange(len(self._all_images))
        return ds

    def get_fold(self, fold_index):
        if Subset is None:
            raise ImportError(
                "PyTorch is required for get_fold(). Install with: pip install medsegmnist[torch]"
            )
        fold = self.meta["cv_folds"][f"fold_{fold_index}"]
        full = self._make_full_view()
        return Subset(full, fold["train"]), Subset(full, fold["test"])

    def get_label_names(self):
        return self.meta.get("label_names", {})

    def info(self):
        import pprint
        keys = [
            "flag",
            "class_name",
            "name",
            "version",
            "dimensionality",
            "modality",
            "anatomy",
            "available_sizes",
            "label_names",
        ]
        filtered = {k: self.meta.get(k) for k in keys}
        pprint.pprint(filtered)

    def __repr__(self):
        return (
            f"{self.class_name}(split={self._split!r}, size={self.size!r}, "
            f"n={len(self)})"
        )

    def download(self):
        raise NotImplementedError("download() not yet implemented")

    def download_native(self):
        raise NotImplementedError("download_native() not yet implemented")


class MedSegMNIST3D(_MedSegMNISTBase):
    dimensionality = "3D"

    def __getitem__(self, index):
        actual = self._indices[index]
        image = self._all_images[actual].copy() if self.mmap_mode else self._all_images[actual]
        mask = self._all_masks[actual].copy() if self.mmap_mode else self._all_masks[actual]

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        image = np.expand_dims(image, 0).astype(np.float32)
        mask = mask.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def get_data(self):
        return self._all_images, self._all_masks

    def get_voxel_spacing(self):
        if self.size == "native":
            return tuple(self.meta.get("native_voxel_spacing_mm", []))
        sz = str(self.size)
        return tuple(
            self.meta.get("standardised_sizes", {})
            .get(sz, {})
            .get("voxel_spacing_mm", [])
        )


class MedSegMNIST2D(_MedSegMNISTBase):
    dimensionality = "2D"

    def __getitem__(self, index):
        actual = self._indices[index]
        image = self._all_images[actual].copy() if self.mmap_mode else self._all_images[actual]
        mask = self._all_masks[actual].copy() if self.mmap_mode else self._all_masks[actual]

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        if image.ndim == 2:
            image = np.expand_dims(image, 0).astype(np.float32)
        else:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = mask.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def get_data(self):
        return self._all_images, self._all_masks

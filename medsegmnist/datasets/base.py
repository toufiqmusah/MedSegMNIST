import os
import json
import numpy as np

try:
    from torch.utils.data import Dataset, Subset
except ImportError:
    Dataset = object
    Subset = None

DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".medsegmnist")


class MedSegMNIST3D(Dataset):
    """Base class for 3D volumetric medical image segmentation datasets.

    Loads pre-processed NPZ files from disk and provides a unified PyTorch
    ``Dataset`` interface. Subclasses should set class-level attributes
    (``flag``, ``class_name``, ``available_sizes``, ``n_classes``,
    ``modality``, ``n_channels``) and decorate the class with ``@register``.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    transform : callable, optional
        Applied to the image tensor (channel-first float32).
    target_transform : callable, optional
        Applied to the mask tensor (uint8).
    download : bool
        If ``True`` and the NPZ is missing, call ``self.download()``.
        Not yet implemented for most datasets.
    root : str
        Directory containing the ``{flag}_{size}.npz`` and
        ``{flag}.json`` files.  Defaults to ``~/.medsegmnist``.
    size : int or str, optional
        One of ``self.available_sizes``.  Falls back to the first
        available size when ``None``.
    mmap_mode : str, optional
        Passed to ``numpy.load``.  Use ``"r"`` to memory-map large files
        without loading them fully into RAM.
    verify : bool
        Reserved for checksum verification (not yet implemented).

    Attributes
    ----------
    images : ndarray
        Image array of shape ``(N, D, H, W)``.
    masks : ndarray
        Label array of shape ``(N, D, H, W)``.
    meta : dict
        Full dataset metadata loaded from the companion JSON file.
    """

    flag: str = None
    class_name: str = None
    available_sizes: list = []
    n_classes: int = None
    modality: str = None
    dimensionality: str = "3D"
    n_channels: int = 1

    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        download=False,
        download_native=False,
        root=DEFAULT_ROOT,
        size=None,
        mmap_mode=None,
        verify=False,
    ):
        assert split in ("train", "test"), (
            f"split must be 'train' or 'test', got {split!r}"
        )
        self.split = split
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
        self.images = (
            loader["train_images"] if split == "train" else loader["test_images"]
        )
        self.masks = loader["train_masks"] if split == "train" else loader["test_masks"]

        json_path = os.path.join(root, f"{self.flag}.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].copy() if self.mmap_mode else self.images[index]
        mask = self.masks[index].copy() if self.mmap_mode else self.masks[index]

        image = np.expand_dims(image, 0).astype(np.float32)
        mask = mask.astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def get_data(self):
        """Return the raw numpy arrays.

        Returns
        -------
        images : ndarray
            Shape ``(N, D, H, W)``.
        masks : ndarray
            Shape ``(N, D, H, W)``.
        """
        return self.images, self.masks

    def get_fold(self, fold_index):
        """Return train/validation subsets for cross-validation.

        Parameters
        ----------
        fold_index : int
            Fold index (0–4).

        Returns
        -------
        train_subset : Subset
        val_subset : Subset
        """
        if Subset is None:
            raise ImportError(
                "PyTorch is required for get_fold(). Install with: pip install medsegmnist[torch]"
            )
        assert self.split == "train", "get_fold() only valid for split='train'"
        fold = self.meta["cv_folds"][f"fold_{fold_index}"]
        return Subset(self, fold["train"]), Subset(self, fold["val"])

    def get_label_names(self):
        """Map class IDs to human-readable names.

        Returns
        -------
        dict
            ``{"0": "background", "1": "tumour", ...}``
        """
        return self.meta.get("label_names", {})

    def get_voxel_spacing(self):
        """Voxel spacing in millimetres.

        Returns
        -------
        tuple
            ``(spacing_x, spacing_y, spacing_z)`` or ``()`` if unknown.
        """
        if self.size == "native":
            return tuple(self.meta.get("native_voxel_spacing_mm", []))
        sz = str(self.size)
        return tuple(
            self.meta.get("standardised_sizes", {})
            .get(sz, {})
            .get("voxel_spacing_mm", [])
        )

    def info(self):
        """Pretty-print key metadata fields."""
        import pprint

        keys = [
            "flag",
            "class_name",
            "name",
            "version",
            "dimensionality",
            "modality",
            "anatomy",
            "n_train",
            "n_test",
            "available_sizes",
            "label_names",
        ]
        filtered = {k: self.meta.get(k) for k in keys}
        pprint.pprint(filtered)

    def __repr__(self):
        return (
            f"{self.class_name}(split={self.split!r}, size={self.size!r}, "
            f"n={len(self)})"
        )

    def download(self):
        raise NotImplementedError("download() not yet implemented")

    def download_native(self):
        raise NotImplementedError("download_native() not yet implemented")


class MedSegMNIST2D(Dataset):
    """Base class for 2D medical image segmentation datasets.

    Same interface as ``MedSegMNIST3D``, adapted for 2D data.  Images with
    shape ``(H, W)`` get a channel dimension prepended; images with shape
    ``(H, W, C)`` are transposed to ``(C, H, W)``.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    transform : callable, optional
        Applied to the image tensor (channel-first float32).
    target_transform : callable, optional
        Applied to the mask tensor (uint8).
    download : bool
        If ``True`` and the NPZ is missing, call ``self.download()``.
    root : str
        Directory containing NPZ and JSON files.  Defaults to
        ``~/.medsegmnist``.
    size : int or str, optional
        One of ``self.available_sizes``.  Falls back to the first
        available size when ``None``.
    mmap_mode : str, optional
        Passed to ``numpy.load``.  Use ``"r"`` for memory-mapped access.
    verify : bool
        Reserved for checksum verification (not yet implemented).

    Attributes
    ----------
    images : ndarray
        Image array of shape ``(N, H, W)`` or ``(N, H, W, C)``
        (before channel-first conversion in ``__getitem__``).
    masks : ndarray
        Label array of shape ``(N, H, W)``.
    meta : dict
        Full dataset metadata loaded from the companion JSON file.
    """

    flag: str = None
    class_name: str = None
    available_sizes: list = []
    n_classes: int = None
    modality: str = None
    dimensionality: str = "2D"
    n_channels: int = 1

    def __init__(
        self,
        split,
        transform=None,
        target_transform=None,
        download=False,
        download_native=False,
        root=DEFAULT_ROOT,
        size=None,
        mmap_mode=None,
        verify=False,
    ):
        assert split in ("train", "test"), (
            f"split must be 'train' or 'test', got {split!r}"
        )
        self.split = split
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
        self.images = (
            loader["train_images"] if split == "train" else loader["test_images"]
        )
        self.masks = loader["train_masks"] if split == "train" else loader["test_masks"]

        json_path = os.path.join(root, f"{self.flag}.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].copy() if self.mmap_mode else self.images[index]
        mask = self.masks[index].copy() if self.mmap_mode else self.masks[index]

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
        """Return the raw numpy arrays.

        Returns
        -------
        images : ndarray
            Shape ``(N, H, W)`` or ``(N, H, W, C)``.
        masks : ndarray
            Shape ``(N, H, W)``.
        """
        return self.images, self.masks

    def get_fold(self, fold_index):
        """Return train/validation subsets for cross-validation.

        Parameters
        ----------
        fold_index : int
            Fold index (0–4).

        Returns
        -------
        train_subset : Subset
        val_subset : Subset
        """
        if Subset is None:
            raise ImportError(
                "PyTorch is required for get_fold(). Install with: pip install medsegmnist[torch]"
            )
        assert self.split == "train", "get_fold() only valid for split='train'"
        fold = self.meta["cv_folds"][f"fold_{fold_index}"]
        return Subset(self, fold["train"]), Subset(self, fold["val"])

    def get_label_names(self):
        """Map class IDs to human-readable names.

        Returns
        -------
        dict
            ``{"0": "background", "1": "nuclei", ...}``
        """
        return self.meta.get("label_names", {})

    def info(self):
        """Pretty-print key metadata fields."""
        import pprint

        keys = [
            "flag",
            "class_name",
            "name",
            "version",
            "dimensionality",
            "modality",
            "anatomy",
            "n_train",
            "n_test",
            "available_sizes",
            "label_names",
        ]
        filtered = {k: self.meta.get(k) for k in keys}
        pprint.pprint(filtered)

    def __repr__(self):
        return (
            f"{self.class_name}(split={self.split!r}, size={self.size!r}, "
            f"n={len(self)})"
        )

    def download(self):
        raise NotImplementedError("download() not yet implemented")

    def download_native(self):
        raise NotImplementedError("download_native() not yet implemented")

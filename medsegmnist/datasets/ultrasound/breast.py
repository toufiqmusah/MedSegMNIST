import os
import json
import numpy as np
from ..base import MedSegMNIST2D
from ...registry import register


def _normalize_mammogram(image):
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, [1.0, 99.0])
    img = np.clip(img, lo, hi)
    mean = img.mean()
    std = img.std()
    if std > 0:
        img = (img - mean) / std
    return img.astype(np.float32)


@register
class BreastSegMNIST2D(MedSegMNIST2D):
    flag = "breast2d"
    class_name = "BreastSegMNIST2D"
    organ = "breast"
    available_sizes = [128, 256, 512, "native"]
    n_classes = 2
    modality = "Mammography"
    n_channels = 1

    citation = (
        'Oza, Parita, et al. "Digital mammography dataset for breast cancer diagnosis '
        'research (DMID) with breast mass segmentation analysis." '
        'Biomedical Engineering Letters 14.2 (2024): 317-330.'
    )

    zenodo_record_id = "20694762"
    zenodo_file_ids = {}
    hf_repo_id = None

    def __init__(self, split="all", transform=None, target_transform=None,
                 download=False, download_native=False, root=None, size=None,
                 mmap_mode=None, verify=False):
        if root is None:
            from ..base import DEFAULT_ROOT
            root = DEFAULT_ROOT
        if size == "native" or (size is None and "native" in self.available_sizes):
            self._split = split
            self.transform = transform
            self.target_transform = target_transform
            self.root = root
            self.mmap_mode = mmap_mode
            self.size = size if size is not None else "native"
            self._validate_size()

            json_path = os.path.join(self._organ_root(), f"{self.flag}.json")
            if os.path.isfile(json_path):
                with open(json_path) as f:
                    self.meta = json.load(f)
            else:
                self.meta = {}

            n_total = self.meta.get("n_total", 0)
            self._all_images = np.zeros(n_total)
            self._all_masks = np.zeros(n_total)
            self._resolve_indices()

            self._native_dir = os.path.join(self._organ_root(), f"{self.flag}_native")
            self._native_ids = [f"{self.flag}_{i}" for i in range(n_total)]

            from ..base import _show_citation
            _show_citation(type(self))
        else:
            MedSegMNIST2D.__init__(
                self, split=split, transform=transform,
                target_transform=target_transform, download=download,
                download_native=download_native, root=root, size=size,
                mmap_mode=mmap_mode, verify=verify,
            )
            self._native_dir = os.path.join(self._organ_root(), f"{self.flag}_native")
            self._native_ids = [
                f"{self.flag}_{i}" for i in range(len(self._all_images))
            ]

    def _resolve_npz_path(self):
        if self.size == "native":
            return os.path.join(self._organ_root(), f"{self.flag}_native.npz")
        return super()._resolve_npz_path()

    def __getitem__(self, index):
        if self.size != "native":
            return super().__getitem__(index)

        actual = self._indices[index]
        vol_id = self._native_ids[actual]
        npz_path = os.path.join(self._native_dir, f"{vol_id}.npz")
        npz = np.load(npz_path)
        image = _normalize_mammogram(npz["image"])
        mask = npz["mask"].astype(np.uint8)

        image = np.expand_dims(image, 0).astype(np.float32)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def get_data(self):
        if self.size == "native":
            raise RuntimeError(
                "get_data() not supported for native size. "
                "Access individual samples via indexing instead."
            )
        return super().get_data()

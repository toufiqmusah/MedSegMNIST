import os
import json
import numpy as np
from ..base import DEFAULT_ROOT, MedSegMNIST2D
from ...registry import register


@register
class SkinSegMNIST2D(MedSegMNIST2D):
    flag = "derm2d"
    class_name = "SkinSegMNIST2D"
    organ = "dermoscopy"
    available_sizes = [128, 256, 512, "native"]
    n_classes = 2
    modality = "Dermoscopy"
    n_channels = 3

    citation = (
        'Codella, Noel, et al. "Skin Lesion Analysis Toward Melanoma Detection '
        '2018: A Challenge Hosted by the International Skin Imaging Collaboration '
        '(ISIC)." arXiv preprint arXiv:1902.03368 (2019).'
    )

    zenodo_record_id = "20694762"
    zenodo_file_ids = {}
    hf_repo_id = None

    def __init__(self, split="all", transform=None, target_transform=None,
                 download=False, download_native=False, root=None, size=None,
                 mmap_mode=None, verify=False):
        if root is None:
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

            n_total = self.meta.get("n_samples", 0)
            self._all_images = np.zeros(n_total)
            self._all_masks = np.zeros(n_total)
            self._resolve_indices()

            self._native_dir = os.path.join(self._organ_root(), "derm2d_native")
            self._native_meta = self.meta.get("native_metadata", [])

            from ..base import _show_citation
            _show_citation(type(self))
        else:
            MedSegMNIST2D.__init__(
                self, split=split, transform=transform,
                target_transform=target_transform, download=download,
                download_native=download_native, root=root, size=size,
                mmap_mode=mmap_mode, verify=verify,
            )
            self._native_dir = os.path.join(self._organ_root(), "derm2d_native")
            self._native_meta = self.meta.get("native_metadata", [])

    def _resolve_npz_path(self):
        if self.size == "native":
            return os.path.join(self._organ_root(), "derm2d_native.npz")
        return super()._resolve_npz_path()

    def __getitem__(self, index):
        if self.size != "native":
            actual = self._indices[index]
            image = self._all_images[actual].copy() if self.mmap_mode else self._all_images[actual]
            mask = self._all_masks[actual].copy() if self.mmap_mode else self._all_masks[actual]

            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0

            if image.ndim == 2:
                image = np.expand_dims(image, 0)
            else:
                image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            mask = mask.astype(np.uint8)

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)

            return image, mask

        # Native: load per-image NPZ
        actual = self._indices[index]
        entry = self._native_meta[actual]
        npz_path = os.path.join(self._native_dir, entry["file"])
        npz = np.load(npz_path)
        image = npz["image"].astype(np.float32) / 255.0
        mask = npz["mask"].astype(np.uint8)

        if image.ndim == 2:
            image = np.expand_dims(image, 0)
        else:
            image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)

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

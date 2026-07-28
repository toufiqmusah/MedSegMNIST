import os
import json
import numpy as np
from ..base import DEFAULT_ROOT, MedSegMNIST3D
from ...registry import register

HU_RANGE = (-1000, 1000)


def _decode_hu(image_uint8):
    return image_uint8.astype(np.float32) / 255.0 * 2000.0 - 1000.0


def _normalize_ct(volume_np):
    volume = volume_np.astype(np.float32)
    volume = np.clip(volume, HU_RANGE[0], HU_RANGE[1])
    foreground = volume[volume > HU_RANGE[0]]
    if len(foreground) > 0:
        mean = foreground.mean()
        std = foreground.std()
        if std > 0:
            volume = (volume - mean) / std
    return volume


@register
class AbdomenSegMNIST3D(MedSegMNIST3D):
    flag = "abdomen3d"
    class_name = "AbdomenSegMNIST3D"
    organ = "abdomen"
    available_sizes = [64, 96, 128, 192, "native"]
    n_classes = 6
    modality = "CT"
    n_channels = 1
    rot90_k = 2

    citation = (
        'Rister, Blaine, et al. "CT-ORG, a new dataset for multiple organ segmentation '
        'in computed tomography." Scientific Data 7.1 (2020): 381.'
    )

    zenodo_record_id = "20694762"
    zenodo_file_ids = {}
    hf_repo_id = None

    def __init__(self, split="all", transform=None, target_transform=None,
                 download=False, download_native=False, root=None, size=None,
                 mmap_mode=None, verify=False):
        if root is None:
            root = DEFAULT_ROOT
        # For native, skip base loading and set up metadata manually
        if size == "native" or (size is None and "native" in self.available_sizes):
            self._split = split
            self.transform = transform
            self.target_transform = target_transform
            self.root = root
            self.mmap_mode = mmap_mode
            self.size = size if size is not None else "native"
            self._validate_size()

            json_path = os.path.join(self.root, f"{self.flag}.json")
            if os.path.isfile(json_path):
                with open(json_path) as f:
                    self.meta = json.load(f)
            else:
                self.meta = {}

            n_total = self.meta.get("n_total", 0)
            self._all_images = np.zeros(n_total)
            self._all_masks = np.zeros(n_total)
            self._resolve_indices()

            self._native_dir = os.path.join(self.root, "abdomen3d_native")
            self._native_ids = [f"ct_org_volume-{i}" for i in range(n_total)]

            from ..base import _show_citation
            _show_citation(type(self))
        else:
            MedSegMNIST3D.__init__(
                self, split=split, transform=transform,
                target_transform=target_transform, download=download,
                download_native=download_native, root=root, size=size,
                mmap_mode=mmap_mode, verify=verify,
            )
            self._native_dir = os.path.join(self.root, "abdomen3d_native")
            self._native_ids = [
                f"ct_org_volume-{i}" for i in range(len(self._all_images))
            ]

    def _resolve_npz_path(self):
        if self.size == "native":
            return os.path.join(self.root, "abdomen3d_native.npz")
        return super()._resolve_npz_path()

    def __getitem__(self, index):
        if self.size != "native":
            return super().__getitem__(index)

        actual = self._indices[index]
        vol_id = self._native_ids[actual]
        npz_path = os.path.join(self._native_dir, f"{vol_id}.npz")
        npz = np.load(npz_path)
        image = _decode_hu(npz["image"])
        image = _normalize_ct(image)
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

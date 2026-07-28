import os
import json
import numpy as np
from ..base import DEFAULT_ROOT, MedSegMNIST3D
from ...registry import register


@register
class SpineSegMNIST3D(MedSegMNIST3D):
    flag = "spine3d"
    class_name = "SpineSegMNIST3D"
    organ = "spine"
    available_sizes = [64, 96, 128, 192, "native"]
    n_classes = 3
    modality = "MR"
    n_channels = 1

    citation = (
        'Zhou, Longfei, et al. "The Duke University Cervical Spine MRI Segmentation '
        'Dataset (CSpineSeg)." Scientific Data 12.1 (2025): 1695.'
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

            self._native_dir = os.path.join(self._organ_root(), "spine3d_native")
            self._native_meta = self.meta.get("native_metadata", [])
            self._encode_scale = self.meta.get("encode_scale", 255.0 / 6.0)
            self._encode_offset = self.meta.get("encode_offset", -3.0)

            from ..base import _show_citation
            _show_citation(type(self))
        else:
            MedSegMNIST3D.__init__(
                self, split=split, transform=transform,
                target_transform=target_transform, download=download,
                download_native=download_native, root=root, size=size,
                mmap_mode=mmap_mode, verify=verify,
            )
            self._native_dir = os.path.join(self._organ_root(), "spine3d_native")
            self._native_meta = self.meta.get("native_metadata", [])
            self._encode_scale = self.meta.get("encode_scale", 255.0 / 6.0)
            self._encode_offset = self.meta.get("encode_offset", -3.0)

    def _resolve_npz_path(self):
        if self.size == "native":
            return os.path.join(self._organ_root(), "spine3d_native.npz")
        return super()._resolve_npz_path()

    def _decode_uint8(self, image_uint8):
        return image_uint8.astype(np.float32) / self._encode_scale + self._encode_offset

    def __getitem__(self, index):
        if self.size != "native":
            actual = self._indices[index]
            image = self._all_images[actual].copy() if self.mmap_mode else self._all_images[actual]
            mask = self._all_masks[actual].copy() if self.mmap_mode else self._all_masks[actual]

            if image.dtype == np.uint8:
                image = self._decode_uint8(image)

            mask = mask.astype(np.uint8)
            # FIXME(orientation): rotate 180° until preprocessing is corrected
            mask = np.rot90(mask, k=2, axes=(0, 1))

            image = np.expand_dims(image, 0).astype(np.float32)

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)

            return image, mask

        # Native: load per-volume NPZ
        actual = self._indices[index]
        entry = self._native_meta[actual]
        npz_path = os.path.join(self._native_dir, entry["file"])
        npz = np.load(npz_path)
        image = self._decode_uint8(npz["image"])
        mask = npz["mask"].astype(np.uint8)
        # FIXME(orientation): rotate 180° until preprocessing is corrected
        mask = np.rot90(mask, k=2, axes=(0, 1))

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

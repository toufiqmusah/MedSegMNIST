import os
import json
import numpy as np
import threading

# Module-level context: set by __getitem__ so viz functions auto-detect
# the active dataset's view_axis / rot90_k without extra parameters.
_viz_ctx = threading.local()
_viz_ctx.view_axis = -1
_viz_ctx.rot90_k = 0
_viz_ctx.label_names = {}

def _set_viz_context(view_axis=-1, rot90_k=0, label_names=None):
    _viz_ctx.view_axis = view_axis
    _viz_ctx.rot90_k = rot90_k
    _viz_ctx.label_names = label_names or {}

try:
    from torch.utils.data import Dataset, Subset
except ImportError:
    Dataset = object
    Subset = None

DEFAULT_ROOT = os.path.normpath(
    os.path.join(os.path.expanduser("~"), ".medsegmnist", "data")
)
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
    view_axis: int = -1  # 3D viz: -1=last, 0=axial, 1=coronal; override per dataset
    rot90_k: int = 0      # 90° rotations to apply to 2D slices for display

    SIZE_ALIASES = {"low": None, "mid": None, "high": None, "native": "native"}

    @classmethod
    def _resolve_size_alias(cls, size):
        if size in cls.SIZE_ALIASES:
            alias_map = cls._build_size_aliases()
            if size in alias_map:
                return alias_map[size]
            return size
        return size

    @classmethod
    def _build_size_aliases(cls):
        int_sizes = sorted([s for s in cls.available_sizes if isinstance(s, int)])
        if not int_sizes:
            return {"low": None, "mid": None, "high": None, "native": "native"}
        mid = min(int_sizes, key=lambda x: abs(x - 256))
        return {
            "low": int_sizes[0],
            "mid": mid,
            "high": int_sizes[-1],
            "native": "native",
        }

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
        self.root = root if root is not None else DEFAULT_ROOT
        self.mmap_mode = mmap_mode

        if size is None:
            size = self.available_sizes[0]
        self.size = self._resolve_size_alias(size)
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

        json_path = os.path.join(self._organ_root(), f"{self.flag}.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        self._resolve_indices()

        _show_citation(type(self))

    def _validate_size(self):
        valid = self.available_sizes + list(self.SIZE_ALIASES)
        if self.size not in valid:
            raise ValueError(
                f"Invalid size {self.size!r} for {self.class_name}. "
                f"Available sizes: {self.available_sizes}, "
                f"aliases: {list(self.SIZE_ALIASES)}"
            )

    organ: str = None

    def _organ_root(self):
        organ_dir = getattr(self, "organ", None)
        return os.path.join(self.root, organ_dir) if organ_dir else self.root

    def _resolve_npz_path(self):
        filename = (
            f"{self.flag}_native.npz"
            if self.size == "native"
            else f"{self.flag}_{self.size}.npz"
        )
        return os.path.join(self._organ_root(), filename)

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

    def set_label_names(self, label_names):
        self.meta["label_names"] = label_names

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

    def download(self, download_all=False):
        import hashlib
        import json as _json
        import os as _os
        import urllib.request
        import urllib.parse
        import sys

        record_id = getattr(self, "zenodo_record_id", None)
        if not record_id:
            raise NotImplementedError(
                f"download() not available for {self.class_name}. "
                "No Zenodo record ID configured."
            )

        organ_dir = self._organ_root()
        _os.makedirs(organ_dir, exist_ok=True)

        api_url = f"https://zenodo.org/api/records/{record_id}"
        print(f"[MedSegMNIST] Fetching file manifest for {self.class_name}...")
        try:
            with urllib.request.urlopen(api_url, timeout=30) as resp:
                record = _json.loads(resp.read().decode())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Zenodo record {record_id}: {e}")

        dataset_files = [
            f for f in record["files"] if f["key"].startswith(self.flag)
        ]
        if not dataset_files:
            print(f"  No files found for flag '{self.flag}' on Zenodo.")
            return

        for file_info in dataset_files:
            filename = file_info["key"]
            expected_md5 = file_info["checksum"].replace("md5:", "")
            dest_path = _os.path.join(organ_dir, filename)

            if _os.path.isfile(dest_path):
                if self._check_md5(dest_path, expected_md5):
                    print(f"  {filename} OK, skipping.")
                    continue
                print(f"  {filename} checksum mismatch, re-downloading.")

            file_url = file_info["links"]["self"]
            print(f"  Downloading {filename}...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(file_url, dest_path)
            except Exception as e:
                print(f"FAILED: {e}")
                if _os.path.isfile(dest_path):
                    _os.remove(dest_path)
                raise

            if not self._check_md5(dest_path, expected_md5):
                _os.remove(dest_path)
                raise RuntimeError(
                    f"MD5 mismatch for {filename} after download."
                )
            print("done.")

    @staticmethod
    def _check_md5(filepath, expected_md5):
        import hashlib
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                md5.update(chunk)
        return md5.hexdigest() == expected_md5

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

        _set_viz_context(self.view_axis, self.rot90_k, self.get_label_names())
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

        _set_viz_context(self.view_axis, self.rot90_k, self.get_label_names())
        return image, mask

    def get_data(self):
        return self._all_images, self._all_masks

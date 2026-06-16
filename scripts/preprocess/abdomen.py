import argparse
import os
import tempfile
import shutil
import zipfile
import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold
from common import (
    resample_and_resize,
    normalize_ct,
    save_large_npz,
    save_metadata,
    write_checksum,
    compute_padded_shape,
    pad_to_shape,
)

FLAG = "abdomen3d"
CLASS_NAME = "AbdomenSegMNIST3D"
SOURCE_NAME = "CT-ORG"
MODALITY = "CT"
ANATOMY = "Abdomen"
DIMENSIONALITY = "3D"

STANDARDISED_SIZES = {
    64: {"shape": (64, 64, 64), "spacing": (6.22, 6.22, 7.11)},
    96: {"shape": (96, 96, 96), "spacing": (4.15, 4.15, 4.74)},
    128: {"shape": (128, 128, 128), "spacing": (3.11, 3.11, 3.55)},
    192: {"shape": (192, 192, 192), "spacing": (2.07, 2.07, 2.37)},
}

NATIVE_SPACING_APPROX = (0.78, 0.78, 1.0)
HU_RANGE = (-1000, 1000)

LABEL_NAMES = {
    "0": "background",
    "1": "lung",
    "2": "liver",
    "3": "kidney",
    "4": "bone",
    "5": "urinary_bladder",
}

LABEL_ORIGINAL_VALUES = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


def collect_volumes(raw_dir):
    img_dir = os.path.join(raw_dir, "images")
    seg_dir = os.path.join(raw_dir, "segmentations")
    volumes = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".nii.gz"):
            continue
        vol_id = fname.replace("_0000.nii.gz", "")
        img_path = os.path.join(img_dir, fname)
        seg_path = os.path.join(seg_dir, vol_id)
        if not os.path.isdir(seg_path):
            continue
        parts = sorted(os.listdir(seg_path))
        if not parts:
            continue
        seg_file = os.path.join(seg_path, parts[-1])
        volumes.append({"id": vol_id, "image": img_path, "seg": seg_file})
    return volumes


def process_one_volume(entry, target_shape):
    img = nib.load(entry["image"]).get_fdata(dtype=np.float32)
    seg = nib.load(entry["seg"]).get_fdata().astype(np.uint8)
    mask = np.where((seg >= 1) & (seg <= 5), seg, 0).astype(np.uint8)
    if target_shape is not None:
        img_rs, msk_rs = resample_and_resize(img, mask, target_shape)
    else:
        img_rs, msk_rs = img.copy(), mask.copy()
    img_norm = normalize_ct(img_rs, HU_RANGE)
    return img_norm, msk_rs


def crop_or_pad_to_shape(volume, target_shape, constant=0):
    result = np.full(target_shape, constant, dtype=volume.dtype)
    slices = tuple(
        slice(0, min(s, t)) for s, t in zip(volume.shape, target_shape)
    )
    result[slices] = volume[slices]
    return result


def load_volume_shapes(volumes):
    shapes = []
    for v in volumes:
        img = nib.load(v["image"])
        shapes.append(list(img.shape))
    return shapes


def compute_native_out_shape(all_shapes, percentile=95):
    z_sizes = [s[2] for s in all_shapes]
    p95 = int(np.percentile(z_sizes, percentile))
    return [512, 512, p95]


def main(raw_dir, out_dir, sizes):
    os.makedirs(out_dir, exist_ok=True)

    print("Collecting volumes...")
    volumes = collect_volumes(raw_dir)
    n_total = len(volumes)
    print(f"  Found {n_total} volumes")

    print("Loading volume shapes...")
    all_shapes = load_volume_shapes(volumes)
    print(f"  Z range: {min(s[2] for s in all_shapes)}-{max(s[2] for s in all_shapes)}")

    print("Creating 5-fold CV splits...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = {}
    for i, (tr, te) in enumerate(kf.split(np.arange(n_total))):
        cv_folds[f"fold_{i}"] = {"train": tr.tolist(), "test": te.tolist()}

    per_sample_meta = []
    for v, shp in zip(volumes, all_shapes):
        per_sample_meta.append(
            {
                "original_id": v["id"],
                "original_spacing_mm": list(NATIVE_SPACING_APPROX),
                "original_shape": list(shp),
                "split": "all",
            }
        )

    generated_sizes = []
    native_padded_shape = None

    for size_val in sizes:
        print(f"Processing size={size_val}...")

        if size_val == "native":
            spec = None
            target_shape = None
            out_shape = compute_native_out_shape(all_shapes)
            n_vox = np.prod(out_shape) * n_total
            print(f"  Native padded shape: {out_shape} ({n_vox/1e6:.0f}M voxels, "
                  f"{n_vox*4/1e9:.1f} GB float32)")

            temp_dir = tempfile.mkdtemp()
            try:
                imgs_path = os.path.join(temp_dir, "train_images.npy")
                msks_path = os.path.join(temp_dir, "train_masks.npy")
                test_img_path = os.path.join(temp_dir, "test_images.npy")
                test_msk_path = os.path.join(temp_dir, "test_masks.npy")

                out_images = np.lib.format.open_memmap(
                    imgs_path, mode="w+", dtype=np.float32,
                    shape=(n_total, *out_shape)
                )
                out_masks = np.lib.format.open_memmap(
                    msks_path, mode="w+", dtype=np.uint8,
                    shape=(n_total, *out_shape)
                )

                for i, v in enumerate(volumes):
                    if (i + 1) % 30 == 0 or i == 0:
                        print(f"    {i + 1}/{n_total}")
                    img_norm, msk = process_one_volume(v, target_shape)
                    if out_images.shape[1:] != img_norm.shape:
                        img_norm = crop_or_pad_to_shape(img_norm, out_shape, constant=-1000)
                        msk = crop_or_pad_to_shape(msk, out_shape, constant=0)
                    out_images[i] = img_norm
                    out_masks[i] = msk

                out_images.flush()
                out_masks.flush()
                del out_images, out_masks

                np.save(test_img_path, np.zeros((0, *out_shape), dtype=np.float32))
                np.save(test_msk_path, np.zeros((0, *out_shape), dtype=np.uint8))

                npz_path = os.path.join(out_dir, "abdomen3d_native.npz")
                print(f"    Zipping NPZ...")
                with zipfile.ZipFile(npz_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name in ["train_images", "train_masks", "test_images", "test_masks"]:
                        p = os.path.join(temp_dir, f"{name}.npy")
                        zf.write(p, f"{name}.npy")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            spec = STANDARDISED_SIZES[size_val]
            target_shape = spec["shape"]
            out_shape = target_shape

            out_images = np.zeros((n_total, *out_shape), dtype=np.float32)
            out_masks = np.zeros((n_total, *out_shape), dtype=np.uint8)

            for i, v in enumerate(volumes):
                if (i + 1) % 30 == 0 or i == 0:
                    print(f"    {i + 1}/{n_total}")
                img_norm, msk = process_one_volume(v, target_shape)
                if out_images.shape[1:] != img_norm.shape:
                    img_norm = crop_or_pad_to_shape(img_norm, out_shape, constant=-1000)
                    msk = crop_or_pad_to_shape(msk, out_shape, constant=0)
                out_images[i] = img_norm
                out_masks[i] = msk

            npz_path = os.path.join(out_dir, f"abdomen3d_{size_val}.npz")
            print(f"    Saving NPZ ({out_images.nbytes / 1e6:.0f} MB images)...")
            save_large_npz(npz_path, out_images, out_masks, out_images[:0], out_masks[:0])
            del out_images, out_masks

        if size_val == "native":
            native_padded_shape = list(out_shape)

        size_key = str(size_val)
        print(f"  Saved: {npz_path}")

        checksum_name = f"abdomen3d_{size_key}.sha256"
        write_checksum(npz_path, os.path.join(out_dir, "checksums"), checksum_name)

        generated_sizes.append(size_val)

    meta = {
        "flag": FLAG,
        "class_name": CLASS_NAME,
        "name": SOURCE_NAME,
        "version": "1.0.0",
        "dimensionality": DIMENSIONALITY,
        "modality": MODALITY,
        "anatomy": ANATOMY,
        "source_url": "https://www.cancerimagingarchive.net/collection/ct-org/",
        "license": "CC BY 3.0",
        "redistribution_allowed": True,
        "paper_doi": "10.1038/s41597-020-00715-2",
        "split_seed": 42,
        "split_strategy": "5-fold_cv",
        "n_folds": 5,
        "available_sizes": sorted([s for s in generated_sizes if isinstance(s, int)])
        + (["native"] if "native" in generated_sizes else []),
        "native_voxel_spacing_mm": list(NATIVE_SPACING_APPROX),
        "native_padded_shape": list(native_padded_shape) if native_padded_shape else list(STANDARDISED_SIZES[128]["shape"]),
        "standardised_sizes": {
            str(k): {"shape": list(v["shape"]), "voxel_spacing_mm": list(v["spacing"])}
            for k, v in STANDARDISED_SIZES.items()
            if k in [s for s in generated_sizes if isinstance(s, int)]
        },
        "hu_range": list(HU_RANGE),
        "normalization": "hu_clip_zscore",
        "label_names": LABEL_NAMES,
        "label_original_values": LABEL_ORIGINAL_VALUES,
        "n_total": n_total,
        "cv_folds": cv_folds,
        "per_sample_metadata": per_sample_meta,
    }

    meta_path = os.path.join(out_dir, f"{FLAG}.json")
    save_metadata(meta_path, meta)
    print(f"  Metadata: {meta_path}")


def parse_size(s):
    if s == "native":
        return "native"
    return int(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Preprocess {SOURCE_NAME} -> {CLASS_NAME}"
    )
    parser.add_argument(
        "--raw_dir", required=True, help="Path to raw CT-ORG dataset directory"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for processed npz files"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["128"],
        type=parse_size,
        help="Sizes to generate: 64, 96, 128, 192, native",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.sizes)

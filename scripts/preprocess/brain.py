import argparse
import os
import numpy as np
import nibabel as nib
from common import (
    resample_and_resize,
    normalize_mri,
    make_splits,
    compute_padded_shape,
    pad_to_shape,
    save_npz,
    save_metadata,
    write_checksum,
)

FLAG = "brain3d"
CLASS_NAME = "BrainSegMNIST3D"
SOURCE_NAME = "BraTS-Africa"
MODALITY = "MRI"
ANATOMY = "Brain"
DIMENSIONALITY = "3D"

STANDARDISED_SIZES = {
    96: {"shape": (96, 96, 64), "spacing": (2.5, 2.5, 2.42)},
    128: {"shape": (128, 128, 80), "spacing": (1.88, 1.88, 1.94)},
    224: {"shape": (224, 224, 144), "spacing": (1.07, 1.07, 1.08)},
}

NATIVE_SPACING = (1.0, 1.0, 1.0)
NATIVE_SHAPE = (240, 240, 155)

LABEL_NAMES = {
    "0": "background",
    "1": "necrotic_core",
    "2": "peritumoral_edema",
    "3": "enhancing_tumor",
}

LABEL_ORIGINAL_VALUES = {"0": 0, "1": 1, "2": 2, "3": 3}


def collect_patients(raw_dir):
    patients = []
    for subdir in sorted(os.listdir(raw_dir)):
        subpath = os.path.join(raw_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        for pat in sorted(os.listdir(subpath)):
            patpath = os.path.join(subpath, pat)
            if not os.path.isdir(patpath):
                continue
            t2f_file = os.path.join(patpath, f"{pat}-t2f.nii.gz")
            seg_file = os.path.join(patpath, f"{pat}-seg.nii.gz")
            if os.path.exists(t2f_file) and os.path.exists(seg_file):
                patients.append({"id": pat, "t2f": t2f_file, "seg": seg_file})
    return patients


def load_volumes(patients):
    images, masks = [], []
    per_sample_meta = []
    for p in patients:
        img = nib.load(p["t2f"]).get_fdata(dtype=np.float32)
        seg = nib.load(p["seg"]).get_fdata().astype(np.uint8)
        images.append(img)
        masks.append(seg)
        per_sample_meta.append(
            {
                "original_id": p["id"],
                "original_spacing_mm": list(NATIVE_SPACING),
                "original_shape": list(img.shape),
            }
        )
    return np.stack(images), np.stack(masks), per_sample_meta


def main(raw_dir, out_dir, sizes):
    os.makedirs(out_dir, exist_ok=True)

    print("Collecting patients...")
    patients = collect_patients(raw_dir)
    print(f"  Found {len(patients)} patients")

    print("Loading volumes...")
    all_images, all_masks, per_sample_meta = load_volumes(patients)
    print(f"  Images shape: {all_images.shape}, Masks shape: {all_masks.shape}")

    print("Creating splits...")
    n_total = len(patients)
    train_idx, test_idx, cv_folds = make_splits(
        n_total, test_size=0.20, n_folds=5, seed=42
    )
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    for p_idx in train_idx:
        per_sample_meta[p_idx]["split"] = "train"
    for p_idx in test_idx:
        per_sample_meta[p_idx]["split"] = "test"

    train_images_raw = all_images[train_idx]
    train_masks_raw = all_masks[train_idx]
    test_images_raw = all_images[test_idx]
    test_masks_raw = all_masks[test_idx]

    generated_sizes = []
    native_padded_shape = None

    for size_val in sizes:
        print(f"Processing size={size_val}...")

        if size_val == "native":
            spacing = NATIVE_SPACING
            target_shape = None
        else:
            spec = STANDARDISED_SIZES[size_val]
            spacing = spec["spacing"]
            target_shape = spec["shape"]

        train_images_list, train_masks_list = [], []
        test_images_list, test_masks_list = [], []

        for i in range(len(train_images_raw)):
            img, msk = train_images_raw[i], train_masks_raw[i]
            if size_val == "native":
                img_rs, msk_rs = img.copy(), msk.copy()
            else:
                img_rs, msk_rs = resample_and_resize(img, msk, spacing, target_shape)
            img_norm = normalize_mri(img_rs)
            train_images_list.append(img_norm)
            train_masks_list.append(msk_rs)

        for i in range(len(test_images_raw)):
            img, msk = test_images_raw[i], test_masks_raw[i]
            if size_val == "native":
                img_rs, msk_rs = img.copy(), msk.copy()
            else:
                img_rs, msk_rs = resample_and_resize(img, msk, spacing, target_shape)
            img_norm = normalize_mri(img_rs)
            test_images_list.append(img_norm)
            test_masks_list.append(msk_rs)

        if size_val == "native":
            all_train_shapes = [v.shape for v in train_images_list]
            all_test_shapes = [v.shape for v in test_images_list]
            padded_shape = compute_padded_shape(
                all_train_shapes + all_test_shapes, percentile=95
            )
            native_padded_shape = padded_shape
            print(f"  Native padded shape: {padded_shape}")
            train_images_list = [
                pad_to_shape(v, padded_shape) for v in train_images_list
            ]
            train_masks_list = [pad_to_shape(v, padded_shape) for v in train_masks_list]
            test_images_list = [pad_to_shape(v, padded_shape) for v in test_images_list]
            test_masks_list = [pad_to_shape(v, padded_shape) for v in test_masks_list]

        train_images = np.stack(train_images_list)
        train_masks = np.stack(train_masks_list)
        test_images = np.stack(test_images_list)
        test_masks = np.stack(test_masks_list)

        size_key = str(size_val)
        suffix = "_native" if size_val == "native" else f"_{size_val}"

        npz_path = os.path.join(out_dir, f"{FLAG}{suffix}.npz")
        save_npz(npz_path, train_images, train_masks, test_images, test_masks)
        print(f"  Saved: {npz_path} ({train_images.nbytes / 1e6:.0f} MB)")

        checksum_name = f"{FLAG}_{size_key}.sha256"
        write_checksum(npz_path, os.path.join(out_dir, "checksums"), checksum_name)
        print(f"  Checksum: checksums/{checksum_name}")

        generated_sizes.append(size_val)

    meta = {
        "flag": FLAG,
        "class_name": CLASS_NAME,
        "name": SOURCE_NAME,
        "version": "1.0.0",
        "dimensionality": DIMENSIONALITY,
        "modality": MODALITY,
        "anatomy": ANATOMY,
        "source_url": "https://www.cancerimagingarchive.net/collection/brats-africa/",
        "license": "TCIA Restricted",
        "redistribution_allowed": True,
        "paper_doi": "",
        "split_seed": 42,
        "split_ratios": {"train": 0.80, "test": 0.20},
        "split_strategy": "patient-level",
        "available_sizes": sorted([s for s in generated_sizes if isinstance(s, int)])
        + (["native"] if "native" in generated_sizes else []),
        "native_voxel_spacing_mm": list(NATIVE_SPACING),
        "native_padded_shape": list(native_padded_shape)
        if native_padded_shape
        else list(NATIVE_SHAPE),
        "native_percentile_box": 95,
        "standardised_sizes": {
            str(k): {"shape": list(v["shape"]), "voxel_spacing_mm": list(v["spacing"])}
            for k, v in STANDARDISED_SIZES.items()
            if k in [s for s in generated_sizes if isinstance(s, int)]
        },
        "normalization": "percentile_clip_zscore",
        "normalization_percentiles": [0.5, 99.5],
        "label_names": LABEL_NAMES,
        "label_original_values": LABEL_ORIGINAL_VALUES,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
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
        "--raw_dir", required=True, help="Path to raw BraTS-Africa dataset directory"
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for processed npz files"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["128"],
        type=parse_size,
        help="Sizes to generate: 96, 128, 224, native",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.sizes)

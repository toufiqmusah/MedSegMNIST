import argparse
import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold
from common import (
    resample_and_resize,
    normalize_mri,
    save_large_npz,
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


def process_one_volume(patient, target_shape):
    img = nib.load(patient["t2f"]).get_fdata(dtype=np.float32)
    seg = nib.load(patient["seg"]).get_fdata().astype(np.uint8)
    if target_shape is not None:
        img_rs, msk_rs = resample_and_resize(img, seg, target_shape)
    else:
        img_rs, msk_rs = img.copy(), seg.copy()
    return normalize_mri(img_rs), msk_rs


def main(raw_dir, out_dir, sizes):
    os.makedirs(out_dir, exist_ok=True)

    print("Collecting patients...")
    patients = collect_patients(raw_dir)
    n_total = len(patients)
    print(f"  Found {n_total} patients")

    print("Creating 5-fold CV splits...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = {}
    for i, (tr, te) in enumerate(kf.split(np.arange(n_total))):
        cv_folds[f"fold_{i}"] = {"train": tr.tolist(), "test": te.tolist()}

    per_sample_meta = []
    for p in patients:
        per_sample_meta.append(
            {
                "original_id": p["id"],
                "original_spacing_mm": list(NATIVE_SPACING),
                "original_shape": list(NATIVE_SHAPE),
                "split": "all",
            }
        )

    generated_sizes = []
    native_padded_shape = None

    for size_val in sizes:
        print(f"Processing size={size_val}...")

        if size_val == "native":
            target_shape = None
            out_shape = NATIVE_SHAPE
        else:
            spec = STANDARDISED_SIZES[size_val]
            target_shape = spec["shape"]
            out_shape = target_shape

        out_images = np.zeros((n_total, *out_shape), dtype=np.float32)
        out_masks = np.zeros((n_total, *out_shape), dtype=np.uint8)

        for i, p in enumerate(patients):
            img_norm, msk = process_one_volume(p, target_shape)
            out_images[i] = img_norm
            out_masks[i] = msk

            if (i + 1) % 30 == 0:
                print(f"    {i + 1}/{n_total}")

        if size_val == "native":
            native_padded_shape = out_shape

        size_key = str(size_val)
        suffix = "_native" if size_val == "native" else f"_{size_val}"
        npz_path = os.path.join(out_dir, f"{FLAG}{suffix}.npz")
        print(f"    Saving NPZ ({out_images.nbytes / 1e6:.0f} MB images)...")
        save_large_npz(npz_path, out_images, out_masks, out_images[:0], out_masks[:0])

        checksum_name = f"{FLAG}_{size_key}.sha256"
        write_checksum(npz_path, os.path.join(out_dir, "checksums"), checksum_name)
        print(f"  Saved: {npz_path}")

        generated_sizes.append(size_val)
        del out_images, out_masks

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
        "split_strategy": "5-fold_cv",
        "n_folds": 5,
        "available_sizes": sorted([s for s in generated_sizes if isinstance(s, int)])
        + (["native"] if "native" in generated_sizes else []),
        "native_voxel_spacing_mm": list(NATIVE_SPACING),
        "native_padded_shape": list(native_padded_shape) if native_padded_shape else list(NATIVE_SHAPE),
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

import argparse
import os
import sys

import numpy as np
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    normalize_ultrasound,
    compute_padded_shape,
    pad_to_shape,
    save_npz,
    save_metadata,
    write_checksum,
)

FLAG = "breast2d"
CLASS_NAME = "BreastSegMNIST"
MODALITY = "Ultrasound"
ANATOMY = "Breast"
DIMENSIONALITY = "2D"

STANDARDISED_SIZES = {
    128: 128,
    256: 256,
}

LABEL_NAMES = {
    "0": "background",
    "1": "lesion",
}

N_FOLDS = 5
SPLIT_SEED = 42


def load_samples_from_zip(zip_path, img_prefix, label_prefix):
    import zipfile

    samples = []
    with zipfile.ZipFile(zip_path, "r") as z:
        all_files = set(z.namelist())
        image_files = sorted(
            f for f in all_files
            if f.startswith(img_prefix) and f.endswith(".png")
            and not f.startswith("__MACOSX") and "/._" not in f
        )
        for img_path in image_files:
            basename = os.path.basename(img_path)
            if "_0000" in basename:
                mask_basename = basename.replace("_0000.png", ".png")
            else:
                mask_basename = basename
            mask_name = f"{label_prefix}/{mask_basename}"
            if mask_name not in all_files:
                continue

            with z.open(img_path) as f:
                img_np = np.array(Image.open(f), dtype=np.float32)
            with z.open(mask_name) as f:
                msk_np = np.array(Image.open(f), dtype=np.uint8)

            msk_binary = (msk_np > 128).astype(np.uint8)

            stem = os.path.splitext(mask_basename)[0]
            cls_name = stem.split("_")[0] if "_" in stem and stem.split("_")[0] in ("normal", "benign", "malignant") else ""

            samples.append({
                "image": img_np,
                "mask": msk_binary,
                "source": "busi" if "busi" in zip_path.lower().replace("-clean", "") else "breats_usg",
                "class": cls_name,
                "original_id": stem,
                "has_lesion": bool(msk_binary.sum() > 0),
            })
    return samples


def main(busi_zip, breats_zip, out_dir, sizes):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading BUSI-Clean...")
    busi = load_samples_from_zip(busi_zip, "imagesTr", "labelsTr")
    print(f"  {len(busi)} samples")

    print("Loading Breats-USG...")
    breats = load_samples_from_zip(breats_zip, "imagesTs", "labelsTs")
    print(f"  {len(breats)} samples")

    all_samples = busi + breats
    n_total = len(all_samples)
    print(f"\nTotal combined: {n_total}")

    has_lesion = np.array([s["has_lesion"] for s in all_samples])
    n_lesion = has_lesion.sum()
    n_normal = n_total - n_lesion
    print(f"  Lesion: {n_lesion}, Normal (empty mask): {n_normal}")

    indices = np.arange(n_total)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    cv_folds = {}
    for i, (tr, te) in enumerate(kf.split(indices)):
        cv_folds[f"fold_{i}"] = {
            "train": tr.tolist(),
            "test": te.tolist(),
        }

    all_images = [s["image"] for s in all_samples]
    all_masks = [s["mask"] for s in all_samples]
    all_meta = [{
        "original_id": s["original_id"],
        "source": s["source"],
        "class": s["class"],
        "has_lesion": s["has_lesion"],
        "original_shape": list(s["image"].shape),
    } for s in all_samples]

    for i in range(n_total):
        all_meta[i]["split"] = "all"

    generated_sizes = []
    native_padded_shape = None
    all_shapes = [s["image"].shape for s in all_samples]

    for size_val in sizes:
        print(f"Processing size={size_val}...")

        if size_val == "native":
            padded_shape = compute_padded_shape(all_shapes, percentile=90)
            native_padded_shape = padded_shape
            print(f"  Native padded shape: {padded_shape}")

            def crop_or_pad(arr, target):
                cur_h, cur_w = arr.shape[:2]
                t_h, t_w = target
                top = max(0, (cur_h - t_h) // 2)
                left = max(0, (cur_w - t_w) // 2)
                cropped = arr[top:top + t_h, left:left + t_w] if cur_h > t_h or cur_w > t_w else arr
                return pad_to_shape(cropped, target)

            all_images_proc = [
                normalize_ultrasound(crop_or_pad(img, padded_shape))
                for img in all_images
            ]
            all_masks_proc = [
                crop_or_pad(msk, padded_shape) for msk in all_masks
            ]
        else:
            target_size = STANDARDISED_SIZES[size_val]
            target_shape = (target_size, target_size)

            def proc(img, msk):
                img_rs = resize(img, target_shape, preserve_range=True, order=1).astype(np.float32)
                msk_rs = resize(msk, target_shape, preserve_range=True, order=0).astype(np.uint8)
                msk_rs = (msk_rs > 0.5).astype(np.uint8)
                return normalize_ultrasound(img_rs), msk_rs

            all_images_proc = []
            all_masks_proc = []
            for img, msk in zip(all_images, all_masks):
                img_n, msk_n = proc(img, msk)
                all_images_proc.append(img_n)
                all_masks_proc.append(msk_n)

        all_images_stacked = np.stack(all_images_proc)
        all_masks_stacked = np.stack(all_masks_proc)

        size_key = str(size_val)
        suffix = "_native" if size_val == "native" else f"_{size_val}"

        npz_path = os.path.join(out_dir, f"{FLAG}{suffix}.npz")
        save_npz(npz_path, all_images_stacked, all_masks_stacked, all_images_stacked[:0], all_masks_stacked[:0])
        print(f"  Saved: {npz_path} ({all_images_stacked.nbytes / 1e6:.0f} MB, "
              f"total={len(all_images_stacked)})")

        checksum_name = f"{FLAG}_{size_key}.sha256"
        write_checksum(npz_path, os.path.join(out_dir, "checksums"), checksum_name)
        print(f"  Checksum: checksums/{checksum_name}")

        generated_sizes.append(size_val)

    meta = {
        "flag": FLAG,
        "class_name": CLASS_NAME,
        "name": "BUSI + Breats-USG",
        "version": "1.0.0",
        "dimensionality": DIMENSIONALITY,
        "modality": MODALITY,
        "anatomy": ANATOMY,
        "source_urls": [
            "https://scholar.cu.edu.eg/?q=afahmy/pages/BUSI",
        ],
        "license": "Research use",
        "redistribution_allowed": False,
        "paper_doi": "",
        "split_seed": SPLIT_SEED,
        "split_strategy": "5-fold_cv",
        "source_counts": {"busi_clean": len(busi), "breats_usg": len(breats)},
        "available_sizes": sorted([s for s in generated_sizes if isinstance(s, int)])
        + (["native"] if "native" in generated_sizes else []),
        "native_padded_shape": list(native_padded_shape) if native_padded_shape else None,
        "native_percentile_box": 90,
        "standardised_sizes": {
            str(k): {"shape": [v, v]}
            for k, v in STANDARDISED_SIZES.items()
            if k in [s for s in generated_sizes if isinstance(s, int)]
        },
        "normalization": "percentile_clip_zscore",
        "normalization_percentiles": [1.0, 99.0],
        "label_names": LABEL_NAMES,
        "n_total": n_total,
        "n_folds": N_FOLDS,
        "cv_folds": cv_folds,
        "per_sample_metadata": all_meta,
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
        description=f"Preprocess BUSI-Clean + Breats-USG -> {CLASS_NAME}"
    )
    parser.add_argument("--busi-zip", required=True, help="Path to BUSI-Clean.zip")
    parser.add_argument("--breats-zip", required=True, help="Path to Breats-USG.zip")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["128"],
        type=parse_size,
        help="Sizes to generate: 128, 256, native",
    )
    args = parser.parse_args()
    main(args.busi_zip, args.breats_zip, args.out_dir, args.sizes)

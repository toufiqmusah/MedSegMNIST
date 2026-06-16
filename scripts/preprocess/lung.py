import argparse
import os
import numpy as np
from PIL import Image
from skimage.transform import resize
from common import (
    normalize_xray,
    compute_padded_shape,
    pad_to_shape,
    save_large_npz,
    save_metadata,
    write_checksum,
)
from sklearn.model_selection import KFold

FLAG = "lung2d"
CLASS_NAME = "LungSegMNIST2D"
MODALITY = "X-ray"
ANATOMY = "Lung"
DIMENSIONALITY = "2D"

SUBSET_SOURCES = {
    "Darwin": {
        "name": "Darwin COVID-19 Chest X-ray",
        "url": "https://github.com/nico-curtiz/Darwin-CXR",
        "license": "Research use",
    },
    "Shenzhen": {
        "name": "Shenzhen Hospital Chest X-ray",
        "url": "https://openi.nlm.nih.gov/",
        "license": "Public (NIH)",
    },
    "Montgomery": {
        "name": "Montgomery County TB Chest X-ray",
        "url": "https://openi.nlm.nih.gov/",
        "license": "Public (NIH)",
    },
}

STANDARDISED_SIZES = {
    128: 128,
    256: 256,
    512: 512,
}

LABEL_NAMES = {
    "0": "background",
    "1": "lung",
}


def collect_samples(raw_dir):
    samples = []
    for subset_name in sorted(SUBSET_SOURCES.keys()):
        subset_dir = os.path.join(raw_dir, subset_name)
        if not os.path.isdir(subset_dir):
            print(f"  Subset dir not found: {subset_dir}, skipping.")
            continue
        img_dir = os.path.join(subset_dir, "img")
        mask_dir = os.path.join(subset_dir, "mask")
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            print(f"  Skipping {subset_name}: no img/ or mask/ directory")
            continue

        print(f"Collecting {subset_name}...")
        img_files = sorted(os.listdir(img_dir))
        mask_files = sorted(os.listdir(mask_dir))
        img_stems = set(os.path.splitext(f)[0] for f in img_files)
        mask_stems = set(os.path.splitext(f)[0] for f in mask_files)
        common_stems = sorted(img_stems & mask_stems)

        if len(common_stems) < min(len(img_files), len(mask_files)):
            print(
                f"  Warning {subset_name}: {len(img_files)} images, {len(mask_files)} masks, "
                f"{len(common_stems)} matched"
            )

        for stem in common_stems:
            img_f = next(f for f in img_files if os.path.splitext(f)[0] == stem)
            mask_f = next(f for f in mask_files if os.path.splitext(f)[0] == stem)
            img_path = os.path.join(img_dir, img_f)
            mask_path = os.path.join(mask_dir, mask_f)

            mask_np = np.array(Image.open(mask_path), dtype=np.uint8)
            samples.append(
                {
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "meta": {
                        "original_id": stem,
                        "source_subset": subset_name,
                        "original_shape": list(mask_np.shape),
                    },
                }
            )

    return samples


def process_one(img_path, mask_path, size_val, target_shape, padded_shape):
    img_np = np.array(Image.open(img_path), dtype=np.float32)
    mask_np = np.array(Image.open(mask_path), dtype=np.uint8)
    if img_np.ndim == 3:
        img_np = img_np.mean(axis=2)
    mask_binary = (mask_np > 128).astype(np.uint8)

    if size_val == "native":
        return (
            normalize_xray(pad_to_shape(img_np, padded_shape)),
            pad_to_shape(mask_binary, padded_shape),
        )

    target_size = STANDARDISED_SIZES[size_val]
    target_shape = (target_size, target_size)
    img_rs = resize(img_np, target_shape, preserve_range=True, order=1).astype(np.float32)
    msk_rs = resize(mask_binary, target_shape, preserve_range=True, order=0).astype(np.uint8)
    msk_rs = (msk_rs > 0.5).astype(np.uint8)
    return normalize_xray(img_rs), msk_rs


def main(raw_dir, out_dir, sizes):
    os.makedirs(out_dir, exist_ok=True)

    print("Collecting samples (paths only)...")
    all_samples = collect_samples(raw_dir)
    n_total = len(all_samples)
    print(f"  Total samples: {n_total}")

    print("Creating 5-fold CV splits...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = {}
    for i, (tr, te) in enumerate(kf.split(np.arange(n_total))):
        cv_folds[f"fold_{i}"] = {"train": tr.tolist(), "test": te.tolist()}

    for i in range(n_total):
        all_samples[i]["meta"]["split"] = "all"

    generated_sizes = []
    native_padded_shape = None

    for size_val in sizes:
        print(f"Processing size={size_val}...")

        if size_val == "native":
            all_shapes = [s["meta"]["original_shape"] for s in all_samples]
            padded_shape = compute_padded_shape(all_shapes, percentile=95)
            native_padded_shape = padded_shape
            out_shape = padded_shape
            target_shape = None
            print(f"  Native padded shape: {padded_shape}")
        else:
            ts = STANDARDISED_SIZES[size_val]
            out_shape = (ts, ts)
            target_shape = (ts, ts)
            padded_shape = None

        out_images = np.zeros((n_total, *out_shape), dtype=np.float32)
        out_masks = np.zeros((n_total, *out_shape), dtype=np.uint8)

        for i, s in enumerate(all_samples):
            img_out, msk_out = process_one(
                s["img_path"], s["mask_path"], size_val, target_shape, padded_shape
            )
            out_images[i] = img_out
            out_masks[i] = msk_out

            if (i + 1) % 1000 == 0:
                print(f"    {i + 1}/{n_total}")

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

    source_names = []
    source_urls = []
    for subset_name in sorted(SUBSET_SOURCES.keys()):
        source_names.append(SUBSET_SOURCES[subset_name]["name"])
        source_urls.append(SUBSET_SOURCES[subset_name]["url"])

    all_meta = [s["meta"] for s in all_samples]
    meta = {
        "flag": FLAG,
        "class_name": CLASS_NAME,
        "name": " + ".join(source_names),
        "version": "1.0.0",
        "dimensionality": DIMENSIONALITY,
        "modality": MODALITY,
        "anatomy": ANATOMY,
        "source_urls": source_urls,
        "license": "Mixed (see per_sample_metadata for per-subset licenses)",
        "redistribution_allowed": True,
        "paper_doi": "",
        "split_seed": 42,
        "split_strategy": "5-fold_cv",
        "available_sizes": sorted([s for s in generated_sizes if isinstance(s, int)])
        + (["native"] if "native" in generated_sizes else []),
        "native_padded_shape": list(native_padded_shape) if native_padded_shape else None,
        "native_percentile_box": 95,
        "standardised_sizes": {
            str(k): {"shape": [v, v]}
            for k, v in STANDARDISED_SIZES.items()
            if k in [s for s in generated_sizes if isinstance(s, int)]
        },
        "normalization": "percentile_clip_zscore",
        "normalization_percentiles": [1.0, 99.0],
        "label_names": LABEL_NAMES,
        "n_total": n_total,
        "n_folds": 5,
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
        description=f"Preprocess chest X-ray lung datasets ({', '.join(SUBSET_SOURCES)}) -> {CLASS_NAME}"
    )
    parser.add_argument(
        "--raw_dir",
        required=True,
        help="Path to chest-xray-lungs parent directory containing subdirectories "
        "(Montgomery/, Shenzhen/, Darwin/)",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["128"],
        type=parse_size,
        help="Sizes to generate: 128, 256, 512, native",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.sizes)

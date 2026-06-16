import argparse
import os
import numpy as np
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import KFold
from common import (
    save_large_npz,
    save_metadata,
    write_checksum,
)

FLAG = "fives2d"
CLASS_NAME = "FundusSegMNIST2D"
SOURCE_NAME = "FIVES – A Fundus Image Dataset for AI-based Vessel Segmentation"
MODALITY = "Fundus photography"
ANATOMY = "Retina"
DIMENSIONALITY = "2D"

STANDARDISED_SIZES = {
    256: 256,
    512: 512,
    1024: 1024,
}

LABEL_NAMES = {
    "0": "background",
    "1": "vessel",
}


def collect_samples(raw_dir):
    samples = []
    for split_name in ("train", "test"):
        base = os.path.join(raw_dir, split_name)
        img_dir = os.path.join(base, "Original")
        msk_dir = os.path.join(base, "Ground truth")
        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
            print(f"  Skipping {split_name}: missing img/ or mask/")
            continue

        img_files = sorted(os.listdir(img_dir))
        msk_files = sorted(os.listdir(msk_dir))
        img_stems = set(os.path.splitext(f)[0] for f in img_files)
        msk_stems = set(os.path.splitext(f)[0] for f in msk_files)
        common_stems = sorted(img_stems & msk_stems)

        print(f"  {split_name}: {len(common_stems)} matched ({len(img_files)} images, {len(msk_files)} masks)")

        for stem in common_stems:
            img_f = next(f for f in img_files if os.path.splitext(f)[0] == stem)
            msk_f = next(f for f in msk_files if os.path.splitext(f)[0] == stem)
            img_path = os.path.join(img_dir, img_f)
            msk_path = os.path.join(msk_dir, msk_f)

            msk_np = np.array(Image.open(msk_path))
            samples.append({
                "img_path": img_path,
                "mask_path": msk_path,
                "meta": {
                    "original_id": stem,
                    "original_split": split_name,
                    "original_shape": list(msk_np.shape[:2]),
                },
            })

    return samples


def process_one(img_path, mask_path, size_val, target_shape):
    img_np = np.array(Image.open(img_path))
    msk_np = np.array(Image.open(mask_path)).astype(np.uint8)

    if img_np.ndim == 3 and img_np.shape[2] == 3:
        img_rgb = img_np
    elif img_np.ndim == 2:
        img_rgb = np.stack([img_np] * 3, axis=-1)
    else:
        img_rgb = img_np

    msk_gray = msk_np[..., 0] if msk_np.ndim == 3 else msk_np
    msk_binary = (msk_gray > 128).astype(np.uint8)

    if size_val == "native":
        return img_rgb.astype(np.uint8), msk_binary

    target_size = STANDARDISED_SIZES[size_val]
    target_shape = (target_size, target_size)
    img_rs = resize(img_rgb.astype(np.float32), target_shape, preserve_range=True, order=1)
    img_u8 = np.clip(np.round(img_rs), 0, 255).astype(np.uint8)
    msk_rs = resize(msk_binary, target_shape, preserve_range=True, order=0).astype(np.uint8)
    msk_rs = (msk_rs > 0.5).astype(np.uint8)
    return img_u8, msk_rs


def main(raw_dir, out_dir, sizes):
    os.makedirs(out_dir, exist_ok=True)

    print("Collecting samples...")
    all_samples = collect_samples(raw_dir)
    n_total = len(all_samples)
    print(f"  Total: {n_total}")

    n_train = sum(1 for s in all_samples if s["meta"]["original_split"] == "train")
    n_test = sum(1 for s in all_samples if s["meta"]["original_split"] == "test")
    print(f"  Train: {n_train}, Test: {n_test}")

    train_idx = [i for i, s in enumerate(all_samples) if s["meta"]["original_split"] == "train"]
    test_idx = [i for i, s in enumerate(all_samples) if s["meta"]["original_split"] == "test"]

    print("Creating CV folds (fold 0 = official test)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = {}
    for i, (tr, te) in enumerate(kf.split(np.arange(n_train))):
        tr_abs = [train_idx[j] for j in tr]
        te_abs = [train_idx[j] for j in te]
        cv_folds[f"fold_{i}"] = {"train": tr_abs, "test": te_abs}

    cv_folds["fold_0"] = {"train": train_idx, "test": test_idx}

    for i in range(n_total):
        all_samples[i]["meta"]["split"] = "all"

    generated_sizes = []
    native_shape = None

    for size_val in sizes:
        print(f"Processing size={size_val}...")

        if size_val == "native":
            all_shapes = [s["meta"]["original_shape"] for s in all_samples]
            native_shape = all_shapes[0]
            out_shape = native_shape
            target_shape = None
        else:
            ts = STANDARDISED_SIZES[size_val]
            out_shape = (ts, ts, 3)
            target_shape = (ts, ts)

        if size_val == "native":
            out_masks = np.zeros((n_total, *native_shape), dtype=np.uint8)
            out_images = np.zeros((n_total, *native_shape, 3), dtype=np.uint8)
        else:
            out_masks = np.zeros((n_total, ts, ts), dtype=np.uint8)
            out_images = np.zeros((n_total, ts, ts, 3), dtype=np.uint8)

        for i, s in enumerate(all_samples):
            img_out, msk_out = process_one(
                s["img_path"], s["mask_path"], size_val, target_shape
            )
            out_images[i] = img_out
            out_masks[i] = msk_out

            if (i + 1) % 200 == 0:
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

    all_meta = [s["meta"] for s in all_samples]
    meta = {
        "flag": FLAG,
        "class_name": CLASS_NAME,
        "name": SOURCE_NAME,
        "version": "1.0.0",
        "dimensionality": DIMENSIONALITY,
        "modality": MODALITY,
        "anatomy": ANATOMY,
        "source_url": "https://github.com/HaoLucas/FIVES",
        "license": "Research use",
        "redistribution_allowed": True,
        "paper_doi": "10.1038/s41597-024-03958-1",
        "split_seed": 42,
        "split_strategy": "5-fold_cv",
        "official_test_fold": 0,
        "available_sizes": sorted([s for s in generated_sizes if isinstance(s, int)])
        + (["native"] if "native" in generated_sizes else []),
        "native_padded_shape": list(native_shape) if native_shape else None,
        "standardised_sizes": {
            str(k): {"shape": [k, k, 3]}
            for k, v in STANDARDISED_SIZES.items()
            if k in [s for s in generated_sizes if isinstance(s, int)]
        },
        "normalization": "rgb_div255",
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
        description=f"Preprocess FIVES fundus dataset -> {CLASS_NAME}"
    )
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["256"],
        type=parse_size,
        help="Sizes to generate: 256, 512, native",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.sizes)

import argparse, os, json
import numpy as np
from PIL import Image
from skimage.transform import resize
from common import (
    normalize_xray,
    make_splits, compute_padded_shape, pad_to_shape,
    save_npz, save_metadata, write_checksum,
)

FLAG = "lung2d"
CLASS_NAME = "LungSegMNIST"
MODALITY = "X-ray"
ANATOMY = "Lung"
DIMENSIONALITY = "2D"
N_CLASSES = 2
N_CHANNELS = 1

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


def load_subset(subset_dir, subset_name):
    img_dir = os.path.join(subset_dir, "img")
    mask_dir = os.path.join(subset_dir, "mask")
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        print(f"  Skipping {subset_name}: no img/ or mask/ directory")
        return [], []

    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    img_stems = set(os.path.splitext(f)[0] for f in img_files)
    mask_stems = set(os.path.splitext(f)[0] for f in mask_files)
    common_stems = sorted(img_stems & mask_stems)

    if len(common_stems) < min(len(img_files), len(mask_files)):
        print(f"  Warning {subset_name}: {len(img_files)} images, {len(mask_files)} masks, "
              f"{len(common_stems)} matched")

    images, masks, meta = [], [], []
    for stem in common_stems:
        img_f = [f for f in img_files if os.path.splitext(f)[0] == stem][0]
        mask_f = [f for f in mask_files if os.path.splitext(f)[0] == stem][0]

        img_np = np.array(Image.open(os.path.join(img_dir, img_f)), dtype=np.float32)
        mask_np = np.array(Image.open(os.path.join(mask_dir, mask_f)), dtype=np.uint8)

        if img_np.ndim == 3:
            img_np = img_np.mean(axis=2)

        mask_binary = (mask_np > 128).astype(np.uint8)

        images.append(img_np)
        masks.append(mask_binary)
        meta.append({
            "original_id": stem,
            "source_subset": subset_name,
            "original_shape": list(mask_binary.shape),
        })

    return images, masks, meta


def main(raw_dir, out_dir, sizes):
    os.makedirs(out_dir, exist_ok=True)

    all_images, all_masks, all_meta = [], [], []
    total_loaded = 0

    for subset_name in sorted(SUBSET_SOURCES.keys()):
        subset_dir = os.path.join(raw_dir, subset_name)
        if not os.path.isdir(subset_dir):
            print(f"  Subset dir not found: {subset_dir}, skipping.")
            continue
        print(f"Loading {subset_name}...")
        imgs, msks, meta = load_subset(subset_dir, subset_name)
        if imgs:
            all_images.extend(imgs)
            all_masks.extend(msks)
            all_meta.extend(meta)
            total_loaded += len(imgs)
            print(f"  {len(imgs)} samples loaded from {subset_name}")

    print(f"Total samples loaded: {total_loaded}")

    all_shapes = [m.shape for m in all_masks]

    print("Creating splits...")
    n_total = len(all_images)
    train_idx, test_idx, cv_folds = make_splits(n_total, test_size=0.20, n_folds=5, seed=42)
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    for p_idx in train_idx:
        all_meta[p_idx]["split"] = "train"
    for p_idx in test_idx:
        all_meta[p_idx]["split"] = "test"

    for size_val in sizes:
        print(f"Processing size={size_val}...")

        if size_val == "native":
            padded_shape = compute_padded_shape(all_shapes, percentile=95)
            print(f"  Native padded shape: {padded_shape}")

            train_images_list = [
                normalize_xray(pad_to_shape(all_images[i], padded_shape))
                for i in train_idx
            ]
            train_masks_list = [
                pad_to_shape(all_masks[i], padded_shape)
                for i in train_idx
            ]
            test_images_list = [
                normalize_xray(pad_to_shape(all_images[i], padded_shape))
                for i in test_idx
            ]
            test_masks_list = [
                pad_to_shape(all_masks[i], padded_shape)
                for i in test_idx
            ]
        else:
            target_size = STANDARDISED_SIZES[size_val]
            target_shape = (target_size, target_size)

            def process_sample(img, msk):
                img_rs = resize(img, target_shape, preserve_range=True, order=1).astype(np.float32)
                msk_rs = resize(msk, target_shape, preserve_range=True, order=0).astype(np.uint8)
                msk_rs = (msk_rs > 0.5).astype(np.uint8)
                return normalize_xray(img_rs), msk_rs

            train_images_list, train_masks_list = [], []
            for i in train_idx:
                img_n, msk_n = process_sample(all_images[i], all_masks[i])
                train_images_list.append(img_n)
                train_masks_list.append(msk_n)

            test_images_list, test_masks_list = [], []
            for i in test_idx:
                img_n, msk_n = process_sample(all_images[i], all_masks[i])
                test_images_list.append(img_n)
                test_masks_list.append(msk_n)

        train_images = np.stack(train_images_list)
        train_masks = np.stack(train_masks_list)
        test_images = np.stack(test_images_list)
        test_masks = np.stack(test_masks_list)

        size_suffix = "" if size_val == "native" else f"_{size_val}"
        npz_path = os.path.join(out_dir, f"{FLAG}{size_suffix}.npz")
        save_npz(npz_path, train_images, train_masks, test_images, test_masks)
        print(f"  Saved: {npz_path} ({train_images.nbytes / 1e6:.0f} MB, "
              f"train={len(train_images)}, test={len(test_images)})")

        if size_val == "native":
            meta_available_sizes = sorted([s for s in sizes if isinstance(s, int)]) + ["native"]
        else:
            meta_available_sizes = [s for s in sizes if isinstance(s, int)]

        source_names = []
        source_urls = []
        for subset_name in sorted(os.listdir(raw_dir)):
            if subset_name in SUBSET_SOURCES:
                source_names.append(SUBSET_SOURCES[subset_name]["name"])
                source_urls.append(SUBSET_SOURCES[subset_name]["url"])

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
            "split_ratios": {"train": 0.80, "test": 0.20},
            "split_strategy": "image-level",
            "available_sizes": meta_available_sizes,
            "native_padded_shape": list(padded_shape) if size_val == "native" else None,
            "native_percentile_box": 95,
            "standardised_sizes": {
                str(k): {"shape": [v, v]} for k, v in STANDARDISED_SIZES.items()
                if k in [s for s in sizes if isinstance(s, int)]
            },
            "normalization": "percentile_clip_zscore",
            "normalization_percentiles": [1.0, 99.0],
            "label_names": LABEL_NAMES,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "cv_folds": cv_folds,
            "per_sample_metadata": all_meta,
        }

        meta_path = os.path.join(out_dir, f"{FLAG}{size_suffix}.json")
        save_metadata(meta_path, meta)
        print(f"  Saved: {meta_path}")

        write_checksum(npz_path, os.path.join(out_dir, "checksums"))
        print(f"  Checksum written")


def parse_size(s):
    if s == "native":
        return "native"
    return int(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Preprocess chest X-ray lung datasets ({', '.join(SUBSET_SOURCES)}) → {CLASS_NAME}"
    )
    parser.add_argument("--raw_dir", required=True,
                        help="Path to chest-xray-lungs parent directory containing subdirectories "
                             "(Montgomery/, Shenzhen/, Darwin/)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", nargs="+", default=["128"], type=parse_size,
                        help="Sizes to generate: 128, 256, 512, native")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.sizes)

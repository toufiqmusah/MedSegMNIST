#!/usr/bin/env python3
"""
Preprocess ISIC 2018 Task 1 (2,594 dermoscopic images with lesion masks)
→ SkinLesionSegMNIST2D.

Binary segmentation: 0=background, 1=lesion.
"""

import os, sys, json, time
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.preprocess.common import save_large_npz, save_metadata

# ── Config ──────────────────────────────────────────────────
DATASET_NAME = "derm2d"
OUT_DIR = "/teamspace/studios/this_studio/MedSegMNIST/datasets/dermoscopy"
DATA_DIR = "/teamspace/studios/this_studio/dataset-raw/isic2018/extracted"
IMG_DIR = os.path.join(DATA_DIR, "ISIC2018_Task1-2_Training_Input")
MASK_DIR = os.path.join(DATA_DIR, "ISIC2018_Task1_Training_GroundTruth")
SIZES = [128, 256, 512]
N_FOLDS = 5
SEED = 42
PROGRESS_EVERY = 200


def build_pairs():
    img_stems = set()
    for f in os.listdir(IMG_DIR):
        if f.endswith(".jpg"):
            img_stems.add(f.replace(".jpg", ""))

    pairs = []
    for f in sorted(os.listdir(MASK_DIR)):
        if not f.endswith("_segmentation.png"):
            continue
        stem = f.replace("_segmentation.png", "")
        if stem in img_stems:
            pairs.append((
                stem,
                os.path.join(IMG_DIR, f"{stem}.jpg"),
                os.path.join(MASK_DIR, f),
            ))
    return pairs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building image-mask pairs...")
    pairs = build_pairs()
    n_total = len(pairs)
    print(f"  Found {n_total} pairs")

    print("\nPre-allocating output arrays...")
    arrays = {}
    for size in SIZES:
        arrays[size] = {
            "img": np.zeros((n_total, size, size, 3), dtype=np.uint8),
            "mask": np.zeros((n_total, size, size), dtype=np.uint8),
        }

    native_dir = os.path.join(OUT_DIR, f"{DATASET_NAME}_native")
    os.makedirs(native_dir, exist_ok=True)
    native_meta = []
    native_shapes = []

    print("\nProcessing images...")
    start_time = time.time()

    for i, (img_id, img_path, mask_path) in enumerate(pairs):
        try:
            img_np = np.array(Image.open(img_path))
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)

            mask_np = np.array(Image.open(mask_path))
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            mask_bin = (mask_np > 128).astype(np.uint8)

            for size in SIZES:
                target = (size, size)
                img_pil = Image.fromarray(img_np)
                img_rs = np.array(img_pil.resize(target, Image.LANCZOS))
                arrays[size]["img"][i] = img_rs

                mask_pil = Image.fromarray(mask_bin * 255)
                mask_rs = np.array(mask_pil.resize(target, Image.NEAREST))
                arrays[size]["mask"][i] = (mask_rs > 128).astype(np.uint8)

            np.savez_compressed(
                os.path.join(native_dir, f"{img_id}.npz"),
                image=img_np.astype(np.uint8),
                mask=mask_bin,
            )
            native_shapes.append(list(img_np.shape[:2]))
            native_meta.append({
                "index": i,
                "file": f"{img_id}.npz",
                "id": img_id,
                "shape": list(img_np.shape[:2]),
            })

        except Exception as e:
            print(f"\n  ERROR image {i} ({img_id}): {e}")
            import traceback; traceback.print_exc()
            continue

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_total - i - 1) / rate
            print(f"  [{i+1}/{n_total}] {elapsed/60:.1f}m elap, "
                  f"{remaining/60:.1f}m rem, {rate:.1f} img/min")

    failed = set()
    for i in range(n_total):
        if arrays[SIZES[0]]["img"][i].sum() == 0 and \
           arrays[SIZES[0]]["mask"][i].sum() == 0:
            failed.add(i)
    valid_idx = np.array([i for i in range(n_total) if i not in failed])
    n_processed = len(valid_idx)
    print(f"\n  Processed {n_processed}/{n_total} ({len(failed)} failed)")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_folds = {}
    for fi, (tr, te) in enumerate(kf.split(np.arange(n_processed))):
        cv_folds[f"fold_{fi}"] = {"train": tr.tolist(), "test": te.tolist()}
    fold0 = cv_folds["fold_0"]
    print(f"  CV folds: {len(fold0['train'])} train, {len(fold0['test'])} test")

    print("\nSaving NPZ files...")
    for size in SIZES:
        print(f"  Size {size}...")
        compact_img = arrays[size]["img"][valid_idx]
        compact_mask = arrays[size]["mask"][valid_idx]
        out_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_{size}.npz")
        save_large_npz(out_path, compact_img, compact_mask,
                       np.zeros((0, *compact_img.shape[1:]), dtype=np.uint8),
                       np.zeros((0, *compact_mask.shape[1:]), dtype=np.uint8))
        sz = os.path.getsize(out_path)
        print(f"    {out_path}: {sz/1e9:.2f} GB")
        del arrays[size]

    meta = {
        "name": "SkinSegMNIST2D",
        "description": "ISIC 2018 Task 1 dermoscopic images with lesion boundary segmentation",
        "source": "ISIC 2018 Challenge",
        "modality": "Dermoscopy",
        "n_samples": n_processed,
        "n_classes": 2,
        "labels": {"background": 0, "lesion": 1},
        "available_sizes": SIZES + ["native"],
        "cv_folds": cv_folds,
        "native_shapes": native_shapes,
        "native_metadata": native_meta,
        "official_test_fold": 0,
        "seed": SEED,
    }
    meta_path = os.path.join(OUT_DIR, f"{DATASET_NAME}.json")
    save_metadata(meta_path, meta)
    print(f"\nMetadata: {meta_path}")

    print("\nSummary:")
    total = 0
    for size in SIZES:
        fp = os.path.join(OUT_DIR, f"{DATASET_NAME}_{size}.npz")
        if os.path.exists(fp):
            sz = os.path.getsize(fp)
            total += sz
            print(f"  {DATASET_NAME}_{size}.npz: {sz/1e9:.2f} GB")
    native_total = sum(os.path.getsize(os.path.join(native_dir, f))
                       for f in os.listdir(native_dir) if f.endswith(".npz"))
    total += native_total
    print(f"  native/: {native_total/1e9:.2f} GB")
    print(f"  TOTAL: {total/1e9:.2f} GB")
    print("Done ✓")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Preprocess OAIZIB-CM (507 knee DESS MRI volumes) → KneeSegMNIST3D.

Labels: 0=background, 1=Femur, 2=Femoral Cartilage, 3=Tibia,
         4=Medial Tibial Cartilage, 5=Lateral Tibial Cartilage → 6 classes.

Single-pass processing: read NIfTI once, resize to all sizes simultaneously.
"""

import os, sys, json, tempfile, time
import zipfile
import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.preprocess.common import resample_and_resize, save_large_npz, save_metadata

# ── Config ──────────────────────────────────────────────────
DATASET_NAME = "knee3d"
OUT_DIR = "/teamspace/studios/this_studio/MedSegMNIST/datasets/knee"
DATA_DIR = "/teamspace/studios/this_studio/dataset-raw/knee"
SIZES = [64, 96, 128, 192]
N_FOLDS = 5
SEED = 42
ENCODE_SCALE = 255.0 / 6.0
ENCODE_OFFSET = -3.0
PROGRESS_EVERY = 25
TMPDIR = "/tmp/knee_memmap"

LABEL_NAMES = {
    0: "background",
    1: "femur",
    2: "femoral_cartilage",
    3: "tibia",
    4: "medial_tibial_cartilage",
    5: "lateral_tibial_cartilage",
}


def build_pairs(data_dir):
    pairs = []
    for split in ["Tr", "Ts"]:
        img_zip = os.path.join(data_dir, f"images{split}.zip")
        lbl_zip = os.path.join(data_dir, f"labels{split}.zip")
        if not os.path.isfile(img_zip) or not os.path.isfile(lbl_zip):
            print(f"  Warning: missing {split} zips, skipping")
            continue

        with zipfile.ZipFile(img_zip) as zf_img, \
                zipfile.ZipFile(lbl_zip) as zf_lbl:
            img_names = sorted(
                [n for n in zf_img.namelist() if n.endswith(".nii.gz")]
            )
            lbl_names = set(
                n.split("/")[1] for n in zf_lbl.namelist() if n.endswith(".nii.gz")
            )

            for img_path in img_names:
                img_id = os.path.basename(img_path).replace("_0000.nii.gz", ".nii.gz")
                if img_id in lbl_names:
                    pairs.append((split, img_path, f"labels{split}/{img_id}"))
    return pairs


def read_nifti_from_zip(zf, path):
    data = zf.read(path)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        img = nib.load(tmp.name)
        return img.get_fdata(), img.header
    finally:
        os.unlink(tmp.name)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMPDIR, exist_ok=True)

    # 1. Build pairs
    print("Building image-mask pairs...")
    pairs = build_pairs(DATA_DIR)
    n_total = len(pairs)
    print(f"  Found {n_total} pairs")

    # 2. Verify first 3
    print("\nVerifying alignment...")
    for i in range(min(3, n_total)):
        split, img_path, lbl_path = pairs[i]
        with zipfile.ZipFile(os.path.join(DATA_DIR, f"images{split}.zip")) as zf:
            vol, _ = read_nifti_from_zip(zf, img_path)
        with zipfile.ZipFile(os.path.join(DATA_DIR, f"labels{split}.zip")) as zf:
            seg, _ = read_nifti_from_zip(zf, lbl_path)
        seg_u8 = seg.astype(np.uint8)
        print(f"  {i}: vol={vol.shape} seg={seg_u8.shape} "
              f"vals={np.unique(seg_u8)} ✓")

    # 3. Pre-allocate arrays indexed by pair position
    print("\nPre-allocating output arrays...")
    arrays = {}
    for size in SIZES:
        shape = (n_total, size, size, size)
        arrays[size] = {
            "img": np.memmap(os.path.join(TMPDIR, f"img_{size}.npy"),
                             dtype=np.uint8, mode="w+", shape=shape),
            "mask": np.memmap(os.path.join(TMPDIR, f"mask_{size}.npy"),
                              dtype=np.uint8, mode="w+", shape=shape),
        }

    # Native: per-volume NPZs (uint8)
    native_dir = os.path.join(OUT_DIR, f"{DATASET_NAME}_native")
    os.makedirs(native_dir, exist_ok=True)
    native_meta = []
    native_shapes = []

    # 4. Single-pass processing
    print("\nProcessing volumes (single pass)...")
    start_time = time.time()

    for i, (split, img_path, lbl_path) in enumerate(pairs):
        try:
            with zipfile.ZipFile(os.path.join(DATA_DIR, f"images{split}.zip")) as zf:
                vol, _ = read_nifti_from_zip(zf, img_path)
            with zipfile.ZipFile(os.path.join(DATA_DIR, f"labels{split}.zip")) as zf:
                seg_data, _ = read_nifti_from_zip(zf, lbl_path)
            seg_data = seg_data.astype(np.uint8)

            # Normalize MRI: percentile clip + z-score on foreground
            fg = vol[vol > 0]
            lo, hi = np.percentile(fg, [0.5, 99.5])
            vol_clip = np.clip(vol, lo, hi)
            fg_norm = vol_clip[vol_clip > 0]
            mean, std = fg_norm.mean(), fg_norm.std()
            vol_norm = np.zeros_like(vol_clip)
            if std > 0:
                vol_norm[vol > 0] = (vol_clip[vol > 0] - mean) / std

            # Resize to all standard sizes
            for size in SIZES:
                img_r, mask_r = resample_and_resize(vol_norm, seg_data,
                                                    (size, size, size))
                img_u8 = np.clip((img_r + 3.0) * ENCODE_SCALE, 0, 255).astype(np.uint8)
                arrays[size]["img"][i] = img_u8
                arrays[size]["mask"][i] = mask_r

            # Native: per-volume NPZ
            vol_uint8 = np.clip(
                (vol_norm + 3.0) * ENCODE_SCALE, 0, 255
            ).astype(np.uint8)
            img_id = os.path.basename(img_path).replace("_0000.nii.gz", "")
            npz_name = f"volume_{i:04d}.npz"
            np.savez_compressed(
                os.path.join(native_dir, npz_name),
                image=vol_uint8, mask=seg_data,
            )
            native_shapes.append(list(seg_data.shape))
            native_meta.append({
                "index": i,
                "file": npz_name,
                "shape": list(seg_data.shape),
                "id": img_id,
                "split": split,
                "norm_mean": float(mean),
                "norm_std": float(std),
                "clip_lo": float(lo),
                "clip_hi": float(hi),
                "encode_offset": ENCODE_OFFSET,
                "encode_scale": ENCODE_SCALE,
            })

        except Exception as e:
            print(f"\n  ERROR volume {i} ({img_path}): {e}")
            import traceback; traceback.print_exc()
            continue

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_total - i - 1) / rate
            print(f"  [{i+1}/{n_total}] {elapsed/60:.1f}m elap, "
                  f"{remaining/60:.1f}m rem, {rate:.1f} vol/min")
            for size in SIZES:
                for arr in arrays[size].values():
                    arr.flush()

    # Determine which indices succeeded
    failed = set()
    for i in range(n_total):
        if arrays[SIZES[0]]["img"][i].sum() == 0 and \
           arrays[SIZES[0]]["mask"][i].sum() == 0:
            failed.add(i)
    valid_idx = np.array([i for i in range(n_total) if i not in failed])
    n_processed = len(valid_idx)
    n_failed = len(failed)
    print(f"\n  Processed {n_processed}/{n_total} ({n_failed} failed)")

    # Create splits on valid-only indices
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_folds = {}
    for fi, (tr, te) in enumerate(kf.split(np.arange(n_processed))):
        cv_folds[f"fold_{fi}"] = {"train": tr.tolist(), "test": te.tolist()}
    fold0 = cv_folds["fold_0"]
    n_train, n_test = len(fold0["train"]), len(fold0["test"])
    print(f"  CV folds created: {n_train} train, {n_test} test")

    # Final flush
    for size in SIZES:
        for arr in arrays[size].values():
            arr.flush()
            del arr

    # 5. Save NPZs
    print("\nSaving NPZ files...")
    for size in SIZES:
        print(f"  Size {size}...")
        src_img = np.memmap(os.path.join(TMPDIR, f"img_{size}.npy"),
                            dtype=np.uint8, mode="r",
                            shape=(n_total, size, size, size))
        src_mask = np.memmap(os.path.join(TMPDIR, f"mask_{size}.npy"),
                             dtype=np.uint8, mode="r",
                             shape=(n_total, size, size, size))

        compact_img = src_img[valid_idx]
        compact_mask = src_mask[valid_idx]

        out_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_{size}.npz")
        save_large_npz(out_path, compact_img, compact_mask,
                       np.zeros((0, size, size, size), dtype=np.uint8),
                       np.zeros((0, size, size, size), dtype=np.uint8))
        sz = os.path.getsize(out_path)
        print(f"    {out_path}: {sz/1e9:.2f} GB")

        del src_img, src_mask
        os.remove(os.path.join(TMPDIR, f"img_{size}.npy"))
        os.remove(os.path.join(TMPDIR, f"mask_{size}.npy"))

    # 6. Save metadata
    meta = {
        "name": "KneeSegMNIST3D",
        "description": "Knee DESS MRI with femur, femoral cartilage, tibia, "
                       "medial tibial cartilage, and lateral tibial cartilage segmentation",
        "source": "OAIZIB-CM",
        "modality": "MR",
        "n_samples": n_processed,
        "n_classes": 6,
        "labels": LABEL_NAMES,
        "available_sizes": SIZES + ["native"],
        "cv_folds": cv_folds,
        "encode_scale": ENCODE_SCALE,
        "encode_offset": ENCODE_OFFSET,
        "native_shapes": native_shapes,
        "native_metadata": native_meta,
        "official_test_fold": 0,
        "seed": SEED,
    }
    meta_path = os.path.join(OUT_DIR, f"{DATASET_NAME}.json")
    save_metadata(meta_path, meta)
    print(f"\nMetadata: {meta_path}")

    # Summary
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

#!/usr/bin/env python3
"""
Preprocess DukeCSpineSeg (1254 cervical spine MRI volumes)
→ SpineSegMNIST3D with vertebral body (1) + spinal canal (2) labels.

Labels: 0=background, 1=vertebral body, 2=spinal canal → 3 classes.

Single-pass processing: read DICOM once, resize to all sizes simultaneously.
"""

import os, sys, json, glob, time
import zipfile
from io import BytesIO
import numpy as np
import nibabel as nib
import pydicom
from sklearn.model_selection import train_test_split, KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.preprocess.common import resample_and_resize, save_large_npz, save_metadata, normalize_mri

# ── Config ──────────────────────────────────────────────────
DATASET_NAME = "spine3d"
OUT_DIR = "/teamspace/studios/this_studio/MedSegMNIST/datasets"
SEG_DIR = "/teamspace/studios/this_studio/dataset-raw/spine/segmentations"
IMG_ZIP = "/teamspace/studios/this_studio/dataset-raw/spine/imaging.zip"
SIZES = [64, 96, 128, 192]
N_FOLDS = 5
TEST_SIZE = 0.20
SEED = 42
PROGRESS_EVERY = 25
TMPDIR = "/tmp/spine_memmap"


def build_index(zip_path):
    print("Building series index...")
    index = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = sorted([n for n in zf.namelist() if n.endswith(".zip")])
        for i, entry in enumerate(entries):
            parts = entry.split("/")
            patient_id = parts[1]
            try:
                data = zf.read(entry)
                inner = zipfile.ZipFile(BytesIO(data))
                dcm_names = sorted([n for n in inner.namelist() if n.endswith(".dcm")])
                if not dcm_names:
                    continue
                ds = pydicom.dcmread(BytesIO(inner.read(dcm_names[0])), stop_before_pixels=True)
                series_num = int(ds.SeriesNumber)
                index[(patient_id, series_num)] = {
                    "zip_path": entry,
                    "patient_id": patient_id,
                    "series_num": series_num,
                    "rows": int(ds.Rows),
                    "cols": int(ds.Columns),
                    "num_slices": len(dcm_names),
                }
            except Exception as e:
                print(f"  Warning: {entry}: {e}")
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{len(entries)}")
    print(f"  Index: {len(index)} series")
    return index


def match_segs(index, seg_dir):
    seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith(".nii.gz")])
    pairs = []
    for sf in seg_files:
        base = sf.replace(".nii.gz", "")
        parts = base.split("_")
        pid = parts[0]
        sn = None
        for p in parts:
            if p.startswith("Series-"):
                sn = int(p.split("-")[1])
        key = (pid, sn)
        if key in index:
            pairs.append((index[key], os.path.join(seg_dir, sf)))
    print(f"  Matched {len(pairs)}/{len(seg_files)}")
    return pairs


def read_dicom_volume(outer_zip, entry):
    inner_data = outer_zip.read(entry["zip_path"])
    inner = zipfile.ZipFile(BytesIO(inner_data))
    dcm_names = sorted([n for n in inner.namelist() if n.endswith(".dcm")])
    slice_data = []
    for dname in dcm_names:
        ds = pydicom.dcmread(BytesIO(inner.read(dname)))
        instance = int(getattr(ds, "InstanceNumber", 0))
        arr = ds.pixel_array.astype(np.float32)
        slice_data.append((instance, arr))
    slice_data.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slice_data], axis=-1)


def verify_alignment(volume, seg_data, idx):
    v = volume.shape
    s = seg_data.shape
    if v == s:
        return
    if (v[1], v[0], v[2]) == s:
        return "transpose"
    raise ValueError(f"Volume {idx}: vol={v} vs seg={s}")


def create_splits(n, test_size, n_folds, seed):
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv = {}
    for i, (tr, va) in enumerate(kf.split(train_idx)):
        cv[f"fold_{i}"] = {"train": tr.tolist(), "val": va.tolist()}
    return train_idx.tolist(), test_idx.tolist(), cv


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMPDIR, exist_ok=True)

    # 1. Build + match
    index = build_index(IMG_ZIP)
    pairs = match_segs(index, SEG_DIR)
    n_total = len(pairs)
    print(f"Total: {n_total} volumes")

    # 2. Splits
    train_idx, test_idx, cv_folds = create_splits(n_total, TEST_SIZE, N_FOLDS, SEED)
    is_test = np.zeros(n_total, dtype=bool)
    is_test[test_idx] = True
    n_train, n_test = len(train_idx), len(test_idx)
    print(f"Train: {n_train}, Test: {n_test}")

    # 3. Verify first 5
    print("\nVerifying alignment...")
    with zipfile.ZipFile(IMG_ZIP, "r") as zf:
        for i in range(min(5, n_total)):
            entry, seg_path = pairs[i]
            vol = read_dicom_volume(zf, entry)
            seg = nib.load(seg_path).get_fdata().astype(np.uint8)
            verify_alignment(vol, seg, i)
            print(f"  {i}: vol={vol.shape} seg={seg.shape} ✓")

    # 4. Pre-allocate arrays (memmap for all sizes to save RAM)
    print("\nPre-allocating output arrays...")
    arrays = {}  # {size: {split: {img/mask: memmap}}}
    for size in SIZES:
        img_shape = (n_train, size, size, size)
        mask_shape = (n_train, size, size, size)
        img_test_shape = (n_test, size, size, size)
        mask_test_shape = (n_test, size, size, size)

        arrays[size] = {
            "train_img": np.memmap(os.path.join(TMPDIR, f"train_img_{size}.npy"),
                                   dtype=np.float32, mode="w+", shape=img_shape),
            "train_mask": np.memmap(os.path.join(TMPDIR, f"train_mask_{size}.npy"),
                                    dtype=np.uint8, mode="w+", shape=mask_shape),
            "test_img": np.memmap(os.path.join(TMPDIR, f"test_img_{size}.npy"),
                                  dtype=np.float32, mode="w+", shape=img_test_shape),
            "test_mask": np.memmap(os.path.join(TMPDIR, f"test_mask_{size}.npy"),
                                   dtype=np.uint8, mode="w+", shape=mask_test_shape),
        }

    # Native: per-volume NPZs (uint8)
    native_dir = os.path.join(OUT_DIR, f"{DATASET_NAME}_native")
    os.makedirs(native_dir, exist_ok=True)
    native_meta = []
    native_shapes = []
    native_test_indices = []

    # 5. Single-pass processing
    print("\nProcessing volumes (single pass)...")
    counters = {size: {"train": 0, "test": 0} for size in SIZES}
    start_time = time.time()
    n_processed = 0

    with zipfile.ZipFile(IMG_ZIP, "r") as zf:
        for i, (entry, seg_path) in enumerate(pairs):
            try:
                # Read
                vol = read_dicom_volume(zf, entry)
                seg_data = nib.load(seg_path).get_fdata().astype(np.uint8)

                result = verify_alignment(vol, seg_data, i)
                if result == "transpose":
                    seg_data = seg_data.transpose(1, 0, 2)

                # Normalize MRI
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
                    img_r, mask_r = resample_and_resize(vol_norm, seg_data, (size, size, size))
                    if is_test[i]:
                        idx_l = counters[size]["test"]
                        arrays[size]["test_img"][idx_l] = img_r
                        arrays[size]["test_mask"][idx_l] = mask_r
                        counters[size]["test"] += 1
                    else:
                        idx_l = counters[size]["train"]
                        arrays[size]["train_img"][idx_l] = img_r
                        arrays[size]["train_mask"][idx_l] = mask_r
                        counters[size]["train"] += 1

                # Native: encode to uint8 and save per-volume NPZ
                scale = 255.0 / 6.0  # map [-3, 3] z-score → [0, 255]
                vol_uint8 = np.clip((vol_norm + 3.0) * scale, 0, 255).astype(np.uint8)
                vol_path = os.path.join(native_dir, f"volume_{i:04d}.npz")
                np.savez_compressed(vol_path, image=vol_uint8, mask=seg_data)

                native_shapes.append(list(seg_data.shape))
                native_test_indices.append(int(is_test[i]))
                native_meta.append({
                    "index": i,
                    "file": f"volume_{i:04d}.npz",
                    "shape": list(seg_data.shape),
                    "patient": entry["patient_id"],
                    "series": entry["series_num"],
                    "is_test": bool(is_test[i]),
                    "norm_mean": float(mean),
                    "norm_std": float(std),
                    "clip_lo": float(lo),
                    "clip_hi": float(hi),
                    "encode_offset": -3.0,
                    "encode_scale": scale,
                })
                n_processed += 1

            except Exception as e:
                print(f"\n  ERROR volume {i}: {e}")
                import traceback; traceback.print_exc()
                continue

            if (i + 1) % PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (n_total - i - 1) / rate
                print(f"  [{i+1}/{n_total}] {elapsed/60:.1f}m elap, "
                      f"{remaining/60:.1f}m rem, {rate:.1f} vol/min")
                # Flush memmaps
                for size in SIZES:
                    for arr in arrays[size].values():
                        arr.flush()

    print(f"  Processed {n_processed}/{n_total} volumes")

    # Final flush
    for size in SIZES:
        for arr in arrays[size].values():
            arr.flush()
            del arr

    # 6. Save NPZs (pass memmap objects directly to avoid RAM blowup)
    print("\nSaving NPZ files...")
    for size in SIZES:
        print(f"  Size {size}...")

        # Re-open memmaps (fresh file handles after processing)
        mm_train_img = np.memmap(
            os.path.join(TMPDIR, f"train_img_{size}.npy"),
            dtype=np.float32, mode="r",
            shape=(n_train, size, size, size))
        mm_train_mask = np.memmap(
            os.path.join(TMPDIR, f"train_mask_{size}.npy"),
            dtype=np.uint8, mode="r",
            shape=(n_train, size, size, size))
        mm_test_img = np.memmap(
            os.path.join(TMPDIR, f"test_img_{size}.npy"),
            dtype=np.float32, mode="r",
            shape=(n_test, size, size, size))
        mm_test_mask = np.memmap(
            os.path.join(TMPDIR, f"test_mask_{size}.npy"),
            dtype=np.uint8, mode="r",
            shape=(n_test, size, size, size))

        out_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_{size}.npz")
        # Always use save_large_npz for memmap-backed arrays; it writes .npy to
        # temp, keeping peak RAM at buffer size rather than full array.
        save_large_npz(out_path, mm_train_img, mm_train_mask, mm_test_img, mm_test_mask)
        sz = os.path.getsize(out_path)
        print(f"    {out_path}: {sz/1e9:.2f} GB")

        # Cleanup memmap files (close first, then delete)
        del mm_train_img, mm_train_mask, mm_test_img, mm_test_mask
        os.remove(os.path.join(TMPDIR, f"train_img_{size}.npy"))
        os.remove(os.path.join(TMPDIR, f"train_mask_{size}.npy"))
        os.remove(os.path.join(TMPDIR, f"test_img_{size}.npy"))
        os.remove(os.path.join(TMPDIR, f"test_mask_{size}.npy"))

    # 7. Save metadata
    meta = {
        "name": "SpineSegMNIST3D",
        "description": "Cervical spine MRI with vertebral body (1) and spinal canal (2) segmentation",
        "source": "DukeCSpineSeg",
        "modality": "MR",
        "n_samples": n_processed,
        "n_classes": 3,
        "labels": {"background": 0, "vertebral_body": 1, "spinal_canal": 2},
        "n_train": n_train,
        "n_test": n_test,
        "available_sizes": SIZES + ["native"],
        "test_indices": test_idx,
        "train_indices": train_idx,
        "cv_folds": cv_folds,
        "native_shapes": native_shapes,
        "native_test_indices": native_test_indices,
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
    ns = sum(os.path.getsize(os.path.join(native_dir, f))
             for f in os.listdir(native_dir) if f.endswith(".npz"))
    total += ns
    print(f"  native/: {ns/1e9:.2f} GB")
    print(f"  TOTAL: {total/1e9:.2f} GB")
    print("Done ✓")


if __name__ == "__main__":
    main()

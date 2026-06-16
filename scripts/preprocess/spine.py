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
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.preprocess.common import resample_and_resize, save_large_npz, save_metadata

# ── Config ──────────────────────────────────────────────────
DATASET_NAME = "spine3d"
OUT_DIR = "/teamspace/studios/this_studio/MedSegMNIST/datasets"
SEG_DIR = "/teamspace/studios/this_studio/dataset-raw/spine/segmentations"
IMG_ZIP = "/teamspace/studios/this_studio/dataset-raw/spine/imaging.zip"
SIZES = [64, 96, 128, 192]
N_FOLDS = 5
SEED = 42
ENCODE_SCALE = 255.0 / 6.0
ENCODE_OFFSET = -3.0
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


def reorient_seg(seg_data, vol_shape, idx):
    """
    Reorient NIfTI segmentation to match DICOM volume orientation.
    
    NIfTI axis 1 is systematically flipped relative to DICOM column axis
    (verified across all 1254 volumes). Also handles shape mismatches.
    """
    v = vol_shape
    s = seg_data.shape

    if v == s:
        pass
    elif len(s) == 3 and s[0] == v[-1] and s[1] == v[1] and s[2] == v[0]:
        seg_data = seg_data.transpose(1, 2, 0)
    elif (v[1], v[0], v[2]) == s:
        seg_data = seg_data.transpose(1, 0, 2)
    else:
        raise ValueError(f"Volume {idx}: vol={v} vs seg={s}")

    # NIfTI axis 1 is always flipped relative to DICOM column axis
    seg_data = np.flip(seg_data, axis=1)
    return seg_data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMPDIR, exist_ok=True)

    # 1. Build + match
    index = build_index(IMG_ZIP)
    pairs = match_segs(index, SEG_DIR)
    n_total = len(pairs)
    print(f"Total: {n_total} volumes")

    # 2. Splits are created AFTER processing to handle errors

    # 3. Verify first 5
    print("\nVerifying alignment...")
    with zipfile.ZipFile(IMG_ZIP, "r") as zf:
        for i in range(min(5, n_total)):
            entry, seg_path = pairs[i]
            vol = read_dicom_volume(zf, entry)
            seg = nib.load(seg_path).get_fdata().astype(np.uint8)
            seg2 = reorient_seg(seg.copy(), vol.shape, i)
            print(f"  {i}: vol={vol.shape} seg={seg.shape} → {seg2.shape} ✓")

    # 4. Pre-allocate arrays indexed by pair position (all volumes)
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

    # 5. Single-pass processing (store at pair index i; errors leave zero entries)
    print("\nProcessing volumes (single pass)...")
    start_time = time.time()

    with zipfile.ZipFile(IMG_ZIP, "r") as zf:
        for i, (entry, seg_path) in enumerate(pairs):
            try:
                # Read
                vol = read_dicom_volume(zf, entry)
                seg_data = nib.load(seg_path).get_fdata().astype(np.uint8)

                seg_data = reorient_seg(seg_data, vol.shape, i)

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
                    img_u8 = np.clip((img_r + 3.0) * ENCODE_SCALE, 0, 255).astype(np.uint8)
                    arrays[size]["img"][i] = img_u8
                    arrays[size]["mask"][i] = mask_r

                # Native: per-volume NPZ
                vol_uint8 = np.clip((vol_norm + 3.0) * ENCODE_SCALE, 0, 255).astype(np.uint8)
                np.savez_compressed(
                    os.path.join(native_dir, f"volume_{i:04d}.npz"),
                    image=vol_uint8, mask=seg_data,
                )
                native_shapes.append(list(seg_data.shape))
                native_meta.append({
                    "index": i,
                    "file": f"volume_{i:04d}.npz",
                    "shape": list(seg_data.shape),
                    "patient": entry["patient_id"],
                    "series": entry["series_num"],
                    "norm_mean": float(mean),
                    "norm_std": float(std),
                    "clip_lo": float(lo),
                    "clip_hi": float(hi),
                    "encode_offset": ENCODE_OFFSET,
                    "encode_scale": ENCODE_SCALE,
                })

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
                for size in SIZES:
                    for arr in arrays[size].values():
                        arr.flush()

    # Determine which indices succeeded by checking first size array for zeros
    failed = set()
    for i in range(n_total):
        if arrays[SIZES[0]]["img"][i].sum() == 0 and arrays[SIZES[0]]["mask"][i].sum() == 0:
            failed.add(i)
    valid_idx = np.array([i for i in range(n_total) if i not in failed])
    n_processed = len(valid_idx)
    print(f"  Processed {n_processed}/{n_total} volumes")

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

    # 6. Save NPZs (compact → remove failed indices → save)
    print("\nSaving NPZ files...")
    for size in SIZES:
        print(f"  Size {size}...")

        src_img = np.memmap(os.path.join(TMPDIR, f"img_{size}.npy"),
                            dtype=np.uint8, mode="r",
                            shape=(n_total, size, size, size))
        src_mask = np.memmap(os.path.join(TMPDIR, f"mask_{size}.npy"),
                             dtype=np.uint8, mode="r",
                             shape=(n_total, size, size, size))

        # Compact: select only valid row indices
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

    # 7. Save metadata
    meta = {
        "name": "SpineSegMNIST3D",
        "description": "Cervical spine MRI with vertebral body (1) and spinal canal (2) segmentation",
        "source": "DukeCSpineSeg",
        "modality": "MR",
        "n_samples": n_processed,
        "n_classes": 3,
        "labels": {"background": 0, "vertebral_body": 1, "spinal_canal": 2},
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
    ns = sum(os.path.getsize(os.path.join(native_dir, f))
             for f in os.listdir(native_dir) if f.endswith(".npz"))
    total += ns
    print(f"  native/: {ns/1e9:.2f} GB")
    print(f"  TOTAL: {total/1e9:.2f} GB")
    print("Done ✓ ❤️")


if __name__ == "__main__":
    main()

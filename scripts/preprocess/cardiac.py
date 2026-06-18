"""
Preprocess ACDC (cardiac MRI) → CardiacSegMNIST3D.
Labels: 0=background, 1=RV, 2=Myocardium, 3=LV → 4 classes.

Source: https://huggingface.co/datasets/mathpluscode/ACDC
Resampled to 1×1×10mm, center-cropped to 192×192 in-plane.
100 patients × 2 timepoints (ED/ES) = 200 labeled volumes.
"""

import os, sys, json, time, gc, tempfile, shutil
import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.preprocess.common import resample_and_resize, save_large_npz, save_metadata

DATASET_NAME = "cardiac3d"
OUT_DIR = "/teamspace/studios/this_studio/MedSegMNIST/datasets"

# Standard sizes (all cubic)
SIZES = [64, 96, 128]
N_FOLDS = 5
SEED = 42
PROGRESS_EVERY = 25
TMPDIR = "/tmp/cardiac_memmap"


def download_data(target_dir):
    """Download ACDC from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"]
        )
        from huggingface_hub import snapshot_download

    print("Downloading ACDC from Hugging Face...")
    data_dir = snapshot_download(
        repo_id="mathpluscode/ACDC",
        allow_patterns=["*.nii.gz", "*.csv"],
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    print(f"  Downloaded to {data_dir}")
    return data_dir


def collect_samples(data_dir):
    """Collect all (image_path, gt_path, patient_id, frame) pairs."""
    samples = []
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        # maybe flat structure
        train_dir = data_dir

    patients = sorted([d for d in os.listdir(train_dir)
                       if os.path.isdir(os.path.join(train_dir, d))])
    for pid in patients:
        pdir = os.path.join(train_dir, pid)
        for fname in sorted(os.listdir(pdir)):
            if fname.endswith("_gt.nii.gz"):
                base = fname.replace("_gt.nii.gz", "")
                gt_path = os.path.join(pdir, fname)
                img_path = os.path.join(pdir, f"{base}.nii.gz")
                if os.path.isfile(img_path):
                    frame_str = base.split("frame")[-1]
                    samples.append((img_path, gt_path, pid, int(frame_str)))
    return samples


def normalize_mri(volume):
    """P0.5/P99.5 clip + z-score over non-zero foreground."""
    fg = volume[volume > 0]
    if len(fg) == 0:
        return volume
    lo, hi = np.percentile(fg, [0.5, 99.5])
    clipped = np.clip(volume, lo, hi)
    fg2 = clipped[clipped > 0]
    mean, std = fg2.mean(), fg2.std()
    normed = np.zeros_like(clipped)
    if std > 0:
        normed[clipped > 0] = (clipped[clipped > 0] - mean) / std
    return normed


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMPDIR, exist_ok=True)

    raw_dir = os.path.join(OUT_DIR, "..", "dataset-raw", "cardiac")
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Download
    data_dir = download_data(raw_dir)

    # 2. Collect samples
    samples = collect_samples(data_dir)
    n_total = len(samples)
    print(f"Total: {n_total} samples ({n_total // 2} patients × 2 timepoints)")

    # 3. Pre-allocate output arrays
    print("\nPre-allocating output arrays...")
    arrays = {}
    for size in SIZES:
        shape = (n_total, size, size, size)
        arrays[size] = {
            "img": np.memmap(
                os.path.join(TMPDIR, f"img_{size}.npy"),
                dtype=np.uint8, mode="w+", shape=shape,
            ),
            "mask": np.memmap(
                os.path.join(TMPDIR, f"mask_{size}.npy"),
                dtype=np.uint8, mode="w+", shape=shape,
            ),
        }

    native_dir = os.path.join(OUT_DIR, f"{DATASET_NAME}_native")
    os.makedirs(native_dir, exist_ok=True)
    native_meta = []
    native_shapes = []

    # 4. Process volumes
    print("\nProcessing volumes...")
    start_time = time.time()

    for i, (img_path, gt_path, pid, frame) in enumerate(samples):
        try:
            vol = nib.load(img_path).get_fdata().astype(np.float32)
            seg = nib.load(gt_path).get_fdata().astype(np.uint8)

            if vol.shape != seg.shape:
                print(f"  WARN: shape mismatch {i}: vol={vol.shape} seg={seg.shape}")
                continue

            vol_norm = normalize_mri(vol)

            for size in SIZES:
                img_r, mask_r = resample_and_resize(
                    vol_norm, seg, (size, size, size)
                )
                img_u8 = np.clip(img_r * 255, 0, 255).astype(np.uint8)
                arrays[size]["img"][i] = img_u8
                arrays[size]["mask"][i] = mask_r

            vol_uint8 = np.clip(vol_norm * 255, 0, 255).astype(np.uint8)
            np.savez_compressed(
                os.path.join(native_dir, f"volume_{i:04d}.npz"),
                image=vol_uint8, mask=seg,
            )
            native_shapes.append(list(seg.shape))
            native_meta.append({
                "index": i,
                "file": f"volume_{i:04d}.npz",
                "shape": list(seg.shape),
                "patient": pid,
                "frame": frame,
            })

        except Exception as e:
            print(f"\n  ERROR volume {i} ({pid}, frame {frame}): {e}")
            import traceback
            traceback.print_exc()
            continue

        if (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_total - i - 1) / rate
            print(f"  [{i+1}/{n_total}] {elapsed/60:.1f}m elap, "
                  f"{remaining/60:.1f}m rem, {rate:.1f} vol/min")

    # 5. Determine valid indices
    failed = set()
    for i in range(n_total):
        if arrays[SIZES[0]]["img"][i].sum() == 0 and arrays[SIZES[0]]["mask"][i].sum() == 0:
            failed.add(i)
    valid_idx = np.array([i for i in range(n_total) if i not in failed])
    n_processed = len(valid_idx)
    print(f"  Processed {n_processed}/{n_total} volumes")

    # 6. Create folds on valid indices
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_folds = {}
    for fi, (tr, te) in enumerate(kf.split(np.arange(n_processed))):
        cv_folds[f"fold_{fi}"] = {"train": tr.tolist(), "test": te.tolist()}
    fold0 = cv_folds["fold_0"]
    print(f"  CV folds: {len(fold0['train'])} train, {len(fold0['test'])} test")

    # 7. Flush and close memmaps
    for size in SIZES:
        for arr in arrays[size].values():
            arr.flush()
            del arr

    # 8. Save NPZs
    print("\nSaving NPZ files...")
    for size in SIZES:
        print(f"  Size {size}...")
        src_img = np.memmap(
            os.path.join(TMPDIR, f"img_{size}.npy"),
            dtype=np.uint8, mode="r",
            shape=(n_total, size, size, size),
        )
        src_mask = np.memmap(
            os.path.join(TMPDIR, f"mask_{size}.npy"),
            dtype=np.uint8, mode="r",
            shape=(n_total, size, size, size),
        )

        compact_img = src_img[valid_idx]
        compact_mask = src_mask[valid_idx]

        out_path = os.path.join(OUT_DIR, f"{DATASET_NAME}_{size}.npz")
        save_large_npz(
            out_path, compact_img, compact_mask,
            np.zeros((0, size, size, size), dtype=np.uint8),
            np.zeros((0, size, size, size), dtype=np.uint8),
        )
        print(f"    {out_path}: {os.path.getsize(out_path)/1e9:.2f} GB")

        del src_img, src_mask
        os.remove(os.path.join(TMPDIR, f"img_{size}.npy"))
        os.remove(os.path.join(TMPDIR, f"mask_{size}.npy"))

    # 9. Save metadata
    meta = {
        "name": "CardiacSegMNIST3D",
        "description": "Cardiac cine MRI with RV (1), Myocardium (2), LV (3) segmentation",
        "source": "ACDC (mathpluscode/ACDC on Hugging Face)",
        "modality": "MR",
        "n_samples": n_processed,
        "n_classes": 4,
        "labels": {
            "background": 0, "right_ventricle": 1,
            "myocardium": 2, "left_ventricle": 3,
        },
        "available_sizes": SIZES + ["native"],
        "cv_folds": cv_folds,
        "native_shapes": native_shapes,
        "native_metadata": native_meta,
        "official_test_fold": 0,
        "seed": SEED,
    }
    save_metadata(os.path.join(OUT_DIR, f"{DATASET_NAME}.json"), meta)

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
    print("Done")


if __name__ == "__main__":
    main()

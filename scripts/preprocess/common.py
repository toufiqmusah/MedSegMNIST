import os
import json
import hashlib
import tempfile
import zipfile
import shutil
import numpy as np
import torchio as tio
from sklearn.model_selection import train_test_split, KFold


def resample_and_resize(image_np, mask_np, target_shape):
    image_4d = image_np[np.newaxis].astype(np.float32)
    mask_4d = mask_np[np.newaxis].astype(np.int32)
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image_4d),
        mask=tio.LabelMap(tensor=mask_4d),
    )
    out = tio.Resize(target_shape)(subject)
    return out.image.numpy()[0], out.mask.numpy()[0].astype(np.uint8)


def normalize_mri(volume_np, low=0.5, high=99.5):
    volume = volume_np.astype(np.float32)
    non_zero = volume[volume > 0]
    if len(non_zero) == 0:
        return volume
    lo, hi = np.percentile(non_zero, [low, high])
    volume = np.clip(volume, lo, hi)
    mean = volume[volume > 0].mean()
    std = volume[volume > 0].std()
    if std > 0:
        volume[volume > 0] = (volume[volume > 0] - mean) / std
    return volume.astype(np.float32)


def normalize_xray(image_np, low=1.0, high=99.0):
    image = image_np.astype(np.float32)
    lo, hi = np.percentile(image, [low, high])
    image = np.clip(image, lo, hi)
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std
    return image.astype(np.float32)


def normalize_ct(volume_np, hu_range):
    volume = volume_np.astype(np.float32)
    volume = np.clip(volume, hu_range[0], hu_range[1])
    foreground = volume[volume > hu_range[0]]
    if len(foreground) == 0:
        return volume.astype(np.float32)
    mean = foreground.mean()
    std = foreground.std()
    if std > 0:
        volume = (volume - mean) / std
    return volume.astype(np.float32)


def normalize_rgb(image_np):
    return (image_np / 255.0).astype(np.float32)


def normalize_ultrasound(image_np, low=1.0, high=99.0):
    image = image_np.astype(np.float32)
    lo, hi = np.percentile(image, [low, high])
    image = np.clip(image, lo, hi)
    mean = image.mean()
    std = image.std()
    if std > 0:
        image = (image - mean) / std
    return image.astype(np.float32)


def make_splits(n_samples, test_size=0.20, n_folds=5, seed=42):
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True
    )
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_folds = {}
    for i, (tr, va) in enumerate(kf.split(train_idx)):
        cv_folds[f"fold_{i}"] = {
            "train": tr.tolist(),
            "val": va.tolist(),
        }
    return train_idx.tolist(), test_idx.tolist(), cv_folds


def compute_padded_shape(all_shapes, percentile=95):
    shapes = np.array(all_shapes)
    return tuple(
        int(np.percentile(shapes[:, i], percentile)) for i in range(shapes.shape[1])
    )


def pad_to_shape(volume, target_shape, constant=0):
    pad = [(0, max(0, t - s)) for s, t in zip(volume.shape, target_shape)]
    return np.pad(volume, pad, mode="constant", constant_values=constant)


def save_npz(out_path, train_images, train_masks, test_images, test_masks):
    np.savez_compressed(
        out_path,
        train_images=train_images,
        train_masks=train_masks,
        test_images=test_images,
        test_masks=test_masks,
    )


def save_large_npz(out_path, train_images, train_masks, test_images, test_masks):
    temp_dir = tempfile.mkdtemp()
    try:
        arrays = {
            "train_images": train_images,
            "train_masks": train_masks,
            "test_images": test_images,
            "test_masks": test_masks,
        }
        npy_paths = {}
        for name, arr in arrays.items():
            path = os.path.join(temp_dir, f"{name}.npy")
            np.save(path, arr)
            npy_paths[name] = path
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ["train_images", "train_masks", "test_images", "test_masks"]:
                zf.write(npy_paths[name], f"{name}.npy")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def save_metadata(out_path, meta_dict):
    with open(out_path, "w") as f:
        json.dump(meta_dict, f, indent=2)


def compute_sha256(file_path):
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_checksum(npz_path, checksum_dir, checksum_name=None):
    os.makedirs(checksum_dir, exist_ok=True)
    sha = compute_sha256(npz_path)
    base = os.path.basename(npz_path)
    if checksum_name is None:
        checksum_name = f"{base}.sha256"
    out_path = os.path.join(checksum_dir, checksum_name)
    with open(out_path, "w") as f:
        f.write(f"{sha}  {base}\n")
    return sha

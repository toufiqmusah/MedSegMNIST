import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import KFold
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(__file__))
from common import save_npz, write_checksum as _write_checksum

RAW_DIR = "/teamspace/studios/this_studio/dataset-raw/nuclei-seg"
OUT_DIR = "/teamspace/studios/this_studio/datasets"
FLAG = "nuclei2d"

NATIVE_SHAPE = (1024, 1024)
NATIVE_PAD_SHAPE = (1024, 1024)

STANDARDISED_SIZES = {
    256: {"shape": (256, 256)},
    512: {"shape": (512, 512)},
}


def load_nusec():
    base = os.path.join(RAW_DIR, "NuSeC")
    train_samples = []
    test_samples = []

    tmpdir = tempfile.mkdtemp()
    for split_name, rar_name, img_dir in [
        ("train", "mask of train nuclei.rar", "train nuclei"),
        ("test", "mask of test nuclei.rar", "test nuclei"),
    ]:
        rar_path = os.path.join(base, rar_name)
        subprocess.run(
            ["unrar", "e", rar_path, tmpdir + "/"],
            capture_output=True,
            check=True,
        )
        mask_files = {}
        for f in os.listdir(tmpdir):
            if f.endswith(".tif.tif"):
                mask_files[f] = os.path.join(tmpdir, f)

        img_path = os.path.join(base, img_dir)
        for f in sorted(os.listdir(img_path)):
            if not f.endswith(".tif.tif"):
                continue
            img = np.array(Image.open(os.path.join(img_path, f)))
            if f in mask_files:
                msk = np.array(Image.open(mask_files[f]))
            else:
                msk = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

            msk_binary = (msk > 0).astype(np.uint8)
            sample = {
                "image": img,
                "mask": msk_binary,
                "source": "nusec",
            }
            if split_name == "train":
                train_samples.append(sample)
            else:
                test_samples.append(sample)
            del mask_files[f]

        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))

    return train_samples, test_samples


def rasterize_xml_mask(xml_path, width, height):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mask = Image.new("L", (width, height), 0)
    for region in root.findall(".//Region"):
        polygon = [
            (float(v.get("X")), float(v.get("Y")))
            for v in region.findall("Vertices/Vertex")
        ]
        if len(polygon) > 2:
            ImageDraw.Draw(mask).polygon(polygon, fill=255)
    return np.array(mask, dtype=np.uint8)


def load_monuseg():
    base = os.path.join(RAW_DIR, "MoNuSeg 2018 Training Data")

    train_img_dir = os.path.join(base, "Training", "Tissue Images")
    train_ann_dir = os.path.join(base, "Training", "Annotations")
    test_dir = os.path.join(base, "Testing")

    train_samples = []
    test_samples = []

    for f in sorted(os.listdir(train_img_dir)):
        if not f.endswith(".tif"):
            continue
        stem = os.path.splitext(f)[0]
        img = np.array(Image.open(os.path.join(train_img_dir, f)))
        h, w = img.shape[:2]
        xml_path = os.path.join(train_ann_dir, stem + ".xml")
        if os.path.exists(xml_path):
            msk = rasterize_xml_mask(xml_path, w, h)
        else:
            msk = np.zeros((h, w), dtype=np.uint8)
        train_samples.append(
            {
                "image": img,
                "mask": msk,
                "source": "monuseg",
            }
        )

    for f in sorted(os.listdir(test_dir)):
        if not f.endswith(".tif"):
            continue
        stem = os.path.splitext(f)[0]
        img = np.array(Image.open(os.path.join(test_dir, f)))
        h, w = img.shape[:2]
        xml_path = os.path.join(test_dir, stem + ".xml")
        if os.path.exists(xml_path):
            msk = rasterize_xml_mask(xml_path, w, h)
        else:
            msk = np.zeros((h, w), dtype=np.uint8)
        test_samples.append(
            {
                "image": img,
                "mask": msk,
                "source": "monuseg",
            }
        )

    return train_samples, test_samples


def pad_to_shape(image, mask, target_shape):
    h, w = image.shape[:2]
    th, tw = target_shape
    pad_h = th - h
    pad_w = tw - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if image.ndim == 3:
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )
    else:
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
        )
    mask = np.pad(
        mask,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
    )
    return image, mask


def normalize_rgb(image):
    return (image / 255.0).astype(np.float32)


def resize_image_mask(image, mask, target_size):
    if image.shape[:2] == target_size:
        return image.copy(), mask.copy()
    order = 1 if image.ndim == 3 else 1
    img_resized = resize(
        image,
        (target_size[0], target_size[1]),
        order=order,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    msk_resized = resize(
        mask.astype(np.float32),
        (target_size[0], target_size[1]),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    msk_resized = (msk_resized > 0.5).astype(np.uint8)
    return img_resized, msk_resized


def main():
    parser = argparse.ArgumentParser(description="Preprocess NucleiSegMNIST2D")
    parser.add_argument("--raw_dir", default=RAW_DIR)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument(
        "--sizes",
        type=str,
        default="256,512,native",
        help="Sizes to generate: 256, 512, native (comma-separated)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    checksums_dir = os.path.join(args.out_dir, "checksums")
    os.makedirs(checksums_dir, exist_ok=True)

    print("Loading NuSeC...")
    nusec_train, nusec_test = load_nusec()
    print(f"  {len(nusec_train)} train, {len(nusec_test)} test")

    print("Loading MoNuSeg...")
    monuseg_train, monuseg_test = load_monuseg()
    print(f"  {len(monuseg_train)} train, {len(monuseg_test)} test")

    # Preserve original dataset splits: train = NuSeC_train + MoNuSeg_train,
    # test = NuSeC_test + MoNuSeg_test
    train_samples = nusec_train + monuseg_train
    test_samples = nusec_test + monuseg_test
    print(f"\nTotal: {len(train_samples)} train, {len(test_samples)} test")

    def process_samples(samples):
        images = []
        masks = []
        meta = []
        for s in samples:
            img, msk = pad_to_shape(s["image"], s["mask"], NATIVE_PAD_SHAPE)
            images.append(img)
            masks.append(msk)
            meta.append({"source": s["source"]})
        return np.stack(images, axis=0), np.stack(masks, axis=0), meta

    train_images, train_masks, train_meta = process_samples(train_samples)
    test_images, test_masks, test_meta = process_samples(test_samples)
    print(f"Train images: {train_images.shape}, masks: {train_masks.shape}")
    print(f"Test images: {test_images.shape}, masks: {test_masks.shape}")

    # CV folds on training set only
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds = {}
    for i, (tr, va) in enumerate(kf.split(train_images)):
        cv_folds[f"fold_{i}"] = {
            "train": tr.tolist(),
            "val": va.tolist(),
        }

    sizes_to_process = [s.strip() for s in args.sizes.split(",")]

    for size_str in sizes_to_process:
        if size_str == "native":
            suffix = "native"
            img_out = normalize_rgb(train_images.copy())
            msk_out = train_masks.copy()
            tst_img_out = normalize_rgb(test_images.copy())
            tst_msk_out = test_masks.copy()
        else:
            size = int(size_str)
            suffix = str(size)
            config = STANDARDISED_SIZES[size]
            target_shape = config["shape"]

            train_resized = [
                resize_image_mask(train_images[i], train_masks[i], target_shape)
                for i in range(len(train_images))
            ]
            img_out = normalize_rgb(np.stack([r[0] for r in train_resized], axis=0))
            msk_out = np.stack([r[1] for r in train_resized], axis=0)

            test_resized = [
                resize_image_mask(test_images[i], test_masks[i], target_shape)
                for i in range(len(test_images))
            ]
            tst_img_out = normalize_rgb(np.stack([r[0] for r in test_resized], axis=0))
            tst_msk_out = np.stack([r[1] for r in test_resized], axis=0)

        npz_path = os.path.join(args.out_dir, f"{FLAG}_{suffix}.npz")
        save_npz(npz_path, img_out, msk_out, tst_img_out, tst_msk_out)
        print(f"  Saved: {npz_path} ({os.path.getsize(npz_path) / 1e6:.0f} MB)")

        _write_checksum(
            npz_path,
            checksums_dir,
            checksum_name=f"{FLAG}_{suffix}.sha256",
        )

    label_names = {"0": "background", "1": "nuclei"}

    meta_dict = {
        "flag": FLAG,
        "class_name": "NucleiSegMNIST2D",
        "name": "NuSeC + MoNuSeg 2018",
        "version": "1.0.0",
        "dimensionality": "2D",
        "modality": "Pathology",
        "anatomy": "Multi-organ (nuclei)",
        "source_urls": [
            "https://github.com/baovuong96/nusec",
            "https://monuseg.grand-challenge.org",
        ],
        "license": "Research purposes",
        "redistribution_allowed": False,
        "split_strategy": "by_source_original_splits",
        "original_split_details": {
            "nusec": {"train": 75, "test": 25},
            "monuseg": {"train": 37, "test": 14},
            "combined": {"train": 112, "test": 39},
        },
        "available_sizes": (
            [int(s) for s in sizes_to_process if s != "native"]
            + (["native"] if "native" in sizes_to_process else [])
        ),
        "native_padded_shape": list(NATIVE_PAD_SHAPE),
        "native_percentile_box": 100,
        "standardised_sizes": {
            k: v for k, v in STANDARDISED_SIZES.items() if str(k) in sizes_to_process
        },
        "normalization": "rgb_255",
        "label_names": label_names,
        "label_original_values": {"0": 0, "1": 1},
        "n_train": len(train_images),
        "n_test": len(test_images),
        "cv_folds": cv_folds,
        "per_sample_metadata": train_meta + test_meta,
    }

    json_path = os.path.join(args.out_dir, f"{FLAG}.json")
    with open(json_path, "w") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"  Metadata: {json_path}")


if __name__ == "__main__":
    main()

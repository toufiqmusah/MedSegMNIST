import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from ..datasets.base import _viz_ctx

_SEG_CMAP = mcolors.ListedColormap(
    ["black", "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
     "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4"]
)


def _stretch_rgb(img):
    if img.ndim != 3 or img.shape[-1] != 3:
        return img
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return img
    s = np.clip((img - lo) / (hi - lo), 0, 1).astype(img.dtype)
    return s


def _cmap_for(n_labels):
    n = max(2, min(n_labels, _SEG_CMAP.N))
    return mcolors.ListedColormap(_SEG_CMAP.colors[:n])


def _ensure_2d(image, mask, slice_idx=None, slice_axis=None, rot90_k=0):
    if slice_axis is None:
        slice_axis = getattr(_viz_ctx, 'view_axis', -1)
    if rot90_k == 0:
        rot90_k = getattr(_viz_ctx, 'rot90_k', 0)
    if image.ndim == 4:
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        elif image.shape[0] == 3 and image.shape[-1] != 3:
            if slice_idx is None:
                fg = (mask > 0).sum(axis=(0, 1))
                slice_idx = int(np.argmax(fg)) if fg.sum() > 0 else image.shape[-1] // 2
            image = image[:, :, :, slice_idx]
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0, :, :, slice_idx]
            elif mask.ndim == 4:
                mask = mask[:, :, slice_idx]
            elif mask.ndim == 3:
                mask = mask[:, :, slice_idx]
            image = np.transpose(image, (1, 2, 0))
            return image, mask
        else:
            image, mask = image[0], mask[0]

    if image.ndim == 3:
        if image.shape[-1] == 3:
            return image, mask
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
            return image, mask
        if image.shape[0] == 1:
            image = image[0]
        if image.ndim == 3:
            if slice_idx is None:
                _axis = slice_axis % image.ndim
                other_axes = tuple(a for a in range(image.ndim) if a != _axis)
                fg = (mask > 0).sum(axis=other_axes).astype(np.float64)
                if fg.sum() > 0:
                    indices = np.arange(len(fg))
                    slice_idx = int(np.round(np.average(indices, weights=fg)))
                else:
                    slice_idx = image.shape[_axis] // 2
            else:
                _axis = slice_axis % image.ndim
            image = np.take(image, slice_idx, axis=_axis)
            if mask.ndim == 3:
                mask = np.take(mask, slice_idx, axis=_axis)
    if rot90_k:
        image = np.rot90(image, k=rot90_k)
        if mask.ndim == 2:
            mask = np.rot90(mask, k=rot90_k)
    return image, mask


def plot_sample(image, mask, slice_idx=None, slice_axis=None, rot90_k=0, ax=None):
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(1, 2, figsize=(10, 5))

    img_slice, msk_slice = _ensure_2d(image, mask, slice_idx, slice_axis, rot90_k)

    is_rgb = img_slice.ndim == 3 and img_slice.shape[-1] == 3
    if is_rgb:
        img_slice = _stretch_rgb(img_slice)
    cmap = None if is_rgb else "gray"
    ax[0].imshow(img_slice, cmap=cmap, aspect="auto")
    ax[0].set_title("Image")
    ax[0].axis("off")

    n_labels = int(msk_slice.max()) + 1
    ax[1].imshow(msk_slice, cmap=_cmap_for(n_labels), vmin=0, vmax=n_labels - 1, aspect="auto")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    if own_fig:
        plt.tight_layout()
        plt.show()

    return ax


def plot_overlay(image, mask, alpha=0.5, slice_idx=None, slice_axis=None, rot90_k=0,
                 ax=None):
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(6, 6))

    img_slice, msk_slice = _ensure_2d(image, mask, slice_idx, slice_axis, rot90_k)

    is_rgb = img_slice.ndim == 3 and img_slice.shape[-1] == 3
    if is_rgb:
        img_slice = _stretch_rgb(img_slice)
    cmap = None if is_rgb else "gray"
    ax.imshow(img_slice, cmap=cmap, aspect="auto")
    msk_masked = np.ma.masked_where(msk_slice == 0, msk_slice)
    n_labels = int(msk_slice.max()) + 1
    ax.imshow(msk_masked, cmap=_cmap_for(n_labels), alpha=alpha,
              vmin=0, vmax=n_labels - 1, aspect="auto")
    ax.set_title("Overlay")
    ax.axis("off")

    if own_fig:
        plt.tight_layout()
        plt.show()

    return ax


def plot_grid(dataset=None, images=None, masks=None, indices=None, n_samples=9, n_cols=None, slice_idx=None, slice_axis=None, seed=None):
    if slice_axis is None:
        slice_axis = getattr(dataset, 'view_axis', -1) if dataset is not None else -1
    rot90_k = getattr(dataset, 'rot90_k', 0) if dataset is not None else 0

    if dataset is not None:
        n_total = len(dataset)
        if indices is not None:
            idxs = indices
        else:
            rng = np.random.default_rng(seed)
            idxs = rng.integers(0, n_total, size=min(n_samples, n_total)).tolist()
        images, masks = zip(*[dataset[i] for i in idxs]) if idxs else ([], [])

    n = len(images)
    if n_cols is None:
        n_cols = max(1, int(np.ceil(np.sqrt(n))))
    n_rows = (n + n_cols - 1) // n_cols
    n_sub_cols = n_cols * 2

    fig, axes = plt.subplots(n_rows, n_sub_cols,
                             figsize=(n_sub_cols * 1.5, n_rows * 1.5))
    if n_rows == 1 and n_sub_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    n_labels = max(int(msk.max()) + 1 for msk in masks) if masks else 0
    cmap = _cmap_for(n_labels) if n_labels else _SEG_CMAP
    vmax = n_labels - 1 if n_labels else 0

    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    for i in range(n):
        row = i // n_cols
        col = (i % n_cols) * 2
        img_slice, msk_slice = _ensure_2d(images[i], masks[i], slice_idx, slice_axis, rot90_k)
        is_rgb = img_slice.ndim == 3 and img_slice.shape[-1] == 3
        if is_rgb:
            img_slice = _stretch_rgb(img_slice)
        im_cmap = None if is_rgb else "gray"
        axes[row, col].imshow(img_slice, cmap=im_cmap, aspect="auto")
        axes[row, col + 1].imshow(msk_slice, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for a in axes[row, col:col + 2]:
            a.axis("off")

    for i in range(n, n_rows * n_cols):
        row = i // n_cols
        col = (i % n_cols) * 2
        axes[row, col].axis("off")
        axes[row, col + 1].axis("off")

    for i in range(n):
        row = i // n_cols
        col = (i % n_cols) * 2
        bbox_img = axes[row, col].get_position()
        bbox_msk = axes[row, col + 1].get_position()
        x0, y0 = bbox_img.x0, bbox_img.y0
        x1, y1 = bbox_msk.x1, bbox_img.y1
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         fill=False, edgecolor="lime", linewidth=0.6,
                         transform=fig.transFigure, clip_on=False)
        fig.add_artist(rect)

    return fig

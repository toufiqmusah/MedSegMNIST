import numpy as np
import matplotlib.pyplot as plt


def _ensure_2d(image, mask, slice_idx=None):
    if image.ndim == 4:
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        elif image.shape[0] == 3 and image.shape[-1] != 3:
            image = image[0]
        else:
            image, mask = image[0], mask[0]

    if image.ndim == 3:
        if image.shape[-1] == 3:
            return image, mask
        if image.shape[0] in (1, 3):
            image = image[0]
        if image.ndim == 3:
            if slice_idx is None:
                fg = (mask > 0).sum(axis=(0, 1))
                slice_idx = int(np.argmax(fg)) if fg.sum() > 0 else image.shape[-1] // 2
            image = image[..., slice_idx]
            if mask.ndim == 3:
                mask = mask[..., slice_idx]
    return image, mask


def plot_sample(image, mask, slice_idx=None, label_names=None, ax=None):
    """Plot an image and its segmentation mask side-by-side.

    Parameters
    ----------
    image : ndarray
        Image array (2D, 3D, or 4D with channel dim).
    mask : ndarray
        Segmentation mask (same spatial dims as ``image``).
    slice_idx : int, optional
        For 3D data, the slice index to display.  Auto-selected
        on the slice with the most foreground pixels if not given.
    label_names : dict, optional
        Not currently used (reserved).
    ax : array-like of Axes, optional
        A pair of matplotlib Axes.  Creates a new figure if not given.

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(10, 5))

    img_slice, msk_slice = _ensure_2d(image, mask, slice_idx)

    ax[0].imshow(img_slice, cmap="gray")
    ax[0].set_title("Image")

    n_labels = len(np.unique(msk_slice))
    ax[1].imshow(
        msk_slice,
        cmap="tab10" if n_labels <= 10 else "viridis",
        vmin=0,
        vmax=n_labels - 1,
    )
    ax[1].set_title("Mask")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    return ax


def plot_overlay(image, mask, alpha=0.4, slice_idx=None, ax=None):
    """Overlay a segmentation mask on the image with transparency.

    Parameters
    ----------
    image : ndarray
    mask : ndarray
    alpha : float
        Transparency of the overlay (0 = invisible, 1 = opaque).
    slice_idx : int, optional
        For 3D data, the slice index to display.
    ax : Axes, optional
        Matplotlib Axes.  Creates a new figure if not given.

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    img_slice, msk_slice = _ensure_2d(image, mask, slice_idx)

    ax.imshow(img_slice, cmap="gray")
    ax.imshow(
        msk_slice, cmap="tab10", alpha=alpha, vmin=0, vmax=max(1, msk_slice.max())
    )
    ax.set_title("Overlay")
    ax.axis("off")
    return ax


def plot_grid(images, masks, n_cols=4, slice_idx=None):
    """Arrange multiple image–mask pairs in a grid.

    Parameters
    ----------
    images : list or ndarray
    masks : list or ndarray
    n_cols : int
        Number of columns (each column shows image + mask).
    slice_idx : int, optional
        For 3D data, the slice index to display.

    Returns
    -------
    Figure
    """
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 5, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        row = i // n_cols
        col = (i % n_cols) * 2
        plot_sample(
            images[i], masks[i], slice_idx=slice_idx, ax=axes[row, col : col + 2]
        )
    for i in range(n, n_rows * n_cols):
        row = i // n_cols
        col = (i % n_cols) * 2
        axes[row, col].axis("off")
        axes[row, col + 1].axis("off")
    plt.tight_layout()
    return fig

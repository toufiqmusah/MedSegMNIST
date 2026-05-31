import pytest
import numpy as np

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


class TestEnsure2D:
    def test_2d_grayscale(self):
        from medsegmnist.utils.visualize import _ensure_2d

        img = np.random.randn(64, 64)
        msk = np.random.randint(0, 2, (64, 64))
        img_out, msk_out = _ensure_2d(img, msk)
        assert img_out.shape == (64, 64)
        assert msk_out.shape == (64, 64)

    def test_3d_select_middle(self):
        from medsegmnist.utils.visualize import _ensure_2d

        img = np.random.randn(64, 64, 32)
        msk = np.random.randint(0, 2, (64, 64, 32))
        img_out, msk_out = _ensure_2d(img, msk)
        assert img_out.ndim == 2
        assert msk_out.ndim == 2

    def test_4d_with_channel(self):
        from medsegmnist.utils.visualize import _ensure_2d

        img = np.random.randn(1, 64, 64, 32)
        msk = np.random.randint(0, 2, (64, 64, 32))
        img_out, msk_out = _ensure_2d(img, msk)
        assert img_out.ndim == 2
        assert msk_out.ndim == 2

    def test_rgb(self):
        from medsegmnist.utils.visualize import _ensure_2d

        img = np.random.randn(64, 64, 3)
        msk = np.random.randint(0, 2, (64, 64))
        img_out, msk_out = _ensure_2d(img, msk)
        assert img_out.shape == (64, 64, 3)
        assert msk_out.shape == (64, 64)

    def test_4d_with_3_channels(self):
        from medsegmnist.utils.visualize import _ensure_2d

        img = np.random.randn(3, 64, 64, 32)
        msk = np.random.randint(0, 2, (64, 64, 32))
        img_out, msk_out = _ensure_2d(img, msk)
        assert img_out.ndim == 2
        assert msk_out.ndim == 2


class TestPlotSample:
    def test_creates_axes(self):
        from medsegmnist.utils.visualize import plot_sample
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2)
        img = np.random.randn(64, 64)
        msk = np.random.randint(0, 2, (64, 64))
        result = plot_sample(img, msk, ax=ax)
        assert result is ax
        plt.close(fig)

    def test_with_overlay_slice_idx(self):
        from medsegmnist.utils.visualize import plot_overlay
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        img = np.random.randn(64, 64, 32)
        msk = np.random.randint(0, 2, (64, 64, 32))
        plot_overlay(img, msk, alpha=0.5, ax=ax)
        plt.close(fig)


class TestPlotGrid:
    def test_smoke(self):
        from medsegmnist.utils.visualize import plot_grid
        import matplotlib.pyplot as plt

        images = [np.random.randn(64, 64) for _ in range(4)]
        masks = [np.random.randint(0, 2, (64, 64)) for _ in range(4)]
        fig = plot_grid(images, masks, n_cols=2)
        assert fig is not None
        plt.close(fig)

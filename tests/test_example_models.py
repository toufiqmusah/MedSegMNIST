"""Tests for the example U-Net implementations.

These test the reference implementations in ``examples/``, which are not
part of the ``medsegmnist`` package itself.
"""

import sys
import os
import pytest

torch = pytest.importorskip("torch")

_examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
if _examples_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_examples_dir))


class TestUNet2D:
    def test_init(self):
        from unet import UNet2D

        model = UNet2D(in_channels=1, n_classes=2)
        assert model.in_channels == 1
        assert model.n_classes == 2

    def test_forward_output_shape(self):
        from unet import UNet2D

        model = UNet2D(in_channels=1, n_classes=2)
        x = torch.randn(4, 1, 128, 128)
        out = model(x)
        assert out.shape == (4, 2, 128, 128)

    def test_multi_class(self):
        from unet import UNet2D

        model = UNet2D(in_channels=1, n_classes=4)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 4, 64, 64)

    def test_batch_1(self):
        from unet import UNet2D

        model = UNet2D(in_channels=1, n_classes=2)
        x = torch.randn(1, 1, 128, 128)
        out = model(x)
        assert out.shape == (1, 2, 128, 128)

    def test_different_depths(self):
        from unet import UNet2D

        for depth in [3, 4]:
            model = UNet2D(in_channels=1, n_classes=2, depth=depth)
            x = torch.randn(2, 1, 128, 128)
            out = model(x)
            assert out.shape == (2, 2, 128, 128)

    def test_transposed_conv(self):
        from unet import UNet2D

        model = UNet2D(in_channels=1, n_classes=2, bilinear=False)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        assert out.shape == (2, 2, 128, 128)

    def test_gradients_flow(self):
        from unet import UNet2D

        model = UNet2D(in_channels=1, n_classes=2)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for {name}"

    def test_base_filters_configurable(self):
        from unet import UNet2D

        model = UNet2D(in_channels=1, n_classes=2, base_filters=32)
        x = torch.randn(2, 1, 128, 128)
        out = model(x)
        assert out.shape == (2, 2, 128, 128)


class TestUNet3D:
    def test_init(self):
        from unet3d import UNet3D

        model = UNet3D(in_channels=1, n_classes=4)
        assert model.in_channels == 1
        assert model.n_classes == 4

    def test_forward_output_shape(self):
        from unet3d import UNet3D

        model = UNet3D(in_channels=1, n_classes=4)
        x = torch.randn(2, 1, 64, 64, 48)
        out = model(x)
        assert out.shape == (2, 4, 64, 64, 48)

    def test_batch_1(self):
        from unet3d import UNet3D

        model = UNet3D(in_channels=1, n_classes=2)
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32, 32)

    def test_gradients_flow(self):
        from unet3d import UNet3D

        model = UNet3D(in_channels=1, n_classes=2, base_filters=16)
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for {name}"

    def test_base_filters_configurable(self):
        from unet3d import UNet3D

        model = UNet3D(in_channels=1, n_classes=2, base_filters=16)
        x = torch.randn(2, 1, 32, 32, 32)
        out = model(x)
        assert out.shape == (2, 2, 32, 32, 32)

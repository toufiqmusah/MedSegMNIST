Contributing
============

MedSegMNIST welcomes contributions! Here's how you can help.

Setting up a development environment
------------------------------------

.. code-block:: bash

   git clone https://github.com/MedSegMNIST/MedSegMNIST.git
   cd MedSegMNIST
   pip install -e ".[dev,preprocess]"

Code style
----------

We use `ruff <https://docs.astral.sh/ruff/>`_ for both linting and
formatting:

.. code-block:: bash

   ruff check medsegmnist/ tests/
   ruff format --check medsegmnist/ tests/

Running tests
-------------

.. code-block:: bash

   python -m pytest tests/

Tests that require the actual NPZ data files are marked with
``@pytest.mark.requires_data`` and will skip automatically in CI where
the data is not available.

Adding a new dataset
--------------------

1. **Create the dataset class** in ``medsegmnist/datasets/{modality}/``::

    @register
    class MyDataset(MedSegMNIST2D):
        flag = "my2d"
        class_name = "MyDataset"
        available_sizes = [128, 256]
        n_classes = 3
        modality = "CT"
        n_channels = 1

2. **Export it** in ``medsegmnist/__init__.py``.

3. **Add a preprocessing script** in ``scripts/preprocess/`` that generates
   the NPZ and JSON files in the dataset root.

4. **Add tests** in ``tests/`` covering at least length, shape, and fold
   access.

5. **Update the docs** — add the dataset to ``docs/source/datasets.rst``.

Project structure
-----------------

.. code-block:: text

   medsegmnist/
   ├── __init__.py              # Public API (dataset classes, info, list_datasets)
   ├── registry.py              # Dataset registry (@register, info, list_datasets)
   ├── datasets/
   │   ├── base.py              # MedSegMNIST2D / MedSegMNIST3D
   │   ├── mri/brain.py         # BrainSegMNIST3D
   │   ├── xray/lung.py         # LungSegMNIST2D
   │   └── pathology/nuclei.py  # NucleiSegMNIST2D
   ├── cli/                     # CLI entry points (medsegmnist train/eval)
   ├── training/                # DiceScore, DiceLoss, MedSegModule
   └── utils/                   # Visualisation utilities
   scripts/preprocess/          # Preprocessing scripts
   examples/                    # Reference model implementations
   docs/                        # Sphinx documentation

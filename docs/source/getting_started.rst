Getting started
===============

.. highlight:: python

Installation
------------

.. code-block:: bash

   pip install medsegmnist

To run preprocessing scripts for building datasets from raw sources:

.. code-block:: bash

   pip install "medsegmnist[preprocess]"

For development:

.. code-block:: bash

   pip install "medsegmnist[dev]"

Quick start
-----------

.. code-block:: python

   from medsegmnist import LungSegMNIST2D, list_datasets

   # List all available datasets
   print(list_datasets())

   # Load a dataset
   ds = LungSegMNIST2D(split="train", size=128, root="/path/to/datasets")
   print(len(ds))  # 5448

   # Access a sample
   image, mask = ds[0]
   print(image.shape)  # (1, 128, 128) — channel-first float32
   print(mask.shape)   # (128, 128) — uint8

Load any dataset
----------------

.. code-block:: python

   from medsegmnist import info

   info("brain3d")

   from medsegmnist import BrainSegMNIST3D

   ds = BrainSegMNIST3D(split="train", size=96)
   image, mask = ds[0]        # (1, 96, 96, 64), (96, 96, 64)

Data shape convention
---------------------

+----------------+------------------------------+------------------+
| Dimensionality | Image shape                  | Mask shape       |
+================+==============================+==================+
| 2D (1-channel) | ``(1, H, W)`` float32        | ``(H, W)`` uint8 |
+----------------+------------------------------+------------------+
| 2D (3-channel) | ``(3, H, W)`` float32        | ``(H, W)`` uint8 |
+----------------+------------------------------+------------------+
| 3D             | ``(1, D, H, W)`` float32     | ``(D, H, W)``    |
+----------------+------------------------------+------------------+

Cross-validation folds
----------------------

.. code-block:: python

   ds = LungSegMNIST2D(split="train", size=128)
   train_subset, val_subset = ds.get_fold(0)
   print(len(train_subset), len(val_subset))

Training with your own model
----------------------------

.. code-block:: python

   from medsegmnist import LungSegMNIST2D
   from medsegmnist.training import MedSegModule
   import lightning as L
   from torch.utils.data import DataLoader

   ds = LungSegMNIST2D(split="train", size=128)
   train_subset, val_subset = ds.get_fold(0)

   train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
   val_loader = DataLoader(val_subset, batch_size=16)

   model = ...  # your segmentation model
   module = MedSegModule(model=model, num_classes=2)

   trainer = L.Trainer(max_epochs=50)
   trainer.fit(module, train_loader, val_loader)

Or use the CLI:

.. code-block:: bash

   medsegmnist train --model "mymodel.MyModel" --flag lung2d --size 128 \\
       --epochs 50 --batch-size 16

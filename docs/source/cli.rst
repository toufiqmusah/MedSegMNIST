CLI Reference
=============

The ``medsegmnist`` command-line tool provides training and evaluation
functionality.

.. code-block:: text

   medsegmnist train --help
   medsegmnist eval --help

train
-----

Trains a user-provided segmentation model on any MedSegMNIST dataset.

.. code-block:: bash

   medsegmnist train --model "examples.unet.UNet2D" \\
       --flag lung2d --size 128 \\
       --epochs 50 --batch-size 16 \\
       --lr 1e-3 --fold 0

Required arguments:

* ``--model`` — Dotted import path to the model class (e.g.,
  ``"mypackage.models.MyModel"``)

Optional arguments:

* ``--model-kwargs`` — JSON string of keyword arguments for the model
  constructor (default: ``{}``)
* ``--flag`` — Dataset flag (default: ``"lung2d"``)
* ``--size`` — Image size (default: first available size)
* ``--root`` — Dataset root directory
* ``--epochs`` — Number of epochs (default: 50)
* ``--batch-size`` — Batch size (default: 8)
* ``--lr`` — Learning rate (default: 1e-3)
* ``--weight-decay`` — Weight decay (default: 1e-4)
* ``--fold`` — Cross-validation fold 0–4 (default: 0)
* ``--seed`` — Random seed (default: 42)
* ``--devices`` — Number of devices (default: 1)
* ``--accelerator`` — ``auto``, ``cpu``, or ``gpu`` (default: ``auto``)
* ``--fast-dev-run`` — Run a single batch for smoke-testing
* ``--output-dir`` — Checkpoint directory (default: ``./checkpoints``)

eval
----

Evaluates a checkpoint on the test set and reports per-class Dice and IoU.

.. code-block:: bash

   medsegmnist eval --checkpoint checkpoints/lung2d-128-epoch=42-val_dice=0.97.ckpt

Required arguments:

* ``--checkpoint`` — Path to a ``.ckpt`` file

Optional arguments:

* ``--flag`` — Dataset flag (auto-detected from checkpoint filename if
  omitted)
* ``--size`` — Image size (auto-detected from checkpoint filename if
  omitted)
* ``--root`` — Dataset root directory
* ``--batch-size`` — Batch size (default: 8)
* ``--devices`` — Number of devices (default: 1)
* ``--accelerator`` — ``auto``, ``cpu``, or ``gpu`` (default: ``auto``)

Example output::

    Class           Dice      IoU
    ─────────────────────────────
    background      0.9923    0.9847
    lung            0.9718    0.9452
    ─────────────────────────────
    Macro average   0.9820    0.9649

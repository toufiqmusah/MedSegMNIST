Datasets
========

.. highlight:: python

Supported datasets
------------------

.. list-table::
   :header-rows: 1
   :widths: 12 18 12 22 5 8 8 8 28

   * - Flag
     - Dataset
     - Modality
     - Anatomy
     - Dim
     - Classes
     - Train
     - Test
     - Sizes
   * - ``brain3d``
     - BrainSegMNIST3D
     - MRI
     - Brain (gliomas)
     - 3D
     - 4
     - 116
     - 30
     - 96, 128, 224, native
   * - ``lung2d``
     - LungSegMNIST
     - X-ray
     - Chest / Lungs
     - 2D
     - 2
     - 5,448
     - 1,362
     - 128, 256, 512
   * - ``nuclei2d``
     - NucleiSegMNIST
     - Pathology
     - Multi-organ (nuclei)
     - 2D
     - 2
     - 112
     - 39
     - 256, 512, native

Individual dataset details
--------------------------

BrainSegMNIST3D
~~~~~~~~~~~~~~~

Brain tumour sub-region segmentation from BraTS-Africa (T2-FLAIR).

- **Labels:** 0 = background, 1 = necrotic core, 2 = oedema, 3 = enhancing tumour
- **Native resolution:** 240 × 240 × 155 at 1.0 mm isotropic
- **Train:** 116 patients, **Test:** 30 patients
- **Sizes:** 96, 128, 224, native

LungSegMNIST
~~~~~~~~~~~~

Lung field segmentation from chest X-rays (Darwin + Montgomery + Shenzhen).

- **Labels:** 0 = background, 1 = lung
- **Native resolution:** 512 × 512, grayscale (converted from RGB)
- **Train:** 5,448 images, **Test:** 1,362 images
- **Sizes:** 128, 256, 512

NucleiSegMNIST
~~~~~~~~~~~~~~

Nuclei segmentation from NuSeC + MoNuSeg 2018.

- **Labels:** 0 = background, 1 = nuclei
- **Channels:** 3 (RGB)
- **Native resolution:** 1024 × 1024 (MoNuSeg centre-padded to match NuSeC)
- **Train:** 112 images, **Test:** 39 images
- **Sizes:** 256, 512, native
- **Split strategy:** Original dataset boundaries preserved (NuSeC train/test + MoNuSeg train/test)

Adding a new dataset
--------------------

To add a new dataset to MedSegMNIST:

1. Create a dataset class inheriting from ``MedSegMNIST2D`` or ``MedSegMNIST3D``
2. Decorate with ``@register``
3. Add a preprocessing script in ``scripts/preprocess/``
4. Generate the NPZ files and JSON metadata

.. code-block:: python

   from medsegmnist.datasets.base import MedSegMNIST2D
   from medsegmnist.registry import register

   @register
   class MyDataset(MedSegMNIST2D):
       flag = "my2d"
       class_name = "MyDataset"
       available_sizes = [128, 256]
       n_classes = 3
       modality = "CT"
       n_channels = 1

The rest — data loading, folds, ``info()``, ``list_datasets()`` — works automatically.

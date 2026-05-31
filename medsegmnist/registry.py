"""Registry for dataset classes.

New datasets are registered via the ``@register`` decorator on a concrete
``MedSegMNIST2D`` or ``MedSegMNIST3D`` subclass. Once registered, they appear
in ``list_datasets()`` and ``info()`` automatically.
"""

DATASET_REGISTRY = {}


def register(cls):
    DATASET_REGISTRY[cls.flag] = {
        "class": cls.class_name,
        "flag": cls.flag,
        "dimensionality": cls.dimensionality,
        "modality": cls.modality,
        "n_classes": cls.n_classes,
        "available_sizes": cls.available_sizes,
    }
    return cls


def info(flag):
    entry = DATASET_REGISTRY.get(flag)
    if entry is None:
        raise KeyError(
            f"Unknown dataset flag {flag!r}. Call list_datasets() to see available flags."
        )
    import pprint

    pprint.pprint(entry)
    return entry


def list_datasets(dimensionality=None):
    results = []
    for flag, entry in DATASET_REGISTRY.items():
        if dimensionality and entry["dimensionality"] != dimensionality:
            continue
        results.append((flag, entry["class"], entry["modality"]))
    return results

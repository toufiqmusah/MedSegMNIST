import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = "/teamspace/studios/this_studio/datasets"
HAS_DATA = os.path.isdir(DATASET_DIR) and any(
    f.endswith(".npz") for f in os.listdir(DATASET_DIR)
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_data: test requires NPZ dataset files")


def pytest_collection_modifyitems(items):
    if not HAS_DATA:
        skip = pytest.mark.skip(reason=f"test data not found in {DATASET_DIR}")
        for item in items:
            if "requires_data" in item.keywords:
                item.add_marker(skip)

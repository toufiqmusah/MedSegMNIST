"""Sphinx configuration for MedSegMNIST documentation."""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../medsegmnist"))

import medsegmnist

project = "MedSegMNIST"
copyright = f"{datetime.now().year}, MedSegMNIST Contributors"
author = "MedSegMNIST Contributors"
release = medsegmnist.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = None
html_favicon = None

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}

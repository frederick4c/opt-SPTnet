"""Sphinx configuration for the SPTnet documentation."""

from __future__ import annotations

import os
import sys
from datetime import date


ROOT = os.path.abspath("..")
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

project = "SPTnet"
author = "SPTnet contributors"
copyright = f"{date.today().year}, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "h5py",
    "matplotlib",
    "matplotlib.animation",
    "matplotlib.pyplot",
    "numpy",
    "positional_encodings",
    "scipy",
    "tifffile",
    "torch",
    "tqdm",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

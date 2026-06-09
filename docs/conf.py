"""Sphinx configuration for TorchInspector."""

import sys
from pathlib import Path

# Add src to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "TorchInspector"
copyright = "2026, TorchInspector contributors"
author = "TorchInspector contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

html_theme = "furo"
html_title = "TorchInspector"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autodoc_typehints = "description"
napoleon_google_docstring = True

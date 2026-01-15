"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))

# -- Project information -----------------------------------------------------
project = "JESTER"
copyright = "2025, JESTER Contributors"
author = "JESTER Contributors"
release = "0.1.1"
version = "0.1.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "colon_fence",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]

html_theme_options = {
    "use_fullscreen_button": False,
    "use_download_button": False,
    "use_repository_button": True,
    "repository_url": "https://github.com/nuclear-multimessenger-astronomy/jester",
    "home_page_in_toc": True,
    "logo": {
        "image_light": "_static/logo_light.svg",
        "image_dark": "_static/logo_dark.svg",
    },
}

html_title = "JESTER"
html_favicon = "_static/icon.svg"

pygments_style = "xcode"

# -- Autodoc configuration --------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Mock imports for optional dependencies that may not be available during docs build
autodoc_mock_imports = [
    "flowMC.nfmodel",
    "flowMC.proposal",
    "flowMC.Sampler",
]

add_module_names = False
autodoc_inherit_docstrings = False
python_maximum_signature_line_length = 88

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_rtype = False
napoleon_attr_annotations = True
napoleon_use_ivar = True

# -- Copy button configuration -----------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- MathJax configuration ---------------------------------------------------
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- Suppress warnings -------------------------------------------------------
suppress_warnings = [
    "ref.python",  # Suppress ambiguous python cross-reference warnings
    "sphinx_autodoc_typehints.forward_reference",  # Suppress forward reference warnings
    "sphinx_autodoc_typehints.guarded_import",  # Suppress guarded import warnings
]

# -- Nitpicky mode configuration ---------------------------------------------
# Ignore ambiguous references to common field names
nitpick_ignore = [
    ("py:class", "type"),
]

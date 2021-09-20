from sphinx_mdolab_theme.config import *

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../"))


project = "preFoil"

# mock import for autodoc
autodoc_mock_imports = ["numpy", "pyspline", "scipy", "pygeo"]

# intersphinx
intersphinx_mapping = {
    "adflow": (f"https://mdolab-adflow.readthedocs-hosted.com/en/latest", None),
    "mach-aero": (f"https://mdolab-mach-aero.readthedocs-hosted.com/en/latest", None),
    "pyspline": (f"https://mdolab-pyspline.readthedocs-hosted.com/en/latest", None),

}

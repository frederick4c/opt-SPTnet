SPTnet Documentation
====================

SPTnet is a refactored package for model components, datasets, training
utilities, and inference helpers used in single-particle tracking workflows.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   training_data_generation
   api/index

API Reference
-------------

The API reference is generated from docstrings in ``src/sptnet`` using Sphinx
``autodoc`` and ``autosummary``.

Build locally with:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   sphinx-build -b html docs docs/_build/html

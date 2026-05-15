# SPTnet

**WIP**
Refactor of SPTnet (https://github.com/HuanglabPurdue/SPTnet) to optimise the code and make the package accessible 


The refactored package lives under `src/sptnet/`. 

## Documentation

API documentation is built with Sphinx from the docstrings in `src/sptnet`.

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The generated site will be written to `docs/_build/html`.

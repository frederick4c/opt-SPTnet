# Contributing to opt-SPTnet

Thanks for your interest in the project. This guide covers the local
development workflow.

## Development install

```bash
python -m pip install -e ".[test,docs,dev]"
```

For an exact, pinned environment use the committed lock file:

```bash
python -m pip install -e ".[test,docs,dev]" -c constraints.txt
```

## Running the tests

```bash
pytest
```

The suite is CPU-only and uses small synthetic HDF5/TIFF files and tensors, so
it runs comfortably on a laptop. Torch-backed tests skip automatically where
PyTorch is not installed.

## Building the documentation

```bash
sphinx-build -W -b html docs docs/_build/html
```

`-W` turns warnings into errors, matching CI. Note that every public module
under `src/sptnet` must be listed in `docs/api/index.rst`; the
`test_docs_api_coverage` test enforces this.

## Code style

Style is checked with [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check src tests
```

Optionally install the git hooks so checks run on commit:

```bash
pre-commit install
pre-commit run --all-files
```

The lint job in CI is currently non-blocking while existing style is brought up
to standard; please keep new code ruff-clean.

## Continuous integration

`.github/workflows/ci.yml` runs the test matrix (Python 3.10 and 3.12), the
docs build, a wheel build + clean-environment import, and ruff.

## Repository layout and the two release remotes

The core package lives in `src/sptnet`. Experiment scripts, SLURM job files, and
benchmark tooling live in `experiments/` and `slurm/` and are not part of the
installable package.

This project is published to two remotes with different ignore profiles:

- **Public GitHub** (code only): uses the default `.gitignore`, which excludes
  the report, notebooks, raw data, and internal working notes.
- **GitLab submission** (examiner-facing): additionally includes the report
  sources/PDF and the small reproducibility artifacts. See `SUBMISSION.md` and
  `.gitignore.submission`.

## License

By contributing you agree that your contributions are licensed under the MIT
License (see `LICENSE`). opt-SPTnet is a derivative of the original Huang Lab
SPTnet and retains its upstream copyright notice.

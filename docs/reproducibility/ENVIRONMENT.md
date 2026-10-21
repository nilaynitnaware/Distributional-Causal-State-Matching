# Execution environment

## Supported release environment

The public release is tested with Python 3.12. Historical executions recorded
Python 3.12.13; `pyproject.toml` therefore declares `>=3.12,<3.13`.
`requirements.in` records direct dependency intent and `requirements.txt` is a
fully pinned lock regenerated with `pip-compile` under Python 3.12.13. The lock
includes pandas for the physical Mk1 workflows.

Create an isolated environment and install from the repository root:

```bash
python3.12 -m venv dcsm-env
dcsm-env/bin/python -m pip install -r requirements.txt
```

A GPU is not required. The tested public workflows use the CPU backend. Torch
remains in the lock because selected numerical backends and tests import it.

## Thread settings

For deterministic resource use and parity with the recorded robustness runs,
set one thread for the common BLAS/OpenMP implementations before launching a
campaign:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Parallel campaign workers are separate processes and are controlled by each
runner's `--workers` argument.

## Historically recorded software and platform

The calibrated and calibration-free provenance records report Python 3.12.13,
NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0, NetworkX 3.6.1, Matplotlib
3.11.1, x86_64, and glibc 2.39
(`docs/reproducibility/execution-provenance.md:71-76,101-139`). Robustness also
records Linux 7.0.0-28, eight workers, 32 logical CPUs, one-thread BLAS
variables, CPU execution, and a visible but unused GPU
(`docs/reproducibility/execution-provenance.md:155-185`).

The physical analyses record Python 3.12.13, NumPy 2.5.1, pandas 3.0.5, and
SciPy 1.18.0. Their host identity is unavailable
(`docs/reproducibility/execution-provenance.md:192-227`). These are historical
records, not a claim that the current verification host was the original host.

## Historical linkage limitation

Historical byte-exact source linkage is unavailable for Stage B,
calibration-free, robustness, and Mk1. The released code is the best recovered
working-tree source and its copy provenance is recorded, but the records do not
prove that those exact bytes were loaded by the historical processes. Seed and
aggregate agreement do not remove this limitation.

Hardware details not present in the historical records—especially the physical
analysis host, memory, CPU model, and BLAS build—remain unknown.

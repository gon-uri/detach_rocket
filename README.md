# Detach-ROCKET

<img align="right" src="logo/detach_logo.png" alt="Logo" width="150"/>

![Python](https://img.shields.io/badge/Python-%E2%89%A53.11-blue)
![License](https://img.shields.io/badge/License-BSD--3--Clause-green)
![Install](https://img.shields.io/badge/install-from%20GitHub-orange)

Official repository for:

- [Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels](https://link.springer.com/article/10.1007/s10618-024-01062-7)
- [Classification of raw MEG/EEG data with detach-rocket ensemble: an improved rocket algorithm for multivariate time series analysis](https://www.arxiv.org/abs/2408.02760)

## Why Detach-ROCKET?

ROCKET models generate thousands of random convolutional features, but most are redundant. Detach-ROCKET automatically prunes them — keeping only the features that matter. The result: **equal or better accuracy, faster inference, and a fraction of the model size.**

| | Full ROCKET | Detach-ROCKET |
|---|---|---|
| Test Accuracy | 80.49% | **82.10%** |
| Features Retained | 100% | **2.36%** |
| Inference Time | 21.96s | **1.01s (22x faster)** |

*FordB dataset (UCR archive) — 10,000 kernels. See the [full example notebook](examples/Detach_ROCKET_example_UCR.ipynb).*

## Overview

Detach-ROCKET applies **Sequential Feature Detachment (SFD)** to ROCKET-family models for time-series classification. It iteratively removes the least important features, selects the optimal model size, and physically rebuilds a smaller, faster transformer. The API follows scikit-learn conventions — just `fit`, `predict`, and `score`.

The library provides four main classes:

- **`DetachRocket`** — End-to-end model: wraps any ROCKET-family transformer (Rocket, MiniRocket, MultiRocket), prunes it with SFD, and rebuilds a smaller transformer for fast inference. [aeon](https://www.aeon-toolkit.org/) ROCKET-family transformers are supported out of the box (aeon is a base dependency). Physical kernel rebuilding is currently implemented for aeon's `Rocket` and the CuPy MiniRocket backend; other transformers use an exact feature-masking fallback (identical predictions, without the inference speedup).

- **`DetachEnsemble`** — Ensemble of independently randomized Detach-MiniRocket models. Designed for multivariate time series, especially high-dimensional data (e.g. MEG/EEG). Provides class probability estimation and channel relevance scores. Supports `backend="pytorch"` (CPU/GPU) and `backend="cuda"` (CuPy).

- **`DetachMatrix`** — Applies SFD to any precomputed feature matrix. Use this when your features come from an external pipeline (tsfresh, catch22, or any other transformer).

- **`PrunedRocketModel`** — Lightweight inference-only model returned by `DetachRocket.detach()`. Contains only the pruned transformer, scaler, and classifier — the minimum needed for deployment.

For a detailed explanation of the methods please refer to the [Detach-ROCKET article](https://link.springer.com/article/10.1007/s10618-024-01062-7) and the [Detach-ROCKET Ensemble article](https://www.arxiv.org/abs/2408.02760).

## Features

- Multiclass classification support
- Built-in PyTorch MiniRocket implementation (CPU and GPU)
- CuPy/CUDA MiniRocket backend for maximum GPU throughput
- Detach-ROCKET Ensemble for high-dimensional multivariate time series
- Channel relevance estimation and label probability for multivariate data
- Physical kernel pruning for faster inference via `model.detach()`
- Works with [aeon](https://www.aeon-toolkit.org/)'s ROCKET-family transformers out of the box, and with any object exposing `transform(X)` (scikit-learn transformers included) through an exact feature-masking fallback

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/gon-uri/detach_rocket
```

The base install pulls in [aeon](https://www.aeon-toolkit.org/), which provides both the ROCKET-family transformers and the UCR/UEA dataset loaders — no extra is needed for either. Requires Python ≥3.11 and NumPy ≥2.

With optional dependencies:

```bash
# DetachEnsemble with PyTorch backend (CPU or CUDA)
pip install "detach_rocket[torch] @ git+https://github.com/gon-uri/detach_rocket"

# DetachEnsemble with CuPy/CUDA backend (requires CUDA GPU)
pip install "detach_rocket[cuda] @ git+https://github.com/gon-uri/detach_rocket"

# Dependencies for running the example notebooks
pip install "detach_rocket[examples] @ git+https://github.com/gon-uri/detach_rocket"

# Everything
pip install "detach_rocket[all] @ git+https://github.com/gon-uri/detach_rocket"
```

Available extras: `torch`, `cuda`, `examples`, `dev`, `all`.

> **Note:** The `cuda` extra installs `cupy-cuda12x`. If your system uses a different CUDA version, see the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

For development:

```bash
git clone https://github.com/gon-uri/detach_rocket.git
cd detach_rocket
pip install -e ".[dev]"
```

## Quick Start — DetachRocket

The model follows the scikit-learn API: `fit`, `predict`, `score`.

aeon ships the UCR/UEA archive loaders, so no extra download helper is needed to get data:

```python
from aeon.datasets import load_classification

X_train, y_train = load_classification("FordB", split="train")
X_test, y_test = load_classification("FordB", split="test")
```

Datasets are downloaded from [timeseriesclassification.com](https://timeseriesclassification.com) on first use and cached locally. `load_classification` returns `(X, y)` per split, with `X` already in the 3D form and `y` an array of string labels.

```python
from detach_rocket import DetachRocket
from aeon.transformations.collection.convolution_based import Rocket

# Instantiate model
rocket = Rocket(n_kernels=10_000)
detach_model = DetachRocket(transformer=rocket, trade_off=0.1)

# Train (validation set required when set_percentage=None)
detach_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# Evaluate
y_pred = detach_model.predict(X_test)
test_acc = detach_model.score(X_test, y_test)
full_model_acc = detach_model.score_full(X_test, y_test)  # Unpruned baseline
summary = detach_model.get_summary()

# Export lightweight model for deployment
pruned_model = detach_model.detach()
y_pred = pruned_model.predict(X_new)
```

If you prefer a fixed pruning level, pass `set_percentage` and fit without a validation set:

```python
detach_model = DetachRocket(transformer=rocket, set_percentage=50)
detach_model.fit(X_train, y_train)
```

**Input shapes:**
- Univariate: `(n_instances, 1, n_timepoints)`. Plain 2D `(n_instances, n_timepoints)` also works — aeon's collection transformers treat it as a single-channel collection and reshape it internally, giving identical results
- Multivariate: `(n_instances, n_channels, n_timepoints)`
- `DetachEnsemble` also accepts 2D univariate input: it reshapes it to a single channel and warns, since the ensemble is designed primarily for multivariate data (with one channel, member diversity comes only from bias sampling, but the ensemble still provides label probabilities)

## Quick Start — DetachEnsemble

Ensemble of Detach-MiniRocket models for multivariate time series. Provides class probability and channel relevance estimation.

```python
from detach_rocket import DetachEnsemble

# Create ensemble (use backend="cuda" for CuPy/CUDA acceleration)
ensemble = DetachEnsemble(
    num_models=20,
    num_kernels=5_000,
    backend="pytorch",
)

# Train and predict
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# Class probabilities (soft voting weighted by training accuracy)
probs = ensemble.predict_proba(X_test, proba="soft")

# Channel relevance estimation
channel_relevance = ensemble.estimate_channel_relevance()
```

<p align="center">
  <img src="logo/channel_relevance.png" alt="Channel relevance estimation example" width="600"/>
</p>

## Quick Start — DetachMatrix

Apply SFD to any precomputed feature matrix — works with tsfresh, catch22, or any feature extraction pipeline.

```python
from detach_rocket import DetachMatrix

# X_features: (n_instances, n_features) — any feature matrix
model = DetachMatrix(trade_off=0.1)
model.fit(X_features, y, X_val=X_features_val, y_val=y_val)

y_pred = model.predict(X_features_test)
test_acc = model.score(X_features_test, y_test)
```

## Notebook Examples

Detailed usage examples are available in the [examples folder](examples/):

- **[Detach-ROCKET (UCR)](examples/Detach_ROCKET_example_UCR.ipynb)** — Univariate classification with Rocket on the FordB dataset. Includes SFD curve visualization, accuracy and inference time comparison.
- **[Detach-ROCKET Ensemble (UEA)](examples/Detach_Ensemble_example_UEA.ipynb)** — Multivariate classification on an EEG dataset (SelfRegulationSCP1). Demonstrates channel relevance estimation.
- **[SFD with tsfresh](examples/SFD_example_tsfresh.ipynb)** — Feature selection on tsfresh features using `DetachMatrix`, showing that SFD works with any feature pipeline.

## Core Modules

- `detach_rocket/detach_classes.py`: Main model classes (`DetachRocket`, `DetachMatrix`, `DetachEnsemble`, `PrunedRocketModel`).
- `detach_rocket/sfd.py`: Sequential Feature Detachment core logic (`feature_detachment`).
- `detach_rocket/model_selection.py`: Model-size selection and final retraining utilities.
- `detach_rocket/pruner.py`: Transformer pruning — `AeonRocketTransformerPruner` physically rebuilds aeon's `Rocket`, `CudaMiniRocketTransformerPruner` does the same for the CuPy MiniRocket backend, and `GenericTransformerPruner` provides the exact feature-masking fallback for every other transformer.
- `detach_rocket/pytorch_minirocket.py`: PyTorch MiniRocket implementation (CPU/GPU).
- `detach_rocket/cuda_minirocket.py`: CuPy/CUDA MiniRocket implementation.

## Migrating from 0.1.x

Version 0.2.0 replaces [sktime](https://www.sktime.net/) with [aeon](https://www.aeon-toolkit.org/) as the base transformer and dataset dependency. The model classes, their parameters, and their behavior are unchanged — what changes is how you build the transformer you pass in and how you load data:

| 0.1.x | 0.2.0 |
|---|---|
| `from sktime.transformations.panel.rocket import Rocket` | `from aeon.transformations.collection.convolution_based import Rocket` |
| `Rocket(num_kernels=10_000)` | `Rocket(n_kernels=10_000)` — aeon renamed the argument |
| `from detach_rocket.utils_datasets import fetch_ucr_dataset` (or `fetch_uea_dataset`) | `from aeon.datasets import load_classification` — one loader for both archives |
| `fetch_ucr_dataset("FordB")` returned both splits in one `Bunch` | `load_classification("FordB", split="train")` returns `(X, y)` for a single split, so call it once per split |
| Univariate `X` came back 2D `(n_instances, n_timepoints)` | Univariate `X` comes back 3D `(n_instances, 1, n_timepoints)` — both are accepted as model input |
| `pip install "detach_rocket[datasets]"` | The `datasets` extra is gone — aeon's loaders come with the base install |
| `PrunedRocketTransformer`, `RocketTransformerPruner` | `PrunedAeonRocketTransformer`, `AeonRocketTransformerPruner` |
| Python ≥3.10, NumPy 1 or 2 | Python ≥3.11, NumPy ≥2 (both required by aeon) |

Transformers you supply yourself keep working regardless of their origin: sktime transformers, scikit-learn transformers, and anything else exposing `transform(X)` still fit and predict correctly through the feature-masking fallback. They simply do not get a physically rebuilt transformer, so `detach()` gives identical predictions without the inference speedup.

> **Intel Mac / x86_64 users needing the `[torch]` extra:** the newest macOS x86_64 torch wheel requires NumPy 1, which conflicts with the NumPy ≥2 floor aeon imposes. Stay on the last sktime-based release, which is tagged and supports NumPy 1 on that platform:
>
> ```bash
> pip install "detach_rocket[torch] @ git+https://github.com/gon-uri/detach_rocket@v0.1.0"
> ```

## Migrating from 0.0.x

Version 0.1.0 is a rewrite with a cleaner, scikit-learn-style API. The main breaking changes:

| 0.0.x | 0.1.0+ |
|---|---|
| `DetachRocket(model_type="rocket", num_kernels=10000)` | `DetachRocket(transformer=Rocket(n_kernels=10_000))` — pass any transformer instance |
| `fit(X, y)` with a silent internal train/val split | Explicit `fit(X, y, X_val=..., y_val=...)`, or `set_percentage=...` to skip validation |
| `score(X, y)` returned a `(pruned_acc, full_acc)` tuple | `score(X, y)` returns a float; the unpruned baseline is `score_full(X, y)` |
| Private attributes (`_feature_matrix`, `_classifier`, ...) | scikit-learn style public attributes (`feature_matrix_`, `classifier_`, ...) |
| `multilabel_type` | `multiclass_type` (renamed; default remains `"max"`, as in the paper) |
| `utils.py` (`feature_detachment`, `select_optimal_model` with built-in plotting) | `sfd.py` (`feature_detachment`) and `model_selection.py` (`select_optimal_pruning`, plotting moved to the notebooks) |

New in 0.1.0: `DetachEnsemble` with PyTorch and CuPy/CUDA MiniRocket backends, physical transformer pruning, `detach()` for lightweight deployment models, and channel relevance estimation.

Two behavior fixes worth knowing: 0.0.x retrained the final classifier on the feature set of the step *before* the selected one (an off-by-one in the mask reconstruction) and fitted the scaler before the internal train/val split; 0.1.0 retrains exactly the selected feature set and keeps validation data out of the scaler fit.

## Troubleshooting

**`llvmlite` fails to build during `pip install`.** `numba` (pulled in by aeon, which compiles the ROCKET kernels with it) stopped publishing macOS x86_64 wheels partway through the version range aeon accepts. On Intel Macs — and on x86_64/Rosetta condas running on Apple Silicon — pip therefore picks a `numba`/`llvmlite` pair that exists only as a source archive and tries to compile LLVM from scratch. Tell pip to prefer versions that ship a wheel over newer source-only releases:

```bash
pip install --prefer-binary git+https://github.com/gon-uri/detach_rocket
```

This resolves to the newest `numba` that still has an x86_64 macOS wheel, which is inside aeon's supported range, so nothing else is affected. Platforms with wheels for the newest `numba` (Linux, Windows, Apple Silicon) resolve identically with or without the flag.

Alternatively, install `numba` from conda first and then install the package:

```bash
conda install "numba>=0.58,<0.64"
pip install git+https://github.com/gon-uri/detach_rocket
```

**`[torch]` extra on Intel Mac / x86_64 Rosetta conda.** The newest macOS x86_64 torch wheel requires `numpy<2`, which cannot coexist with the NumPy ≥2 floor that aeon imposes on this version. Install the last sktime-based release instead, which supports NumPy 1 on that platform:

```bash
pip install "detach_rocket[torch] @ git+https://github.com/gon-uri/detach_rocket@v0.1.0"
```

**`Intel MKL WARNING` about SSE4.2/AVX** (conda installs MKL by default):

```bash
conda install nomkl
```

## License

This project is licensed under the BSD-3-Clause License.

## Citation

If you find these methods useful in your research, please cite the following articles:

*APA*
```
Uribarri, G., Barone, F., Ansuini, A., & Fransén, E. (2024). Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels. Data Mining and Knowledge Discovery, 1-26.

Solana, A., Fransén, E., & Uribarri, G. (2024). Classification of raw MEG/EEG data with detach-rocket ensemble: an improved rocket algorithm for multivariate time series analysis. arXiv preprint arXiv:2408.02760.
```

*BIBTEX*
```bibtex
@article{uribarri2024detach,
  title={Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels},
  author={Uribarri, Gonzalo and Barone, Federico and Ansuini, Alessio and Frans{\'e}n, Erik},
  journal={Data Mining and Knowledge Discovery},
  pages={1--26},
  year={2024},
  publisher={Springer}
}

@article{solana2024classification,
  title={Classification of raw MEG/EEG data with detach-rocket ensemble: an improved rocket algorithm for multivariate time series analysis},
  author={Solana, Adri{\`a} and Frans{\'e}n, Erik and Uribarri, Gonzalo},
  journal={arXiv preprint arXiv:2408.02760},
  year={2024}
}
```

<img src="logo/detach_logo.png" align="centered"
     alt="repo logo" width="80" height="80">

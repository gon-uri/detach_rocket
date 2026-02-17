# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Detach-ROCKET is a Python library for time series classification using pruned random convolutional kernel models. It implements Sequential Feature Detachment (SFD) to select optimal feature subsets from ROCKET-family models (ROCKET, MiniRocket, MultiRocket). Based on two papers: the original Detach-ROCKET paper and the Detach-ROCKET Ensemble paper for multivariate time series (MEG/EEG).

## Installation

```bash
pip install numpy scikit-learn pyts torch matplotlib sktime==0.30.0
pip install git+https://github.com/gon-uri/detach_rocket --quiet
```

For local development:
```bash
pip install -e .
```

## Architecture

The package lives entirely in `detach_rocket/` with three modules:

- **`detach_classes.py`** — Main model classes following scikit-learn's fit/predict API:
  - `DetachRocket` — End-to-end model: transforms time series via a ROCKET variant, runs SFD to prune features, selects optimal size, retrains a RidgeClassifier. Supports `model_type` in `{"rocket", "minirocket", "multirocket", "pytorch_minirocket"}`.
  - `DetachMatrix` — Same SFD pipeline but operates on pre-computed feature matrices `(n_instances, n_features)` instead of raw time series. Useful when you already have features from an external source.
  - `DetachEnsemble` — Ensemble of `DetachRocket` models (currently only `pytorch_minirocket`). Provides `predict_proba()` (soft/hard voting) and `estimate_channel_relevance()` for multivariate time series.
  - `PytorchMiniRocketMultivariate` — PyTorch reimplementation of MiniRocket with multivariate channel combination support. Used internally by `DetachRocket` when `model_type="pytorch_minirocket"`. Includes `get_kernel_features()` for introspection of kernel parameters (biases, channels, weights, dilations, paddings).

- **`utils.py`** — Core SFD algorithm functions:
  - `feature_detachment()` — Iterative feature pruning loop: drops `drop_percentage` of least-important features each step, retrains RidgeClassifier, records accuracy curves.
  - `select_optimal_model()` — Picks optimal pruning step by maximizing `acc_size_tradeoff_coef * compression + smoothed_relative_accuracy`.
  - `retrain_optimal_model()` — Retrains a RidgeClassifier (with optional alpha CV) on the selected feature subset.

- **`utils_datasets.py`** — Dataset fetching utilities for UCR (univariate) and UEA (multivariate) time series archives. Modified from pyts to use updated download URLs.

## Key Design Patterns

- All models use a two-phase fit: (1) SFD curve computation using a validation split or fixed percentage, (2) optimal model selection and retraining on the full training set.
- `fit()` accepts either a `val_set`/`val_set_y` pair, or auto-splits using `val_ratio` and `stratify`. When `fixed_percentage` is set, `X_test`/`y_test` are required instead (used only for SFD curve plotting, not training).
- Feature importance for multi-class problems is derived from RidgeClassifier coefficients using `multilabel_type` (`"norm"`, `"max"`, or `"avg"` corresponding to L2, Linf, L1 norms across class weights).
- `score()` returns a tuple: `(detach_model_accuracy, full_model_accuracy)`.

## Input Shapes

- Univariate: `X_train` shape `(n_instances, n_timepoints)`
- Multivariate: `X_train` shape `(n_instances, n_variables, n_timepoints)`

## Dependencies

numpy, scikit-learn, pyts, torch, matplotlib, sktime (pinned to 0.30.0), scipy

# Test Plan

Status of test coverage for the 0.1.0 rework. Implemented tests live in this
directory; this file tracks what is covered and what remains open.

## Covered

- **`feature_detachment` (sfd.py)** — output shapes, monotonically decreasing
  schedule, irrelevant features pruned first, no-validation path returns `None`
  test scores (`test_feature_detachment.py`).
- **`AeonRocketTransformerPruner` (pruner.py)** — specialized pruner is
  dispatched (never a silent fallback), retained kernel count matches the mask,
  output shape, exact numerical equivalence with the masked full transform, and
  the copied kernels survive `fit`/`fit_transform`/`reset`
  (`test_detach_rocket.py`).
- **Generic pruner fallback** — unsupported transformers get a masking wrapper
  instead of an error (`test_detach_rocket.py`).
- **`DetachRocket`** — fit with trade-off selection and with fixed percentage,
  `get_summary`, model-level pruned-path equivalence, and `detach()` parity
  with the parent model (`test_detach_rocket.py`).
- **`DetachMatrix`** — auto-split fit, `get_summary`, and 3-class fits with
  every `multiclass_type` (`test_detach_matrix.py`).
- **`DetachEnsemble`** (skipped automatically when PyTorch is not installed) —
  fit, `predict`, `predict_proba` (soft and hard), `score`, channel-relevance
  estimation, and `get_kernel_features` bias ordering and univariate channels
  (`test_detach_ensemble.py`).

## Open

- `select_optimal_pruning`: `trade_off=0` should pick the best-accuracy step;
  a very large `trade_off` should pick a heavily pruned step.
- `retrain_optimal_model`: fixed alpha vs. re-computed alpha by CV.
- Input-validation errors of `DetachRocket` / `DetachMatrix` (missing labels,
  missing validation set when `set_percentage` is not used).
- **CUDA backend** (`cuda_minirocket.py`, `CudaMiniRocketTransformerPruner`):
  requires a CUDA GPU, so it is not covered by CI. Run
  `python tests/manual_cuda_equivalence.py` once on a GPU machine after
  changes to the CUDA code or the pruner.

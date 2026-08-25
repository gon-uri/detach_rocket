# aeon port — working plan (branch `feat/aeon-port`)

Goal: migrate detach_rocket's transformer and dataset dependencies from sktime to
aeon (v1.5+), gradually and without ever breaking the library. This file is the
task list for the port; each task is self-contained enough to hand to a separate
agent. Delete this file before the final merge to main.

## Ground rules (apply to every task)

1. Work only on `feat/aeon-port`. One task = one or a few clearly-messaged commits.
2. Before a task is "done": `ruff check .`, `ruff format --check .`, and the full
   `pytest` suite must pass. Never weaken or delete the existing equivalence tests.
3. The sacred invariant: for any pruned transformer, `pruned.transform(X)` must be
   numerically identical (atol 1e-6) to `full.transform(X)[:, feature_mask]`, and
   `detach()` models must reproduce parent predictions exactly. Every stage that
   touches the pruner must re-verify this.
4. aeon facts the tasks rely on (verified 2026-08): aeon 1.5.0 requires
   Python >=3.11 and numpy>=2.0,<2.5; `aeon...Rocket` stores fitted kernels in the
   same 7-tuple as sktime's (`weights, lengths, biases, dilations, paddings,
   num_channel_indices, channel_indices`); parameter is `n_kernels` (not
   `num_kernels`); transform takes 3D numpy and returns 2D float32 numpy.
5. Environment note for this machine: the resident conda is x86_64-under-Rosetta.
   aeon core installs there (numpy 2.x + numba 0.62 have x86_64 mac wheels), but
   aeon + torch cannot coexist on macOS x86_64 (torch 2.2.2 needs numpy<2).
   Tasks needing aeon+torch together must use a native arm64 environment
   (e.g. miniforge arm64) or run on Linux/Colab.

## Stage 0 — Groundwork (no behavior change)

- **T0.1 — aeon dev environment.** Create a fresh environment with Python 3.11+,
  `aeon>=1.5`, plus the repo's dev deps. Smoke: `Rocket(n_kernels=100)` fit/transform
  on random 3D data; confirm the `kernels` 7-tuple attribute and numpy output.
  Record the environment name and any platform quirks in this file.
- **T0.2 — optional `aeon` extra.** Add `aeon = ["aeon>=1.5,<2"]` to
  `[project.optional-dependencies]`. Do NOT touch base deps or requires-python yet
  (dual-support phase; on Python 3.10 the extra simply won't resolve — acceptable).

## Stage 1 — Dual support (aeon works alongside sktime)

- **T1.1 — fallback-path tests.** New `tests/test_aeon_transformers.py`, guarded by
  `pytest.importorskip("aeon")`: fit `DetachRocket` with aeon `Rocket`, `MiniRocket`,
  `MultiRocket` (small sizes) via the existing generic fallback; assert the
  model-level equivalence invariants, predict shapes, and `detach()` parity.
- **T1.2 — specialized aeon Rocket pruner.** In `pruner.py`, add
  `PrunedAeonRocketTransformer(aeon Rocket subclass)` and `AeonRocketTransformerPruner`,
  selected in `get_transformer_pruner` via a lazy `try/except ImportError` import of
  `aeon.transformations.collection.convolution_based` (LESSON from the
  `transformer_models` bug: add a test asserting that, when aeon is installed, an
  aeon Rocket gets the specialized pruner, so a stale import can never silently
  fall back again). Port notes: kernel-copy loop is identical to the sktime pruner
  (same 7-tuple); constructor takes `n_kernels`; the subclass must satisfy aeon's
  `BaseCollectionTransformer` machinery — investigate what its `transform` checks
  (fitted flag, `fit_min_length_`, tags) and set them explicitly, mirroring what
  `PrunedRocketTransformer` does for sktime. Acceptance: pruned≡masked equivalence
  univariate + multivariate, predict-path and `detach()` parity, `get_summary()`
  reports a correct `retained_kernel_count`.
- **T1.3 — docs touch.** README: one sentence that both sktime and aeon
  ROCKET-family transformers are supported, aeon Rocket with physical pruning.

## Stage 2 — Dataset utilities migration

- **T2.1 — aeon-backed loaders.** Reimplement `fetch_ucr_dataset` /
  `fetch_uea_dataset` as thin wrappers over `aeon.datasets.load_classification`
  keeping the current signatures and Bunch-style returns (backward compatible).
  Write a one-off comparison script (scratch, not committed) proving shapes,
  dtypes, and label sets match the old pyts-based loaders for FordB and
  SelfRegulationSCP1 before deleting the old code.
- **T2.2 — dependency cleanup.** `[datasets]` extra becomes `["aeon>=1.5,<2"]`;
  remove `pyts` (and `scipy` from the extra — it stays transitively via
  scikit-learn). Update the friendly ImportError hint in `utils_datasets.py`.

## Stage 3 — Switchover (aeon becomes the default)

Gate: requires Gonzalo's sign-off on the decision points below.

- **T3.1 — base dependency swap.** Replace `sktime` with `aeon` in base deps.
  Decide (decision point A) whether sktime support stays as an optional extra
  (keeping `RocketTransformerPruner` behind a lazy import) or is dropped.
- **T3.2 — pins and floors.** `requires-python >=3.11` (drop the 3.10 classifier);
  align numpy with aeon (`numpy>=2.0,<2.5`); REMOVE the darwin-x86_64 `numpy<2`
  marker from the torch extra (it becomes unsatisfiable) and instead document in
  README Troubleshooting that `[torch]` on Intel/Rosetta macs requires
  detach_rocket 0.1.x (decision point B: whether to keep a `0.1.x` maintenance
  note/branch for those users). Re-check the numba floor against aeon's own
  `numba>=0.55,<0.64` cap and rewrite the llvmlite troubleshooting if the
  resolver behavior changed.
- **T3.3 — notebooks.** Update imports (`aeon` Rocket, `n_kernels=`), dataset
  cells, and install cells; re-execute all three end-to-end in a fresh aeon
  environment; commit fresh outputs; sync the README results table with the new
  committed FordB run.
- **T3.4 — README overhaul.** Install section, quick starts, Core Modules, and a
  "0.1.x → 0.2.0" migration subsection (sktime→aeon, `num_kernels`→`n_kernels`
  where users construct transformers, Python/numpy floors, Intel-mac torch note).
- **T3.5 — CI.** Test job Python version to 3.11 (or a small matrix); ensure aeon
  installs in the test job; keep lint job as is.

## Stage 4 — Full verification (before merging to main)

- **T4.1 — fresh-environment user test.** Clean env, install from the branch via
  `pip install git+...@feat/aeon-port`, `pip check`, import smoke, base-only
  import simulation (blocked-imports trick) confirming dependency boundaries,
  full pytest from a fresh clone.
- **T4.2 — numerical regression vs 0.1.0.** Script comparing sktime-based (0.1.0)
  and aeon-based DetachRocket on the same FordB subset: pruner equivalences exact
  on both; end accuracies within normal run-to-run variance (kernel RNG streams
  differ, so no bit-identical expectation across libraries).
- **T4.3 — GPU re-verification.** Re-run the Colab CUDA notebook against this
  branch (CUDA backend is untouched, but the numpy>=2 environment is new for it).
- **T4.4 — release mechanics.** Version 0.2.0, update the migration section, final
  ruff/pytest gate, then merge per the same procedure as 0.1.0.

## Decision points for Gonzalo

- **A.** Keep sktime as an optional supported backend after the switchover, or
  drop it entirely in 0.2.0?
- **B.** Intel/Rosetta-mac + `[torch]` users after 0.2.0: point them at 0.1.x
  (tag or maintenance branch), or is a README note enough?
- **C.** Keep the `fetch_ucr_dataset`/`fetch_uea_dataset` wrapper API long-term,
  or deprecate in favor of documenting `aeon.datasets.load_classification`
  directly?

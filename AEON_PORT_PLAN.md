# aeon port — working plan (branch `feat/aeon-port`)

Goal: migrate detach_rocket from sktime to aeon (transformers AND datasets),
gradually, without ever breaking the library. This file is the authoritative
task list; individual stages are executed by dispatched agents. Delete this
file before the final merge to main.

**Resolved decisions (Gonzalo, 2026-08):**
- **A.** sktime support is dropped entirely in this release (no optional extra).
  User-supplied sktime transformers still *work* through the generic masking
  fallback (anything with `transform(X)`), they just don't get physical pruning.
- **B.** macOS x86_64 (Intel / Rosetta) users needing the `[torch]` extra are
  referred to the tagged release **v0.1.0** (tag + GitHub release exist) via a
  README Troubleshooting note. No maintenance branch.
- **C.** The `fetch_ucr_dataset` / `fetch_uea_dataset` wrappers and the whole
  `utils_datasets.py` are removed; docs and notebooks use
  `aeon.datasets.load_classification` directly. The `[datasets]` extra and the
  `pyts` dependency disappear (aeon is a base dep and includes the loaders).

---

## Context every agent must know

- Repo: `/Users/uribarri/Documents/detach_rocket`. Work ONLY on branch
  `feat/aeon-port` (verify with `git status -sb` before touching anything).
  Commit in small logical units with descriptive messages; end each commit
  message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
  Do NOT push; the coordinator pushes after review.
- Environment: conda is **Miniforge at `~/miniforge3`** (native arm64). The
  working env is **`detach_aeon`** (Python 3.12.14, aeon 1.5.0, sktime 1.1.0,
  torch 2.13.0, numpy 2.3.5, pytest, ruff — all verified working together).
  Run everything as:
  `~/miniforge3/bin/conda run --no-capture-output -n detach_aeon <cmd>`
  Tests import the local package because `python -m pytest` from the repo root
  puts the cwd on `sys.path` — always run pytest that way, from the repo root.
- Definition of done for every task: from the repo root, all three pass:
  1. `python -m pytest` (full suite, no failures, no new skips except
     documented importorskips)
  2. `ruff check .`
  3. `ruff format --check .`
  Run `ruff format .` on files you create/edit before the check.
- **The sacred invariant** (never weaken, always re-verify when touching the
  pruner): for any pruned transformer, `pruned.transform(X)` must equal
  `full.transform(X)[:, feature_mask]` numerically (atol 1e-6), and
  `model.detach()` must reproduce the parent model's predictions exactly.
  Guarded today by `tests/test_detach_rocket.py::test_pruned_transformer_consistency`
  and `::test_model_level_pruned_path_equivalence`.
- Never touch `detach_rocket/pytorch_minirocket.py`, `detach_rocket/cuda_minirocket.py`,
  or `tests/manual_cuda_equivalence.py` — the ensemble backends are independent
  of the sktime/aeon choice.
- Style: match the existing code (numpydoc docstrings, 120-col ruff format).
  Comments only for constraints the code can't express.

## Verified aeon facts (checked live against aeon 1.5.0, 2026-08)

- Import: `from aeon.transformations.collection.convolution_based import Rocket, MiniRocket, MultiRocket`
- `Rocket(n_kernels=..., normalise=..., n_jobs=..., random_state=...)` — note
  **`n_kernels`**, not sktime's `num_kernels`.
- After `fit`, `rocket.kernels` is the SAME 7-tuple as sktime's:
  `(weights, lengths, biases, dilations, paddings, num_channel_indices,
  channel_indices)`; `lengths` dtype int32. The existing sktime kernel-copy
  loop in `RocketTransformerPruner.prune_transformer` therefore ports as-is.
- `transform` takes 3D numpy `(n_cases, n_channels, n_timepoints)` and returns
  2D numpy float32 `(n_cases, 2 * n_kernels)` — PPV and max per kernel, same
  column pairing as sktime. No DataFrame conversion involved.
- aeon transformers inherit `BaseCollectionTransformer`; its `transform`
  performs fitted/metadata checks. The pruned subclass must satisfy them —
  investigate the base class source in the env
  (`~/miniforge3/envs/detach_aeon/lib/python3.12/site-packages/aeon/...`)
  and set whatever fitted-state attributes are required (at minimum the fitted
  flag and `fit_min_length_`), mirroring how `PrunedRocketTransformer` handles
  sktime's `_tags`/`_is_fitted` today.
- Datasets: `from aeon.datasets import load_classification`;
  `X, y = load_classification("FordB", split="train")` (and `split="test"`),
  downloads from timeseriesclassification.com with local caching;
  multivariate sets (e.g. "SelfRegulationSCP1") return 3D numpy.
- aeon 1.5 requires Python >=3.11,<3.15 and numpy>=2.0,<2.5; it depends on
  numba (>=0.55,<0.64), pandas, scikit-learn (>=1.6,<1.9), scipy itself.

---

## Stage 0 — Groundwork  ✅ DONE (coordinator)

`detach_aeon` env created and smoke-tested; `v0.1.0` tag + GitHub release
published on the last sktime-based commit (`ddb2f6e`).

## Stage 1 — AGENT TASK 1: aeon support alongside sktime

Everything in this stage is additive; sktime remains the default. The library
must be fully green with and without aeon installed.

1. **pyproject:** add an optional extra `aeon = ["aeon>=1.5,<2"]` under
   `[project.optional-dependencies]`, and add it to the `all` extra's list.
   Touch nothing else in pyproject (base deps, python floor stay as they are
   in this stage).
2. **Specialized aeon pruner** in `detach_rocket/pruner.py`:
   - `PrunedAeonRocketTransformer`: subclass of aeon's `Rocket` that stores a
     `features_mask` and whose `transform` returns only retained columns —
     the aeon analogue of the existing `PrunedRocketTransformer`. It is built
     unfitted, so set the fitted-state attributes aeon's `transform` checks
     (see "Verified aeon facts"). Define it so the module still imports when
     aeon is absent (e.g. build the class lazily inside the pruner, or guard
     the definition — choose the cleanest pattern that keeps `import
     detach_rocket.pruner` working without aeon).
   - `AeonRocketTransformerPruner(TransformerPruner)`: near-copy of
     `RocketTransformerPruner.prune_transformer` (the kernels tuple is
     identical); constructor call uses `n_kernels=`; validate mask size
     `2 * n_kernels`; unfitted transformer (no `kernels` attribute) raises
     `ValueError`.
   - `get_transformer_pruner`: add a lazy `try/except ImportError` branch
     importing aeon's Rocket and returning the new pruner on isinstance match,
     BEFORE the generic fallback (same pattern as the CUDA branch).
3. **Anti-silent-fallback regression test** (lesson from the old
   `transformer_models` bug, where a stale import silently disabled a pruner
   for months): a test, guarded by `pytest.importorskip("aeon")`, asserting
   `get_transformer_pruner(fitted_aeon_rocket)` returns an
   `AeonRocketTransformerPruner` instance — so a broken import can never again
   degrade silently to the generic fallback.
4. **Tests** in a new `tests/test_aeon_transformers.py` (all guarded by
   `pytest.importorskip("aeon")`), small sizes (~100-500 kernels, ~40-60
   series of length 40-60), fast (<30 s total):
   - DetachRocket + aeon `Rocket` (univariate AND multivariate,
     `set_percentage` path): the sacred invariant — `_prepare_X(X)` equals
     `scaler_.transform(full)[:, feature_mask_]`, `detach()` predict parity,
     pruned transformer is a `PrunedAeonRocketTransformer` (not generic).
   - Same invariant after `fit_set_optimal(...)` re-selection.
   - DetachRocket + aeon `MiniRocket` and `MultiRocket` via the generic
     fallback: fit, predict shape, invariant holds, pruned transformer is a
     `GenericPrunedTransformer`.
   - `get_summary()["retained_kernel_count"]` is a correct positive int for
     the aeon Rocket path (mind sktime-vs-aeon attribute naming:
     `num_kernels` vs `n_kernels` — make the summary work for both, without
     breaking the existing sktime test).
5. **README (one small touch only):** in the DetachRocket bullet of Overview,
   note that aeon ROCKET-family transformers are also supported, with physical
   pruning for aeon's `Rocket`. Leave the rest of the README alone (Stage 3
   rewrites it).
6. Run the full gate. Also run the suite once WITHOUT aeon importable to
   prove the skips work: `python -m pytest` with
   `PYTHONPATH= python -c` trickery is unreliable — instead temporarily
   `pip uninstall -y aeon` in a THROWAWAY copy is overkill; acceptable proxy:
   assert in a quick manual check that `tests/test_aeon_transformers.py`
   contains the importorskip guard at module level and that
   `detach_rocket.pruner` imports cleanly in a subprocess where
   `sys.modules["aeon"]` is blocked (same blocked-import technique as
   `tests/test_warnings.py` uses conceptually).

## Stage 2 — AGENT TASK 2: the switchover (code, dependencies, tests)

Prereq: Stage 1 merged into `feat/aeon-port` and green.

1. **pyproject:**
   - Base deps become: `numpy>=2.0,<2.5`, `scikit-learn>=1.7.2,<1.9`,
     `aeon>=1.5,<2`. REMOVE `sktime` and the explicit `numba` pin (aeon
     brings numba itself). Keep the comments style, updating their content.
   - `requires-python = ">=3.11"`; drop the 3.10 classifier.
   - REMOVE the `aeon` extra (now a base dep) and the `datasets` extra
     entirely; update the `all` extra accordingly.
   - `[torch]` extra: remove the darwin-x86_64 `numpy<2` marker line and its
     comment (unsatisfiable with numpy>=2 base; Intel-mac users are pointed
     to v0.1.0 in Stage 3's README work). Keep `torch>=2.2.2`… actually bump
     to `torch>=2.3` (first numpy-2-tolerant torch) so pip fails loudly on
     platforms with only older torch instead of installing a broken pair.
2. **pruner.py:** remove the sktime import, `PrunedRocketTransformer`, and
   `RocketTransformerPruner`; aeon's branch becomes the primary specialized
   pruner (import aeon's Rocket at module top now — aeon is a base dep — and
   drop the try/except around it; KEEP the try/except pattern for the CUDA
   branch). Update the module docstring.
3. **utils_datasets.py:** delete the file. Grep the whole repo for
   `utils_datasets`, `fetch_ucr_dataset`, `fetch_uea_dataset`, `pyts` and
   clean every reference (code, tests, pyproject comments; notebooks/README
   are Stage 3/4's job — leave a note in the progress log for them).
4. **Tests migration:** `tests/test_detach_rocket.py` currently builds sktime
   `Rocket(num_kernels=512)` — switch fixtures to aeon
   `Rocket(n_kernels=512, random_state=...)`; merge what remains of
   `tests/test_aeon_transformers.py` sensibly (avoid duplicate coverage; the
   anti-silent-fallback test and the invariant tests stay). The
   `test_invalid_pruning` mock-transformer test and everything in
   `test_detach_matrix.py`, `test_feature_detachment.py`, `test_warnings.py`,
   `test_detach_ensemble.py` should pass unchanged — if one doesn't,
   understand why before touching it.
5. **detach_classes.py:** docstrings mention sktime in several places
   (`DetachRocket` transformer param, fit shape notes, examples) — update to
   aeon (`n_kernels`, aeon import path in the Examples sections). The code
   itself should need no changes (`_to_numpy` stays — harmless with numpy
   input, still needed for arbitrary user transformers).
6. Full gate, plus: `python -c "import detach_rocket"` in a subprocess with
   sktime blocked (blocked-import technique) proving sktime is truly gone
   from the import graph.

## Stage 3 — AGENT TASK 3: documentation and CI

Prereq: Stage 2 green.

1. **README.md** full pass:
   - Install section: base install now includes transformers + datasets
     (aeon); extras are `[torch]`, `[cuda]`, `[examples]`, `[dev]`, `[all]`.
   - Quick Starts: aeon imports (`from aeon.transformations.collection.convolution_based import Rocket`),
     `Rocket(n_kernels=10_000)`; dataset examples use
     `from aeon.datasets import load_classification` /
     `X_train, y_train = load_classification("FordB", split="train")`.
   - Features/Overview wording: "compatible with aeon ROCKET-family
     transformers (and any object with `transform(X)` via the masking
     fallback)"; physical pruning = aeon Rocket + CUDA MiniRocket.
   - Core Modules list: remove `utils_datasets.py`, adjust pruner line.
   - Requirements badge: Python ≥3.11.
   - **Migration section:** add a "0.1.x → 0.2.0" subsection ABOVE the
     existing 0.0.x one: sktime→aeon (`num_kernels`→`n_kernels` when
     constructing transformers), Python ≥3.11 and NumPy ≥2 floors,
     `fetch_ucr_dataset`/`fetch_uea_dataset` removed in favor of
     `aeon.datasets.load_classification`, and the Intel-mac/[torch] pointer:
     `pip install "detach_rocket[torch] @ git+https://github.com/gon-uri/detach_rocket@v0.1.0"`.
   - Troubleshooting: re-check the numba/llvmlite section wording (numba now
     arrives via aeon, whose `<0.64` cap changes resolver behavior — keep the
     `--prefer-binary` tip, verified still valid, and the conda-numba
     fallback); replace the old Intel-mac numpy<2 paragraph with the v0.1.0
     pointer.
2. **CI (`.github/workflows/ci.yml`):** test job Python to `"3.12"` (lint can
   stay or match); everything else unchanged. Optionally bump `checkout`/
   `setup-python` action majors (they warn about Node 20 deprecation).
3. **pyproject classifiers/keywords sanity** after Stage 2's edits.
4. Full gate (docs changes shouldn't move tests, but run it anyway).

## Stage 4 — AGENT TASK 4: notebooks

Prereq: Stage 3 done (README's new numbers come from this stage's runs).

1. Update all three notebooks in `examples/`:
   - Install cells: `!pip install "detach_rocket[torch] @ git+https://github.com/gon-uri/detach_rocket.git" matplotlib`
     style (drop `[datasets]`; tsfresh notebook keeps its extra `tsfresh`).
   - Data loading: `aeon.datasets.load_classification` (FordB;
     SelfRegulationSCP1; PhalangesOutlinesCorrect), adapting variable names
     minimally. Mind return conventions: `load_classification` returns
     `(X, y)` per split; univariate X is 3D `(n, 1, length)` — the UCR/tsfresh
     notebooks currently get 2D from the old loaders, so reshape or pass 3D
     (DetachRocket handles both for transformer input, but check the tsfresh
     cell which feeds TSFreshFeatureExtractor).
   - UCR notebook: sktime Rocket import → aeon; `num_kernels` → `n_kernels`.
   - Re-execute all three headless in `detach_aeon` env, skipping only the
     pip-install cells (there is a runner pattern in the repo history:
     a `SkipPipExecutePreprocessor` — reimplement briefly in scratch), write
     fresh outputs back in place, verify zero error outputs and that plots
     rendered, and report the headline numbers (FordB full vs detach accuracy,
     retained %, timing; tsfresh accuracies; UEA ensemble accuracy).
2. Sync the README results table with the new committed FordB run.
3. Full gate + a JSON sanity check that each notebook parses and cell counts
   are as expected.

## Stage 5 — Final verification (coordinator + Gonzalo)

- Fresh-env first-time-user test from the branch (clean env, GitHub install,
  pip check, import smoke, base-only blocked-import simulation, pytest from a
  fresh clone).
- Numerical regression: pruner invariants exact; FordB accuracy in the same
  band as 0.1.0 (kernel RNG differs across libraries — parity, not identity).
- Colab GPU notebook re-run against the branch (Gonzalo).
- Version → `0.2.0` in pyproject AND `detach_rocket/__init__.py`; delete this
  file; final gates; merge to main per the 0.1.0 procedure; tag `v0.2.0`.

---

## Progress log (agents append here, newest first)

- 2026-08-25: **Stage 4 done** (agent). All three notebooks ported to aeon,
  re-executed in place with fresh outputs, and the README results table synced
  from the new FordB run. Gates: `pytest` **35 passed** (5.2 s), `ruff check .`
  and `ruff format --check .` clean (17 files; notebooks are ruff-excluded).
  JSON sanity check passes for all three: plain-JSON parse + `nbformat.validate`,
  cell counts 11 / 10 / 13 unchanged, **zero** `error` outputs, one `image/png`
  per plotting cell, and each `!pip install` cell left unexecuted and
  output-free.
  - Env prep, installed into `detach_aeon`: matplotlib 3.11.1, nbconvert 7.17.1,
    nbformat 5.11.1, ipykernel 7.3.0, tsfresh 0.21.2. (Already present: aeon
    1.5.0, numpy 2.3.5, scikit-learn 1.7.2, pandas 2.3.3, numba 0.63.1, torch
    2.13.0, pytest 9.1.1, ruff 0.16.4, Python 3.12.14.)
  - **tsfresh extractor: aeon has a working equivalent, so sktime was NOT added
    back.** `aeon.transformations.collection.feature_based.TSFresh` takes the
    same three kwargs the notebook used (`default_fc_parameters="comprehensive"`,
    `show_warnings=False`, `disable_progressbar=True` — identical names) and its
    `_transform` ends in `Xt.to_numpy()`, i.e. a **2D** `(n_cases, n_features)`
    array, exactly what `DetachMatrix` wants. Decisive evidence: the re-run
    reproduces the sktime-era numbers **exactly** — 783 features, 75.41% full,
    74.94% detached, 23.75% optimal size. As a bonus the two sklearn
    "X does not have valid feature names, but StandardScaler was fitted with
    feature names" warnings are gone, because aeon returns numpy where sktime
    returned a DataFrame. `fit_is_empty=True` on that class, and the
    `fit_transform(train)` → `transform(test)` sequence the notebook uses was
    verified to survive aeon's `reset()` (unlike the pruned-Rocket hazard from
    Stage 2) — `default_fc_parameters_` is still intact afterwards.
  - **`np.random.seed` no longer reaches ROCKET's kernels.** aeon's
    `Rocket._fit` hands `self.random_state` straight to the numba-jitted
    `_generate_kernels`, and numba keeps its own RNG state: verified that two
    fits under the same `np.random.seed(42)` give **different** kernels, while
    `random_state=42` is reproducible across different numpy seeds. The UCR
    notebook already carried `np.random.seed(42)` and a seeded
    `train_test_split`, so that seed had silently become a no-op for the part
    that matters. `random_state=42` is therefore passed explicitly to
    `Rocket(...)` — a deviation from the literal task text, taken to preserve
    the notebook's existing reproducibility intent and to make the committed
    README numbers reproducible. `np.random.seed(42)` was left in place.
  - **Shape handling.** `load_classification` returns exactly the same cases and
    lengths as the old loaders, just 3D: FordB (3636, 1, 500)/(810, 1, 500),
    PhalangesOutlinesCorrect (1800, 1, 80)/(858, 1, 80), SelfRegulationSCP1
    (268, 6, 896)/(293, 6, 896) — the last identical to before. In the UCR and
    tsfresh "Prepare Dataset Matrices" cells the NaN/inf masks moved from
    `axis=1` to `axis=(1, 2)`, the manual `reshape(n, 1, t)` was **deleted** (the
    data already arrives 3D), and the printed header became the three-component
    `( # of instances , # of channels , time series length )` the UEA notebook
    already used — so the cell stays informative and now matches what it prints.
  - **README table, old → new** (line ~20, FordB, 10,000 kernels):
    accuracy 79.26% → **80.49%** full and 81.85% → **82.10%** detached; features
    retained 0.69% → **2.36%**; inference 34.66 s → **21.96 s** full and 0.47 s →
    **1.01 s** detached; speedup 73x → **22x**. Accuracy is squarely in the same
    band (parity, as Stage 5 expects). The speedup moved for two independent
    reasons worth not confusing: SFD kept ~3.4x more features this time (a
    genuine kernel-RNG difference between the libraries), and the machine is
    faster (the *full* model got quicker, 34.66 → 21.96 s). Nothing about the
    method regressed.
  - Runtimes on this machine, each executed alone so the UCR timing cell was not
    perturbed by competing load: UCR **141.8 s**, tsfresh **110.2 s**, UEA
    ensemble **825.1 s** (~13.8 min).
  - UEA ensemble: 98.13% train / **92.49% test** — the test figure is identical
    to the old run. Its output is also far cleaner now: the previous version had
    ~100 lines of `Singular matrix in solving dual problem` / `LinAlgWarning`
    noise, and commit ddb2f6e's warning suppression reduces stderr to the single
    expected "PyTorch is running on CPU" `UserWarning`.
  - Runner (scratch, **not committed**): `nbformat` + an `ExecutePreprocessor`
    subclass whose `preprocess_cell` short-circuits cells whose source starts
    with `!pip install`, clearing their outputs and `execution_count`.
    `kernel_name="python3"`, `timeout=3600`, cwd = `examples/`.
  - **Gotcha for Stage 5:** `detach_rocket` is **not** installed into
    `detach_aeon` (pytest only works because `python -m pytest` from the repo
    root puts cwd on `sys.path`). A notebook kernel runs with cwd = `examples/`,
    so it cannot import the package — the first run died with
    `ModuleNotFoundError: No module named 'detach_rocket'`. Executing the
    notebooks needs `PYTHONPATH=/Users/uribarri/Documents/detach_rocket` (or a
    real `pip install -e .`). Stage 5's fresh-env test installs from GitHub, so
    it will not hit this, but anyone re-running the notebooks locally will.
  - Install cells now read `detach_rocket @ git+…` (UCR), `detach_rocket[torch] @
    git+…` (UEA) and `detach_rocket @ git+… matplotlib tsfresh` (tsfresh); they
    point at the plain repo URL, which resolves to the new main post-merge.
    `nbformat` round-trips all three files byte-identically, so the diffs contain
    only real changes. The one surviving `num_kernels` in the UEA notebook is
    `DetachEnsemble(num_models=…, num_kernels=…)` — the library's own parameter
    name, correctly untouched. The ensemble cell was left without a
    `random_state` (that notebook never had a seed; its numbers are stochastic by
    design). The tsfresh notebook's `multiclass_type` output now prints `'max'`
    rather than the old `'norm'` — that is commit 87b6c2c's restored default, not
    a Stage 4 change.

- 2026-08-25: **Stage 3 done** (agent). README full pass, CI bumped, pyproject
  classifiers checked. Gates: `pytest` **35 passed** (5.5 s), `ruff check .`
  and `ruff format --check .` clean — docs-only changes, no test movement.
  - README: Python badge →3.11; the Features "scikit-learn or sktime" line now
    reads aeon + "any object exposing `transform(X)`"; Install drops
    `[datasets]`, says aeon (base dep) supplies transformers *and* loaders, and
    lists the real extras (`torch`, `cuda`, `examples`, `dev`, `all`); Quick
    Start imports aeon's `Rocket` with `n_kernels=10_000`; a
    `load_classification` snippet was added **before** the model code (the old
    text used `X_train` with no indication of where it came from); Core Modules
    drops `utils_datasets.py` and names all three pruners.
  - **2D input, verified rather than assumed:** aeon's collection transformers
    *do* accept 2D `(n_instances, n_timepoints)` — `_preprocess_collection`
    reshapes it to `(n, 1, t)` and the features are bit-identical to passing 3D
    (`np.allclose` on `fit_transform`, and on `fit(3D)` + `transform(2D)`). The
    whole DetachRocket path was exercised both ways: same score, same
    `retained_kernel_count` (153/153), `detach()` prediction parity, and a
    `PrunedAeonRocketTransformer` either way. So the README says 2D works,
    which is the *opposite* of the "sktime only" claim the task assumed.
  - **numba/llvmlite advice re-verified and it changed meaning.** numba stopped
    publishing macOS x86_64 wheels at **0.63.0** (llvmlite 0.46.0); the last
    ones with an x86_64 mac wheel are numba 0.62.1 / llvmlite 0.45.1. aeon caps
    `numba<0.64`, so a plain pip install on Intel/Rosetta now resolves to
    0.63.1 — source-only there — and compiles LLVM. A wheels-only
    cross-platform resolve (`--platform macosx_12_0_x86_64 --python-version
    312`) lands on numba 0.62.1 + llvmlite 0.45.1 with numpy 2.3.5, i.e. what
    `--prefer-binary` picks. **The tip still holds and is now strictly
    necessary on that platform**, so it was kept and its explanation rewritten.
    On arm64 both dry-runs resolve identically (0.63.1), as expected. The
    conda fallback is now `conda install "numba>=0.58,<0.64"` — unbounded, it
    would install a numba aeon rejects. The `nomkl` note is untouched.
  - Migration: new "Migrating from 0.1.x" above the 0.0.x section, every row
    checked against the `v0.1.0` tag rather than memory (`requires-python
    >=3.10`, `numpy>=1.26.4,<2.5`, `PrunedRocketTransformer`/
    `RocketTransformerPruner`, `datasets = [scipy, pyts]`, the darwin-x86_64
    `numpy<2` torch marker, and `fetch_ucr_dataset` returning a Bunch with both
    splits). It flags the **2D→3D change in what the loaders return**, which
    the plan only noted for Stage 4 but which users hit too. Confirmed
    empirically that a user-supplied **sktime** Rocket still fits, predicts and
    `detach()`es correctly through `GenericPrunedTransformer` (decision A), so
    the README promises that explicitly. The 0.0.x table's right column became
    `0.1.0+` with `n_kernels`, so nobody copies a now-invalid `num_kernels`.
  - Troubleshooting: the Intel-mac `numpy<2` torch paragraph is replaced by the
    `@v0.1.0` install line (the old pin is unsatisfiable under a NumPy-2 base).
  - **CI gotcha worth remembering:** `astral-sh/setup-uv` stopped publishing
    floating major tags in v8, so `@v10` **404s** — it is pinned to the exact
    `v10.0.1`, with a comment in the workflow saying why. `actions/checkout@v7`
    and `actions/setup-python@v7` do exist as major tags. All three declare
    `using: node24`, which is what silences the runner deprecation warning.
    Both jobs moved 3.10 → 3.12. setup-uv v10's cache-poisoning default change
    only affects `pull_request_target`/`workflow_run`/`release`, none of which
    trigger this workflow.
  - pyproject: added the `Programming Language :: Python :: 3.14` classifier —
    the list stopped at 3.13 while `requires-python` is open-ended `>=3.11`,
    and a wheels-only cp314 resolve succeeds (aeon 1.5.0, numba 0.63.1, numpy
    2.3.5, sklearn 1.8.0). Resolver-verified, not runtime-tested; **Stage 5
    could smoke-test 3.14** if it wants the claim exercised. Keywords are
    clean (nothing sktime-flavoured). `requires-python` was left without an
    upper bound on purpose — mirroring aeon's `<3.15` would duplicate a
    constraint aeon already enforces and force a release each time aeon widens
    it. Flagged as a maintainer call, not fixed unilaterally.
  - **Stage 4 inherits:** the README results table (FordB 79.26% / 81.85% /
    0.69% / 34.66s vs 0.47s, 10,000 kernels, line ~24) is still the 0.1.0
    sktime run and must be re-synced from the new notebook execution. The
    notebook links and descriptions in "Notebook Examples" were left as-is and
    still read correctly. Note for the tsfresh notebook: `load_classification`
    returns univariate X as 3D, whereas the old loaders gave 2D.
  - Nothing in the README references the version number `0.2.0` except the new
    migration section's heading text, which Stage 5's version bump should stay
    consistent with.

- 2026-08-25: **Stage 2 done** (agent). aeon is now a base dependency and
  sktime is gone from the code and the metadata. pyproject: base deps are
  `numpy>=2.0,<2.5`, `scikit-learn>=1.7.2,<1.9`, `aeon>=1.5,<2` (numba and
  scipy arrive with aeon); `requires-python = ">=3.11"`; 3.10 classifier and
  the `[aeon]`/`[datasets]` extras removed; `[torch]` is now just
  `torch>=2.3` (the darwin-x86_64 `numpy<2` marker is unsatisfiable under a
  NumPy-2 base); `[tool.ruff] target-version` bumped to `py311` to match the
  Python floor. `pruner.py`: sktime import, `PrunedRocketTransformer` and
  `RocketTransformerPruner` deleted; `PrunedAeonRocketTransformer` is a plain
  top-level class again (lazy factory, class cache and module `__getattr__`
  removed); only the CUDA branch keeps its `try/except ImportError`.
  `utils_datasets.py` deleted. `detach_classes.py` docstrings point at aeon.
  Gates: `pytest` 36 → **35 passed** (4.8 s), `ruff check .` and
  `ruff format --check .` clean. With sktime blocked in a subprocess (same
  `sitecustomize` `ModuleNotFoundError` finder as Stage 1): `import
  detach_rocket` clean with **no `sktime` entry in `sys.modules`**, and the
  full suite still 35 passed / 0 skipped.
  - Test accounting: `tests/test_aeon_transformers.py` (14) was merged into
    `tests/test_detach_rocket.py` (6 → 19) and deleted — with aeon the only
    backend, a separate module and its `importorskip("aeon")` guard no longer
    made sense, and this restores both invariant guards to the file the plan
    names them in. Dropped as duplicates: `test_rocket_pruner` (subsumed by
    `test_pruned_transformer_is_specialized` plus the two invariant tests) and
    the old single-variate `test_model_level_pruned_path_equivalence` (the
    surviving one is parametrized univariate + multivariate). Added:
    `test_pruned_transformer_survives_refitting_calls`. Net 20 → 19; the other
    16 tests are untouched and still pass. The dead
    `pytest.importorskip("numba")` in the old fixture is gone (importing aeon
    already needs numba). `N_KERNELS` is 512 suite-wide (the plan's fixture
    size) rather than Stage 1's 256.
  - `fit_transform` safeguard: `PrunedAeonRocketTransformer.reset()` is
    overridden to always add `"kernels"` to aeon's own `reset(keep=...)` list.
    `fit`/`transform`/`fit_transform` are all `@final`, but `reset` is not, and
    it is the single place the damage happened. Verified in aeon 1.5.0:
    `_fit_transform` skips `_fit` when `fit_is_empty` is set, so the *only*
    hazard was the unconditional `reset()` at the top of `fit_transform` —
    which used to delete `kernels` and leave the instance permanently broken
    (`AttributeError` on every later `transform`, not just the one call).
    With the override, `fit_transform(X)` now returns exactly `transform(X)`,
    and `fit()` / `reset()` are harmless no-ops. Covered by
    `test_pruned_transformer_survives_refitting_calls`.
  - `get_summary()["retained_kernel_count"]` keeps its two-step lookup, but for
    a new reason: `num_kernels` is still needed by the **CUDA** pruned
    transformer, then `n_kernels` for aeon, then `None`. The generic wrapper
    yields `None` (it exposes `num_kernels=None`); both the aeon int and the
    generic `None` are now asserted by tests.
  - Naming kept as-is: `PrunedAeonRocketTransformer` /
    `AeonRocketTransformerPruner` (explicitly aeon's Rocket, vs. the CUDA and
    generic pruners). Stage 3 docs should use these names.
  - **Stage 3 must bump CI's Python**: `.github/workflows/ci.yml` pins
    `python-version: "3.10"` in both jobs, and `uv pip install -e ".[dev]"`
    now fails that floor. Nothing else in CI references a removed extra.
  - Also for Stage 3's README pass: the Core Modules list still has
    `detach_rocket/utils_datasets.py`, and the Install section still offers
    `[aeon]`/`[datasets]`. Stage 4: all three notebooks still import sktime's
    Rocket and/or `fetch_ucr_dataset`/`fetch_uea_dataset`.
  - Minor, for Stage 5 to decide: `detach_rocket/_warnings.py` imports
    `scipy.linalg` directly while scipy is only a transitive dependency (via
    aeon `>=1.9,<1.18`, and sklearn). It was equally transitive before this
    stage (via sktime), so nothing regressed; an explicit scipy pin would just
    make the direct use honest.
- 2026-08-25: **Stage 1 done** (agent). `[aeon]` extra added (and to `all`);
  `AeonRocketTransformerPruner` + `PrunedAeonRocketTransformer` in
  `pruner.py`; aeon branch in `get_transformer_pruner` before the CUDA one;
  `tests/test_aeon_transformers.py` (14 tests, 3 s); README Overview bullet.
  Gates: `pytest` 22 → **36 passed**, `ruff check .` and
  `ruff format --check .` clean. With aeon blocked in a subprocess (a
  `sitecustomize` meta-path finder raising `ModuleNotFoundError`):
  22 passed + 1 skipped, and `import detach_rocket.pruner` is clean.
  Facts for later stages:
  - aeon's `BaseCollectionTransformer.transform` is `@final` — override
    `_transform(X, y=None)`, never `transform`.
  - Unfitted-but-populated state needed by aeon's transform path:
    `_tags = {"fit_is_empty": True}` (skips `_check_is_fitted`/`_check_shape`
    **and** makes a stray `fit()` a no-op instead of regenerating random
    kernels), `is_fitted = True`, `_n_jobs` (read by `Rocket._transform`), and
    `kernels`. `metadata_` is `{}` from the base `__init__`, which makes
    `_check_shape` a no-op; `fit_min_length_` is copied for introspection
    parity only (transform does not read it).
  - `normalise` is now copied from the parent transformer: with
    `normalise=False` parents, defaulting it to True breaks the invariant.
    sktime's `PrunedRocketTransformer` has that latent bug; it dies with the
    class in Stage 2.
  - `fit_transform()` on a pruned aeon transformer would `reset()` away
    `kernels`; nothing in the library calls it (only `transform`), so it was
    left alone.
  - aeon `MiniRocket`/`MultiRocket` do not subclass aeon `Rocket` (generic
    fallback, as planned), and aeon `Rocket` is unrelated to sktime's, so the
    two specialized branches cannot collide.
  - `get_summary()["retained_kernel_count"]` now falls back from sktime's
    `num_kernels` to aeon's `n_kernels`; Stage 2 can drop the first lookup.
  - pytest 9.1 `importorskip` skips on `ModuleNotFoundError` only, so a
    *broken* aeon install errors loudly instead of silently skipping.
  - Unrelated flake for Stage 5: `test_predict_proba_and_predict` builds a
    `DetachEnsemble` without `random_state`, so its kernels come from OS
    entropy; a degenerate draw occasionally emits a NaN-matmul
    `RuntimeWarning` (test still passes).
- 2026-08: Stage 0 done (coordinator): `detach_aeon` env, v0.1.0 tag+release.

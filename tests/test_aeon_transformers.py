"""Tests for the aeon ROCKET-family transformers.

aeon is an optional dependency in this release, so the whole module is
skipped when it is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("aeon")

from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket, Rocket  # noqa: E402

from detach_rocket.detach_classes import DetachRocket  # noqa: E402
from detach_rocket.pruner import (  # noqa: E402
    AeonRocketTransformerPruner,
    GenericPrunedTransformer,
    PrunedAeonRocketTransformer,
    get_transformer_pruner,
)

N_KERNELS = 256
N_SAMPLES = 50
N_TIMEPOINTS = 50
N_CHANNELS = 3


def _make_data(n_channels, seed=0):
    """Return a small random (X, y) classification problem."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_SAMPLES, n_channels, N_TIMEPOINTS))
    y = rng.integers(0, 2, N_SAMPLES)
    return X, y


def _assert_pruned_path_matches_masked_full(model, X):
    """Assert the sacred invariant: pruned path == full transform then mask."""
    full = np.asarray(model.transformer.transform(X))
    masked = model.scaler_.transform(full)[:, model.feature_mask_]
    prepared = model._prepare_X(X)

    assert prepared.shape == masked.shape, f"Shape mismatch: pruned {prepared.shape} vs masked {masked.shape}"
    assert np.allclose(prepared, masked, atol=1e-6), (
        "Pruned inference path diverged from the full-transform-then-mask path"
    )


@pytest.fixture(scope="module", params=["univariate", "multivariate"])
def fitted_model(request):
    """Fitted DetachRocket on an aeon Rocket, for both input dimensionalities."""
    n_channels = 1 if request.param == "univariate" else N_CHANNELS
    X, y = _make_data(n_channels)
    model = DetachRocket(
        transformer=Rocket(n_kernels=N_KERNELS, random_state=42),
        set_percentage=50,
    )
    model.fit(X, y)
    return model, X, y


def test_dispatch_returns_specialized_aeon_pruner():
    """A fitted aeon Rocket must never silently fall back to generic pruning."""
    X, _ = _make_data(N_CHANNELS)
    transformer = Rocket(n_kernels=N_KERNELS, random_state=42)
    transformer.fit(X)

    pruner = get_transformer_pruner(transformer)

    assert isinstance(pruner, AeonRocketTransformerPruner), (
        f"Expected the specialized aeon pruner, got {type(pruner).__name__}"
    )


def test_pruned_transformer_matches_masked_full_transform():
    """Transformer-level invariant, with a mask that splits kernel feature pairs."""
    X, _ = _make_data(N_CHANNELS)
    transformer = Rocket(n_kernels=N_KERNELS, random_state=42)
    full_features = transformer.fit_transform(X)

    rng = np.random.default_rng(1)
    feature_mask = rng.random(full_features.shape[1]) < 0.4
    feature_mask[0] = True  # keep at least one feature

    pruned_transformer = get_transformer_pruner(transformer).prune_transformer(transformer, feature_mask)
    pruned_features = pruned_transformer.transform(X)

    assert pruned_features.shape == (N_SAMPLES, int(np.sum(feature_mask)))
    assert np.allclose(pruned_features, full_features[:, feature_mask], atol=1e-6), (
        "Pruned transformer output does not match masked full transformer output"
    )


def test_pruned_transformer_is_specialized(fitted_model):
    """The fitted model must hold a physically pruned transformer, not a wrapper."""
    model, _, _ = fitted_model

    assert isinstance(model.pruned_transformer_, PrunedAeonRocketTransformer), (
        f"Expected PrunedAeonRocketTransformer, got {type(model.pruned_transformer_).__name__}"
    )

    retained_n_kernels = np.sum(model.feature_mask_[0::2] | model.feature_mask_[1::2])
    assert model.pruned_transformer_.n_kernels == retained_n_kernels, "Pruned transformer kernel count mismatch"


def test_model_level_pruned_path_equivalence(fitted_model):
    """Model-level invariant plus detach() prediction parity."""
    model, X, y = fitted_model

    _assert_pruned_path_matches_masked_full(model, X)

    pruned_model = model.detach()
    assert np.array_equal(pruned_model.predict(X), model.predict(X)), (
        "detach() model predictions differ from the parent model"
    )
    assert np.isclose(pruned_model.score(X, y), model.score(X, y))


def test_get_summary_retained_kernel_count(fitted_model):
    """The summary must count retained kernels through aeon's n_kernels attribute."""
    model, _, _ = fitted_model

    summary = model.get_summary()
    retained_kernel_count = summary["retained_kernel_count"]

    assert isinstance(retained_kernel_count, int)
    assert 0 < retained_kernel_count <= N_KERNELS
    assert retained_kernel_count == model.pruned_transformer_.n_kernels


def test_invariant_after_fit_set_optimal():
    """Re-selecting the pruning level must rebuild a consistent pruned transformer."""
    X, y = _make_data(N_CHANNELS)
    model = DetachRocket(
        transformer=Rocket(n_kernels=N_KERNELS, random_state=42),
        set_percentage=50,
    )
    model.fit(X, y)
    features_before = int(np.sum(model.feature_mask_))

    model.fit_set_optimal(20)

    assert int(np.sum(model.feature_mask_)) < features_before, "Re-selection did not prune further"
    assert isinstance(model.pruned_transformer_, PrunedAeonRocketTransformer)
    _assert_pruned_path_matches_masked_full(model, X)
    assert np.array_equal(model.detach().predict(X), model.predict(X))


def test_univariate_two_dimensional_input():
    """aeon converts 2D univariate input to 3D; the pruned path must follow."""
    X, y = _make_data(1)
    X_2d = X[:, 0, :]

    model = DetachRocket(
        transformer=Rocket(n_kernels=N_KERNELS, random_state=42),
        set_percentage=50,
    )
    model.fit(X_2d, y)

    assert isinstance(model.pruned_transformer_, PrunedAeonRocketTransformer)
    _assert_pruned_path_matches_masked_full(model, X_2d)
    assert np.array_equal(model.predict(X_2d), model.predict(X))


@pytest.mark.parametrize("transformer_class", [MiniRocket, MultiRocket])
def test_generic_fallback_transformers(transformer_class):
    """MiniRocket and MultiRocket are pruned through the masking fallback."""
    X, y = _make_data(N_CHANNELS)
    model = DetachRocket(
        transformer=transformer_class(n_kernels=N_KERNELS, random_state=42),
        set_percentage=50,
    )
    model.fit(X, y)

    assert isinstance(model.pruned_transformer_, GenericPrunedTransformer), (
        f"Expected the generic fallback, got {type(model.pruned_transformer_).__name__}"
    )
    assert model.predict(X).shape == (N_SAMPLES,)
    _assert_pruned_path_matches_masked_full(model, X)
    assert np.array_equal(model.detach().predict(X), model.predict(X))


def test_prune_unfitted_transformer_raises():
    """Pruning an unfitted aeon Rocket must fail loudly."""
    transformer = Rocket(n_kernels=N_KERNELS)
    feature_mask = np.ones(2 * N_KERNELS, dtype=bool)

    with pytest.raises(ValueError, match="must be fit"):
        AeonRocketTransformerPruner().prune_transformer(transformer, feature_mask)


def test_prune_mask_size_mismatch_raises():
    """The mask must cover the two features produced per kernel."""
    X, _ = _make_data(N_CHANNELS)
    transformer = Rocket(n_kernels=N_KERNELS, random_state=42)
    transformer.fit(X)

    with pytest.raises(ValueError, match="mask size mismatch"):
        AeonRocketTransformerPruner().prune_transformer(transformer, np.ones(N_KERNELS, dtype=bool))

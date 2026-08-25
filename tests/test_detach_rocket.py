"""Tests for DetachRocket and the ROCKET-family transformer pruners."""

import numpy as np
import pytest
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket, Rocket
from sklearn.model_selection import train_test_split

from detach_rocket.aeon_multirocket import AeonMultiRocket
from detach_rocket.detach_classes import DetachRocket
from detach_rocket.pruner import (
    AeonRocketTransformerPruner,
    GenericPrunedTransformer,
    PrunedAeonRocketTransformer,
    get_transformer_pruner,
)

N_KERNELS = 512
N_SAMPLES = 50
N_TIMEPOINTS = 50
N_CHANNELS = 3


def _make_data(n_channels, n_samples=N_SAMPLES, seed=0):
    """Return a small random (X, y) classification problem."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_channels, N_TIMEPOINTS))
    y = rng.integers(0, 2, n_samples)
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


@pytest.fixture(scope="module")
def data():
    """Fixture that creates a simple dataset with a validation split."""
    X, y = _make_data(N_CHANNELS, n_samples=100, seed=7)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }


@pytest.fixture(scope="function")
def detach_rocket():
    """DetachRocket on an aeon Rocket, pruned at a fixed percentage."""
    return DetachRocket(
        transformer=Rocket(n_kernels=N_KERNELS, random_state=42),
        set_percentage=50,  # Fixed pruning percentage
        verbose=True,
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


# -- Pruner dispatch ---------------------------------------------------------


def test_dispatch_returns_specialized_aeon_pruner():
    """A fitted aeon Rocket must never silently fall back to generic pruning."""
    X, _ = _make_data(N_CHANNELS)
    transformer = Rocket(n_kernels=N_KERNELS, random_state=42)
    transformer.fit(X)

    pruner = get_transformer_pruner(transformer)

    assert isinstance(pruner, AeonRocketTransformerPruner), (
        f"Expected the specialized aeon pruner, got {type(pruner).__name__}"
    )


def test_invalid_pruning():
    """Test that unsupported transformers fall back to generic pruning."""

    # Create a mock transformer (not a Rocket instance)
    class MockTransformer:
        def transform(self, X):
            # Return a simple dummy transformation (4 features)
            return np.random.rand(X.shape[0], 4)

    mock_transformer = MockTransformer()
    feature_mask = np.array([True, False, True, False])

    # Should return a GenericTransformerPruner (no error)
    pruner = get_transformer_pruner(mock_transformer)
    pruned = pruner.prune_transformer(mock_transformer, feature_mask)

    # Verify it's a GenericPrunedTransformer
    assert isinstance(pruned, GenericPrunedTransformer), "Expected GenericPrunedTransformer fallback"

    # Verify it masks features correctly
    X_dummy = np.random.rand(10, 5)  # Dummy input
    X_pruned = pruned.transform(X_dummy)
    assert X_pruned.shape[1] == 2, "Expected 2 retained features"


# -- Transformer-level pruning ------------------------------------------------


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


def test_pruned_transformer_survives_refitting_calls():
    """fit / fit_transform / reset must not discard the copied kernels.

    aeon's ``fit_transform`` calls ``reset()`` first, which would otherwise
    delete kernels that this transformer can never regenerate.
    """
    X, _ = _make_data(N_CHANNELS)
    transformer = Rocket(n_kernels=N_KERNELS, random_state=42)
    full_features = transformer.fit_transform(X)

    feature_mask = np.zeros(full_features.shape[1], dtype=bool)
    feature_mask[::3] = True
    expected = full_features[:, feature_mask]

    pruned = get_transformer_pruner(transformer).prune_transformer(transformer, feature_mask)

    assert np.allclose(pruned.fit_transform(X), expected, atol=1e-6), (
        "fit_transform on a pruned transformer must equal the masked full output"
    )
    for call in (pruned.reset, lambda: pruned.fit(X)):
        call()
        assert np.allclose(pruned.transform(X), expected, atol=1e-6), (
            "Pruned transformer stopped reproducing the masked full output"
        )


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


# -- Model-level behaviour ----------------------------------------------------


def test_fit_with_optimal_pruning(detach_rocket, data):
    """Test the fit function without a fixed pruning percentage, allowing optimal selection."""
    # Set set_percentage to None to allow optimal pruning
    detach_rocket.set_percentage = None

    # Call the fit method
    detach_rocket.fit(data["X_train"], data["y_train"], X_val=data["X_val"], y_val=data["y_val"])

    # Assert that the pruned transformer was created
    assert detach_rocket.pruned_transformer_ is not None, "Pruned transformer should be initialized"

    # Check that the optimal feature mask has been generated
    assert detach_rocket.feature_mask_ is not None, "Optimal feature mask should be initialized"

    # Assert that the max index and max percentage were calculated correctly
    assert detach_rocket.selected_step_index_ >= 0, "Max index should be non-negative"
    assert detach_rocket.selected_ratio_ >= 0, "Max percentage should be non-negative"


def test_pruned_transformer_is_specialized(fitted_model):
    """The fitted model must hold a physically pruned transformer, not a wrapper."""
    model, _, _ = fitted_model

    assert isinstance(model.pruned_transformer_, PrunedAeonRocketTransformer), (
        f"Expected PrunedAeonRocketTransformer, got {type(model.pruned_transformer_).__name__}"
    )

    retained_n_kernels = np.sum(model.feature_mask_[0::2] | model.feature_mask_[1::2])
    assert model.pruned_transformer_.n_kernels == retained_n_kernels, "Pruned transformer kernel count mismatch"


def test_pruned_transformer_consistency(detach_rocket, data):
    """Test that the pruned transformer produces the same features as
    applying the feature mask to the full transformer output."""
    detach_rocket.fit(data["X_train"], data["y_train"])

    X = data["X_train"]
    full_features = np.asarray(detach_rocket.transformer.transform(X))
    pruned_features = np.asarray(detach_rocket.pruned_transformer_.transform(X))
    masked_full_features = full_features[:, detach_rocket.feature_mask_]

    assert pruned_features.shape == masked_full_features.shape, (
        f"Shape mismatch: pruned {pruned_features.shape} vs masked {masked_full_features.shape}"
    )
    assert np.allclose(pruned_features, masked_full_features), (
        "Pruned transformer output does not match masked full transformer output"
    )


def test_model_level_pruned_path_equivalence(fitted_model):
    """The pruned inference path must equal the full-transform-then-mask path,
    and the detached lightweight model must reproduce the parent's predictions."""
    model, X, y = fitted_model

    _assert_pruned_path_matches_masked_full(model, X)

    pruned_model = model.detach()
    assert np.array_equal(pruned_model.predict(X), model.predict(X)), (
        "detach() model predictions differ from the parent model"
    )
    assert np.isclose(pruned_model.score(X, y), model.score(X, y))


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


@pytest.mark.parametrize("transformer_class", [MiniRocket, MultiRocket, AeonMultiRocket])
def test_generic_fallback_transformers(transformer_class):
    """MiniRocket, MultiRocket, and AeonMultiRocket use the masking fallback."""
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
    assert model.get_summary()["retained_kernel_count"] is None, (
        "Kernel semantics are undefined for the generic fallback"
    )


# -- Summary ------------------------------------------------------------------


def test_get_summary(detach_rocket, data):
    detach_rocket.set_percentage = None
    detach_rocket.fit(data["X_train"], data["y_train"], X_val=data["X_val"], y_val=data["y_val"])

    summary = detach_rocket.get_summary()

    assert summary["estimator"] == "DetachRocket"
    assert summary["is_fitted"] is True
    assert summary["selected_feature_count"] <= summary["full_feature_count"]
    assert 0 <= summary["selected_ratio"] <= 1
    assert summary["retained_kernel_count"] == detach_rocket.pruned_transformer_.n_kernels
    assert summary["final_model_alpha"] > 0


def test_get_summary_retained_kernel_count(fitted_model):
    """The summary must count retained kernels through aeon's n_kernels attribute."""
    model, _, _ = fitted_model

    summary = model.get_summary()
    retained_kernel_count = summary["retained_kernel_count"]

    assert isinstance(retained_kernel_count, int)
    assert 0 < retained_kernel_count <= N_KERNELS
    assert retained_kernel_count == model.pruned_transformer_.n_kernels

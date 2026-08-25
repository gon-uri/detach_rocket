"""Tests for DetachEnsemble with aeon-backed members (MultiRocket, and
MiniRocket on backend="aeon").

Deliberately torch-free (unlike tests/test_detach_ensemble.py, which guards
on torch at module level): the aeon paths must work without the optional
PyTorch dependency.
"""

import numpy as np
import pytest

from detach_rocket.aeon_minirocket import AeonMiniRocket
from detach_rocket.aeon_multirocket import AeonMultiRocket
from detach_rocket.detach_classes import DetachEnsemble

# num_kernels is a FEATURE budget: 672 -> 84 aeon kernels -> 672 features,
# the smallest valid MultiRocket configuration (see DetachEnsemble docs).
NUM_KERNELS = 672


@pytest.fixture(scope="module")
def ensemble_data():
    """Small multivariate dataset where channel 0 carries the class signal."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3, 50)).astype(np.float32)
    y = (X[:, 0, :].mean(axis=1) > 0).astype(int)
    X[:, 0, :] += y[:, None] * 2.0
    return X, y


@pytest.fixture(scope="module")
def fitted_ensemble(ensemble_data):
    X, y = ensemble_data
    ensemble = DetachEnsemble(
        num_models=2, num_kernels=NUM_KERNELS, set_percentage=50, model_type="multirocket", random_state=0
    )
    ensemble.fit(X, y)
    return ensemble


def test_members_are_aeon_multirocket(fitted_ensemble):
    """The dispatch builds AeonMultiRocket members with the //8 kernel budget."""
    assert fitted_ensemble.backend == "aeon"
    for model in fitted_ensemble.derockets:
        assert isinstance(model.transformer, AeonMultiRocket)
        assert model.transformer.n_kernels == NUM_KERNELS // 8


def test_predict_proba_and_predict(fitted_ensemble, ensemble_data):
    X, y = ensemble_data

    proba_soft = fitted_ensemble.predict_proba(X, proba="soft")
    assert proba_soft.shape == (len(y), 2)
    assert np.allclose(proba_soft.sum(axis=1), 1.0)

    proba_hard = fitted_ensemble.predict_proba(X, proba="hard")
    assert np.allclose(proba_hard.sum(axis=1), 1.0)

    y_pred = fitted_ensemble.predict(X)
    assert y_pred.shape == y.shape
    assert set(np.unique(y_pred)) <= set(np.unique(y))


def test_score(fitted_ensemble, ensemble_data):
    X, y = ensemble_data
    accuracy = fitted_ensemble.score(X, y)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_channel_relevance(fitted_ensemble):
    relevance = fitted_ensemble.estimate_channel_relevance()
    assert relevance.shape == (3,)
    assert np.isclose(relevance.sum(), 1.0)
    # Channel 0 carries the class signal, so it should dominate.
    assert relevance.argmax() == 0

    # The paper's aggregation (median) is available as an option.
    relevance_median = fitted_ensemble.estimate_channel_relevance(aggregation="median")
    assert relevance_median.shape == (3,)
    assert np.isclose(relevance_median.sum(), 1.0)
    assert relevance_median.argmax() == 0

    with pytest.raises(ValueError, match="aggregation"):
        fitted_ensemble.estimate_channel_relevance(aggregation="mode")


def test_fit_with_explicit_validation(ensemble_data):
    """An explicit validation set replaces the internal random split."""
    X, y = ensemble_data
    ensemble = DetachEnsemble(num_models=1, num_kernels=NUM_KERNELS, model_type="multirocket")
    ensemble.fit(X[:30], y[:30], X_val=X[30:], y_val=y[30:])
    assert ensemble.is_fitted_
    assert ensemble.predict(X[30:]).shape == y[30:].shape

    with pytest.raises(ValueError, match="y_val"):
        DetachEnsemble(num_models=1, num_kernels=NUM_KERNELS, model_type="multirocket").fit(
            X[:30], y[:30], X_val=X[30:]
        )


def test_random_state_reproducibility(ensemble_data):
    """Same random_state must give identical ensemble outputs."""
    X, y = ensemble_data

    kwargs = dict(num_models=2, num_kernels=NUM_KERNELS, set_percentage=50, model_type="multirocket", random_state=0)
    e1 = DetachEnsemble(**kwargs)
    e2 = DetachEnsemble(**kwargs)
    e1.fit(X, y)
    e2.fit(X, y)
    assert np.allclose(e1.predict_proba(X), e2.predict_proba(X))


def test_univariate_2d_input_warns_and_works(ensemble_data):
    """2D input is reshaped to one channel with a warning, and the ensemble runs."""
    X, y = ensemble_data
    X_2d = X[:, 0, :]

    ensemble = DetachEnsemble(
        num_models=2, num_kernels=NUM_KERNELS, set_percentage=50, model_type="multirocket", random_state=1
    )
    with pytest.warns(UserWarning, match="univariate"):
        ensemble.fit(X_2d, y)

    assert ensemble.num_channels == 1
    pred_2d = ensemble.predict(X_2d)
    pred_3d = ensemble.predict(X_2d[:, None, :])
    assert np.array_equal(pred_2d, pred_3d)
    assert np.allclose(ensemble.predict_proba(X_2d).sum(axis=1), 1.0)

    relevance = ensemble.estimate_channel_relevance()
    assert relevance.shape == (1,)
    assert np.isclose(relevance.sum(), 1.0)


def test_dispatch_validation():
    """Unsupported model_type/backend combinations fail loudly — no silent
    fallback to another implementation."""
    with pytest.raises(ValueError, match="MiniRocket-only"):
        DetachEnsemble(model_type="multirocket", backend="pytorch")
    with pytest.raises(ValueError, match="MiniRocket-only"):
        DetachEnsemble(model_type="multirocket", backend="cuda")
    with pytest.raises(ValueError, match="model_type"):
        DetachEnsemble(model_type="rocket")
    with pytest.raises(ValueError, match="backend"):
        DetachEnsemble(model_type="multirocket", backend="numba")

    with pytest.raises(ValueError, match="num_kernels"):
        DetachEnsemble(model_type="multirocket", num_kernels=600)
    with pytest.raises(ValueError, match="num_kernels"):
        DetachEnsemble(model_type="minirocket", num_kernels=80)

    # Defaults: backend=None resolves to aeon for both model types, and
    # n_jobs=None resolves to -1 (all cores) on the aeon backend.  The
    # pytorch side of the n_jobs resolution is asserted in the torch-guarded
    # test file.
    default = DetachEnsemble(num_models=1, num_kernels=NUM_KERNELS)
    assert default.model_type == "minirocket"
    assert default.backend == "aeon"
    assert default.n_jobs == -1

    multi = DetachEnsemble(num_models=1, num_kernels=NUM_KERNELS, model_type="multirocket")
    assert multi.backend == "aeon"
    assert multi.n_jobs == -1

    assert DetachEnsemble(num_models=1, num_kernels=NUM_KERNELS, n_jobs=2).n_jobs == 2


# -- MiniRocket on the aeon backend --------------------------------------------


@pytest.fixture(scope="module")
def fitted_minirocket_aeon(ensemble_data):
    """Deliberately constructed with all defaults (plus size/pruning): this
    exercises the default aeon backend and n_jobs resolution end-to-end."""
    X, y = ensemble_data
    ensemble = DetachEnsemble(num_models=2, num_kernels=336, set_percentage=50, random_state=0)
    ensemble.fit(X, y)
    return ensemble


def test_minirocket_aeon_members(fitted_minirocket_aeon):
    """The default configuration builds AeonMiniRocket members on all cores;
    num_kernels passes straight through (it already counts features for
    MiniRocket, like the torch/cuda backends' num_features)."""
    assert fitted_minirocket_aeon.model_type == "minirocket"
    assert fitted_minirocket_aeon.backend == "aeon"
    assert fitted_minirocket_aeon.n_jobs == -1
    for model in fitted_minirocket_aeon.derockets:
        assert isinstance(model.transformer, AeonMiniRocket)
        assert model.transformer.n_kernels == 336
        assert model.feature_mask_.size == 336


def test_minirocket_aeon_predict_and_relevance(fitted_minirocket_aeon, ensemble_data):
    X, y = ensemble_data

    proba = fitted_minirocket_aeon.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert fitted_minirocket_aeon.predict(X).shape == y.shape
    assert 0.0 <= fitted_minirocket_aeon.score(X, y) <= 1.0

    relevance = fitted_minirocket_aeon.estimate_channel_relevance()
    assert relevance.shape == (3,)
    assert np.isclose(relevance.sum(), 1.0)
    # Channel 0 carries the class signal, so it should dominate.
    assert relevance.argmax() == 0


def test_minirocket_aeon_reproducibility(ensemble_data):
    """Same random_state must give identical ensemble outputs."""
    X, y = ensemble_data

    kwargs = dict(num_models=2, num_kernels=336, set_percentage=50, backend="aeon", random_state=0)
    e1 = DetachEnsemble(**kwargs)
    e2 = DetachEnsemble(**kwargs)
    e1.fit(X, y)
    e2.fit(X, y)
    assert np.allclose(e1.predict_proba(X), e2.predict_proba(X))

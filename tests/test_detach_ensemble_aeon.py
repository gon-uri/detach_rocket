"""Tests for DetachEnsemble with MultiRocket members.

Deliberately torch-free (unlike tests/test_detach_ensemble.py, which guards
on torch at module level): the aeon MultiRocket path must work without the
optional PyTorch dependency.
"""

import numpy as np
import pytest

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
    ensemble = DetachEnsemble(num_models=2, num_kernels=NUM_KERNELS, set_percentage=50, model_type="multirocket")
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
    with pytest.raises(ValueError, match="channel relevance"):
        DetachEnsemble(model_type="minirocket", backend="aeon")
    with pytest.raises(ValueError, match="model_type"):
        DetachEnsemble(model_type="rocket")
    with pytest.raises(ValueError, match="backend"):
        DetachEnsemble(model_type="multirocket", backend="numba")

    with pytest.raises(ValueError, match="num_kernels"):
        DetachEnsemble(model_type="multirocket", num_kernels=600)
    with pytest.raises(ValueError, match="num_kernels"):
        DetachEnsemble(model_type="minirocket", num_kernels=80)

    # backend=None resolves per model_type (construct only the torch-free one here).
    assert DetachEnsemble(num_models=1, num_kernels=NUM_KERNELS, model_type="multirocket").backend == "aeon"

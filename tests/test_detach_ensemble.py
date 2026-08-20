import numpy as np
import pytest

pytest.importorskip("torch")

from detach_rocket.detach_classes import DetachEnsemble
from detach_rocket.pytorch_minirocket import PytorchMiniRocketMultivariate


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
    ensemble = DetachEnsemble(num_models=2, num_kernels=168, set_percentage=50, backend="pytorch")
    ensemble.fit(X, y)
    return ensemble


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


def test_get_kernel_features_bias_order():
    """Bias values returned per feature must match the feature ordering."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 2, 60)).astype(np.float32)
    # 4 features per kernel so at least one dilation has multiple quantiles.
    transformer = PytorchMiniRocketMultivariate(num_features=336)
    transformer.fit(X)

    mask = np.ones(transformer.num_features, dtype=bool)
    got = transformer.get_kernel_features("biases", mask)

    expected = []
    for i in range(transformer.num_dilations):
        biases = getattr(transformer, f"biases_{i}").cpu().numpy()  # (84, q)
        kernel_idx = getattr(transformer, f"kernel_indices_{i}").cpu().numpy()
        quantile_idx = np.tile(np.arange(biases.shape[1]), transformer.num_kernels)
        expected.append(biases[kernel_idx, quantile_idx])
    assert np.allclose(got, np.concatenate(expected))


def test_get_kernel_features_channels_univariate():
    """Univariate models have no channel combinations; channel 0 is used everywhere."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 1, 60)).astype(np.float32)
    transformer = PytorchMiniRocketMultivariate(num_features=168)
    transformer.fit(X)

    mask = np.ones(transformer.num_features, dtype=bool)
    channels = transformer.get_kernel_features("channels", mask)
    assert channels.shape == (transformer.num_features, 1)
    assert np.all(channels == 1.0)

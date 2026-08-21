import warnings

import numpy as np
from scipy.linalg import LinAlgWarning

from detach_rocket._warnings import quiet_ridge_warnings
from detach_rocket.sfd import feature_detachment

SKLEARN_MSG = "Singular matrix in solving dual problem. Using least-squares solution instead."
SCIPY_MSG = "Ill-conditioned matrix (rcond=1e-09): result may not be accurate."


def test_quiet_context_filters_only_the_expected_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quiet_ridge_warnings():
            warnings.warn(SKLEARN_MSG, UserWarning, stacklevel=2)
            warnings.warn(SCIPY_MSG, LinAlgWarning, stacklevel=2)
            warnings.warn("some other user warning", UserWarning, stacklevel=2)
    assert [str(w.message) for w in caught] == ["some other user warning"]


def test_quiet_context_is_scoped():
    """Outside the context, the warnings surface again."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quiet_ridge_warnings():
            pass
        warnings.warn(SKLEARN_MSG, UserWarning, stacklevel=2)
    assert len(caught) == 1


class _WarningClassifier:
    """Stub classifier whose every fit emits both linear-algebra warnings.

    Whether the warnings actually fire from real ridge fits depends on the
    platform's LAPACK, so the stub makes the check deterministic: if the
    fits inside feature_detachment lose their quiet_ridge_warnings wrapper,
    this test fails on any platform.
    """

    def fit(self, X, y):
        warnings.warn(SKLEARN_MSG, UserWarning, stacklevel=2)
        warnings.warn(SCIPY_MSG, LinAlgWarning, stacklevel=2)
        self.coef_ = np.linspace(1.0, 2.0, X.shape[1])
        return self

    def score(self, X, y):
        return 1.0


def test_feature_detachment_fits_are_wrapped():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 20))
    y = rng.integers(0, 2, 10)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        feature_detachment(_WarningClassifier(), X, y_train=y, X_test=X, y_test=y, num_steps=5)

    assert caught == [], [str(w.message)[:80] for w in caught]

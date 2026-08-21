"""Scoped suppression of expected linear-algebra warnings."""

import warnings
from contextlib import contextmanager

from scipy.linalg import LinAlgWarning


@contextmanager
def quiet_ridge_warnings():
    """Silence the two benign warnings that ridge fits emit in bulk.

    Detach-ROCKET fits ridge classifiers many times on wide feature
    matrices (features >> samples) while scanning regularization values
    down to 1e-10, so these warnings are expected by construction:

    - sklearn's "Singular matrix in solving dual problem" ``UserWarning``:
      sklearn already falls back to an exact least-squares solution.
    - scipy's ``LinAlgWarning`` about ill-conditioned solves: fired mostly
      for the smallest regularization candidates, whose solutions are
      discarded by validation/CV scoring anyway.

    The filters are active only inside this context (the library's own
    fit calls), so identical warnings raised by user code are unaffected.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Singular matrix in solving dual problem",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        yield

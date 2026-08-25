"""aeon MiniRocket subclass with per-feature kernel introspection.

:class:`AeonMiniRocket` adds the ``get_kernel_features`` contract (the one
implemented by the custom PyTorch/CuPy backends) on top of aeon's numba-based
``MiniRocket`` transformer, so it can serve as the fast-CPU MiniRocket backend
of :class:`~detach_rocket.detach_classes.DetachEnsemble`
(``model_type="minirocket", backend="aeon"``) with full channel-relevance
support.  Unlike the PyTorch backend, aeon's transform is numba-parallel on
CPU (the torch CPU path is restricted to one thread by default to avoid
OpenMP deadlocks on fragile stacks).

Feature layout of aeon's MiniRocket output (established from the aeon 1.5
source and verified numerically in ``tests/test_aeon_minirocket.py``):

- One PPV feature per (dilation, kernel, quantile): ``84 * (n_kernels // 84)``
  columns (aeon floors ``n_kernels`` up to at least 84), ordered
  dilation-ascending, then kernel 0..83, then quantile.  There is no parity
  reordering: the alternating padding rule
  ``(dilation_index % 2 + kernel_index) % 2`` only decides whether the pooled
  feature map is cropped.
- Each (dilation, kernel) pair has one channel combination
  (``combination_index = dilation_index * 84 + kernel_index``), shared by its
  quantiles.  There is a single representation, so none of MultiRocket's
  cross-representation subtleties apply.
- The fitted attribute is ``parameters`` (plural — unlike MultiRocket's
  ``parameter``/``parameter1``) and is always the 5-tuple
  ``(n_channels_per_combination, channel_indices, dilations,
  n_features_per_dilation, biases)``.  Univariate fits still draw
  combinations (necessarily all of size 1 on channel 0); the univariate
  transform ignores them, and the channel map is identical either way.
"""

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket

from detach_rocket._aeon_kernel_maps import block_map, block_width


class AeonMiniRocket(MiniRocket):
    """aeon ``MiniRocket`` with per-feature kernel parameter lookup.

    A thin subclass of
    :class:`aeon.transformations.collection.convolution_based.MiniRocket`
    (same constructor, same fitted behavior) that additionally implements
    :meth:`get_kernel_features`, mapping every output feature column to the
    parameters of the kernel that produced it.  This is the contract
    :meth:`DetachEnsemble.estimate_channel_relevance` relies on.

    Notes
    -----
    - The feature *order* differs from
      :class:`~detach_rocket.pytorch_minirocket.PytorchMiniRocketMultivariate`
      (which parity-reorders kernels within each dilation); each backend's
      ``get_kernel_features`` describes its own transform, and the ensemble
      only needs per-member consistency.
    - aeon's ``MiniRocket.fit`` seeds the process-global NumPy RNG when
      ``random_state`` is an int; pass per-member seeds rather than relying
      on ``np.random.seed``.

    Examples
    --------
    >>> transformer = AeonMiniRocket(n_kernels=336, random_state=0)
    >>> features = transformer.fit_transform(X)  # (n_instances, 336)
    >>> mask = np.ones(features.shape[1], dtype=bool)
    >>> channels = transformer.get_kernel_features("channels", mask)
    """

    def _fit(self, X, y=None):
        """Record the channel count, then fit as aeon does."""
        self.n_channels_ = X.shape[1]
        return super()._fit(X, y)

    def get_kernel_features(self, which, where):
        """Return the *which* kernel parameter for every output feature.

        Rows are in exact transform-column order.  Rows whose entry in
        *where* is ``True`` carry the parameter value; the remaining rows
        are ``NaN``.

        Parameters
        ----------
        which : str
            One of ``"biases"``, ``"channels"``, ``"weights"``,
            ``"dilations"``, ``"paddings"``.
        where : array-like of bool of shape (n_features,)
            Feature selection mask, e.g. the retained-feature mask of a
            Detach model.

        Returns
        -------
        features : np.ndarray
            Shape ``(n_features, n_channels)`` for ``"channels"``,
            ``(n_features, 9)`` for ``"weights"``, ``(n_features,)``
            otherwise.
        """
        if getattr(self, "parameters", None) is None:
            raise ValueError("Transformer must be fitted before calling get_kernel_features.")

        valid = ("biases", "channels", "weights", "dilations", "paddings")
        if which not in valid:
            raise ValueError(f'"{which}" is not recognized as a feature. Possible features are {valid}.')

        num_features = block_width(self.parameters)
        where = np.asarray(where, dtype=bool)
        if where.shape != (num_features,):
            raise ValueError(f"where must be a boolean array of shape ({num_features},); got {where.shape}.")

        full_features = block_map(which, self.parameters, self.parameters, self.n_channels_, MiniRocket._indices)
        if full_features.ndim == 2:
            where = np.repeat(where[:, np.newaxis], full_features.shape[1], axis=1)
        return np.where(where, full_features, np.nan)

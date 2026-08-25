"""aeon MultiRocket subclass with per-feature kernel introspection.

:class:`AeonMultiRocket` adds the ``get_kernel_features`` contract (the one
implemented by the custom MiniRocket backends) on top of aeon's numba-based
``MultiRocket`` transformer, so it can serve as a :class:`~detach_rocket.detach_classes.DetachEnsemble`
member with full channel-relevance support.

Feature layout of aeon's MultiRocket output (established from the aeon 1.5
source and verified numerically in ``tests/test_aeon_multirocket.py``):

- The output has 8 contiguous blocks of equal width ``n = 84 * (n_kernels // 84)``:
  ``[base: PPV | LSPV | MPV | MIPV][diff: PPV | LSPV | MPV | MIPV]``, where
  "base" is the raw series and "diff" its first-order difference.  Note that
  the *storage* order of the pooling operators is PPV, LSPV, MPV, MIPV.
- Within each block, features are ordered dilation-ascending, then kernel
  0..83, then quantile.  There is no parity reordering: the alternating
  padding rule only decides whether the pooled feature map is cropped.
- Each (dilation, kernel) pair has one channel combination, shared by its
  quantiles and by all 4 pooling operators.  Although the fit draws an
  independent set of combinations for the diff representation (stored in
  ``parameter1``), aeon's transform discards them and reuses the *base*
  combinations for both halves (``_transform_multi`` unpacks ``parameters1``
  as ``_, _, dilations1, n_features_per_dilation1, biases1``); the diff
  combinations only influence bias fitting.  The channel map here reflects
  what the transform computes, not what the fit draws.  (The diff half can
  never have more dilations than the base half — its series is shorter and
  ``_fit_dilations`` is monotonic in length — so indexing base combinations
  with diff dilation indices is always in range.)
- One bias per feature serves all 4 pooling operators of that feature.
- The diff half's convolution is not exactly the symmetric "same" pattern:
  aeon reuses ``end = n_timepoints - padding`` on the length ``L - 1`` diff
  arrays, so taps left of center read one sample to the right of the
  symmetric position.  This does not affect the feature maps, only exact
  reconstructions of the transform (see the layout test).
"""

import numpy as np
from aeon.transformations.collection.convolution_based import MultiRocket

from detach_rocket._aeon_kernel_maps import block_map, block_width


class AeonMultiRocket(MultiRocket):
    """aeon ``MultiRocket`` with per-feature kernel parameter lookup.

    A thin subclass of
    :class:`aeon.transformations.collection.convolution_based.MultiRocket`
    (same constructor, same fitted behavior) that additionally implements
    :meth:`get_kernel_features`, mapping every output feature column to the
    parameters of the kernel that produced it.  This is the contract
    :meth:`DetachEnsemble.estimate_channel_relevance` relies on.

    Notes
    -----
    - ``n_features_per_kernel`` must remain at its default of 4: aeon's
      transform kernels hardcode the four pooling operators (PPV, LSPV,
      MPV, MIPV), and this class' feature maps assume that layout.
    - Requires ``n_timepoints >= 10`` so the first-order difference series
      keeps the 9 points a kernel needs (aeon only validates the raw series).
    - aeon's ``MultiRocket.fit`` seeds the process-global NumPy RNG when
      ``random_state`` is an int; pass per-member seeds rather than relying
      on ``np.random.seed``.

    Examples
    --------
    >>> transformer = AeonMultiRocket(n_kernels=672, random_state=0)
    >>> features = transformer.fit_transform(X)  # (n_instances, 5376)
    >>> mask = np.ones(features.shape[1], dtype=bool)
    >>> channels = transformer.get_kernel_features("channels", mask)
    """

    #: Storage order of the pooling-operator blocks in the transform output,
    #: matching the integer codes returned by ``get_kernel_features("poolings")``.
    POOLING_NAMES = ("ppv", "lspv", "mpv", "mipv")

    def _fit(self, X, y=None):
        """Record the channel count, validate the input, and fit as aeon does."""
        if self.n_features_per_kernel != 4:
            raise ValueError(
                "AeonMultiRocket requires n_features_per_kernel=4: aeon's MultiRocket transform hardcodes "
                "the four pooling operators (PPV, LSPV, MPV, MIPV), which get_kernel_features relies on; "
                f"got n_features_per_kernel={self.n_features_per_kernel}."
            )
        if X.shape[2] < 10:
            raise ValueError(
                "AeonMultiRocket requires n_timepoints >= 10 so that the first-order difference series "
                f"keeps at least the 9 timepoints a kernel spans; got {X.shape[2]}."
            )
        self.n_channels_ = X.shape[1]
        return super()._fit(X, y)

    # -- Per-feature kernel parameter lookup ---------------------------------

    def get_kernel_features(self, which, where):
        """Return the *which* kernel parameter for every output feature.

        Rows are in exact transform-column order.  Rows whose entry in
        *where* is ``True`` carry the parameter value; the remaining rows
        are ``NaN``.

        Parameters
        ----------
        which : str
            One of ``"biases"``, ``"channels"``, ``"weights"``,
            ``"dilations"``, ``"paddings"`` (the contract shared with the
            MiniRocket backends), or the MultiRocket-specific
            ``"poolings"`` (float codes indexing :attr:`POOLING_NAMES`:
            0=PPV, 1=LSPV, 2=MPV, 3=MIPV) and ``"representations"``
            (0 = raw series, 1 = first-order difference).
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
        if self.parameter is None or self.parameter1 is None:
            raise ValueError("Transformer must be fitted before calling get_kernel_features.")

        valid = ("biases", "channels", "weights", "dilations", "paddings", "poolings", "representations")
        if which not in valid:
            raise ValueError(f'"{which}" is not recognized as a feature. Possible features are {valid}.')

        n_base = block_width(self.parameter)
        n_diff = block_width(self.parameter1)
        num_features = 4 * (n_base + n_diff)

        where = np.asarray(where, dtype=bool)
        if where.shape != (num_features,):
            raise ValueError(f"where must be a boolean array of shape ({num_features},); got {where.shape}.")

        if which == "poolings":
            # Block-level label: 4 pooling blocks per representation, in storage order.
            full_features = np.concatenate(
                [np.repeat(np.arange(4, dtype=float), n_base), np.repeat(np.arange(4, dtype=float), n_diff)]
            )
        elif which == "representations":
            full_features = np.repeat(np.array([0.0, 1.0]), [4 * n_base, 4 * n_diff])
        else:
            # The same per-representation map serves all 4 pooling blocks of
            # that representation.  Dilations, biases, and quantile counts
            # come from each representation's own tuple, but the channel
            # combinations always come from the BASE tuple: aeon's transform
            # discards parameter1's combinations (see the module docstring).
            indices = MultiRocket._indices
            base_block = block_map(which, self.parameter, self.parameter, self.n_channels_, indices)
            diff_block = block_map(which, self.parameter1, self.parameter, self.n_channels_, indices)
            reps = (4,) if base_block.ndim == 1 else (4, 1)
            full_features = np.concatenate([np.tile(base_block, reps), np.tile(diff_block, reps)])

        if full_features.ndim == 2:
            where = np.repeat(where[:, np.newaxis], full_features.shape[1], axis=1)
        return np.where(where, full_features, np.nan)

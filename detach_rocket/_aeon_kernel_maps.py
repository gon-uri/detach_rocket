"""Shared per-feature kernel-map builders for the aeon-backed transformers.

The map-building logic is identical for aeon's MiniRocket and MultiRocket
(MultiRocket's transform is a direct extension of MiniRocket's): the features
of one pooling block are ordered dilation-ascending, then kernel 0..83, then
quantile, and each (dilation, kernel) pair owns one channel combination
(``combination_index = dilation_index * 84 + kernel_index``).  How blocks are
assembled into the full output differs per transformer and stays in the
respective class.
"""

import numpy as np

N_KERNELS = 84  # fixed MiniRocket/MultiRocket kernel count
KERNEL_SIZE = 9


def block_width(params):
    """Number of features in one pooling block described by *params*."""
    return N_KERNELS * int(np.sum(params[-2]))


def block_map(which, params, combo_params, n_channels, indices):
    """Build the per-feature parameter map for one pooling block.

    Feature order within a block is dilation-ascending, then kernel 0..83,
    then quantile — exactly the order in which aeon's transforms write the
    columns.

    Parameters
    ----------
    which : str
        One of ``"biases"``, ``"channels"``, ``"weights"``, ``"dilations"``,
        ``"paddings"``.  Callers validate before calling.
    params : tuple
        A fitted aeon parameter tuple ending in ``(..., dilations,
        n_features_per_dilation, biases)`` — 5 elements when channel
        combinations were drawn, 3 for MultiRocket's univariate fit.
        Supplies the layout (dilations, quantile counts) and the biases.
    combo_params : tuple
        The tuple supplying the channel combinations (elements 0 and 1 when
        it has 5 elements).  For MultiRocket this is always the BASE
        ``parameter``: aeon's transform applies the base combinations to
        both representations.
    n_channels : int
        Number of input channels the transformer was fitted on.
    indices : np.ndarray of shape (84, 3)
        The fixed kernel index triplets (the transformer's ``_indices``).

    Returns
    -------
    block : np.ndarray
        Shape ``(block_width, n_channels)`` for ``"channels"``,
        ``(block_width, 9)`` for ``"weights"``, ``(block_width,)`` otherwise.
    """
    dilations = np.asarray(params[-3], dtype=int)
    npd = np.asarray(params[-2], dtype=int)

    if which == "biases":
        # aeon fits biases in (dilation, kernel, quantile) order already.
        return np.asarray(params[-1], dtype=float)
    if which == "dilations":
        return np.repeat(dilations, N_KERNELS * npd).astype(float)
    if which == "paddings":
        return np.repeat((KERNEL_SIZE - 1) * dilations // 2, N_KERNELS * npd).astype(float)

    kernel_of_feature = np.concatenate([np.repeat(np.arange(N_KERNELS), q) for q in npd])

    if which == "weights":
        weights = np.full((N_KERNELS, KERNEL_SIZE), -1.0)
        weights[np.arange(N_KERNELS)[:, np.newaxis], indices] = 2.0
        return weights[kernel_of_feature]

    # which == "channels"
    if len(combo_params) != 5:  # MultiRocket univariate fit: no combinations are drawn
        return np.ones((N_KERNELS * int(npd.sum()), 1), dtype=float)
    n_channels_per_combination, channel_indices = combo_params[0], combo_params[1]

    combination_of_feature = np.concatenate(
        [np.repeat(d * N_KERNELS + np.arange(N_KERNELS), npd[d]) for d in range(len(dilations))]
    )
    n_combinations = len(n_channels_per_combination)
    indicator = np.zeros((n_combinations, n_channels), dtype=float)
    rows = np.repeat(np.arange(n_combinations), n_channels_per_combination)
    indicator[rows, channel_indices] = 1.0
    return indicator[combination_of_feature]

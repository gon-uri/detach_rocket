"""
Transformer pruning utilities for ROCKET-family models.
"""

from abc import ABC, abstractmethod

import numpy as np
from sktime.transformations.panel.rocket import Rocket


def _normalize_feature_mask(feature_mask):
    """Validate and normalize a boolean feature mask."""
    mask = np.asarray(feature_mask)
    if mask.ndim != 1:
        raise ValueError("optimal_feature_mask must be a 1D boolean array.")
    if mask.size == 0:
        raise ValueError("optimal_feature_mask must contain at least one feature.")
    if mask.dtype != np.bool_:
        unique_values = np.unique(mask)
        if np.all(np.isin(unique_values, [0, 1])):
            mask = mask.astype(bool, copy=False)
        else:
            raise ValueError("optimal_feature_mask must contain only boolean (or 0/1) values.")
    return mask


def get_transformer_pruner(transformer):
    """Return the appropriate :class:`TransformerPruner` for *transformer*.

    If a specialized pruner is not available for the transformer type,
    returns a :class:`GenericTransformerPruner` that wraps the original
    transformer and applies feature masking at transform time.

    Parameters
    ----------
    transformer : sktime transformer
        A fitted ROCKET-family transformer.

    Returns
    -------
    pruner : TransformerPruner
    """
    if isinstance(transformer, Rocket):
        return RocketTransformerPruner()
    # elif isinstance(transformer, MiniRocketTransformer):
    #     return MiniRocketTransformerPruner()
    else:
        return GenericTransformerPruner()


class PrunedRocket(Rocket):
    """A pruned Rocket transformer that outputs only retained features.

    Inherits from :class:`~sktime.transformations.panel.rocket.Rocket`
    but overrides ``transform`` to apply the full convolution and then
    select only the columns indicated by ``features_mask``.

    Parameters
    ----------
    num_kernels : int
        Number of retained kernels.
    features_mask : np.ndarray of bool
        Boolean mask of length ``2 * num_kernels`` indicating which
        of the two features (PPV and max) per kernel are retained.
    """

    _tags = {"fit_is_empty": True}

    def __init__(self, num_kernels, features_mask):
        super().__init__(num_kernels=num_kernels)
        self.features_mask = features_mask
        self._is_fitted = True

    def transform(self, X):
        """Transform *X* and return only the retained feature columns."""
        X_transf = super().transform(X)
        X_transf = X_transf.to_numpy()
        return X_transf[:, self.features_mask]


class TransformerPruner(ABC):
    """Abstract base class for transformer pruners.

    Subclasses must implement :meth:`prune_transformer` for a specific
    ROCKET-family transformer type.
    """

    @abstractmethod
    def prune_transformer(self, original_transformer, optimal_feature_mask):
        """Create a pruned copy of *original_transformer*.

        Parameters
        ----------
        original_transformer : sktime transformer
            A **fitted** ROCKET-family transformer.
        optimal_feature_mask : np.ndarray of bool
            Boolean mask indicating which features to retain.

        Returns
        -------
        pruned_transformer
            A new transformer that outputs only the retained features.
        """


class GenericPrunedTransformer:
    """Wrapper for any transformer that applies feature masking.

    This is a fallback for transformers that don't have specialized
    pruning logic.  It keeps the original transformer intact and
    applies the feature mask during :meth:`transform`.

    Parameters
    ----------
    original_transformer : object
        A fitted transformer (typically from sktime or similar).
    feature_mask : np.ndarray of bool
        Boolean mask indicating which output features to retain.

    Attributes
    ----------
    num_kernels : int or None
        Number of retained kernels when known, otherwise ``None`` for
        generic transformer outputs where kernel semantics are undefined.
    """

    def __init__(self, original_transformer, feature_mask):
        self.original_transformer = original_transformer
        self.feature_mask = _normalize_feature_mask(feature_mask)
        self.num_kernels = None

    def transform(self, X):
        """Transform *X* with the original transformer and mask features."""
        X_transf = self.original_transformer.transform(X)
        # Handle pandas DataFrame or numpy array
        if hasattr(X_transf, "to_numpy"):
            X_transf = X_transf.to_numpy()
        else:
            X_transf = np.asarray(X_transf)
        if X_transf.ndim != 2:
            raise ValueError(
                "Fallback pruned transformer expects transform(X) to return a 2D feature matrix."
            )
        if X_transf.shape[1] != self.feature_mask.size:
            raise ValueError(
                "Feature mask length does not match transformer output width: "
                f"mask has {self.feature_mask.size} features but transform(X) produced "
                f"{X_transf.shape[1]}."
            )
        return X_transf[:, self.feature_mask]


class GenericTransformerPruner(TransformerPruner):
    """Fallback pruner for transformers without specialized logic.

    Returns a :class:`GenericPrunedTransformer` that wraps the original
    transformer and applies feature masking.
    """

    def prune_transformer(self, original_transformer, optimal_feature_mask):
        """Create a masked wrapper around *original_transformer*.

        Parameters
        ----------
        original_transformer : object
            A fitted transformer.
        optimal_feature_mask : np.ndarray of bool
            Boolean mask indicating which features to retain.

        Returns
        -------
        pruned_transformer : GenericPrunedTransformer
        """
        if original_transformer is None:
            raise ValueError("Transformer must not be None.")
        if not callable(getattr(original_transformer, "transform", None)):
            raise ValueError(
                "Unsupported transformer for generic fallback: missing callable transform(X)."
            )
        mask = _normalize_feature_mask(optimal_feature_mask)
        return GenericPrunedTransformer(original_transformer, mask)


class RocketTransformerPruner(TransformerPruner):
    """Pruner for :class:`~sktime.transformations.panel.rocket.Rocket`.

    Extracts the kernel parameters (weights, biases, dilations, paddings,
    channel indices) corresponding to retained features and builds a
    :class:`PrunedRocket` instance.
    """

    def prune_transformer(self, original_trf, optimal_feature_mask):
        """Create a pruned Rocket transformer.

        Parameters
        ----------
        original_trf : Rocket
            A **fitted** Rocket transformer (must have ``kernels``
            attribute).
        optimal_feature_mask : np.ndarray of bool
            Boolean mask of length ``2 * num_kernels``.

        Returns
        -------
        pruned_trf : PrunedRocket

        Raises
        ------
        ValueError
            If *original_trf* has not been fitted.
        """

        # check if transformer is fit
        if not hasattr(original_trf, "kernels"):
            raise ValueError("Transformer must be fit before pruning")

        num_kernels = original_trf.num_kernels
        optimal_feature_mask = _normalize_feature_mask(optimal_feature_mask)
        expected_mask_size = 2 * num_kernels
        if optimal_feature_mask.size != expected_mask_size:
            raise ValueError(
                "Rocket feature mask size mismatch: "
                f"expected {expected_mask_size} features for {num_kernels} kernels, "
                f"got {optimal_feature_mask.size}."
            )

        # Precompute number of pruned kernels
        retained_num_kernels = np.sum(optimal_feature_mask[0::2] | optimal_feature_mask[1::2])

        # Preallocate arrays with the exact number of retained kernels
        retained_mask = np.full(2 * retained_num_kernels, True)
        retained_weights = np.zeros(original_trf.kernels[0].shape[0], dtype=np.float32)  # Adjust size later in the loop
        retained_lengths = np.zeros(retained_num_kernels, dtype=np.int32)
        retained_biases = np.zeros(retained_num_kernels, dtype=np.float32)
        retained_dilations = np.zeros(retained_num_kernels, dtype=np.int32)
        retained_paddings = np.zeros(retained_num_kernels, dtype=np.int32)
        retained_num_channel_indices = np.zeros(retained_num_kernels, dtype=np.int32)
        retained_channel_indices = np.zeros(
            original_trf.kernels[6].shape[0], dtype=np.int32
        )  # Adjust size later in the loop

        a1 = 0  # for weights
        a2 = 0  # for channel_indices

        i_retained = 0
        a1_retained = 0  # for retained_weights
        a2_retained = 0  # for retained_channel_indices

        for i in range(num_kernels):
            _length = original_trf.kernels[1][i]
            _num_channels_indices = original_trf.kernels[5][i]

            b1 = a1 + (_num_channels_indices * _length)
            b2 = a2 + _num_channels_indices

            # optimal_feature_maske i or i+1 should be selected
            if optimal_feature_mask[2 * i] or optimal_feature_mask[2 * i + 1]:
                retained_mask[2 * i_retained] = optimal_feature_mask[2 * i]
                retained_mask[2 * i_retained + 1] = optimal_feature_mask[2 * i + 1]

                retained_weights[a1_retained : a1_retained + (b1 - a1)] = original_trf.kernels[0][a1:b1]
                retained_channel_indices[a2_retained : a2_retained + (b2 - a2)] = original_trf.kernels[6][a2:b2]

                retained_lengths[i_retained] = _length
                retained_biases[i_retained] = original_trf.kernels[2][i]
                retained_dilations[i_retained] = original_trf.kernels[3][i]
                retained_paddings[i_retained] = original_trf.kernels[4][i]
                retained_num_channel_indices[i_retained] = _num_channels_indices

                a1_retained += b1 - a1
                a2_retained += b2 - a2
                i_retained += 1

            a1 = b1
            a2 = b2

        retained_weights = retained_weights[:a1_retained]
        retained_channel_indices = retained_channel_indices[:a2_retained]

        # define new retained transformation
        pruned_trf = PrunedRocket(retained_num_kernels, retained_mask)
        # define kernels, they will not exist if it was not fit. They are tuple
        pruned_trf.kernels = (
            retained_weights,
            retained_lengths,
            retained_biases,
            retained_dilations,
            retained_paddings,
            retained_num_channel_indices,
            retained_channel_indices,
        )

        return pruned_trf

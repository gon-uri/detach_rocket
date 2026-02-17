"""
Transformer pruning utilities for ROCKET-family models.
"""

from abc import ABC, abstractmethod
from sktime.transformations.panel.rocket import (
    Rocket,
    MiniRocketMultivariate,
    MultiRocketMultivariate
)

import numpy as np


def get_transformer_pruner(transformer):
    """Return the appropriate :class:`TransformerPruner` for *transformer*.

    Parameters
    ----------
    transformer : sktime transformer
        A fitted ROCKET-family transformer.

    Returns
    -------
    pruner : TransformerPruner

    Raises
    ------
    ValueError
        If no pruner is available for the given transformer type.
    """
    if isinstance(transformer, Rocket):
        return RocketTransformerPruner()
    # elif isinstance(transformer, MiniRocketTransformer):
    #     return MiniRocketTransformerPruner()
    else:
        raise ValueError(f"No pruner available for transformer type: {type(transformer)}")
    

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
        if not hasattr(original_trf, 'kernels'):
            raise ValueError("Transformer must be fit before pruning")
        
        num_kernels = original_trf.num_kernels

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
        retained_channel_indices = np.zeros(original_trf.kernels[6].shape[0], dtype=np.int32)  # Adjust size later in the loop

        a1 = 0 # for weights
        a2 = 0 # for channel_indices

        i_retained = 0
        a1_retained = 0 # for retained_weights
        a2_retained = 0 # for retained_channel_indices

        for i in range(num_kernels):
            _length = original_trf.kernels[1][i]
            _num_channels_indices = original_trf.kernels[5][i]
            
            b1 = a1 + (_num_channels_indices * _length)
            b2 = a2 + _num_channels_indices
            
            # optimal_feature_maske i or i+1 should be selected
            if optimal_feature_mask[2 * i] or optimal_feature_mask[2 * i + 1]:

                retained_mask[2 * i_retained] = optimal_feature_mask[2 * i]
                retained_mask[2 * i_retained+1] = optimal_feature_mask[2 * i + 1]

                retained_weights[a1_retained:a1_retained + (b1 - a1)] = original_trf.kernels[0][a1:b1]
                retained_channel_indices[a2_retained:a2_retained + (b2 - a2)] = original_trf.kernels[6][a2:b2]

                retained_lengths[i_retained] = _length
                retained_biases[i_retained] = original_trf.kernels[2][i]
                retained_dilations[i_retained] = original_trf.kernels[3][i]
                retained_paddings[i_retained] = original_trf.kernels[4][i]
                retained_num_channel_indices[i_retained] = _num_channels_indices
                
                a1_retained += (b1 - a1)
                a2_retained += (b2 - a2)
                i_retained += 1
            
            a1 = b1
            a2 = b2

        retained_weights = retained_weights[:a1_retained]
        retained_channel_indices = retained_channel_indices[:a2_retained]    

        
        # define new retained transformation
        pruned_trf = PrunedRocket(retained_num_kernels, 
                                  retained_mask 
                                )   
        # define kernels, they will not exist if it was not fit. They are tuple
        pruned_trf.kernels = (retained_weights, 
                            retained_lengths, 
                            retained_biases, 
                            retained_dilations, 
                            retained_paddings, 
                            retained_num_channel_indices, 
                            retained_channel_indices
                            )

        return pruned_trf

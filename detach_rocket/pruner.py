

from sktime.transformations.panel.rocket import (
    Rocket,
    MiniRocketMultivariate,
    MultiRocketMultivariate
)

import numpy as np

def get_transformer_pruner(transformer):
    if isinstance(transformer, Rocket):
        return RocketTransformerPruner()
    # elif isinstance(transformer, MiniRocketTransformer):
    #     return MiniRocketTransformerPruner()
    else:
        raise ValueError(f"No pruner available for transformer type: {type(transformer)}")
    

# define a PrunedRocket class, which inherits everything from Rocket but replace the transform method
class PrunedRocket(Rocket):
    _tags = {"fit_is_empty": True}
    def __init__(self, num_kernels, features_mask):
        super().__init__(num_kernels=num_kernels)
        self.features_mask = features_mask # cases where PPV or max are not used
        self._is_fitted = True

    def transform(self, X):        
        X_transf=super().transform(X) # dataframe
        X_transf = X_transf.to_numpy()
        return X_transf[:, self.features_mask]

    
class TransformerPruner:
    """
    Base class for pruning a transformer.
    Subclasses should implement the `prune_transformer` method.
    """
    def prune_transformer(self, original_transformer, optimal_feature_mask):
        raise NotImplementedError("Subclasses must implement this method")

class RocketTransformerPruner(TransformerPruner):
    def prune_transformer(self, original_trf, optimal_feature_mask):

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
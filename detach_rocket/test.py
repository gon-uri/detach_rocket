from sktime.transformations.panel.rocket import Rocket
from sktime.datasets import load_unit_test
import numpy as np
from detach_rocket.pruner import PrunedRocket

X_train, y_train = load_unit_test(split="train")
X_test, y_test = load_unit_test(split="test")
trf = Rocket(num_kernels=512)
trf.fit(X_train)

print(trf.kernels[0].shape) # weights
print(trf.kernels[1].shape) # lenghts
print(trf.kernels[2].shape) # biases
print(trf.kernels[3].shape) # dilations
print(trf.kernels[4].shape) # paddings
print(trf.kernels[5].shape) # num_channel_indices
print(trf.kernels[6].shape) # channel_indices

print(trf.num_kernels)

num_kernels = trf.num_kernels

# build fake optimial_feature_mask, randomly select X% of the features
optimal_feature_mask = np.random.choice([0, 1], size=2*num_kernels, p=[0.5, 0.5])
print(f'optimal_feature_mask: {optimal_feature_mask.shape}')


# Precompute number of pruned kernels
retained_num_kernels = np.sum(optimal_feature_mask[0::2] | optimal_feature_mask[1::2])

# Preallocate arrays with the exact number of retained kernels
retained_mask = np.zeros(2 * retained_num_kernels)
retained_weights = np.zeros(trf.kernels[0].shape[0])  # Adjust size later in the loop
retained_lengths = np.zeros(retained_num_kernels, dtype=np.int32)
retained_biases = np.zeros(retained_num_kernels, dtype=np.float32)
retained_dilations = np.zeros(retained_num_kernels, dtype=np.int32)
retained_paddings = np.zeros(retained_num_kernels, dtype=np.int32)
retained_num_channel_indices = np.zeros(retained_num_kernels, dtype=np.int32)
retained_channel_indices = np.zeros(trf.kernels[6].shape[0])  # Adjust size later in the loop

a1 = 0 # for weights
a2 = 0 # for channel_indices

i_retained = 0
a1_retained = 0 # for retained_weights
a2_retained = 0 # for retained_channel_indices

for i in range(num_kernels):
    _length = trf.kernels[1][i]
    _num_channels_indices = trf.kernels[5][i]
    
    b1 = a1 + (_num_channels_indices * _length)
    b2 = a2 + _num_channels_indices
    
    # optimal_feature_maske i or i+1 should be selected
    if optimal_feature_mask[2 * i] or optimal_feature_mask[2 * i + 1]:

        retained_mask[2 * i_retained] = optimal_feature_mask[2 * i]
        retained_mask[2 * i_retained+1] = optimal_feature_mask[2 * i + 1]

        retained_weights[a1_retained:a1_retained + (b1 - a1)] = trf.kernels[0][a1:b1]
        retained_channel_indices[a2_retained:a2_retained + (b2 - a2)] = trf.kernels[6][a2:b2]

        retained_lengths[i_retained] = _length
        retained_biases[i_retained] = trf.kernels[2][i]
        retained_dilations[i_retained] = trf.kernels[3][i]
        retained_paddings[i_retained] = trf.kernels[4][i]
        
        a1_retained += (b1 - a1)
        a2_retained += (b2 - a2)
        i_retained += 1
    
    a1 = b1
    a2 = b2

retained_weights = retained_weights[:a1_retained]
retained_channel_indices = retained_channel_indices[:a2_retained]


    

# define new retained transformation
pruned_trf = PrunedRocket(retained_num_kernels, retained_mask)
# define kernels, they will not exist if it was not fit. They are tuple
pruned_trf.kernels = (retained_weights, 
                      retained_lengths, 
                      retained_biases, 
                      retained_dilations, 
                      retained_paddings, 
                      retained_num_channel_indices, 
                      retained_channel_indices
                    )

print(pruned_trf.num_kernels)

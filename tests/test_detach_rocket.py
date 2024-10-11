import pytest
import numpy as np
from sktime.transformations.panel.rocket import Rocket
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from detach_rocket.detach_classes import DetachRocket
from detach_rocket.pruner import PrunedRocket, get_transformer_pruner


@pytest.fixture(scope="module")
def data():
    """Fixture that creates a simple dataset and initializes DetachRocket."""

    n_timepoints = 50
    n_dims = 3
    n_samples = 100
    X_multivariate = np.random.randn(n_samples, n_dims, n_timepoints)
    y_multivariate = np.random.randint(0, 2, n_samples)

    # Create a simple classification dataset
    X_train, X_val, y_train, y_val = train_test_split(X_multivariate, y_multivariate, test_size=0.2, random_state=42)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }

@pytest.fixture(scope="function")
def detach_rocket():
    # Initialize a Rocket transformer with 512 kernels
    rocket_transformer = Rocket(num_kernels=512)
    # Initialize the DetachRocket class
    detach_rocket = DetachRocket(
        transformer=rocket_transformer,
        # trade_off=0.1,
        set_percentage=50,  # Fixed pruning percentage
        verbose=True
    )
    return detach_rocket

# def test_fit_with_fixed_pruning(data):
#     """Test the fit function with a fixed pruning percentage."""
#     # Call the fit method
#     data["detach_rocket"].fit(data["X_train"], data["y_train"], X_val=data["X_val"], y_val=data["y_val"])

#     # Assert that the pruned transformer was created
#     assert data["detach_rocket"]._pruned_transformer is not None, "Pruned transformer should be initialized"

#     # Check that the pruned feature matrix has the expected shape
#     expected_num_features = np.sum(data["detach_rocket"]._optimal_feature_mask)
#     pruned_feature_matrix_shape = data["detach_rocket"]._pruned_feature_matrix.shape
#     assert pruned_feature_matrix_shape[1] == expected_num_features, "Pruned feature matrix shape mismatch"


def test_fit_with_optimal_pruning(detach_rocket, data):
    """Test the fit function without a fixed pruning percentage, allowing optimal selection."""
    # Set set_percentage to None to allow optimal pruning
    detach_rocket._set_percentage = None

    # Call the fit method
    detach_rocket.fit(data["X_train"], data["y_train"], X_val=data["X_val"], y_val=data["y_val"])

    # Assert that the pruned transformer was created
    assert detach_rocket._pruned_transformer is not None, "Pruned transformer should be initialized"

    # Check that the optimal feature mask has been generated
    assert detach_rocket._optimal_feature_mask is not None, "Optimal feature mask should be initialized"

    # Assert that the max index and max percentage were calculated correctly
    assert detach_rocket._max_index >= 0, "Max index should be non-negative"
    assert detach_rocket._max_percentage >= 0, "Max percentage should be non-negative"

def test_rocket_pruner(detach_rocket, data):

    """Test that the pruned transformer is correctly created."""

    # Call fit to initialize the pruned transformer
    detach_rocket.fit(data["X_train"], data["y_train"])

    # Assert that the pruned transformer is an instance of PrunedRocket
    pruned_transformer = detach_rocket._pruned_transformer
    assert isinstance(pruned_transformer, PrunedRocket), "Pruned transformer should be a Rocket instance"
    
    # Assert that the pruned transformer has the correct number of kernels
    retained_num_kernels = np.sum(detach_rocket._optimal_feature_mask[0::2] | detach_rocket._optimal_feature_mask[1::2])
    assert pruned_transformer.num_kernels == retained_num_kernels, "Pruned transformer kernel count mismatch"
    
    # Ensure that pruned transformer can transform data
    pruned_features = pruned_transformer.transform(data["X_train"])
    assert pruned_features.shape[1] == np.sum(detach_rocket._optimal_feature_mask), "Pruned features shape mismatch"


def test_invalid_pruning():
    """Test the case where an invalid pruning condition arises."""
    # Test that the pruner raises an error for an invalid transformer
    with pytest.raises(ValueError):
        pruner = get_transformer_pruner("InvalidTransformer")
        pruner.prune_transformer(None, np.array([True, False]))


# check if result is near with threshold  0.001


import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import Rocket

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
    pytest.importorskip("numba")
    # Initialize a Rocket transformer with 512 kernels
    rocket_transformer = Rocket(num_kernels=512)
    # Initialize the DetachRocket class
    detach_rocket = DetachRocket(
        transformer=rocket_transformer,
        # trade_off=0.1,
        set_percentage=50,  # Fixed pruning percentage
        verbose=True,
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
    detach_rocket.set_percentage = None

    # Call the fit method
    detach_rocket.fit(data["X_train"], data["y_train"], X_val=data["X_val"], y_val=data["y_val"])

    # Assert that the pruned transformer was created
    assert detach_rocket.pruned_transformer_ is not None, "Pruned transformer should be initialized"

    # Check that the optimal feature mask has been generated
    assert detach_rocket.feature_mask_ is not None, "Optimal feature mask should be initialized"

    # Assert that the max index and max percentage were calculated correctly
    assert detach_rocket.selected_step_index_ >= 0, "Max index should be non-negative"
    assert detach_rocket.selected_ratio_ >= 0, "Max percentage should be non-negative"


def test_rocket_pruner(detach_rocket, data):
    """Test that the pruned transformer is correctly created."""

    # Call fit to initialize the pruned transformer
    detach_rocket.fit(data["X_train"], data["y_train"])

    # Assert that the pruned transformer is an instance of PrunedRocket
    pruned_transformer = detach_rocket.pruned_transformer_
    assert isinstance(pruned_transformer, PrunedRocket), "Pruned transformer should be a Rocket instance"

    # Assert that the pruned transformer has the correct number of kernels
    retained_num_kernels = np.sum(detach_rocket.feature_mask_[0::2] | detach_rocket.feature_mask_[1::2])
    assert pruned_transformer.num_kernels == retained_num_kernels, "Pruned transformer kernel count mismatch"

    # Ensure that pruned transformer can transform data
    pruned_features = pruned_transformer.transform(data["X_train"])
    assert pruned_features.shape[1] == np.sum(detach_rocket.feature_mask_), "Pruned features shape mismatch"


def test_invalid_pruning():
    """Test that unsupported transformers fall back to generic pruning."""
    # Create a mock transformer (not a Rocket instance)
    class MockTransformer:
        def transform(self, X):
            # Return a simple dummy transformation (4 features)
            return np.random.rand(X.shape[0], 4)

    mock_transformer = MockTransformer()
    feature_mask = np.array([True, False, True, False])

    # Should return a GenericTransformerPruner (no error)
    pruner = get_transformer_pruner(mock_transformer)
    pruned = pruner.prune_transformer(mock_transformer, feature_mask)

    # Verify it's a GenericPrunedTransformer
    from detach_rocket.pruner import GenericPrunedTransformer

    assert isinstance(pruned, GenericPrunedTransformer), "Expected GenericPrunedTransformer fallback"

    # Verify it masks features correctly
    X_dummy = np.random.rand(10, 5)  # Dummy input
    X_pruned = pruned.transform(X_dummy)
    assert X_pruned.shape[1] == 2, "Expected 2 retained features"


def test_get_summary(detach_rocket, data):
    detach_rocket.set_percentage = None
    detach_rocket.fit(data["X_train"], data["y_train"], X_val=data["X_val"], y_val=data["y_val"])

    summary = detach_rocket.get_summary()

    assert summary["estimator"] == "DetachRocket"
    assert summary["is_fitted"] is True
    assert summary["selected_feature_count"] <= summary["full_feature_count"]
    assert 0 <= summary["selected_ratio"] <= 1
    assert summary["retained_kernel_count"] == detach_rocket.pruned_transformer_.num_kernels
    assert summary["final_model_alpha"] > 0

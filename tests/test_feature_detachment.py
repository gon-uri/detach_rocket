import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from detach_rocket.utils import feature_detachment  # Import the function to test

def test_feature_detachment_with_validation():
    # data
    n_samples = 200
    n_relevant_features = 5
    n_irrelevant_features = 5
    drop_ratio=0.1
    steps=1/drop_ratio

    X_relevant = np.random.randn(n_samples,n_relevant_features)
    X_irrelevant = np.random.rand(n_samples, n_irrelevant_features)
    X = np.hstack((X_relevant, X_irrelevant))
    y = np.where(X[:, :n_relevant_features].sum(axis=1) > 0, 1, -1) # need all to decide positive label

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifier = RidgeClassifier()
    retained_ratios, train_scores, test_scores, feature_matrix = feature_detachment(
        classifier, 
        X_train, 
        X_test=X_val, 
        y_train=y_train,
        y_test=y_val,
        drop_ratio=drop_ratio,
        verbose=False,
    )

    expected_ratios = np.arange(1, 0, -drop_ratio)
    np.testing.assert_array_almost_equal(retained_ratios, expected_ratios, decimal=5)
    assert len(train_scores) == steps, "No training scores calculated"
    assert len(test_scores) == steps, "Test scores missing steps"
    assert feature_matrix.shape[1] == X.shape[1], "Feature matrix has incorrect number of features"
    assert (feature_matrix[5, 5:] == 0).all(), "Irrelevant features were not pruned first"


# def test_feature_detachment_multilabel():
    

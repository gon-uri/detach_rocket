import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

from detach_rocket.sfd import feature_detachment


def test_feature_detachment_with_validation():
    # data
    rng = np.random.default_rng(42)
    n_samples = 150
    n_relevant_features = 300
    n_irrelevant_features = 300
    drop_ratio = 0.1
    num_steps = 25

    X_relevant = rng.standard_normal((n_samples, n_relevant_features))
    X_irrelevant = rng.random((n_samples, n_irrelevant_features))
    X = np.hstack((X_relevant, X_irrelevant))
    y = np.where(X[:, :n_relevant_features].sum(axis=1) > 0, 1, -1)  # need all to decide positive label

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = RidgeClassifier()
    retained_ratios, train_scores, test_scores, feature_matrix = feature_detachment(
        classifier,
        X_train,
        X_test=X_val,
        y_train=y_train,
        y_test=y_val,
        drop_ratio=drop_ratio,
        num_steps=num_steps,
        verbose=False,
    )

    assert feature_matrix.shape[1] == X.shape[1], "Feature matrix has incorrect number of features"
    # Check that the number of features retained decreases at each step
    assert np.all(np.diff(retained_ratios) <= 0), "Retained ratios did not decrease monotonically"
    # Check that the retrained ratios, train scores, and test scores have the correct length
    assert len(retained_ratios) == num_steps, "Incorrect length of retained ratios"
    assert len(train_scores) == num_steps, "Incorrect length of train scores"
    assert len(test_scores) == num_steps, "Incorrect length of test scores"
    # Feature ranking is data-dependent; by this step most irrelevant features should be pruned.
    assert np.count_nonzero(feature_matrix[20, n_relevant_features:]) <= 10, (
        "Irrelevant features were not pruned early enough"
    )

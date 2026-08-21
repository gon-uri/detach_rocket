"""
Model size selection and retraining utilities for Detach-ROCKET.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV


def select_optimal_pruning(
    retained_ratios,
    val_accuracies,
    norm_factor=None,
    trade_off: float = 0.1,
    smoothing_points: int = 3,
):
    """
    Select the optimal model size after SFD.

    Parameters
    ----------
    retained_ratios: numpy array
        Proportion of retained features at each detachment step.
    val_accuracies: numpy array
        Validation accuracy at each detachment step.
    norm_factor: float
        Normalization factor for accuracy. Defaults to first val_accuracies value.
    trade_off: float
        Tradeoff between model size and validation accuracy.
    smoothing_points: int
        Smoothing window for validation accuracy.

    Returns
    -------
    max_index: int
        Index of the selected detachment step.
    max_percentage: float
        Size of the selected model (retained feature proportion).
    """
    if norm_factor is None:
        norm_factor = val_accuracies[0]

    x_vec = 1 - retained_ratios
    # Normalization and smoothing are stability tweaks over the paper's plain objective acc(q) + c*q.
    y_vec = val_accuracies / norm_factor
    box = np.ones(smoothing_points) / smoothing_points
    y_vec_smooth = np.convolve(y_vec, box, mode="same")
    optimality_curve = trade_off * x_vec + y_vec_smooth

    max_index = np.argmax(optimality_curve)
    max_percentage = retained_ratios[max_index]

    return max_index, max_percentage


def retrain_optimal_model(
    feature_mask,
    X_train_scaled_transform,
    y_train,
    model_alpha=None,
    verbose=False,
):
    """
    Retrain a Ridge classifier on an optimal selected feature subset.

    Parameters
    ----------
    feature_mask: numpy array
        Boolean mask for selected features.
    X_train_scaled_transform: numpy array
        Training feature matrix with shape (n_instances, n_features).
    y_train: numpy array
        Training labels.
    model_alpha: float
        Fixed alpha to use. If None, alpha is recomputed by CV.
    verbose: bool
        If true, print training summary.

    Returns
    -------
    optimal_classifier: sklearn model
        Ridge classifier trained on selected features.
    optimal_acc_train: float
        Training accuracy for the optimal classifier.
    """
    masked_x_train = X_train_scaled_transform[:, feature_mask]

    if model_alpha is None:
        cv_classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
        cv_classifier.fit(masked_x_train, y_train)
        model_alpha = cv_classifier.alpha_

    optimal_classifier = RidgeClassifier(alpha=model_alpha)
    optimal_classifier.fit(masked_x_train, y_train)
    optimal_acc_train = optimal_classifier.score(masked_x_train, y_train)

    if verbose:
        print("TRAINING RESULTS Detach Model:")
        print(f"Optimal Alpha Detach Model: {model_alpha:.2f}")
        print(f"Train Accuracy Detach Model: {100 * optimal_acc_train:.2f}%")
        print("-------------------------")

    return optimal_classifier, optimal_acc_train

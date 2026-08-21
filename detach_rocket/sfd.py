"""Sequential Feature Detachment (SFD) core methods."""

import numpy as np


def feature_detachment(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray = None,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    drop_ratio: float = 0.05,
    num_steps: int = 150,
    multiclass_type: str = "max",
    verbose: bool = False,
):
    """
    Apply Sequential Feature Detachment (SFD) to a feature matrix.

    Parameters
    ----------
    classifier: sklearn model
        Ridge linear classifier trained on the full training dataset.
    X_train: numpy array
        Training features matrix, shape (n_instances, n_features).
    y_train: numpy array
        Training labels.
    X_test: numpy array
        Validation/test features matrix, shape (n_instances, n_features).
    y_test: numpy array
        Validation/test labels.
    drop_ratio: float
        Proportion of features dropped at each step of the detachment process.
    num_steps: int
        Maximum number of detachment steps (actual steps may be fewer
        when the feature count is small).
    multiclass_type: str
        Method to calculate feature importance for multiclass
        classification: "max" (default, as in the Detach-ROCKET paper),
        "norm", or "avg".
    verbose: bool
        If true, print progress during the detachment process.

    Returns
    -------
    retained_ratios: numpy array
        Proportion of features retained at each detachment step.
    score_list_train: numpy array
        Training accuracy at each detachment step.
    score_list_test: numpy array or None
        Validation/test accuracy at each detachment step.
    feature_importance_matrix: numpy array
        Feature importance matrix at each detachment step.
    """
    if not hasattr(classifier, "coef_"):
        classifier.fit(X_train, y_train)

    multiclass_methods = {
        "norm": lambda coef: np.linalg.norm(coef, axis=0, ord=2),
        "max": lambda coef: np.linalg.norm(coef, axis=0, ord=np.inf),
        "avg": lambda coef: np.linalg.norm(coef, axis=0, ord=1),  # L1 norm (sum of abs values)
    }

    total_features = X_train.shape[1]
    retain_ratio = 1 - drop_ratio
    retained_ratios_uniform = np.power(retain_ratio, np.arange(num_steps))

    retained_features = np.unique((retained_ratios_uniform * total_features).astype(int))
    retained_features = retained_features[::-1]
    retained_features = retained_features[retained_features > 0]
    retained_ratios = retained_features / total_features

    score_list_train = []
    score_list_test = [] if X_test is not None and y_test is not None else None
    selection_mask = np.full(total_features, False)
    feature_importance_matrix = np.zeros((len(retained_features), total_features))

    # In sklearn >=1.6.1, binary classification coef_ is (n_features,) instead of
    # (1, n_features). Multi-class remains (n_classes, n_features).
    if len(np.shape(classifier.coef_)) > 1:
        if multiclass_type not in multiclass_methods:
            raise ValueError('Invalid multiclass_type. Choose from: "norm", "max", or "avg".')
        calc_feature_importance = multiclass_methods[multiclass_type]
    else:

        def calc_feature_importance(coef):
            return np.abs(coef).ravel()

    feature_importance = calc_feature_importance(classifier.coef_)

    for count, num_features in enumerate(retained_features):
        selected_idxs = np.argsort(feature_importance)[-num_features:]
        selection_mask[:] = False
        selection_mask[selected_idxs] = True

        X_train_subsampled = X_train[:, selection_mask]
        classifier.fit(X_train_subsampled, y_train)

        avg_score_train = classifier.score(X_train_subsampled, y_train)
        score_list_train.append(avg_score_train)

        if score_list_test is not None:
            X_test_subsampled = X_test[:, selection_mask]
            avg_score_test = classifier.score(X_test_subsampled, y_test)
            score_list_test.append(avg_score_test)

        feature_importance[~selection_mask] = 0
        feature_importance_matrix[count, :] = feature_importance
        feature_importance[selection_mask] = calc_feature_importance(classifier.coef_)

        if verbose:
            current_percentage = retained_ratios[count] * 100
            print(f"Step {count + 1} out of {len(retained_features)}: {current_percentage:.2f}% of features used")

    if score_list_test is not None:
        score_list_test = np.asarray(score_list_test)

    return retained_ratios, np.asarray(score_list_train), score_list_test, feature_importance_matrix

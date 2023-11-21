"""
Utility functions for SFD feature detachment and Detach-ROCKET model.
"""

from sklearn.linear_model import (RidgeClassifierCV,RidgeClassifier)
import numpy as np
import matplotlib.pyplot as plt


def feature_detachment(classifier,
                        X_train: np.ndarray,
                        X_test: np.ndarray,
                        y_train: np.ndarray,
                        y_test: np.ndarray,
                        drop_percentage: float = 0.05,
                        total_number_steps: int = 150,
                        multilabel_type: str = "norm",
                        verbose = True):
    """
    Applies Sequential Feature Detachment (SFD) to a feature matrix.

    Parameters
    ----------
    classifier: sklearn model
        Ridge linear classifier trained on the full training dataset.
    X_train: numpy array
        Training features matrix, a 2d array of shape (# instances, # features).
    X_test: numpy array
        Test features matrix, a 2d array of shape (# instances, # features).
    y_train: numpy array
        Training labels
    y_test: nparray
        Test labels
    drop_percentage: float
        Proportion of features drop at each step of the detachment process
    total_number_steps: int
        Total number of detachment steps performed during SFD
    verbose: bool
        If true, prints a message at the end of each step

    Returns
    -------
    percentage_vector: numpy array
        Array with the the model size at each detachment step (proportion of the original full model)
    score_list_train: numpy array
        Balanced accuraccy on the training set at each detachment step
    score_list_test: numpy array
        Balanced accuraccy on the test set at each detachment step
    feature_importance_matrix: numpy array
        A 2d array of shape (# steps, # features) with the importance of the features at each detachment step
    """

    # Check if training set is normalized
    mean_vector = X_train.mean(axis=0)
    zeros_vector = np.zeros_like(mean_vector)
    nomalized_condition = np.isclose(mean_vector, zeros_vector,atol=1e-02)
    # assert all(nomalized_condition), "The feature matrix should be normalized before training classifier."
    total_feats = X_train.shape[1]

    # Feature importance from full model
    
    # Check if problem is multilabel
    if np.shape(classifier.coef_)[0]>1:
        if multilabel_type == "norm":
            feature_importance_full = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=2)
        elif multilabel_type == "max":
            feature_importance_full = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=np.inf)
        elif multilabel_type == "avg":
            feature_importance_full = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=1)
        else:
            raise ValueError('Invalid multilabel_type argument. Choose from: "norm", "max", or "avg".')
    else:
        feature_importance_full = np.abs(classifier.coef_)[0,:]

    # Define percentage vector
    keep_percentage = 1-drop_percentage
    powers_vector = np.arange(total_number_steps)
    percentage_vector_unif = np.power(keep_percentage, powers_vector)

    num_feat_per_step = np.unique((percentage_vector_unif*total_feats).astype(int))
    num_feat_per_step = num_feat_per_step[::-1]
    num_feat_per_step = num_feat_per_step[num_feat_per_step>0]
    percentage_vector = num_feat_per_step/total_feats

    # Define lists and matrices
    score_list_train = []
    score_list_test = []
    feature_importance = np.copy(feature_importance_full)
    feature_importance_matrix = np.zeros((len(percentage_vector),len(feature_importance_full)))
    # feature_selection_matrix = np.full((len(percentage_vector),len(feature_importance_full)), False)

    # Begin iterative feature selection
    for count, feat_num in enumerate(num_feat_per_step):

        per = percentage_vector[count]

        # Cumpute mask for selected features
        drop_features = total_feats - feat_num

        selected_idxs = np.argsort(feature_importance)[drop_features:]
        selection_mask = np.full(total_feats, False)
        selection_mask[selected_idxs] = True

        # Apply mask
        X_train_subsampled = X_train[:,selection_mask]
        X_test_subsampled = X_test[:,selection_mask]

        # Train model for selected features
        classifier.fit(X_train_subsampled, y_train)

        # Compute scores for train and test sets
        avg_score_train = classifier.score(X_train_subsampled, y_train)
        avg_score_test = classifier.score(X_test_subsampled, y_test)
        score_list_train.append(avg_score_train)
        score_list_test.append(avg_score_test)

        # Save current feature importance and selected features
        feature_importance_matrix[count,:] = feature_importance
        # feature_selection_matrix[count,:] = selection_mask

        # Kill masked features
        feature_importance[~selection_mask] = 0

        # Compute feature importance taking into account multilabel type
        if np.shape(classifier.coef_)[0]>1:
            if multilabel_type == "norm":
                feature_importance[selection_mask] = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=2)
            elif multilabel_type == "max":
                feature_importance[selection_mask] = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=np.inf)
            elif multilabel_type == "avg":
                feature_importance[selection_mask] = np.linalg.norm(classifier.coef_[:,:],axis=0,ord=1)
            else:
                raise ValueError('Invalid multilabel_type argument. Choose from: "norm", "max", or "avg".')
        else:
            feature_importance[selection_mask] = np.abs(classifier.coef_)[0,:]

        if verbose==True:
            print("Step {} out of {}".format(count+1, total_number_steps))
            print('{:.3f}% of features used'.format(100*per))

    return percentage_vector, np.asarray(score_list_train), np.asarray(score_list_test), feature_importance_matrix #,feature_selection_matrix


def select_optimal_model(percentage_vector,
                            acc_test,
                            full_model_score_test,
                            acc_size_tradeoff_coef: float=0.1, # 0 means only weighting accuracy, +inf only weighting model size
                            smoothing_points: int = 3,
                            graphics = True):

    """
    Function that selects the optimal model size after SFD procces.

    Parameters
    ----------
    percentage_vector: numpy array
        Array with the the model size at each detachment step (proportion of the initial full model)
    acc_test: numpy array
        Balanced accuracy on the test set at each detachment step
    full_model_score_test: float
        Balanced accuracy of the full initial full classifier on the test set
    acc_size_tradeoff_coef: float
        Parameter that governs the tradeoff between size and accuracy. 0 means only weighting accuracy, +inf only weighting model size
    smoothing_points: int
        Level of smoothing applied to the acc_test
    graphics: bool
        If true, prints a matplotlib figure with the desition criteria

    Returns
    -------
    max_index: int
        Index of the optimal model (optimal number of SFD steps)
    max_percentage: float
        Size of the selected optimal model (proportion of the initial full model)
    """

    # Create model percentage vector
    x_vec = (1-percentage_vector)

    # Create smoothed relative test acc vector
    y_vec = (acc_test/full_model_score_test)
    box = np.ones(smoothing_points)/smoothing_points
    y_vec_smooth = np.convolve(y_vec, box, mode='same')

    # Define the functio to optimize
    optimality_curve = acc_size_tradeoff_coef*x_vec+y_vec_smooth

    # Compute max of the function
    max_index = np.argmax(optimality_curve)
    max_x_vec = x_vec[max_index]
    max_percentage = percentage_vector[max_index]

    # Plot results
    if graphics == True:
        margin = int((smoothing_points)/2)
        plt.plot(x_vec,y_vec, label='Relative test accuracy')
        plt.plot(x_vec[margin:-margin],optimality_curve[margin:-margin], label='Function to optimize')
        plt.scatter(max_x_vec,optimality_curve[max_index],c='C2', label='Maximum')
        plt.scatter(max_x_vec,y_vec[max_index],c='C3', label='Selected value')
        plt.legend()
        plt.ylabel('Relative Classification Accuracy')
        plt.xlabel('% of features Dropped')
        plt.show()

    return max_index, max_percentage


def retrain_optimal_model(feature_mask,
                          X_train_scaled_transform,
                          y_train,
                          max_index,
                          model_alpha = None,
                          verbose = True):

    """
    Function that retrains a Ridge classifier with the optimal subset of selected features.

    Parameters
    ----------
    feature_importance_matrix: numpy array
        A 2d array of shape (# steps, # features) with the importance of the features at each detachment step
    X_train_scaled_transform: numpy array
        Training features matrix, a 2d array of shape (# instances, # dimensions)
    X_test_scaled_transform: numpy array
        Test features matrix, a 2d array of shape (# instances, # dimensions)
    max_index: int
        Index of the optimal model (optimal number of SFD steps)
    model_alpha: float
        Alpha regularization parameter to be used with the Ridge classifier.
        If None recompute alpha using CrossValidation on the optimal features.
    verbose: bool
        If true, prints the results

    Returns
    -------
    optimal_classifier: sklearn model
        Ridge classifier trained on selected features, alpha is recomputed from the training set
    optimal_acc_train: float
        Balanced accuracy on the training set with optimal_classifier
    optimal_acc_train: float
        Balanced accuracy on the test set with optimal_classifier
    """

    masked_X_train = X_train_scaled_transform[:,feature_mask]

    if model_alpha==None:
      # CV to find best alpha
      cv_classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
      cv_classifier.fit(masked_X_train, y_train)
      model_alpha = cv_classifier.alpha_

    # Refit with all training set
    optimal_classifier = RidgeClassifier(alpha=model_alpha)
    optimal_classifier.fit(masked_X_train, y_train)
    optimal_acc_train = optimal_classifier.score(masked_X_train, y_train)

    print('TRAINING RESULTS Detach Model:')
    print('Optimal Alpha Detach Model: {:.2f}'.format(model_alpha))
    print('Train Accuraccy Detach Model: {:.2f}%'.format(100*optimal_acc_train))
    print('-------------------------')

    return optimal_classifier, optimal_acc_train

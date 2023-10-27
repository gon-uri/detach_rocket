"""Detach-RCOKET end-to-end model class."""

from detach_rocket.utils import (feature_detachment, select_optimal_model, retrain_optimal_model)

from sklearn.linear_model import (RidgeClassifierCV ,RidgeClassifier)
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import (
    Rocket,
    MiniRocket,
    MultiRocket
)
from sklearn.model_selection import train_test_split
import numpy as np

class DetachRocket:

    def __init__(
        self,
        model_type='rocket',
        num_kernels=10000,
        trade_off=0.1,
        recompute_alpha = True,
        val_ratio=0.33,
        ):

        self._sfd_curve = None
        self._transformer = None
        self._full_transformer = None
        self._scaler = None
        self._classifier = None
        self._acc_train = None
        self._full_classifier = None
        self._percentage_vector = None
        self._feature_matrix = None
        self._labels = None
        self._feature_importance_matrix = None
        self._full_model_alpha = None
        self._max_index = None
        self._max_percentage = None
        self._is_fitted = False
        self._optimal_computed = False
        

        self.num_kernels = num_kernels
        self.trade_off = trade_off
        self.val_ratio = val_ratio
        self.recompute_alpha = recompute_alpha

       # Create rocket model
        if model_type == "rocket":
            self._full_transformer = Rocket(num_kernels=num_kernels)
        elif model_type == "minirocket":
            self._full_transformer = MiniRocket(num_kernels=num_kernels)
        elif model_type == "multirocket":
            self._full_transformer = MultiRocket(num_kernels=num_kernels)
        else:
            raise ValueError('Invalid model_type argument. Choose from: "rocket", "minirocket", or "multirocket".')
        self._full_classifier = RidgeClassifierCV(alphas=np.logspace(-10,10,20))
        return


    def fit(self, X, y=None, verbose=False):
        assert y is not None, "Labels are required to fit Detach Rocket"

        if verbose == True:
            print('Applying Data Transformation')

        self._feature_matrix = self._full_transformer.fit_transform(X)
        self._labels = y

        if verbose == True:
            print('Fitting Full Model')

        # scale feature matrix
        scaler = StandardScaler(with_mean=True)
        self._feature_matrix = scaler.fit_transform(self._feature_matrix)
        self._scaler = scaler

        # TODO: if not enough samples for train/val split
        # if self.trade_off == None:
        #   print("Using default pruning level of 90%")
        # else:

        # Train full rocket as baseline
        self._full_classifier.fit(self._feature_matrix, y)
        self._full_model_alpha = self._full_classifier.alpha_

        print('TRAINING RESULTS FUll ROCKET:')
        print('Optimal Alpha FUll ROCKET: {:.2f}'.format(self._full_model_alpha))
        print('Train Accuraccy FUll ROCKET: {:.2f}%'.format(100*self._full_classifier.score(self._feature_matrix, y)))
        print('-------------------------')

        # Train-Validation split
        X_train, X_val, y_train, y_val = train_test_split(self._feature_matrix,
                                                            y,
                                                            test_size=self.val_ratio,
                                                            random_state=42,
                                                            stratify=y)

        # Train model for selected features
        sfd_classifier = RidgeClassifier(alpha=self._full_model_alpha)
        sfd_classifier.fit(X_train, y_train)

        # Feature Detachment
        if verbose == True:
            print('Applying Sequential Feature Detachment')

        self._percentage_vector, _, self._sfd_curve, self._feature_importance_matrix = feature_detachment(sfd_classifier, X_train, X_val, y_train, y_val, verbose=False)

        # Training Optimal Model
        if verbose == True:
            print('Training Optimal Model')

        self._is_fitted = True
        self.fit_trade_off(self.trade_off)

        return


    def fit_trade_off(self,trade_off=None):

        assert trade_off is not None, "Missing argument"
        assert self._is_fitted == True, "Model not fitted. Call fit method first."

        # Select optimal
        max_index, max_percentage = select_optimal_model(self._percentage_vector, self._sfd_curve, self._sfd_curve[0], self.trade_off, graphics=False)
        self._max_index = max_index
        self._max_percentage = max_percentage

        # Check if alpha will be recomputed
        if self.recompute_alpha:
            alpha_optimal = None
        else:
            alpha_optimal = self._full_model_alpha

        # Create feature mask
        self._feature_mask = self._feature_importance_matrix[max_index]>0

        # Re-train optimal model
        self._classifier, self._acc_train = retrain_optimal_model(self._feature_mask,
                                                                    self._feature_matrix,
                                                                    self._labels,
                                                                    alpha_optimal,
                                                                    max_index,
                                                                    verbose = False)

        return


    def predict(self,X):

        assert self._is_fitted == True, "Model not fitted. Call fit method first."

        # Transform time series to feature matrix
        transformed_X = np.asarray(self._full_transformer.transform(X))
        masked_transformed_X = transformed_X[:,self._feature_mask]

        y_pred = self._classifier.predict(masked_transformed_X)

        return y_pred


    def score(self, X, y):

        assert self._is_fitted == True, "Model not fitted. Call fit method first."

        # Transform time series to feature matrix
        transformed_X = np.asarray(self._full_transformer.transform(X))
        transformed_X = self._scaler.fit_transform(transformed_X)
        masked_transformed_X = transformed_X[:,self._feature_mask]

        return self._classifier.score(masked_transformed_X, y), self._full_classifier.score(transformed_X, y)

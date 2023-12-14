"""
DetachRocket end-to-end model class and DetachMatrix class.
"""

from detach_rocket.utils import (feature_detachment, select_optimal_model, retrain_optimal_model)

from sklearn.linear_model import (RidgeClassifierCV ,RidgeClassifier)
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import (
    Rocket,
    MiniRocketMultivariate,
    MultiRocketMultivariate
)
from sklearn.model_selection import train_test_split
import numpy as np
from fastcore.foundation import ifnone

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DetachRocket:

    """
    Rocket model with feature detachment. 
    For univariate time series, the shape of `X_train` should be (n_instances, n_timepoints).
    For multivariate time series, the shape of `X_train` should be (n_instances, n_variables, n_timepoints).

    Parameters:
    - model_type: Type of the rocket model ("rocket", "minirocket", or "multirocket").
    - num_kernels: Number of kernels for the rocket model.
    - trade_off: Trade-off parameter to set optimal pruning.
    - recompute_alpha: Whether to recompute alpha for optimal model training.
    - val_ratio: Validation set ratio.
    - verbose: Verbosity for logging.
    - multilabel_type: Type of feature ranking in case of multilabel classification ("max" by default).

    Attributes:
    - _sfd_curve: Curve for Sequential Feature Detachment.
    - _full_transformer: Full transformer for rocket model.
    - _full_classifier: Full classifier for baseline.
    - _full_model_alpha: Alpha for full model.
    - _classifier: Classifier for optimal model.
    - _feature_matrix: Feature matrix.
    - _feature_importance_matrix: Matrix for feature importance. Zero values indicate pruned features. Dimension: [Number of steps, Number of features].
    - _percentage_vector: Vector of percentage values.
    - _scaler: Scaler for feature matrix.
    - _labels: Labels.
    - _acc_train: Training accuracy.
    - _max_index: Index for maximum percentage.
    - _max_percentage: Maximum percentage.
    - _is_fitted: Flag indicating if the model is fitted.
    - _optimal_computed: Flag indicating if optimal model is computed.

    Methods:
    - fit: Fit the DetachRocket model.
    - fit_trade_off: Fit the model with a given trade-off.
    - predict: Make predictions using the fitted model.
    - score: Get the accuracy score of the model.

    """
    

    def __init__(
        self,
        model_type='rocket',
        num_kernels=10000,
        trade_off=0.1,
        recompute_alpha = True,
        val_ratio=0.33,
        verbose = False,
        multilabel_type = 'max'
        ):

        self._sfd_curve = None
        #self._transformer = None
        self._full_transformer = None
        self._full_classifier = None
        self._full_model_alpha = None
        self._classifier = None
        self._feature_matrix = None
        self._feature_importance_matrix = None
        self._percentage_vector = None
        self._scaler = None
        self._labels = None
        self._acc_train = None
        self._max_index = None
        self._max_percentage = None
        self._is_fitted = False
        self._optimal_computed = False

        self.num_kernels = num_kernels
        self.trade_off = trade_off
        self.val_ratio = val_ratio
        self.recompute_alpha = recompute_alpha
        self.verbose = verbose
        self.multilabel_type = multilabel_type

       # Create rocket model
        if model_type == "rocket":
            self._full_transformer = Rocket(num_kernels=num_kernels)
        elif model_type == "minirocket":
            self._full_transformer = MiniRocketMultivariate(num_kernels=num_kernels)
        elif model_type == "multirocket":
            self._full_transformer = MultiRocketMultivariate(num_kernels=num_kernels)
        else:
            raise ValueError('Invalid model_type argument. Choose from: "rocket", "minirocket", or "multirocket".')

        self._full_classifier = RidgeClassifierCV(alphas=np.logspace(-10,10,20))
        self._scaler = StandardScaler(with_mean=True)

        return

    def fit(self, X, y=None):
        assert y is not None, "Labels are required to fit Detach Rocket"

        if self.verbose == True:
            print('Applying Data Transformation')

        self._feature_matrix = self._full_transformer.fit_transform(X)
        self._labels = y

        if self.verbose == True:
            print('Fitting Full Model')

        # scale feature matrix
        self._feature_matrix = self._scaler.fit_transform(self._feature_matrix)

        # TODO: if not enough samples for train/val split
        # if self.trade_off == None:
        #   print("Using default pruning level of 90%")
        # else:

        # Train full rocket as baseline
        self._full_classifier.fit(self._feature_matrix, y)
        self._full_model_alpha = self._full_classifier.alpha_

        print('TRAINING RESULTS Full ROCKET:')
        print('Optimal Alpha Full ROCKET: {:.2f}'.format(self._full_model_alpha))
        print('Train Accuraccy Full ROCKET: {:.2f}%'.format(100*self._full_classifier.score(self._feature_matrix, y)))
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
        if self.verbose == True:
            print('Applying Sequential Feature Detachment')

        self._percentage_vector, _, self._sfd_curve, self._feature_importance_matrix = feature_detachment(sfd_classifier, X_train, X_val, y_train, y_val, verbose=self.verbose, multilabel_type = self.multilabel_type)

        # Training Optimal Model
        if self.verbose == True:
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
                                                                    verbose = self.verbose)

        return

    def predict(self,X):

        assert self._is_fitted == True, "Model not fitted. Call fit method first."

        # Transform time series to feature matrix
        transformed_X = np.asarray(self._full_transformer.transform(X))
        transformed_X = self._scaler.transform(transformed_X)
        masked_transformed_X = transformed_X[:,self._feature_mask]

        y_pred = self._classifier.predict(masked_transformed_X)

        return y_pred

    def score(self, X, y):

        assert self._is_fitted == True, "Model not fitted. Call fit method first."

        # Transform time series to feature matrix
        transformed_X = np.asarray(self._full_transformer.transform(X))
        transformed_X = self._scaler.transform(transformed_X)
        masked_transformed_X = transformed_X[:,self._feature_mask]

        return self._classifier.score(masked_transformed_X, y), self._full_classifier.score(transformed_X, y)


class DetachMatrix:
    """
    A class for pruning a feature matrix using feature detachment.
    The shape of the input matrix should be (n_instances, n_features).

    Parameters:
    - trade_off: Trade-off parameter.
    - recompute_alpha: Whether to recompute alpha for optimal model training.
    - val_ratio: Validation set ratio.
    - verbose: Verbosity for logging.
    - multilabel_type: Type of multilabel classification ("max" by default).

    Attributes:
    - _sfd_curve: Curve for Sequential Feature Detachment.
    - _scaler: Scaler for feature matrix.
    - _classifier: Classifier for optimal model.
    - _acc_train: Training accuracy.
    - _full_classifier: Full classifier for baseline.
    - _percentage_vector: Vector for percentage values.
    - _feature_matrix: Feature matrix.
    - _labels: Labels.
    - _feature_importance_matrix: Matrix for feature importance.
    - _full_model_alpha: Alpha for full model.
    - _max_index: Index for maximum percentage.
    - _max_percentage: Maximum percentage.
    - _is_fitted: Flag indicating if the model is fitted.
    - _optimal_computed: Flag indicating if optimal model is computed.
    - trade_off: Trade-off parameter.
    - val_ratio: Validation set ratio.
    - recompute_alpha: Whether to recompute alpha for optimal model training.
    - verbose: Verbosity for logging.
    - multilabel_type: Type of feature ranking in case of multilabel classification ("max" by default).

    Methods:
    - fit: Fit the DetachMatrix model.
    - fit_trade_off: Fit the model with a given trade-off.
    - predict: Make predictions using the fitted model.
    - score: Get the accuracy score of the model.

    """

    def __init__(
        self,
        trade_off=0.1,
        recompute_alpha = True,
        val_ratio=0.33,
        verbose = False,
        multilabel_type = 'max'
        ):

        self._sfd_curve = None
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
        

        self.trade_off = trade_off
        self.val_ratio = val_ratio
        self.recompute_alpha = recompute_alpha
        self.verbose = verbose
        self.multilabells
        _type = multilabel_type

        self._full_classifier = RidgeClassifierCV(alphas=np.logspace(-10,10,20))
        self._scaler = StandardScaler(with_mean=True)

        return

    def fit(self, X, y=None):
        assert y is not None, "Labels are required to fit Detach Matrix"

        self._feature_matrix = X
        self._labels = y

        if self.verbose == True:
            print('Fitting Full Model')

        # scale feature matrix
        self._feature_matrix = self._scaler.fit_transform(self._feature_matrix)
        

        # TODO: if not enough samples for train/val split
        # if self.trade_off == None:
        #   print("Using default pruning level of 90%")
        # else:

        # Train full rocket as baseline
        self._full_classifier.fit(self._feature_matrix, y)
        self._full_model_alpha = self._full_classifier.alpha_

        print('TRAINING RESULTS Full Features:')
        print('Optimal Alpha Full Features: {:.2f}'.format(self._full_model_alpha))
        print('Train Accuraccy Full Features: {:.2f}%'.format(100*self._full_classifier.score(self._feature_matrix, y)))
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
        if self.verbose == True:
            print('Applying Sequential Feature Detachment')

        self._percentage_vector, _, self._sfd_curve, self._feature_importance_matrix = feature_detachment(sfd_classifier, X_train, X_val, y_train, y_val, verbose=self.verbose, multilabel_type = self.multilabel_type)

        # Training Optimal Model
        if self.verbose == True:
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
                                                                    verbose = self.verbose)

        return

    def predict(self,X):

        assert self._is_fitted == True, "Model not fitted. Call fit method first."

        # Transform time series to feature matrix
        scaled_X = self._scaler.transform(X)
        masked_scaled_X = scaled_X[:,self._feature_mask]

        y_pred = self._classifier.predict(masked_scaled_X)

        return y_pred

    def score(self, X, y):

        assert self._is_fitted == True, "Model not fitted. Call fit method first."

        # Transform time series to feature matrix
        scaled_X = self._scaler.transform(X)
        masked_scaled_X = scaled_X[:,self._feature_mask]


        return self._classifier.score(masked_scaled_X, y), self._full_classifier.score(scaled_X, y)



class RocketPytorch(nn.Sequential):
    """
    End-to-edn pytorch implementation of a Rocket classifier.
    """

    def __init__(self, c_in, c_out, seq_len, num_features=10_000):
        
        # Backbone
        backbone =  RocketFeaturesPytorch(c_in, seq_len, n_kernels=num_features, kss=[7, 9, 11], device=None, verbose=False)

        # Head
        self.num_features = num_features
        layers = [nn.Flatten()]
        layers += [nn.BatchNorm1d(num_features)]
        linear = nn.Linear(num_features, c_out)
        nn.init.constant_(linear.weight.data, 0)
        nn.init.constant_(linear.bias.data, 0) 
        layers += [linear]
        head = nn.Sequential(*layers)
        super().__init__(OrderedDict([('backbone', backbone), ('head', head)]))



class RocketFeaturesPytorch(nn.Module):
    def __init__(self, c_in, seq_len, n_kernels=10_000, kss=[7, 9, 11], device=None, verbose=False):

        '''
        Input: is a 3d torch tensor of type torch.float32. When used with univariate TS,
        make sure you transform the 2d to 3d by adding unsqueeze(1).
        c_in: number of channels or features. For univariate c_in is 1.
        seq_len: sequence length
        '''
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0, np.log2((seq_len - 1) // (ks - 1)))
            padding = int((ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.randn(1, c_in, ks)
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - .5)
            layer = nn.Conv1d(c_in, 1, ks, padding=2 * padding, dilation=int(dilation), bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.kss = kss
        self.to(device=device)
        self.verbose=verbose

    def forward(self, x):
        _output = []
        for i in range(self.n_kernels):
            out = self.convs[i](x).cpu()
            _max = out.max(dim=-1)[0]
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            _output.append(_max)
            _output.append(_ppv)
        features = torch.cat(_output, dim=1)
        return features
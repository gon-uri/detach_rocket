"""
DetachRocket end-to-end model class, DetachMatrix class and DetachEnsemble class.
"""

from detach_rocket.utils import (feature_detachment, select_optimal_pruning, retrain_optimal_model)
from detach_rocket.pruner import get_transformer_pruner

from sklearn.linear_model import (RidgeClassifierCV ,RidgeClassifier)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sktime.transformations.panel.rocket import (
    Rocket,
    MiniRocketMultivariate,
    MultiRocketMultivariate
)
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class DetachRocket:
    def __init__(
        self,
        transformer,
        trade_off=0.1,
        set_percentage=None,
        recompute_alpha=None,
        verbose=False
        ):
        self._transformer = transformer
        self._classifier = RidgeClassifier()
        self._scaler = StandardScaler(with_mean=True)
        self._feature_matrix = None
        self._feature_matrix_val = None
        self._trade_off = trade_off
        self._set_percentage = set_percentage
        self._recompute_alpha = recompute_alpha
        self._full_model_alpha = None
        self._verbose = verbose
    
    def log(self, message):
        if self._verbose:
            print(message)

    def _validate_inputs(self, X, y, X_val, y_val):
        if X is None or y is None:
            raise ValueError("Training data (X, y) is required.")

        if self._set_percentage is None and (X_val is None or y_val is None):
            raise ValueError("Validation data (X_val, y_val) is required for calculating optimal pruning.")
        
        if self._set_percentage is not None and (X_val is not None or y_val is not None):
            self.log("Warning: Validation set provided but will be ignored as set_percentage is set.")

        if self._set_percentage is not None:
            self.log("Warning: Using fixed percentage for pruning. trade_off will be ignored.")
            
    def fit(self, X, y=None, X_val=None, y_val=None, **kwargs):
        
        self._validate_inputs(X, y, X_val, y_val)

        self.log("Applying Data Transformation")
        self._feature_matrix = self._transformer.fit_transform(X)
        self._feature_matrix = self._scaler.fit_transform(self._feature_matrix)

        if X_val is not None:
            self._feature_matrix_val = self._transformer.transform(X_val)
            self._feature_matrix_val = self._scaler.transform(self._feature_matrix_val)

        self._fit_params = kwargs

        self.log("Fitting Full Model")
        full_classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
        full_classifier.fit(self._feature_matrix, y)
        self._full_model_alpha = full_classifier.alpha_
        
        self._classifier = RidgeClassifier(alpha=self._full_model_alpha)
        
        self._retained_ratios, self._train_scores, self._val_scores, self._importance_matrix = feature_detachment(
            self._classifier,
            self._feature_matrix,
            X_test=self._feature_matrix_val,
            y_train=y,
            y_test=y_val,
            **kwargs
        )
        
        # Decide on the pruning level
        if self._set_percentage is None:
            self.log("Finding the optimal pruning level")
            self._max_index, self._max_percentage = select_optimal_pruning(
                self._retained_ratios, self._val_scores, trade_off=self._trade_off
            )
        else:
            self.log(f"Using fixed percentage for pruning: {self._set_percentage}%")
            self._max_index = (np.abs(self._retained_ratios - self._set_percentage / 100)).argmin()
            self._max_percentage = self._retained_ratios[self._max_index]

        # Retrain the model with the optimal pruning level
        self.log("Retraining the model with the optimal pruning level")
        self._optimal_feature_mask = self._importance_matrix[self._max_index] > 0
        
        self.log("Initializing pruned transformer with the selected features")
        pruner = get_transformer_pruner(self._transformer)
        self._pruned_transformer = pruner.prune_transformer(self._transformer, self._optimal_feature_mask)

        # Transform the data into the pruned feature space
        self._pruned_feature_matrix = self._pruned_transformer.transform(X)
        
        if self._recompute_alpha:
            self._classifier = RidgeClassifierCV()
            self._classifier.fit(self._pruned_feature_matrix, y)

        return self

class DetachRocket_og:

    """
    Rocket model with feature detachment. 
    For univariate time series, the shape of `X_train` should be (n_instances, n_timepoints).
    For multivariate time series, the shape of `X_train` should be (n_instances, n_variables, n_timepoints).

    Parameters:
    - model_type: Type of the rocket model ("rocket", "minirocket", "multirocket", or "pytorch_minirocket").
    - num_kernels: Number of kernels for the rocket model.
    - trade_off: Trade-off parameter to set optimal pruning.
    - recompute_alpha: Whether to recompute alpha for optimal model training.
    - val_ratio: Validation set ratio.
    - verbose: Verbosity for logging.
    - multilabel_type: Type of feature ranking in case of multilabel classification ("max" by default).
    - fixed_percentage: If not None, the trade_off parameter is ignored and the model is fitted with a fixed percentage of selected features.

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
    - fit_set_optimal: Fit the model with a fixed percentage of features.
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
        multilabel_type = 'max',
        fixed_percentage = None
        ):

        self._sfd_curve = None
        #self._transformer = None
        self._full_transformer = None
        self._full_classifier = None
        self._full_model_alpha = None
        self._classifier = None
        self._feature_matrix = None
        self._feature_matrix_val = None
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
        self.fixed_percentage = fixed_percentage

        # Create rocket model
        if model_type == "rocket":
            self._full_transformer = Rocket(num_kernels=num_kernels)
        elif model_type == "minirocket":
            self._full_transformer = MiniRocketMultivariate(num_kernels=num_kernels)
        elif model_type == "multirocket":
            self._full_transformer = MultiRocketMultivariate(num_kernels=num_kernels)
        elif model_type == "pytorch_minirocket":
            self._full_transformer = PytorchMiniRocketMultivariate(num_features=num_kernels)
        else:
            raise ValueError('Invalid model_type argument. Choose from: "rocket", "minirocket", "multirocket" or "pytorch_minirocket".')

        self._full_classifier = RidgeClassifierCV(alphas=np.logspace(-10,10,20))
        self._scaler = StandardScaler(with_mean=True)

        return

    def fit(self, X, y=None, val_set=None, val_set_y=None, X_test=None, y_test=None):

        assert y is not None, "Labels are required to fit Detach Rocket"

        if self.fixed_percentage is not None:
            # If fixed percentage is provided, no validation set is required
            # Assert there is no validation set
            assert val_set is None, "Validation set is not allowed when using fixed percentage of features, since it is not required for training"
            # Assert that both X_test set and y_test labels are provided
            assert X_test is not None, "X_test is required to fit Detach Rocket with fixed percentage. It is not used for training, but for plotting the feature detachment curve."
            assert y_test is not None, "y_test is required to fit Detach Rocket with fixed percentage. It is not used for training, but for plotting the feature detachment curve."
            
        if self.verbose == True:
            print('Applying Data Transformation')

        self._feature_matrix = self._full_transformer.fit_transform(X)
        self._labels = y

        if self.verbose == True:
            print('Fitting Full Model')

        # scale feature matrix
        self._feature_matrix = self._scaler.fit_transform(self._feature_matrix)

        if val_set is not None:
            self._feature_matrix_val = self._full_transformer.transform(val_set)
            self._feature_matrix_val = self._scaler.transform(self._feature_matrix_val)

        # Train full rocket as baseline
        self._full_classifier.fit(self._feature_matrix, y)
        self._full_model_alpha = self._full_classifier.alpha_

        print('TRAINING RESULTS Full ROCKET:')
        print('Optimal Alpha Full ROCKET: {:.2f}'.format(self._full_model_alpha))
        print('Train Accuraccy Full ROCKET: {:.2f}%'.format(100*self._full_classifier.score(self._feature_matrix, y)))
        print('-------------------------')

        # If fixed percentage is not provided, we set the number of features using the validation set
        if self.fixed_percentage is None:
            
            # Assert no test set is provided
            assert X_test is None, "X_test is not allowed when using trade-off, SFD curves are  computed with a validation set."

            if val_set is not None:
                X_train = self._feature_matrix
                X_val = self._feature_matrix_val
                y_train = y
                y_val = val_set_y
            else:
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

            self._is_fitted = True

            # Training Optimal Model
            if self.verbose == True:
                print('Training Optimal Model')
            
            self.fit_trade_off(self.trade_off)

        else:
            # If fixed percentage is provided, no validation set is required
            # We don't need to split the data into train and validation
            # We are using a fixed percentage of features
            X_train = self._feature_matrix
            y_train = y
            X_test = self._scaler.transform(self._full_transformer.transform(X_test))

            # Train model for selected features
            sfd_classifier = RidgeClassifier(alpha=self._full_model_alpha)
            sfd_classifier.fit(X_train, y_train)

            # Feature Detachment
            if self.verbose == True:
                print('Applying Sequential Feature Detachment')
            
            self._percentage_vector, _, self._sfd_curve, self._feature_importance_matrix = feature_detachment(sfd_classifier, X_train, X_test, y_train, y_test, verbose=self.verbose, multilabel_type = self.multilabel_type)

            self._is_fitted = True

            if self.verbose == True:
                print('Using fixed percentage of features')
            self.fit_set_optimal(self.fixed_percentage)

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
                                                                    self._max_index,
                                                                    alpha_optimal,
                                                                    verbose = self.verbose)

        return
    
    def fit_set_optimal(self, fixed_percentage=None, graphics=True):

        assert fixed_percentage is not None, "Missing argument"
        assert self._is_fitted == True, "Model not fitted. Call fit method first."

        self._max_index = (np.abs(self._percentage_vector - self.fixed_percentage)).argmin()
        self._max_percentage = self._percentage_vector[self._max_index]

        # Check if alpha will be recomputed
        if self.recompute_alpha:
            alpha_optimal = None
        else:
            alpha_optimal = self._full_model_alpha

        # Create feature mask
        self._feature_mask = self._feature_importance_matrix[self._max_index]>0

        # Re-train optimal model
        self._classifier, self._acc_train = retrain_optimal_model(self._feature_mask,
                                                                    self._feature_matrix,
                                                                    self._labels,
                                                                    self._max_index,
                                                                    alpha_optimal,
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
        multilabel_type = 'max',
        fixed_percentage = None
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
        self.multilabel_type = multilabel_type
        self.fixed_percentage = fixed_percentage

        self._full_classifier = RidgeClassifierCV(alphas=np.logspace(-10,10,20))
        self._scaler = StandardScaler(with_mean=True)

        return

    def fit(self, X, y=None, val_set=None, val_set_y=None, X_test=None, y_test=None):

        assert y is not None, "Labels are required to fit Detach Matrix"

        self._feature_matrix = X
        self._labels = y

        if val_set is not None:
            self._feature_matrix_val = val_set

        if self.verbose == True:
            print('Fitting Full Model')

        # scale feature matrix
        self._feature_matrix = self._scaler.fit_transform(self._feature_matrix)
    

        # Train full rocket as baseline
        self._full_classifier.fit(self._feature_matrix, y)
        self._full_model_alpha = self._full_classifier.alpha_

        print('TRAINING RESULTS Full Features:')
        print('Optimal Alpha Full Features: {:.2f}'.format(self._full_model_alpha))
        print('Train Accuraccy Full Features: {:.2f}%'.format(100*self._full_classifier.score(self._feature_matrix, y)))
        print('-------------------------')


        # If fixed percentage is not provided, we set the number of features using the validation set
        if self.fixed_percentage is None:
            
            # Assert no test set is provided
            assert X_test is None, "X_test is not allowed when using trade-off, SFD curves are  computed with a validation set."

            if val_set is not None:
                X_train = self._feature_matrix
                X_val = self._feature_matrix_val
                y_train = y
                y_val = val_set_y
            else:
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

            self._is_fitted = True

            # Training Optimal Model
            if self.verbose == True:
                print('Training Optimal Model')
            
            self.fit_trade_off(self.trade_off)

        # If fixed percentage is provided, no validation set is required
        else:
            # Assert there is no validation set
            assert val_set is None, "Validation set is not allowed when using fixed percentage of features, since it is not required for training"
            # Assert that both X_test set and y_test labels are provided
            assert X_test is not None, "X_test is required to fit Detach Matrix with fixed percentage. It is not used for training, but for plotting the feature detachment curve."
            assert y_test is not None, "y_test is required to fit Detach Matrix with fixed percentage. . It is not used for training, but for plotting the feature detachment curve."

            # We don't need to split the data into train and validation
            # We are using a fixed percentage of features
            X_train = self._feature_matrix
            y_train = y
            X_test = self._scaler.transform(X_test)

            # Train model for selected features
            sfd_classifier = RidgeClassifier(alpha=self._full_model_alpha)
            sfd_classifier.fit(X_train, y_train)

            # Feature Detachment
            if self.verbose == True:
                print('Applying Sequential Feature Detachment')
            
            self._percentage_vector, _, self._sfd_curve, self._feature_importance_matrix = feature_detachment(sfd_classifier, X_train, X_test, y_train, y_test, verbose=self.verbose, multilabel_type = self.multilabel_type)

            self._is_fitted = True

            if self.verbose == True:
                print('Using fixed percentage of features')
            self.fit_set_optimal(self.fixed_percentage)

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
                                                                    max_index,
                                                                    alpha_optimal,
                                                                    verbose = self.verbose)

        return

    def fit_set_optimal(self, fixed_percentage=None, graphics=True):

        assert fixed_percentage is not None, "Missing argument"
        assert self._is_fitted == True, "Model not fitted. Call fit method first."

        self._max_index = (np.abs(self._percentage_vector - self.fixed_percentage)).argmin()
        self._max_percentage = self._percentage_vector[self._max_index]

        # Check if alpha will be recomputed
        if self.recompute_alpha:
            alpha_optimal = None
        else:
            alpha_optimal = self._full_model_alpha

        # Create feature mask
        self._feature_mask = self._feature_importance_matrix[self._max_index]>0

        # Re-train optimal model
        self._classifier, self._acc_train = retrain_optimal_model(self._feature_mask,
                                                                    self._feature_matrix,
                                                                    self._labels,
                                                                    self._max_index,
                                                                    alpha_optimal,
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
    


class PytorchMiniRocketMultivariate(torch.nn.Module):
        """This is a Pytorch implementation of MiniRocket developed by Malcolm McLean and Ignacio Oguiza

        MiniRocket paper citation:
        @article{dempster_etal_2020,
        author  = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
        title   = {{MINIROCKET}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
        year    = {2020},
        journal = {arXiv:2012.08791}
        }
        Original paper: https://arxiv.org/abs/2012.08791
        Original code:  https://github.com/angus924/minirocket
        
        The class was edited by Adrià Solana for the Detach Rocket Ensemble study."""

        kernel_size, num_kernels, fitting = 9, 84, False

        def __init__(self, num_features=10_000, max_dilations_per_kernel=32, device=None):
            super().__init__()
            self.num_features = num_features
            self.max_dilations_per_kernel = max_dilations_per_kernel
            self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        def fit(self, X, chunksize=128):
            self.c_in, self.seq_len = X.shape[1], X.shape[2]
            self.num_features = self.num_features // self.num_kernels * self.num_kernels

            # Define the convolutional kernels
            indices = torch.combinations(torch.arange(self.kernel_size), 3).unsqueeze(1) 
            kernels = (-torch.ones(self.num_kernels, 1, self.kernel_size)).scatter_(2, indices, 2) 
            self.kernels = torch.nn.Parameter(kernels.repeat(self.c_in, 1, 1), requires_grad=False) 

            # Dilations & padding
            self._set_dilations(self.seq_len) 

            # Channel combinations (multivariate)
            if self.c_in > 1:
                self._set_channel_combinations(self.c_in)

            # Define the biases for each dilation
            for i in range(self.num_dilations):
                self.register_buffer(f'biases_{i}', torch.empty((self.num_kernels, self.num_features_per_dilation[i])))
            self.register_buffer('prefit', torch.BoolTensor([False]))
        
            # Move the defined model to device before computation
            self.to(self.device) 

            # Compute biases with the initial forward pass using a subset of samples
            num_samples = X.shape[0]
            if chunksize is None:
                chunksize = min(num_samples, self.num_dilations * self.num_kernels) # Deterministic
            else:
                chunksize = min(num_samples, chunksize) # Stochastic for chunksize < num_samples
            idxs = np.random.choice(num_samples, chunksize, False)
            self.fitting = True
            if isinstance(X, np.ndarray):
                self(torch.from_numpy(X[idxs]).float().to(self.device))
            else:
                self(X[idxs].to(self.device))
            self._set_parameter_indices()
            self.fitting = False

            return self

        def forward(self, x):
            _features = []
            for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)): # Max 32
                _padding1 = i%2

                # Convolution randomly combining channels for MTS
                C = torch.nn.functional.conv1d(x, self.kernels, padding=padding, dilation=dilation, groups=self.c_in) 
                if self.c_in > 1:
                    C = C.reshape(x.shape[0], self.c_in, self.num_kernels, -1)
                    channel_combination = getattr(self, f'channel_combinations_{i}')
                    C = torch.mul(C, channel_combination)
                    C = C.sum(1)

                # Draw the biases (and compute them if the model is fitting)
                if not self.prefit or self.fitting:
                    num_features_this_dilation = self.num_features_per_dilation[i]
                    bias_this_dilation = self._get_bias(C, num_features_this_dilation)
                    setattr(self, f'biases_{i}', bias_this_dilation)
                    if self.fitting:
                        if i < self.num_dilations - 1:
                            continue
                        else:
                            self.prefit = torch.BoolTensor([True])
                            return
                    elif i == self.num_dilations - 1:
                        self.prefit = torch.BoolTensor([True])
                else:
                    bias_this_dilation = getattr(self, f'biases_{i}')

                # Pool into PPVs with alternating padding
                _features.append(self._get_PPVs(C[:, _padding1::2], bias_this_dilation[_padding1::2]))
                _features.append(self._get_PPVs(C[:, 1-_padding1::2, padding:-padding], bias_this_dilation[1-_padding1::2]))
            return torch.cat(_features, dim=1)

        def _set_parameter_indices(self):
            # Simulate a forward pass but keep the indices that match a kernel and bias with a feature
            for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)): # Max 32
                _padding1 = i%2

                # Indices for the kernels / channel combinations & biases
                bias_this_dilation = getattr(self, f'biases_{i}')
                num_kernels, num_quantiles = bias_this_dilation.shape
                bias_indices = torch.arange(num_kernels*num_quantiles, dtype=int).reshape(num_quantiles, num_kernels).transpose(1, 0)

                kernel_indices = torch.arange(num_kernels, dtype=int)
                kernel_indices = torch.stack([kernel_indices]*num_quantiles, dim=-1)

                # Simulated feature maps for the "even" kernels
                C_even = kernel_indices[_padding1::2] # (num kernels / 2, num quantiles)
                bias_this_dilation_even = bias_indices[_padding1::2] # (num kernels / 2, num quantiles)

                # Simulate the PPV reshape
                C_even = C_even.flatten() # replaces .mean(2).flatten(1) and removes placeholder dimensions
                bias_this_dilation_even = bias_this_dilation_even.flatten()

                # Do the same for "odd" kernels
                C_odd = kernel_indices[1-_padding1::2] # (num kernels / 2, num quantiles)
                bias_this_dilation_odd = bias_indices[1-_padding1::2] # (num kernels / 2, num quantiles)

                # Simulate the PPV reshape
                C_odd = C_odd.flatten() # replaces .mean(2).flatten(1) and removes placeholder dimensions
                bias_this_dilation_odd = bias_this_dilation_odd.flatten()

                # Stack into flat arrays
                C_full = torch.cat((C_even, C_odd))
                bias_this_dilation_full = torch.cat((bias_this_dilation_even, bias_this_dilation_odd))

                setattr(self, f'kernel_indices_{i}', C_full)
                setattr(self, f'bias_indices_{i}', bias_this_dilation_full)

            return

        def get_kernel_features(self, which, where):
            # Get the "which" kernel parameters at "where" certain indices 
            full_features = np.empty(shape=(0,), dtype=float)

            if which == 'channels':
                full_features = np.empty(shape=(0, self.c_in), dtype=float)
                where = where[:, np.newaxis]
                where = np.repeat(where, self.c_in, axis=1)
            elif which == 'weights':
                full_features = np.empty(shape=(0, self.kernel_size), dtype=float)
                where = where[:, np.newaxis]
                where = np.repeat(where, self.kernel_size, axis=1)

            for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)):

                biases_this_dilation = getattr(self, f'biases_{i}')
                num_quantiles = biases_this_dilation.shape[1]

                kernel_indices = getattr(self, f'kernel_indices_{i}')
                bias_indices = getattr(self, f'bias_indices_{i}')

                # Biases (=features): as many as num dilations * num kernels * num quantiles
                if which == 'biases':
                    sorted_biases = biases_this_dilation.flatten()[bias_indices]
                    full_features = np.append(full_features, sorted_biases.cpu().numpy())

                # Channel combinations: num_dilations * num_kernel, where each combination has self.c_in indices
                elif which == 'channels':
                    channel_combinations = getattr(self, f'channel_combinations_{i}')

                    for q in range(0, num_quantiles):
                        selected_kernels = kernel_indices[q * self.num_kernels : q * self.num_kernels + self.num_kernels].cpu().numpy()
                        channel_combinations_q = channel_combinations[:, :, selected_kernels]
                        channel_combinations_q = torch.transpose(channel_combinations_q.squeeze(), 0, 1).cpu().numpy()

                        full_features = np.append(full_features, channel_combinations_q, axis=0)

                # Weights: num dilations * num kernels, where each kernel has 9 weights
                elif which == 'weights':
                    weights = self.kernels.view(-1, self.num_kernels, self.kernel_size)[0].cpu().numpy() # Kernels are equal for all channels, pick the first one

                    for q in range(0, num_quantiles):
                        selected_kernels = kernel_indices[q * self.num_kernels : q * self.num_kernels + self.num_kernels].cpu().numpy()
                        weights_q = weights[selected_kernels]

                        full_features = np.append(full_features, weights_q, axis=0)

                elif which == 'dilations':
                    expanded_dilations =  np.repeat(dilation, self.num_kernels*num_quantiles, axis=0)
                    full_features = np.append(full_features, expanded_dilations)

                elif which == 'paddings':
                    expanded_dilations =  np.repeat(padding, self.num_kernels*num_quantiles, axis=0)
                    full_features = np.append(full_features, expanded_dilations)

                else: raise ValueError(f'"{which}" is not recognized as a feature. Possible feaures are "biases", "channels", "weights", "dilations" or "paddings"')

            return np.where(where, full_features, np.nan)

        def _get_PPVs(self, C, bias):
            C = C.unsqueeze(-1)
            bias = bias.view(1, bias.shape[0], 1, bias.shape[1])
            return (C > bias).float().mean(2).flatten(1) 

        def _set_dilations(self, input_length):
            num_features_per_kernel = self.num_features // self.num_kernels
            true_max_dilations_per_kernel = min(num_features_per_kernel, self.max_dilations_per_kernel)
            multiplier = num_features_per_kernel / true_max_dilations_per_kernel
            max_exponent = np.log2((input_length - 1) / (9 - 1))
            dilations, num_features_per_dilation = \
            np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True) 
            num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)
            remainder = num_features_per_kernel - num_features_per_dilation.sum()
            i = 0
            while remainder > 0: 
                num_features_per_dilation[i] += 1
                remainder -= 1
                i = (i + 1) % len(num_features_per_dilation)
            self.num_features_per_dilation = num_features_per_dilation
            self.num_dilations = len(dilations)
            self.dilations = dilations
            self.padding = []
            for i, dilation in enumerate(dilations):
                self.padding.append((((self.kernel_size - 1) * dilation) // 2))

        def _set_channel_combinations(self, num_channels):
            num_combinations = self.num_kernels * self.num_dilations 
            max_num_channels = min(num_channels, 9)
            max_exponent_channels = np.log2(max_num_channels + 1) 
            num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent_channels, num_combinations)).astype(np.int32) 
            channel_combinations = torch.zeros((1, num_channels, num_combinations, 1))
            for i in range(num_combinations):
                channel_combinations[:, np.random.choice(num_channels, num_channels_per_combination[i], False), i] = 1 # From all the channels, set to 1 those that will be combined without repeating
            channel_combinations = torch.split(channel_combinations, self.num_kernels, 2) # split by dilation 
            for i, channel_combination in enumerate(channel_combinations):
                self.register_buffer(f'channel_combinations_{i}', channel_combination) # per dilation

        def _get_quantiles(self, n):
            return torch.tensor([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)]).float() 

        def _get_bias(self, C, num_features_this_dilation):
            # Gets as many biases as features this dilation, from the quantiles of random samples
            idxs = np.random.choice(C.shape[0], self.num_kernels)
            samples = C[idxs].diagonal().T
            biases = torch.quantile(samples, self._get_quantiles(num_features_this_dilation).to(C.device), dim=1).T
            return biases

        def transform(self, o, chunksize=128):
            o = torch.tensor(o).float()
            if isinstance(o, np.ndarray): o = torch.from_numpy(o).to(self.device)
            
            _features = []
            for oi in torch.split(o, chunksize):
                _features.append(self(oi.to(self.device)))

            return torch.cat(_features).cpu()
        
        def fit_transform(self, o):
            return self.fit(o).transform(o)


class DetachEnsemble():
    def __init__(self, 
                num_models=25, 
                num_kernels=10000, 
                model_type='pytorch_minirocket',
                trade_off=0.1, 
                recompute_alpha=True,
                val_ratio=0.33, 
                verbose=False,
                multilabel_type='max',
                fixed_percentage=None,
                ):
            
        assert model_type == 'pytorch_minirocket', f"Incorrect model_type {model_type}: DetachEnsemble currently only supports 'pytorch_minirocket'"
        self.num_models = num_models
        self.num_kernels = num_kernels
        self.model_type = model_type
        self.derockets = []
        for _ in range(num_models):
            _DetachRocket = DetachRocket(
                            model_type='pytorch_minirocket',
                            num_kernels=num_kernels,
                            trade_off=trade_off,
                            recompute_alpha=recompute_alpha,
                            val_ratio=val_ratio,
                            verbose=verbose,
                            multilabel_type=multilabel_type,
                            fixed_percentage=fixed_percentage,
                           )

            self.derockets.append(_DetachRocket)

        self.label_encoder = LabelEncoder()
        self._is_fitted = False

    # Transformer / Classifier methods
    def fit(self, X, y):
        [model.fit(X, y) for model in self.derockets]
        self.num_channels = X.shape[1]
        self.label_encoder.fit(y)

        self._is_fitted = True
        return self

    def predict_proba(self, X, proba='soft'):
        assert self._is_fitted == True, "Model not fitted. Call fit method first."
        weight_matrix = np.zeros((X.shape[0], len(self.label_encoder.classes_), self.num_models)) # (samples, classes, estimators)

        for m, model in enumerate(self.derockets):
            encoded_predictions = self.label_encoder.transform(model.predict(X))

            for p, pred in enumerate(encoded_predictions):
                weight_matrix[p, pred, m] = model._acc_train

        if proba == 'soft':
            votes = weight_matrix.sum(axis=2)
        elif proba == 'hard':
            votes = (weight_matrix != 0).astype(int).sum(axis=2)
            pass
        else: 
            raise ValueError(f'proba={proba} is not valid. Use "soft" or "hard".')
        
        probas = votes / votes.sum(axis=(1), keepdims=True)
        return probas

    def predict(self, X):
        predictions = self.predict_proba(X).argmax(axis=1)
        return self.label_encoder.inverse_transform(predictions)
    
    def estimate_channel_relevance(self):
        channel_relevance_matrix = np.zeros((self.num_models, self.num_channels))

        for m, model in enumerate(self.derockets):
            # Get the weights and the channel combination matrix for the selected features
            feature_weights = model._feature_importance_matrix[model._max_index] # Sparse float array (num_features,)
            selection_mask = feature_weights > 0

            channel_combinations_derocket = model._full_transformer.get_kernel_features('channels', selection_mask) # Indicator matrix (num_features, num_channels)
            num_channels_in_kernel = np.nansum(channel_combinations_derocket, axis=1)

            # Divide weights by the number of channels (num_features,)
            full_weights = (feature_weights[num_channels_in_kernel != 0] / num_channels_in_kernel[num_channels_in_kernel != 0])

            # Get the weighted channel combination matrix (num_features, num_channels)
            weighted_channel_combinations = channel_combinations_derocket[num_channels_in_kernel != 0]*full_weights[:, np.newaxis]

            # Sum contributions and normalize (num_channels,)
            channel_relevance = np.sum(weighted_channel_combinations, axis=0) / np.sum(weighted_channel_combinations)

            # Add to the ensemble matrix
            channel_relevance_matrix[m] = channel_relevance

        return np.median(channel_relevance_matrix, axis=0)

"""
Detach-ROCKET model classes.
"""

from detach_rocket.model_selection import retrain_optimal_model, select_optimal_pruning
from detach_rocket.sfd import feature_detachment
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
try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class BaseDetach:
    """Base class for Detach models.

    Provides shared utilities, learned-attribute initialization, and common
    ``predict`` / ``score`` / ``score_full`` / ``get_summary`` methods used
    by both :class:`DetachRocket` and :class:`DetachMatrix`.

    This class is not meant to be instantiated directly.

    Parameters
    ----------
    trade_off : float, default=0.1
        Weight given to model compression when selecting the optimal pruning
        level.  Higher values favor smaller models.
    set_percentage : float or None, default=None
        If set, forces a fixed pruning level (percentage of features to
        retain, e.g. ``50`` means 50 %).  When not *None*, ``trade_off`` is
        ignored and no validation set is needed for model selection.
    recompute_alpha : bool, default=False
        Whether to re-estimate the Ridge regularization parameter (alpha) by
        cross-validation after pruning.  If *False*, the alpha found on the
        full model is reused.
    verbose : bool, default=False
        If *True*, print progress messages during fitting.
    multilabel_type : str, default="norm"
        Method to aggregate multi-class Ridge coefficients into a single
        feature-importance vector.  One of ``"norm"`` (L2), ``"max"``
        (L∞), or ``"avg"`` (L1).
    """

    def __init__(self, trade_off=0.1, set_percentage=None, recompute_alpha=False, verbose=False, multilabel_type="norm"):
        self.trade_off = trade_off
        self.set_percentage = set_percentage
        self.recompute_alpha = recompute_alpha
        self.verbose = verbose
        self.multilabel_type = multilabel_type

        # Learned attributes (set during fit)
        self.is_fitted_ = False
        self.scaler_ = None
        self.classifier_ = None
        self.full_classifier_ = None
        self.full_model_alpha_ = None
        self.feature_matrix_ = None
        self.feature_matrix_val_ = None
        self.retained_ratios_ = None
        self.train_scores_ = None
        self.val_scores_ = None
        self.importance_matrix_ = None
        self.selected_step_index_ = None
        self.selected_ratio_ = None
        self.feature_mask_ = None
        self.labels_ = None
        self.acc_train_ = None

    # -- Utility helpers -----------------------------------------------------

    def _log(self, message):
        """Print *message* if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _require_fitted(self):
        """Raise ``ValueError`` if the model has not been fitted yet."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit method first.")

    @staticmethod
    def _to_numpy(matrix):
        """Convert a pandas DataFrame or array-like to a NumPy array."""
        if hasattr(matrix, "to_numpy"):
            return matrix.to_numpy()
        return np.asarray(matrix)

    # -- Inference -----------------------------------------------------------

    def _apply_feature_mask(self, X):
        """Scale *X* and select retained features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw (unscaled) feature matrix.

        Returns
        -------
        X_masked : np.ndarray of shape (n_samples, n_selected_features)
        """
        scaled = self.scaler_.transform(self._to_numpy(X))
        return scaled[:, self.feature_mask_]

    def predict(self, X):
        """Predict class labels for *X*.

        Parameters
        ----------
        X : array-like
            Input data.  The expected shape depends on the subclass:
            ``DetachRocket`` expects raw time series, ``DetachMatrix``
            expects a precomputed feature matrix.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        self._require_fitted()
        return self.classifier_.predict(self._prepare_X(X))

    def score(self, X, y):
        """Return the classification accuracy on (*X*, *y*).

        Parameters
        ----------
        X : array-like
            Input data (same format as ``predict``).
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
        """
        self._require_fitted()
        return self.classifier_.score(self._prepare_X(X), y)

    def score_full(self, X, y):
        """Return accuracy of the *unpruned* full model on (*X*, *y*).

        Parameters
        ----------
        X : array-like
            Input data (same format as ``predict``).
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
        """
        self._require_fitted()
        return self.full_classifier_.score(self._prepare_X_full(X), y)

    def _prepare_X(self, X):
        """Transform *X* into the pruned feature space for prediction.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _prepare_X_full(self, X):
        """Transform *X* into the full (unpruned) feature space.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    # -- Pruning step selection -----------------------------------------------

    def _select_pruning_step(self):
        """Set ``selected_step_index_`` and ``selected_ratio_`` from the SFD curve."""
        if self.set_percentage is None:
            self._log("Finding the optimal pruning level")
            self.selected_step_index_, self.selected_ratio_ = select_optimal_pruning(
                self.retained_ratios_, self.val_scores_,
                trade_off=self.trade_off,
            )
        else:
            self._log(f"Using fixed percentage for pruning: {self.set_percentage}%")
            self.selected_step_index_ = int(
                (np.abs(self.retained_ratios_ - self.set_percentage / 100)).argmin()
            )
            self.selected_ratio_ = self.retained_ratios_[self.selected_step_index_]

    # -- Post-fit retraining -------------------------------------------------

    def _retrain_at_step(self, step_index):
        """Retrain the classifier at the given SFD step.

        Sets ``selected_step_index_``, ``selected_ratio_``,
        ``feature_mask_``, ``classifier_``, and ``acc_train_``.
        Subclasses may override to perform additional work (e.g.
        rebuilding a pruned transformer).

        Parameters
        ----------
        step_index : int
            Index into the SFD curve arrays.
        """
        self.selected_step_index_ = step_index
        self.selected_ratio_ = self.retained_ratios_[step_index]

        alpha_optimal = None if self.recompute_alpha else self.full_model_alpha_
        self.feature_mask_ = self.importance_matrix_[step_index] > 0

        self.classifier_, self.acc_train_ = retrain_optimal_model(
            self.feature_mask_,
            self.feature_matrix_,
            self.labels_,
            step_index,
            alpha_optimal,
            verbose=self.verbose,
        )

    def fit_trade_off(self, trade_off=None):
        """Select the optimal pruning level using a trade-off criterion.

        Can be called after :meth:`fit` to re-select the pruning level
        with a different ``trade_off`` value without re-running SFD.

        Parameters
        ----------
        trade_off : float
            Trade-off weight between compression and accuracy.

        Returns
        -------
        self
        """
        if trade_off is None:
            raise ValueError("trade_off argument is required.")
        self._require_fitted()

        max_index, _ = select_optimal_pruning(
            self.retained_ratios_,
            self.val_scores_,
            trade_off=trade_off,
        )
        self._retrain_at_step(max_index)
        return self

    def fit_set_optimal(self, set_percentage=None):
        """Select the pruning level by a fixed percentage.

        Can be called after :meth:`fit` to re-select the pruning level
        with a different ``set_percentage`` without re-running SFD.

        Parameters
        ----------
        set_percentage : float
            Percentage of features to retain (e.g. ``50`` means 50 %).

        Returns
        -------
        self
        """
        if set_percentage is None:
            raise ValueError("set_percentage argument is required.")
        self._require_fitted()

        step_index = int((np.abs(self.retained_ratios_ - set_percentage / 100)).argmin())
        self._retrain_at_step(step_index)
        return self

    # -- Summary -------------------------------------------------------------

    def get_summary(self):
        """Return a dictionary summarizing the fitted model.

        Returns
        -------
        summary : dict
            Keys include ``estimator``, ``is_fitted``, ``selection_mode``,
            ``trade_off``, ``set_percentage``, ``recompute_alpha``,
            ``selected_step_index``, ``selected_ratio``,
            ``selected_feature_ratio``, ``selected_feature_count``,
            ``full_feature_count``, ``full_model_alpha``,
            ``final_model_alpha``, ``selected_step_train_score``,
            ``selected_step_val_score``.  Subclasses may add extra keys.
        """
        self._require_fitted()

        selected_feature_count = int(np.sum(self.feature_mask_))
        full_feature_count = int(self.feature_mask_.size)
        selected_ratio = (
            float(selected_feature_count / full_feature_count)
            if full_feature_count
            else 0.0
        )

        summary = {
            "estimator": type(self).__name__,
            "is_fitted": True,
            "selection_mode": (
                "set_percentage" if self.set_percentage is not None else "trade_off"
            ),
            "trade_off": float(self.trade_off),
            "set_percentage": self.set_percentage,
            "recompute_alpha": bool(self.recompute_alpha),
            "selected_step_index": int(self.selected_step_index_),
            "selected_ratio": float(self.selected_ratio_),
            "selected_feature_ratio": selected_ratio,
            "selected_feature_count": selected_feature_count,
            "full_feature_count": full_feature_count,
            "full_model_alpha": float(self.full_model_alpha_),
            "final_model_alpha": float(self.classifier_.alpha),
            "selected_step_train_score": self._get_train_score(),
            "selected_step_val_score": (
                None
                if self.val_scores_ is None
                else float(self.val_scores_[self.selected_step_index_])
            ),
        }
        return summary

    def _get_train_score(self):
        """Return the training accuracy from the retrained optimal model."""
        return float(self.acc_train_) if self.acc_train_ is not None else None


class DetachRocket(BaseDetach):
    """End-to-end Detach-ROCKET model for time-series classification.

    Transforms raw time series via a ROCKET-family transformer, applies
    Sequential Feature Detachment (SFD) to prune features, selects the
    optimal model size, and retrains a ``RidgeClassifier`` on the pruned
    feature set.

    Parameters
    ----------
    transformer : sktime transformer
        A fitted-or-unfitted ROCKET-family transformer (e.g.
        ``Rocket``, ``MiniRocketMultivariate``,
        ``MultiRocketMultivariate``).
    trade_off : float, default=0.1
        Weight given to model compression when selecting the optimal
        pruning level.
    set_percentage : float or None, default=None
        If set, forces a fixed pruning level (percentage of features to
        retain).  When not *None*, ``trade_off`` is ignored.
    recompute_alpha : bool, default=False
        Whether to re-estimate Ridge alpha by CV after pruning.
    verbose : bool, default=False
        Print progress messages during fitting.
    multilabel_type : str, default="norm"
        Method to aggregate multi-class Ridge coefficients into a single
        feature-importance vector.  One of ``"norm"`` (L2), ``"max"``
        (L∞), or ``"avg"`` (L1).

    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted.
    scaler_ : StandardScaler
        Scaler fitted on the full training feature matrix.
    classifier_ : RidgeClassifier
        Final classifier trained on the pruned feature set.
    full_classifier_ : RidgeClassifierCV
        Classifier trained on the full (unpruned) feature set.
    full_model_alpha_ : float
        Regularization alpha selected on the full model.
    feature_mask_ : np.ndarray of bool
        Boolean mask indicating which features are retained.
    pruned_transformer_ : PrunedRocket
        Transformer that directly produces only the retained features.
    retained_ratios_ : np.ndarray
        Proportion of features retained at each SFD step.
    train_scores_ : np.ndarray
        Training accuracy at each SFD step.
    val_scores_ : np.ndarray or None
        Validation accuracy at each SFD step.
    importance_matrix_ : np.ndarray
        Feature importance values at each SFD step.
    selected_step_index_ : int
        Index of the selected SFD step.
    selected_ratio_ : float
        Proportion of features retained at the selected step.

    Examples
    --------
    >>> from sktime.transformations.panel.rocket import Rocket
    >>> from detach_rocket.detach_classes import DetachRocket
    >>> rocket = Rocket(num_kernels=10_000)
    >>> model = DetachRocket(transformer=rocket, trade_off=0.1)
    >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>> y_pred = model.predict(X_test)

    References
    ----------
    .. [1] Uribarri et al. (2024), *Detach-ROCKET: Sequential feature
       selection for time series classification with random convolutional
       kernels*, Data Mining and Knowledge Discovery.
    """

    def __init__(
        self,
        transformer,
        trade_off=0.1,
        set_percentage=None,
        recompute_alpha=False,
        verbose=False,
        multilabel_type="norm",
    ):
        super().__init__(
            trade_off=trade_off,
            set_percentage=set_percentage,
            recompute_alpha=recompute_alpha,
            verbose=verbose,
            multilabel_type=multilabel_type,
        )
        self.transformer = transformer

        # DetachRocket-specific learned attributes
        self.fit_params_ = None
        self.pruned_transformer_ = None
        self.pruned_feature_matrix_ = None
        self._X_train_raw_ = None

    # -- Input preparation (BaseDetach hooks) --------------------------------

    def _prepare_X(self, X):
        """Transform raw time series into the pruned feature space."""
        transformed = self.pruned_transformer_.transform(X)
        return self._to_numpy(transformed)

    def _prepare_X_full(self, X):
        """Transform raw time series into the full (unpruned) feature space."""
        transformed = self._to_numpy(self.transformer.transform(X))
        return self.scaler_.transform(transformed)

    # -- Validation ----------------------------------------------------------

    def _validate_inputs(self, X, y, X_val, y_val):
        if X is None or y is None:
            raise ValueError("Training data (X, y) is required.")

        if self.set_percentage is None and (X_val is None or y_val is None):
            raise ValueError("Validation data (X_val, y_val) is required for calculating optimal pruning.")
        
        if self.set_percentage is not None and (X_val is not None or y_val is not None):
            self._log("Warning: Validation set provided but will be ignored as set_percentage is set.")

        if self.set_percentage is not None:
            self._log("Warning: Using fixed percentage for pruning. trade_off will be ignored.")
            
    def fit(self, X, y=None, X_val=None, y_val=None, **kwargs):
        """Fit the DetachRocket model.

        Transforms *X* using the ROCKET transformer, runs Sequential Feature
        Detachment (SFD) to obtain an accuracy-vs-size curve, selects the
        optimal pruning level, and retrains a ``RidgeClassifier`` on the
        pruned features.

        Parameters
        ----------
        X : array-like
            Training time series.  Shape ``(n_instances, n_timepoints)`` for
            univariate or ``(n_instances, n_channels, n_timepoints)`` for
            multivariate data.
        y : array-like of shape (n_instances,)
            Training labels.
        X_val : array-like or None, default=None
            Validation time series (same shape convention as *X*).  Required
            when ``set_percentage`` is *None*.
        y_val : array-like or None, default=None
            Validation labels.
        **kwargs
            Extra keyword arguments forwarded to
            :func:`~detach_rocket.sfd.feature_detachment` (e.g.
            ``drop_ratio``, ``num_steps``).  ``verbose`` and
            ``multilabel_type`` are already passed from ``self``.

        Returns
        -------
        self
        """
        self._validate_inputs(X, y, X_val, y_val)

        self._X_train_raw_ = X
        self.scaler_ = StandardScaler(with_mean=True)

        self._log("Applying Data Transformation")
        self.feature_matrix_ = self._to_numpy(self.transformer.fit_transform(X))
        self.feature_matrix_ = self.scaler_.fit_transform(self.feature_matrix_)

        use_validation = self.set_percentage is None and X_val is not None and y_val is not None

        if use_validation:
            self.feature_matrix_val_ = self._to_numpy(self.transformer.transform(X_val))
            self.feature_matrix_val_ = self.scaler_.transform(self.feature_matrix_val_)
        else:
            self.feature_matrix_val_ = None

        self.fit_params_ = kwargs

        self._log("Fitting Full Model")
        full_classifier = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
        full_classifier.fit(self.feature_matrix_, y)
        self.full_classifier_ = full_classifier
        self.full_model_alpha_ = full_classifier.alpha_
        
        self.classifier_ = RidgeClassifier(alpha=self.full_model_alpha_)
        
        self.retained_ratios_, self.train_scores_, self.val_scores_, self.importance_matrix_ = feature_detachment(
            self.classifier_,
            self.feature_matrix_,
            X_test=self.feature_matrix_val_ if use_validation else None,
            y_train=y,
            y_test=y_val if use_validation else None,
            verbose=self.verbose,
            multilabel_type=self.multilabel_type,
            **kwargs
        )
        
        self.labels_ = y
        
        # Decide on the pruning level and retrain
        self._select_pruning_step()
        self._retrain_at_step(self.selected_step_index_)
        self.is_fitted_ = True

        return self

    def _retrain_at_step(self, step_index):
        """Retrain the classifier and rebuild the pruned transformer.

        Extends the base implementation to also rebuild the pruned
        transformer from the updated feature mask.
        """
        super()._retrain_at_step(step_index)

        self._log("Initializing pruned transformer with the selected features")
        pruner = get_transformer_pruner(self.transformer)
        self.pruned_transformer_ = pruner.prune_transformer(self.transformer, self.feature_mask_)
        self.pruned_feature_matrix_ = self._prepare_X(self._X_train_raw_)

    def get_summary(self):
        """Return a dictionary summarizing the fitted model.

        Extends the base summary with ``retained_kernel_count``.

        Returns
        -------
        summary : dict
        """
        summary = super().get_summary()
        summary["retained_kernel_count"] = int(self.pruned_transformer_.num_kernels)
        return summary


class DetachMatrix(BaseDetach):
    """Detach model that operates on precomputed feature matrices.

    Applies Sequential Feature Detachment (SFD) directly to a feature
    matrix of shape ``(n_instances, n_features)`` — useful when features
    have already been extracted by an external pipeline (e.g. tsfresh,
    catch22, or a pre-fitted ROCKET transformer).

    Parameters
    ----------
    trade_off : float, default=0.1
        Weight given to model compression when selecting the optimal
        pruning level.
    set_percentage : float or None, default=None
        If set, forces a fixed pruning level (percentage of features to
        retain).  When not *None*, ``trade_off`` is ignored.
    recompute_alpha : bool, default=False
        Whether to re-estimate Ridge alpha by CV after pruning.
    val_ratio : float, default=0.33
        Fraction of the training data used for validation when no
        explicit validation set is provided.
    verbose : bool, default=False
        Print progress messages during fitting.
    multilabel_type : str, default="norm"
        Method to aggregate multi-class Ridge coefficients into a single
        feature-importance vector.  One of ``"norm"`` (L2), ``"max"``
        (L∞), or ``"avg"`` (L1).

    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted.
    scaler_ : StandardScaler
        Scaler fitted on the training feature matrix.
    classifier_ : RidgeClassifier
        Final classifier trained on the pruned feature set.
    full_classifier_ : RidgeClassifierCV
        Classifier trained on the full (unpruned) feature set.
    full_model_alpha_ : float
        Regularization alpha selected on the full model.
    feature_mask_ : np.ndarray of bool
        Boolean mask indicating which features are retained.
    feature_matrix_ : np.ndarray
        Scaled training feature matrix, stored for post-fit retraining
        via :meth:`fit_trade_off` and :meth:`fit_set_optimal`.
    labels_ : np.ndarray
        Training labels, stored for post-fit retraining.
    retained_ratios_ : np.ndarray
        Proportion of features retained at each SFD step.
    train_scores_ : np.ndarray
        Training accuracy at each SFD step.
    val_scores_ : np.ndarray or None
        Validation accuracy at each SFD step (``None`` when
        ``set_percentage`` is used without a validation set).
    importance_matrix_ : np.ndarray
        Feature importance values at each SFD step.
    selected_step_index_ : int
        Index of the selected SFD step.
    selected_ratio_ : float
        Proportion of features retained at the selected step.

    Examples
    --------
    >>> from detach_rocket.detach_classes import DetachMatrix
    >>> model = DetachMatrix(trade_off=0.1)
    >>> model.fit(X_train_features, y_train, X_val=X_val_features, y_val=y_val)
    >>> y_pred = model.predict(X_test_features)

    References
    ----------
    .. [1] Uribarri et al. (2024), *Detach-ROCKET: Sequential feature
       selection for time series classification with random convolutional
       kernels*, Data Mining and Knowledge Discovery.
    """

    def __init__(
        self,
        trade_off=0.1,
        recompute_alpha=False,
        val_ratio=0.33,
        verbose=False,
        multilabel_type="norm",
        set_percentage=None,
    ):
        super().__init__(
            trade_off=trade_off,
            set_percentage=set_percentage,
            recompute_alpha=recompute_alpha,
            verbose=verbose,
            multilabel_type=multilabel_type,
        )
        self.val_ratio = val_ratio

    # -- Input preparation (BaseDetach hooks) --------------------------------

    def _prepare_X(self, X):
        """Scale *X* and select retained features."""
        return self._apply_feature_mask(X)

    def _prepare_X_full(self, X):
        """Scale *X* using the fitted scaler (no masking)."""
        return self.scaler_.transform(self._to_numpy(X))

    # -- Validation ----------------------------------------------------------

    def _validate_inputs(self, y, X_val, y_val):
        """Validate fit arguments before proceeding."""
        if y is None:
            raise ValueError("Labels are required to fit DetachMatrix.")
        if X_val is not None and y_val is None:
            raise ValueError("y_val is required when X_val is provided.")
        if self.set_percentage is not None and X_val is not None:
            self._log("Warning: X_val provided but ignored when set_percentage is set.")

    # -- Fit -----------------------------------------------------------------

    def fit(self, X, y=None, X_val=None, y_val=None, **kwargs):
        """Fit the DetachMatrix model.

        Parameters
        ----------
        X : array-like of shape (n_instances, n_features)
            Training feature matrix.
        y : array-like of shape (n_instances,)
            Training labels.
        X_val : array-like or None, default=None
            Validation feature matrix.  When ``set_percentage`` is
            *None* and *X_val* is not provided, the training data is
            auto-split using ``val_ratio``.
        y_val : array-like or None, default=None
            Validation labels.
        **kwargs
            Extra keyword arguments forwarded to
            :func:`~detach_rocket.sfd.feature_detachment` (e.g.
            ``drop_ratio``, ``num_steps``).  ``verbose`` and
            ``multilabel_type`` are already passed from ``self``.

        Returns
        -------
        self
        """
        self._validate_inputs(y, X_val, y_val)

        self.labels_ = y

        self._log('Fitting Full Model')

        # Scale feature matrix
        self.scaler_ = StandardScaler(with_mean=True)
        self.feature_matrix_ = self.scaler_.fit_transform(X)

        # Train full model as baseline
        self.full_classifier_ = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
        self.full_classifier_.fit(self.feature_matrix_, y)
        self.full_model_alpha_ = self.full_classifier_.alpha_

        self._log('TRAINING RESULTS Full Features:')
        self._log('Optimal Alpha Full Features: {:.2f}'.format(self.full_model_alpha_))
        self._log('Train Accuracy Full Features: {:.2f}%'.format(100 * self.full_classifier_.score(self.feature_matrix_, y)))
        self._log('-------------------------')

        # Determine SFD split
        if self.set_percentage is not None:
            # Fixed percentage: SFD on full training data, no test scores
            X_sfd_train = self.feature_matrix_
            y_sfd_train = y
            X_sfd_test = None
            y_sfd_test = None
        elif X_val is not None:
            # Explicit validation set
            X_sfd_train = self.feature_matrix_
            y_sfd_train = y
            X_sfd_test = self.scaler_.transform(X_val)
            y_sfd_test = y_val
            self.feature_matrix_val_ = X_sfd_test
        else:
            # Auto-split using val_ratio
            X_sfd_train, X_sfd_test, y_sfd_train, y_sfd_test = train_test_split(
                self.feature_matrix_, y,
                test_size=self.val_ratio,
                random_state=42,
                stratify=y,
            )

        # Run SFD
        sfd_classifier = RidgeClassifier(alpha=self.full_model_alpha_)
        sfd_classifier.fit(X_sfd_train, y_sfd_train)

        self._log('Applying Sequential Feature Detachment')

        self.retained_ratios_, self.train_scores_, self.val_scores_, self.importance_matrix_ = feature_detachment(
            sfd_classifier,
            X_train=X_sfd_train,
            y_train=y_sfd_train,
            X_test=X_sfd_test,
            y_test=y_sfd_test,
            verbose=self.verbose,
            multilabel_type=self.multilabel_type,
            **kwargs
        )

        self.is_fitted_ = True

        # Select pruning level and retrain
        self._select_pruning_step()
        self._retrain_at_step(self.selected_step_index_)

        return self


_TorchModuleBase = torch.nn.Module if torch is not None else object


class PytorchMiniRocketMultivariate(_TorchModuleBase):
        """PyTorch implementation of MiniRocket for multivariate time series.

        Based on the MiniRocket algorithm by Dempster, Schmidt & Webb (2020),
        reimplemented in PyTorch by Malcolm McLean and Ignacio Oguiza, and
        edited by Adrià Solana for the Detach-ROCKET Ensemble study.

        This module generates random convolutional features using fixed
        kernel patterns with varying dilations, and uses Proportion of
        Positive Values (PPV) pooling.  For multivariate time series,
        random channel combinations are applied at each dilation level.

        Parameters
        ----------
        num_features : int, default=10_000
            Total number of features to generate (rounded down to a
            multiple of 84).
        max_dilations_per_kernel : int, default=32
            Maximum number of dilation levels per kernel.
        device : torch.device or None, default=None
            Device to run computations on.  Defaults to CUDA if available,
            otherwise CPU.

        References
        ----------
        .. [1] Dempster, A., Schmidt, D. F., & Webb, G. I. (2020).
               MINIROCKET: A Very Fast (Almost) Deterministic Transform
               for Time Series Classification. arXiv:2012.08791.
        """

        kernel_size, num_kernels, fitting = 9, 84, False

        def __init__(self, num_features=10_000, max_dilations_per_kernel=32, device=None):
            if torch is None:
                raise ImportError(
                    "PytorchMiniRocketMultivariate requires torch. Install torch to use this transformer."
                ) from _TORCH_IMPORT_ERROR
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
            if isinstance(o, np.ndarray):
                o = torch.from_numpy(o).float()
            else:
                o = torch.as_tensor(o).float()
            
            _features = []
            for oi in torch.split(o, chunksize):
                _features.append(self(oi.to(self.device)))

            return torch.cat(_features).cpu()
        
        def fit_transform(self, o):
            return self.fit(o).transform(o)


class DetachEnsemble():
    """Ensemble of Detach-ROCKET models for multivariate time series.

    Creates multiple :class:`DetachRocket` models — each with an
    independently randomized :class:`PytorchMiniRocketMultivariate`
    transformer — fits them independently, and combines their predictions
    via soft or hard voting.  Also provides channel-relevance estimation
    for multivariate data.

    Parameters
    ----------
    num_models : int, default=25
        Number of Detach-ROCKET models in the ensemble.
    num_kernels : int, default=10_000
        Number of kernels for each underlying
        :class:`PytorchMiniRocketMultivariate` transformer.
    trade_off : float, default=0.1
        Trade-off parameter passed to each :class:`DetachRocket`.
    set_percentage : float or None, default=None
        If set, each model uses a fixed pruning percentage instead of
        the trade-off criterion.  When not *None*, ``trade_off`` is
        ignored and no validation split is needed.
    recompute_alpha : bool, default=True
        Whether each model recomputes Ridge alpha after pruning.
    val_ratio : float, default=0.33
        Fraction of the training data used as a validation set for each
        model (only used when ``set_percentage`` is *None*).
    verbose : bool, default=False
        Print progress messages.

    Attributes
    ----------
    derockets : list of DetachRocket
        The individual Detach-ROCKET models.
    label_encoder : LabelEncoder
        Encoder mapping original labels to integer indices.
    is_fitted_ : bool
        Whether the ensemble has been fitted.
    num_channels : int
        Number of input channels (set after :meth:`fit`).

    Examples
    --------
    >>> from detach_rocket.detach_classes import DetachEnsemble
    >>> ensemble = DetachEnsemble(num_models=5, num_kernels=5_000)
    >>> ensemble.fit(X_train, y_train)
    >>> y_pred = ensemble.predict(X_test)
    >>> relevance = ensemble.estimate_channel_relevance()

    References
    ----------
    .. [1] Solana, A., Fransén, E., & Uribarri, G. (2024).
           Classification of raw MEG/EEG data with detach-rocket ensemble.
           arXiv:2408.02760.
    """

    def __init__(
        self,
        num_models=25,
        num_kernels=10_000,
        trade_off=0.1,
        set_percentage=None,
        recompute_alpha=True,
        val_ratio=0.33,
        verbose=False,
    ):
        if torch is None:
            raise ImportError(
                "DetachEnsemble requires PyTorch. Install torch to use this class."
            ) from _TORCH_IMPORT_ERROR

        self.num_models = num_models
        self.num_kernels = num_kernels
        self.trade_off = trade_off
        self.set_percentage = set_percentage
        self.recompute_alpha = recompute_alpha
        self.val_ratio = val_ratio
        self.verbose = verbose

        self.derockets = []
        for _ in range(num_models):
            transformer = PytorchMiniRocketMultivariate(num_features=num_kernels)
            model = DetachRocket(
                transformer=transformer,
                trade_off=trade_off,
                set_percentage=set_percentage,
                recompute_alpha=recompute_alpha,
                verbose=verbose,
            )
            self.derockets.append(model)

        self.label_encoder = LabelEncoder()
        self.is_fitted_ = False

    def fit(self, X, y):
        """Fit all ensemble members on the training data.

        When ``set_percentage`` is *None*, a stratified train/validation
        split is performed (controlled by ``val_ratio``) and passed to
        each :class:`DetachRocket` model.  When ``set_percentage`` is
        set, the full training set is used without splitting.

        Parameters
        ----------
        X : array-like of shape (n_instances, n_channels, n_timepoints)
            Multivariate training time series.
        y : array-like of shape (n_instances,)
            Training labels.

        Returns
        -------
        self
        """
        if self.set_percentage is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.val_ratio,
                random_state=42,
                stratify=y,
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        for model in self.derockets:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        self.num_channels = X.shape[1]
        self.label_encoder.fit(y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X, proba='soft'):
        """Return class probability estimates for *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Input time series.
        proba : {'soft', 'hard'}, default='soft'
            Voting strategy.  ``'soft'`` weights each model's vote by
            its training accuracy at the selected pruning step.
            ``'hard'`` gives equal weight to every model.

        Returns
        -------
        probas : np.ndarray of shape (n_samples, n_classes)
            Normalized class probabilities.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit method first.")

        n_samples = X.shape[0]
        n_classes = len(self.label_encoder.classes_)
        weight_matrix = np.zeros((n_samples, n_classes, self.num_models))

        for m, model in enumerate(self.derockets):
            encoded_predictions = self.label_encoder.transform(model.predict(X))
            train_acc = model.train_scores_[model.selected_step_index_]

            for p, pred in enumerate(encoded_predictions):
                weight_matrix[p, pred, m] = train_acc

        if proba == 'soft':
            votes = weight_matrix.sum(axis=2)
        elif proba == 'hard':
            votes = (weight_matrix != 0).astype(int).sum(axis=2)
        else:
            raise ValueError(f'proba={proba} is not valid. Use "soft" or "hard".')

        probas = votes / votes.sum(axis=1, keepdims=True)
        return probas

    def predict(self, X):
        """Predict class labels for *X* using ensemble voting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Input time series.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        predictions = self.predict_proba(X).argmax(axis=1)
        return self.label_encoder.inverse_transform(predictions)

    def estimate_channel_relevance(self):
        """Estimate the relevance of each input channel.

        Computes a median relevance score across all ensemble members
        by distributing each feature's importance to the channels used
        by its corresponding kernel.

        Returns
        -------
        channel_relevance : np.ndarray of shape (n_channels,)
            Normalized relevance score for each channel.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit method first.")

        channel_relevance_matrix = np.zeros((self.num_models, self.num_channels))

        for m, model in enumerate(self.derockets):
            # Get feature weights at the selected pruning step
            feature_weights = model.importance_matrix_[model.selected_step_index_]
            selection_mask = feature_weights > 0

            # Channel indicator matrix (num_features, num_channels)
            channel_combinations = model.transformer.get_kernel_features(
                'channels', selection_mask
            )
            num_channels_in_kernel = np.nansum(channel_combinations, axis=1)

            # Divide weights by the number of channels per kernel
            valid = num_channels_in_kernel != 0
            full_weights = feature_weights[valid] / num_channels_in_kernel[valid]

            # Weighted channel combination matrix (num_features, num_channels)
            weighted_channel_combinations = (
                channel_combinations[valid] * full_weights[:, np.newaxis]
            )

            # Sum contributions and normalize (num_channels,)
            total = np.sum(weighted_channel_combinations)
            if total > 0:
                channel_relevance = np.sum(weighted_channel_combinations, axis=0) / total
            else:
                channel_relevance = np.zeros(self.num_channels)

            channel_relevance_matrix[m] = channel_relevance

        return np.median(channel_relevance_matrix, axis=0)

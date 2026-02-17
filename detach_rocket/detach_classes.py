"""
Detach-ROCKET model classes.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from detach_rocket.model_selection import retrain_optimal_model, select_optimal_pruning
from detach_rocket.pruner import get_transformer_pruner
from detach_rocket.sfd import feature_detachment


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

    def __init__(
        self, trade_off=0.1, set_percentage=None, recompute_alpha=False, verbose=False, multilabel_type="norm"
    ):
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
                self.retained_ratios_,
                self.val_scores_,
                trade_off=self.trade_off,
            )
        else:
            self._log(f"Using fixed percentage for pruning: {self.set_percentage}%")
            self.selected_step_index_ = int((np.abs(self.retained_ratios_ - self.set_percentage / 100)).argmin())
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
        selected_ratio = float(selected_feature_count / full_feature_count) if full_feature_count else 0.0

        summary = {
            "estimator": type(self).__name__,
            "is_fitted": True,
            "selection_mode": ("set_percentage" if self.set_percentage is not None else "trade_off"),
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
                None if self.val_scores_ is None else float(self.val_scores_[self.selected_step_index_])
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
            **kwargs,
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

        self._log("Fitting Full Model")

        # Scale feature matrix
        self.scaler_ = StandardScaler(with_mean=True)
        self.feature_matrix_ = self.scaler_.fit_transform(X)

        # Train full model as baseline
        self.full_classifier_ = RidgeClassifierCV(alphas=np.logspace(-10, 10, 20))
        self.full_classifier_.fit(self.feature_matrix_, y)
        self.full_model_alpha_ = self.full_classifier_.alpha_

        self._log("TRAINING RESULTS Full Features:")
        self._log(f"Optimal Alpha Full Features: {self.full_model_alpha_:.2f}")
        self._log(f"Train Accuracy Full Features: {100 * self.full_classifier_.score(self.feature_matrix_, y):.2f}%")
        self._log("-------------------------")

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
                self.feature_matrix_,
                y,
                test_size=self.val_ratio,
                random_state=42,
                stratify=y,
            )

        # Run SFD
        sfd_classifier = RidgeClassifier(alpha=self.full_model_alpha_)
        sfd_classifier.fit(X_sfd_train, y_sfd_train)

        self._log("Applying Sequential Feature Detachment")

        self.retained_ratios_, self.train_scores_, self.val_scores_, self.importance_matrix_ = feature_detachment(
            sfd_classifier,
            X_train=X_sfd_train,
            y_train=y_sfd_train,
            X_test=X_sfd_test,
            y_test=y_sfd_test,
            verbose=self.verbose,
            multilabel_type=self.multilabel_type,
            **kwargs,
        )

        self.is_fitted_ = True

        # Select pruning level and retrain
        self._select_pruning_step()
        self._retrain_at_step(self.selected_step_index_)

        return self


class DetachEnsemble:
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
    multilabel_type : str, default="norm"
        Method to aggregate multi-class Ridge coefficients into a single
        feature-importance vector.  Forwarded to each inner
        :class:`DetachRocket`.  One of ``"norm"`` (L2), ``"max"``
        (L∞), or ``"avg"`` (L1).

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
        multilabel_type="norm",
    ):
        try:
            from detach_rocket.pytorch_minirocket import PytorchMiniRocketMultivariate
        except ImportError as exc:
            raise ImportError(
                "DetachEnsemble requires PyTorch. Install it with: pip install detach_rocket[torch]"
            ) from exc

        self.num_models = num_models
        self.num_kernels = num_kernels
        self.trade_off = trade_off
        self.set_percentage = set_percentage
        self.recompute_alpha = recompute_alpha
        self.val_ratio = val_ratio
        self.verbose = verbose
        self.multilabel_type = multilabel_type

        self.derockets = []
        for _ in range(num_models):
            transformer = PytorchMiniRocketMultivariate(num_features=num_kernels)
            model = DetachRocket(
                transformer=transformer,
                trade_off=trade_off,
                set_percentage=set_percentage,
                recompute_alpha=recompute_alpha,
                verbose=verbose,
                multilabel_type=multilabel_type,
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
                X,
                y,
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

    def predict_proba(self, X, proba="soft"):
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
            train_acc = model.acc_train_

            weight_matrix[np.arange(n_samples), encoded_predictions, m] = train_acc

        if proba == "soft":
            votes = weight_matrix.sum(axis=2)
        elif proba == "hard":
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

    def score(self, X, y):
        """Return the classification accuracy on (*X*, *y*).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Input time series.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        # TODO: implement ensemble scoring strategy
        raise NotImplementedError("DetachEnsemble.score() is not yet implemented.")

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
            channel_combinations = model.transformer.get_kernel_features("channels", selection_mask)
            num_channels_in_kernel = np.nansum(channel_combinations, axis=1)

            # Divide weights by the number of channels per kernel
            valid = num_channels_in_kernel != 0
            full_weights = feature_weights[valid] / num_channels_in_kernel[valid]

            # Weighted channel combination matrix (num_features, num_channels)
            weighted_channel_combinations = channel_combinations[valid] * full_weights[:, np.newaxis]

            # Sum contributions and normalize (num_channels,)
            total = np.sum(weighted_channel_combinations)
            if total > 0:
                channel_relevance = np.sum(weighted_channel_combinations, axis=0) / total
            else:
                channel_relevance = np.zeros(self.num_channels)

            channel_relevance_matrix[m] = channel_relevance

        return np.median(channel_relevance_matrix, axis=0)

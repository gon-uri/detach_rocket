"""
Detach-ROCKET model classes.
"""

import warnings

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
    recompute_alpha : bool, default=True
        Whether to re-estimate the Ridge regularization parameter (alpha) by
        cross-validation after pruning.  If *False*, the alpha found on the
        full model is reused.
    verbose : bool, default=False
        If *True*, print progress messages during fitting.
    multiclass_type : str, default="max"
        Method to aggregate multi-class Ridge coefficients into a single
        feature-importance vector.  One of ``"max"`` (L∞, the
        aggregation used in the Detach-ROCKET paper), ``"norm"`` (L2),
        or ``"avg"`` (L1).
    """

    def __init__(self, trade_off=0.1, set_percentage=None, recompute_alpha=True, verbose=False, multiclass_type="max"):
        self.trade_off = trade_off
        self.set_percentage = set_percentage
        self.recompute_alpha = recompute_alpha
        self.verbose = verbose
        self.multiclass_type = multiclass_type

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
        self.labels_val_ = None
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

        # After model selection the validation set has served its purpose,
        # so we combine train + val for the final retraining to maximise
        # the amount of data seen by the classifier.
        if self.feature_matrix_val_ is not None and self.labels_val_ is not None:
            X_retrain = np.concatenate([self.feature_matrix_, self.feature_matrix_val_], axis=0)
            y_retrain = np.concatenate([self.labels_, self.labels_val_], axis=0)

        else:
            X_retrain = self.feature_matrix_
            y_retrain = self.labels_

        # NOTE: ``acc_train_`` is the accuracy of the retrained classifier
        # evaluated on the combined train + val set (i.e. all the data used
        # for retraining).  This is *not* directly comparable to
        # ``train_scores_``, which is recorded during SFD on the training
        # split only.
        self.classifier_, self.acc_train_ = retrain_optimal_model(
            self.feature_mask_,
            X_retrain,
            y_retrain,
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
        if self.val_scores_ is None:
            raise ValueError(
                "No validation scores available: the model was fitted with set_percentage. "
                "Refit with a validation set to use fit_trade_off."
            )

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


class PrunedRocketModel:
    """Lightweight pruned model for inference only.

    Contains only the pruned transformer, its scaler, and the retrained
    classifier — the minimum needed to predict on new time series.
    Returned by :meth:`DetachRocket.detach`.

    Parameters
    ----------
    transformer : object
        A pruned ROCKET-family transformer with a ``transform(X)`` method.
    scaler : StandardScaler
        Scaler fitted on the pruned feature space.
    classifier : RidgeClassifier
        Classifier trained on the pruned, scaled features.

    Examples
    --------
    >>> model = DetachRocket(transformer=rocket, trade_off=0.1)
    >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>> pruned = model.detach()
    >>> y_pred = pruned.predict(X_test)
    """

    def __init__(self, transformer, scaler, classifier):
        self.transformer = transformer
        self.scaler = scaler
        self.classifier = classifier

    def predict(self, X):
        """Predict class labels for *X*.

        Parameters
        ----------
        X : array-like
            Raw time series in the same format used during training.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        transformed = self.transformer.transform(X)
        if hasattr(transformed, "to_numpy"):
            transformed = transformed.to_numpy()
        else:
            transformed = np.asarray(transformed)
        scaled = self.scaler.transform(transformed)
        return self.classifier.predict(scaled)

    def score(self, X, y):
        """Return the classification accuracy on (*X*, *y*).

        Parameters
        ----------
        X : array-like
            Raw time series.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


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
    recompute_alpha : bool, default=True
        Whether to re-estimate Ridge alpha by CV after pruning.
    verbose : bool, default=False
        Print progress messages during fitting.
    multiclass_type : str, default="max"
        Method to aggregate multi-class Ridge coefficients into a single
        feature-importance vector.  One of ``"max"`` (L∞, the
        aggregation used in the Detach-ROCKET paper), ``"norm"`` (L2),
        or ``"avg"`` (L1).

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
    pruned_transformer_ : PrunedRocketTransformer
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
        recompute_alpha=True,
        verbose=False,
        multiclass_type="max",
    ):
        super().__init__(
            trade_off=trade_off,
            set_percentage=set_percentage,
            recompute_alpha=recompute_alpha,
            verbose=verbose,
            multiclass_type=multiclass_type,
        )
        self.transformer = transformer

        # DetachRocket-specific learned attributes
        self.pruned_transformer_ = None
        self.pruned_scaler_ = None

    # -- Input preparation (BaseDetach hooks) --------------------------------

    def _prepare_X(self, X):
        """Transform raw time series into the pruned feature space."""
        transformed = self._to_numpy(self.pruned_transformer_.transform(X))
        return self.pruned_scaler_.transform(transformed)

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
            Training time series of shape
            ``(n_instances, n_channels, n_timepoints)``.  For univariate
            data with sktime transformers a 2D
            ``(n_instances, n_timepoints)`` array is also accepted.
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
            ``multiclass_type`` are already passed from ``self``.

        Returns
        -------
        self
        """
        self._validate_inputs(X, y, X_val, y_val)

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
            multiclass_type=self.multiclass_type,
            **kwargs,
        )

        self.labels_ = y
        self.labels_val_ = y_val if use_validation else None

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

        # Build a scaler for the pruned feature space by extracting the
        # mean and scale of the retained features from the full scaler.
        self.pruned_scaler_ = StandardScaler()
        self.pruned_scaler_.mean_ = self.scaler_.mean_[self.feature_mask_]
        self.pruned_scaler_.scale_ = self.scaler_.scale_[self.feature_mask_]
        self.pruned_scaler_.var_ = self.scaler_.var_[self.feature_mask_]
        self.pruned_scaler_.n_features_in_ = int(np.sum(self.feature_mask_))
        self.pruned_scaler_.n_samples_seen_ = self.scaler_.n_samples_seen_

    def detach(self):
        """Return a lightweight :class:`PrunedRocketModel` for inference.

        The returned object contains only the pruned transformer, its
        scaler, and the retrained classifier — the minimum needed to
        call ``predict`` on new data.

        Returns
        -------
        pruned_model : PrunedRocketModel

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = DetachRocket(transformer=rocket, trade_off=0.1)
        >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        >>> pruned = model.detach()
        >>> y_pred = pruned.predict(X_test)
        """
        self._require_fitted()
        return PrunedRocketModel(
            transformer=self.pruned_transformer_,
            scaler=self.pruned_scaler_,
            classifier=self.classifier_,
        )

    def get_summary(self):
        """Return a dictionary summarizing the fitted model.

        Extends the base summary with ``retained_kernel_count``.

        Returns
        -------
        summary : dict
        """
        summary = super().get_summary()
        retained_kernel_count = getattr(self.pruned_transformer_, "num_kernels", None)
        summary["retained_kernel_count"] = None if retained_kernel_count is None else int(retained_kernel_count)
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
    recompute_alpha : bool, default=True
        Whether to re-estimate Ridge alpha by CV after pruning.
    val_ratio : float, default=0.33
        Fraction of the training data used for validation when no
        explicit validation set is provided.
    random_state : int or None, default=None
        Seed for the internal train/validation split when no explicit
        validation set is given.  *None* keeps the historical fixed
        split (seed 42).
    verbose : bool, default=False
        Print progress messages during fitting.
    multiclass_type : str, default="max"
        Method to aggregate multi-class Ridge coefficients into a single
        feature-importance vector.  One of ``"max"`` (L∞, the
        aggregation used in the Detach-ROCKET paper), ``"norm"`` (L2),
        or ``"avg"`` (L1).

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
        set_percentage=None,
        recompute_alpha=True,
        val_ratio=0.33,
        verbose=False,
        multiclass_type="max",
        random_state=None,
    ):
        super().__init__(
            trade_off=trade_off,
            set_percentage=set_percentage,
            recompute_alpha=recompute_alpha,
            verbose=verbose,
            multiclass_type=multiclass_type,
        )
        self.val_ratio = val_ratio
        self.random_state = random_state

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
            ``multiclass_type`` are already passed from ``self``.

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
            self.labels_val_ = y_val
        else:
            # Auto-split using val_ratio — feature_matrix_ already
            # contains the full training data, so _retrain_at_step
            # will naturally retrain on all of it.
            X_sfd_train, X_sfd_test, y_sfd_train, y_sfd_test = train_test_split(
                self.feature_matrix_,
                y,
                test_size=self.val_ratio,
                random_state=42 if self.random_state is None else self.random_state,
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
            multiclass_type=self.multiclass_type,
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
    independently randomized MiniRocket transformer — fits them
    independently, and combines their predictions via soft or hard
    voting.  Also provides channel-relevance estimation for
    multivariate data.

    Parameters
    ----------
    num_models : int, default=25
        Number of Detach-ROCKET models in the ensemble.
    num_kernels : int, default=10_000
        Number of MiniRocket features per model (passed to the backend
        as ``num_features`` and rounded down to a multiple of 84, the
        number of fixed MiniRocket kernels).
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
    verbose : bool or int, default=False
        Print progress messages.  A value of ``2`` or higher also
        enables verbose output of the individual DetachRocket models.
    multiclass_type : str, default="max"
        Method to aggregate multi-class Ridge coefficients into a single
        feature-importance vector.  Forwarded to each inner
        :class:`DetachRocket`.  One of ``"max"`` (L∞, the aggregation
        used in the Detach-ROCKET paper), ``"norm"`` (L2), or ``"avg"``
        (L1).
    backend : {'pytorch', 'cuda'}, default='pytorch'
        Which MiniRocket implementation to use as the transformer.
        ``'pytorch'`` uses :class:`PytorchMiniRocketMultivariate`
        (requires PyTorch; runs on CPU or CUDA GPU), ``'cuda'`` uses
        :class:`CudaMiniRocketMultivariate` (requires CuPy + a CUDA GPU).
    random_state : int or None, default=None
        Seed controlling the ensemble's randomness: it derives an
        independent seed for every member's transformer (channel
        combinations and bias sampling) and fixes the internal
        train/validation split.  *None* keeps the transformers random
        while the internal split stays at its historical fixed seed (42).

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

    _BACKENDS = ("cuda", "pytorch")

    def __init__(
        self,
        num_models=25,
        num_kernels=10_000,
        trade_off=0.1,
        set_percentage=None,
        recompute_alpha=True,
        val_ratio=0.33,
        verbose=False,
        multiclass_type="max",
        backend="pytorch",
        random_state=None,
    ):
        backend = backend.lower()
        if backend not in self._BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from {self._BACKENDS}.")

        TransformerClass = self._get_transformer_class(backend)

        self.num_models = num_models
        self.num_kernels = num_kernels
        self.trade_off = trade_off
        self.set_percentage = set_percentage
        self.recompute_alpha = recompute_alpha
        self.val_ratio = val_ratio
        self.verbose = verbose
        self.multiclass_type = multiclass_type
        self.backend = backend
        self.random_state = random_state

        # One independent seed per member so the transformers differ from
        # each other but the whole ensemble is reproducible.
        seed_rng = np.random.default_rng(random_state)
        member_seeds = seed_rng.integers(0, 2**32 - 1, size=num_models)

        self.derockets = []
        for m in range(num_models):
            transformer = TransformerClass(num_features=num_kernels, random_state=int(member_seeds[m]))
            model = DetachRocket(
                transformer=transformer,
                trade_off=trade_off,
                set_percentage=set_percentage,
                recompute_alpha=recompute_alpha,
                verbose=(verbose > 1) if isinstance(verbose, int) and not isinstance(verbose, bool) else False,
                multiclass_type=multiclass_type,
            )
            self.derockets.append(model)

        self.label_encoder = LabelEncoder()
        self.is_fitted_ = False

    @staticmethod
    def _get_transformer_class(backend: str):
        """Import and return the transformer class for the given backend."""
        if backend == "cuda":
            from detach_rocket.cuda_minirocket import CudaMiniRocketMultivariate

            return CudaMiniRocketMultivariate
        else:  # "pytorch"
            try:
                from detach_rocket.pytorch_minirocket import PytorchMiniRocketMultivariate
            except ImportError as e:
                raise ImportError(
                    "The 'pytorch' backend requires PyTorch. Install it with: pip install \"detach_rocket[torch]\""
                ) from e
            return PytorchMiniRocketMultivariate

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit all ensemble members on the training data.

        When ``set_percentage`` is *None*, each member needs a
        validation set for pruning-level selection: pass one explicitly
        via *X_val* / *y_val* (e.g. a subject-wise split for MEG/EEG
        data), or leave them *None* to use a stratified random split of
        the training data (controlled by ``val_ratio``).  When
        ``set_percentage`` is set, the full training set is used and any
        provided validation set is ignored.

        Parameters
        ----------
        X : array-like of shape (n_instances, n_channels, n_timepoints)
            Multivariate training time series.
        y : array-like of shape (n_instances,)
            Training labels.
        X_val : array-like or None, default=None
            Optional explicit validation time series.
        y_val : array-like or None, default=None
            Validation labels (required when *X_val* is given).

        Returns
        -------
        self
        """
        if X_val is not None and y_val is None:
            raise ValueError("y_val is required when X_val is provided.")

        if self.backend == "pytorch":
            first_model = self.derockets[0].transformer
            if hasattr(first_model, "device") and first_model.device.type == "cpu":
                msg = "PyTorch is running on CPU as a GPU was not found. Dense operations are restricted to a single thread to prevent OpenMP deadlocks."
                warnings.warn(msg, UserWarning)

        if self.set_percentage is None:
            if X_val is not None:
                X_train, y_train = X, y
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=self.val_ratio,
                    random_state=42 if self.random_state is None else self.random_state,
                    stratify=y,
                )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        for idx, model in enumerate(self.derockets):
            if self.verbose:
                print(f"\rFitting Detach-ROCKET models: {idx + 1}/{self.num_models}...", end="", flush=True)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        if self.verbose:
            print()  # Clear the line after completion

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
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def estimate_channel_relevance(self, aggregation="mean"):
        """Estimate the relevance of each input channel.

        Distributes each feature's importance to the channels used by
        its corresponding kernel, normalizes the scores per ensemble
        member, and aggregates them across members.

        Parameters
        ----------
        aggregation : {'mean', 'median'}, default='mean'
            How to aggregate the per-model relevance scores.  The
            Detach-Rocket Ensemble paper (arXiv:2408.02760) uses
            ``'median'``; the default here is ``'mean'``, which is more
            stable when some models prune (almost) all features.  With
            ``'median'`` the aggregated scores are re-normalized to sum
            to 1, as in the paper.

        Returns
        -------
        channel_relevance : np.ndarray of shape (n_channels,)
            Normalized relevance score for each channel.
        """
        if aggregation not in ("mean", "median"):
            raise ValueError(f"aggregation={aggregation!r} is not valid. Use 'mean' or 'median'.")
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

        if aggregation == "mean":
            return np.mean(channel_relevance_matrix, axis=0)

        median_relevance = np.median(channel_relevance_matrix, axis=0)
        total = median_relevance.sum()
        return median_relevance / total if total > 0 else median_relevance

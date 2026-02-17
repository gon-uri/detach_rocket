# Test Plan

Test ideas collected from existing tests and the deleted `detach_rocket/test.py` script.
To be implemented as proper pytest tests at the end of the refactor.

## 1. `feature_detachment()` (sfd.py)

### 1a. Basic SFD with validation set
Verify output shapes, retained ratios, and that irrelevant features get pruned first.
```python
# Synthetic data: 5 relevant + 5 irrelevant features
X_relevant = np.random.randn(200, 5)
X_irrelevant = np.random.rand(200, 5)
X = np.hstack((X_relevant, X_irrelevant))
y = np.where(X[:, :5].sum(axis=1) > 0, 1, -1)

# After SFD, irrelevant features (columns 5–9) should be pruned first
assert (feature_matrix[5, 5:] == 0).all()
```

### 1b. SFD without validation set
Call with `X_test=None, y_test=None` — should return `None` for test scores.

### 1c. Multi-class (multilabel_type variants)
Test with 3+ classes, verify `"norm"`, `"max"`, `"avg"` all run without error.

### 1d. sklearn compat (binary classification)
Verify it works with both sklearn <1.6 (coef_ shape `(1, n_features)`) and >=1.6.1 (coef_ shape `(n_features,)`).

## 2. `select_optimal_pruning()` (model_selection.py)

### 2a. Returns valid index
`max_index` should be in range, `max_percentage` between 0 and 1.

### 2b. trade_off=0 selects highest accuracy
With zero trade-off, should pick the step with best validation accuracy.

### 2c. Large trade_off selects smallest model
With very large trade-off, should pick a heavily pruned step.

## 3. `retrain_optimal_model()` (model_selection.py)

### 3a. Alpha recomputation
When `model_alpha=None`, alpha should be recomputed via CV.

### 3b. Fixed alpha
When `model_alpha` is provided, the returned classifier should use that alpha.

## 4. `RocketTransformerPruner` (pruner.py)

### 4a. Pruned kernel count matches mask
```python
retained_num_kernels = np.sum(mask[0::2] | mask[1::2])
assert pruned_trf.num_kernels == retained_num_kernels
```

### 4b. Pruned transformer produces correct output shape
```python
pruned_features = pruned_trf.transform(X_train)
assert pruned_features.shape[1] == np.sum(mask)
```

### 4c. Pruned output matches full output at retained indices
Transform with full Rocket, select columns by mask; transform with PrunedRocket.
Results should be equal (or very close).

### 4d. Invalid transformer raises ValueError
```python
with pytest.raises(ValueError):
    get_transformer_pruner("InvalidTransformer")
```

### 4e. Unfitted transformer raises ValueError
```python
with pytest.raises(ValueError):
    pruner.prune_transformer(unfitted_rocket, mask)
```

## 5. `DetachRocket` (detach_classes.py)

### 5a. fit with optimal pruning (trade_off)
```python
dr = DetachRocket(transformer=Rocket(num_kernels=512))
dr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
assert dr._pruned_transformer is not None
assert dr._max_index >= 0
```

### 5b. fit with fixed percentage
```python
dr = DetachRocket(transformer=Rocket(num_kernels=512), set_percentage=50)
dr.fit(X_train, y_train)
```

### 5c. predict and score (once implemented)
Verify predict returns correct shape, score returns sensible values.

### 5d. Input validation
- Missing y raises ValueError
- Missing validation set (when no set_percentage) raises ValueError

## 6. `DetachMatrix` (detach_classes.py)

### 6a. fit / predict / score on pre-computed features
Same SFD pipeline but on a plain feature matrix.

### 6b. fit_trade_off calls select_optimal_pruning correctly
(Was broken — called undefined `select_optimal_model`.)

## 7. `DetachEnsemble` (detach_classes.py)

### 7a. fit / predict / predict_proba
Ensemble of multiple DetachRocket models, soft and hard voting.

### 7b. estimate_channel_relevance
Returns array of shape `(n_channels,)` summing to ~1.

---

**Note:** `DetachEnsemble` currently uses the old `DetachRocket` constructor API
(`model_type=...`) which is incompatible with the new refactored `DetachRocket`.
This must be fixed before these tests can run.

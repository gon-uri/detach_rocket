import numpy as np

from detach_rocket.detach_classes import DetachMatrix


def test_detach_matrix_get_summary():
    rng = np.random.default_rng(7)
    n_samples = 120
    n_features = 12
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, :4].sum(axis=1) > 0).astype(int)

    model = DetachMatrix(trade_off=0.1, recompute_alpha=False, val_ratio=0.2, verbose=False)
    model.fit(X, y)
    summary = model.get_summary()

    assert summary["estimator"] == "DetachMatrix"
    assert summary["is_fitted"] is True
    assert summary["selection_mode"] == "trade_off"
    assert summary["selected_feature_count"] <= summary["full_feature_count"]
    assert 0 <= summary["selected_ratio"] <= 1
    assert summary["full_model_alpha"] > 0
    assert summary["final_model_alpha"] > 0

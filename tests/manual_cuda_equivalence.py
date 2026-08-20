"""One-off GPU check for the CUDA pruner path (run on a machine with CuPy + CUDA).

Verifies, after fixing the import in get_transformer_pruner, that:
  1. get_transformer_pruner selects CudaMiniRocketTransformerPruner (not the generic fallback)
  2. The dilation-skipping pruned transform equals the full transform restricted to the mask
  3. A DetachRocket fitted with the CUDA backend gives identical predictions through
     the pruned path and the full+mask path

Usage:  python tests/manual_cuda_equivalence.py
"""

import numpy as np

from detach_rocket import DetachRocket
from detach_rocket.cuda_minirocket import CudaMiniRocketMultivariate
from detach_rocket.pruner import (
    CudaMiniRocketTransformerPruner,
    get_transformer_pruner,
)

rng = np.random.default_rng(0)
X = rng.standard_normal((40, 4, 128)).astype(np.float32)

t = CudaMiniRocketMultivariate(num_features=840)
F_full = t.fit(X).transform(X)
print("full features:", F_full.shape)

# random mask that empties at least one dilation block
mask = rng.random(t.num_features) < 0.3
offset = 0
first_block = t.num_kernels * int(t.num_features_per_dilation[0])
mask[offset : offset + first_block] = False  # force dilation 0 to be skipped
if not mask.any():
    mask[-5:] = True

pruner = get_transformer_pruner(t)
print("selected pruner:", type(pruner).__name__)
assert isinstance(pruner, CudaMiniRocketTransformerPruner), (
    "get_transformer_pruner did not select the CUDA pruner - import fix missing?"
)

pruned = pruner.prune_transformer(t, mask)
print("skipped dilations:", t.num_dilations - len(pruned.retained_dilation_indices), "of", t.num_dilations)
F_pruned = pruned.transform(X)
F_masked = F_full[:, mask]
print("pruned shape:", F_pruned.shape, "| masked shape:", F_masked.shape)
ok = np.allclose(F_pruned, F_masked, atol=1e-5)
print("pruned == full[:, mask]:", ok)
assert ok, "MISMATCH between pruned transform and masked full transform!"

# end-to-end through DetachRocket
y = (X[:, 0, :].mean(1) > 0).astype(int)
m = DetachRocket(transformer=CudaMiniRocketMultivariate(num_features=840), set_percentage=40)
m.fit(X, y)
full = np.asarray(m.transformer.transform(X))
eq = np.allclose(m._prepare_X(X), m.scaler_.transform(full)[:, m.feature_mask_], atol=1e-5)
print("DetachRocket pruned-path equivalence (cuda backend):", eq)
assert eq
print("ALL CUDA CHECKS PASSED")

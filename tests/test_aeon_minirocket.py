"""Tests for AeonMiniRocket, in particular its get_kernel_features maps.

The layout facts asserted here were established numerically against aeon 1.5
(brute-force reconstruction of sampled features) — never trust a
feature-order map from reading code alone.
"""

import numpy as np
import pytest

from detach_rocket.aeon_minirocket import AeonMiniRocket

N_SAMPLES = 12
N_CHANNELS = 3
N_TIMEPOINTS = 40
N_KERNELS_AEON = 336  # -> 4 features per kernel-position, 336 output features

WHICH_ALL = ("biases", "channels", "weights", "dilations", "paddings")


def _make_X(seed=3, n_channels=N_CHANNELS):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_SAMPLES, n_channels, N_TIMEPOINTS)).astype(np.float32)


@pytest.fixture(scope="module")
def fitted():
    """A fitted multivariate transformer, its input, and its output."""
    X = _make_X()
    transformer = AeonMiniRocket(n_kernels=N_KERNELS_AEON, random_state=11)
    features = np.asarray(transformer.fit_transform(X))
    return transformer, X, features


def _full_mask(features):
    return np.ones(features.shape[1], dtype=bool)


def test_layout_maps_match_bruteforce_transform(fitted):
    """The cornerstone check: recompute sampled features (symmetric dilated
    convolution over the map's claimed channels + PPV against the claimed
    bias, with the parity crop rule) and compare with the actual transform
    columns.  Jointly validates ordering, biases, channels, dilations, and
    paddings."""
    transformer, X, features = fitted
    mask = _full_mask(features)
    maps = {w: transformer.get_kernel_features(w, mask) for w in WHICH_ALL}

    dilations = np.asarray(transformer.parameters[-3], dtype=int)
    npd = np.asarray(transformer.parameters[-2], dtype=int)

    for dilation_index in range(len(dilations)):
        # First/adjacent/middle/last kernels cover both parities of the crop rule.
        for kernel_index in (0, 1, 42, 83):
            for quantile in range(int(npd[dilation_index])):
                f = int(84 * npd[:dilation_index].sum()) + kernel_index * int(npd[dilation_index]) + quantile

                bias = float(maps["biases"][f])
                dilation = int(maps["dilations"][f])
                padding = int(maps["paddings"][f])
                weights = maps["weights"][f]
                member = np.flatnonzero(maps["channels"][f] == 1.0)
                assert dilation == dilations[dilation_index]
                assert member.size > 0

                parity = (dilation_index % 2 + kernel_index) % 2
                for i in range(4):
                    series_sum = X[i].astype(np.float64)[member].sum(axis=0)
                    length = series_sum.shape[0]
                    C = np.zeros(length)
                    for t in range(length):
                        acc = 0.0
                        for j in range(9):
                            pos = t + (j - 4) * dilation
                            if 0 <= pos < length:
                                acc += weights[j] * series_sum[pos]
                        C[t] = acc
                    C_used = C[padding:-padding] if parity == 1 else C
                    expected_ppv = np.mean(C_used > bias)
                    assert np.isclose(expected_ppv, features[i, f], atol=1e-6), (
                        f"Feature map mismatch at dilation_index={dilation_index} "
                        f"kernel={kernel_index} quantile={quantile}: expected {expected_ppv}, got {features[i, f]}"
                    )


def test_channels_map_zeroing_invariance(fitted):
    """Zeroing the channels a feature does NOT use must leave it unchanged;
    zeroing a channel it does use must change it."""
    transformer, X, features = fitted
    channels_map = transformer.get_kernel_features("channels", _full_mask(features))

    candidates = [f for f in range(features.shape[1]) if 0 < channels_map[f].sum() < N_CHANNELS]
    candidates = candidates[:: max(1, len(candidates) // 20)]
    assert candidates, "Expected some kernels to use a strict subset of channels"

    for f in candidates:
        member = channels_map[f] == 1.0

        X_no_others = X.copy()
        X_no_others[:, ~member, :] = 0.0
        features_no_others = np.asarray(transformer.transform(X_no_others))
        assert np.allclose(features_no_others[:, f], features[:, f], atol=1e-6), (
            f"Column {f} changed when zeroing channels its kernel does not use"
        )

        X_no_member = X.copy()
        X_no_member[:, np.flatnonzero(member)[0], :] = 0.0
        features_no_member = np.asarray(transformer.transform(X_no_member))
        assert not np.allclose(features_no_member[:, f], features[:, f]), (
            f"Column {f} did not change when zeroing a channel its kernel uses"
        )


def test_channels_map_matches_parameters(fitted):
    """The channel map must equal an independent unpacking of the fitted
    combination arrays, in (dilation, kernel, quantile) order."""
    transformer, _, features = fitted
    channels_map = transformer.get_kernel_features("channels", _full_mask(features))

    n_cpc, ch_idx = transformer.parameters[0], transformer.parameters[1]
    npd = np.asarray(transformer.parameters[-2], dtype=int)
    offsets = np.concatenate([[0], np.cumsum(n_cpc)])

    f = 0
    for dilation_index in range(len(npd)):
        for kernel_index in range(84):
            combination = dilation_index * 84 + kernel_index
            expected = np.zeros(N_CHANNELS)
            expected[ch_idx[offsets[combination] : offsets[combination + 1]]] = 1.0
            for _ in range(int(npd[dilation_index])):
                assert np.array_equal(channels_map[f], expected)
                f += 1
    assert f == features.shape[1]


def test_scalar_maps_and_masking(fitted):
    """Biases/dilations/paddings mirror the parameter arrays; masked-out
    rows are NaN; weights rows are the fixed MiniRocket kernels."""
    transformer, _, features = fitted
    mask = _full_mask(features)
    dilations = np.asarray(transformer.parameters[-3], dtype=float)
    npd = np.asarray(transformer.parameters[-2], dtype=int)
    biases = np.asarray(transformer.parameters[-1], dtype=float)

    assert np.allclose(transformer.get_kernel_features("biases", mask), biases)
    assert np.array_equal(transformer.get_kernel_features("dilations", mask), np.repeat(dilations, 84 * npd))
    assert np.array_equal(
        transformer.get_kernel_features("paddings", mask), np.repeat((9 - 1) * dilations // 2, 84 * npd)
    )

    weights = transformer.get_kernel_features("weights", mask)
    assert weights.shape == (features.shape[1], 9)
    assert np.all(np.sort(weights, axis=1) == np.array([-1.0] * 6 + [2.0] * 3))

    partial = mask.copy()
    partial[::3] = False
    for which in WHICH_ALL:
        out = transformer.get_kernel_features(which, partial)
        assert np.all(np.isnan(out[::3]))
        assert not np.any(np.isnan(out[partial]))


def test_univariate_channels_all_ones():
    """Univariate combinations are necessarily all size 1 on channel 0."""
    X = _make_X(seed=5, n_channels=1)
    transformer = AeonMiniRocket(n_kernels=N_KERNELS_AEON, random_state=2)
    features = np.asarray(transformer.fit_transform(X))

    channels = transformer.get_kernel_features("channels", _full_mask(features))
    assert channels.shape == (features.shape[1], 1)
    assert np.all(channels == 1.0)


def test_invalid_arguments_raise(fitted):
    transformer, _, features = fitted
    mask = _full_mask(features)

    with pytest.raises(ValueError, match="not recognized"):
        transformer.get_kernel_features("poolings", mask)  # MultiRocket-only name
    with pytest.raises(ValueError, match="shape"):
        transformer.get_kernel_features("biases", mask[:-1])

    unfitted = AeonMiniRocket(n_kernels=N_KERNELS_AEON)
    with pytest.raises(ValueError, match="fitted"):
        unfitted.get_kernel_features("biases", mask)


def test_random_state_reproducibility():
    """Same seed -> identical features; different seed -> different features."""
    X = _make_X(seed=8)
    features_1 = np.asarray(AeonMiniRocket(n_kernels=N_KERNELS_AEON, random_state=7).fit_transform(X))
    features_2 = np.asarray(AeonMiniRocket(n_kernels=N_KERNELS_AEON, random_state=7).fit_transform(X))
    features_3 = np.asarray(AeonMiniRocket(n_kernels=N_KERNELS_AEON, random_state=8).fit_transform(X))

    assert np.array_equal(features_1, features_2)
    assert not np.array_equal(features_1, features_3)

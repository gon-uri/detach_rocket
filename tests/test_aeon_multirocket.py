"""Tests for AeonMultiRocket, in particular its get_kernel_features maps.

The layout facts asserted here (block order, in-block ordering, the base
channel combinations being reused for the diff half, the diff half's
asymmetric convolution) were established numerically against aeon 1.5 —
never trust a feature-order map from reading code alone.
"""

import numpy as np
import pytest

from detach_rocket.aeon_multirocket import AeonMultiRocket

N_SAMPLES = 12
N_CHANNELS = 3
N_TIMEPOINTS = 40
N_KERNELS_AEON = 168  # -> 2 features per kernel-position, 1_344 output features

WHICH_ALL = ("biases", "channels", "weights", "dilations", "paddings", "poolings", "representations")


def _make_X(seed=3, n_channels=N_CHANNELS):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_SAMPLES, n_channels, N_TIMEPOINTS)).astype(np.float32)


@pytest.fixture(scope="module")
def fitted():
    """A fitted multivariate transformer, its input, and its output."""
    X = _make_X()
    transformer = AeonMultiRocket(n_kernels=N_KERNELS_AEON, random_state=11)
    features = np.asarray(transformer.fit_transform(X))
    return transformer, X, features


def _full_mask(features):
    return np.ones(features.shape[1], dtype=bool)


# -- Brute-force layout verification ------------------------------------------


def _manual_conv(series_sum, kernel_weights, dilation, is_diff):
    """Recompute aeon's dilated 'same' convolution for one kernel.

    For the diff representation aeon reuses ``end = n_timepoints - padding``
    on the length ``L - 1`` arrays, which shifts the taps left of center one
    sample to the right of the symmetric position.
    """
    length = series_sum.shape[0]
    left_shift = 1 if is_diff else 0
    C = np.zeros(length)
    for t in range(length):
        acc = 0.0
        for j in range(9):
            pos = t + (j - 4) * dilation + (left_shift if j < 4 else 0)
            if 0 <= pos < length:
                acc += kernel_weights[j] * series_sum[pos]
        C[t] = acc
    return C


def _manual_poolings(C, bias):
    """Replicate aeon's pooling loop exactly: returns (ppv, lspv, mpv, mipv)."""
    ppv = 0
    last_val = 0
    max_stretch = 0.0
    mean_index = 0
    mean = 0.0
    for j in range(C.shape[0]):
        if C[j] > bias:
            ppv += 1
            mean_index += j
            mean += C[j] + bias
        elif C[j] < bias:
            stretch = j - last_val
            if stretch > max_stretch:
                max_stretch = stretch
            last_val = j
    stretch = C.shape[0] - 1 - last_val
    if stretch > max_stretch:
        max_stretch = stretch
    return np.array(
        [
            ppv / C.shape[0],
            max_stretch,
            mean / ppv if ppv > 0 else 0.0,
            mean_index / ppv if ppv > 0 else -1.0,
        ]
    )


def test_layout_maps_match_bruteforce_transform(fitted):
    """The cornerstone check: for sampled features of both representations,
    recompute the convolution and all 4 pooling operators from the maps'
    claimed parameters and compare with the actual transform columns.

    This jointly validates the in-block ordering, the pooling-block order
    (PPV, LSPV, MPV, MIPV), the biases, channels, dilations, and paddings
    maps, and the parity cropping rule.
    """
    transformer, X, features = fitted
    n_rep = features.shape[1] // 8
    mask = _full_mask(features)
    maps = {w: transformer.get_kernel_features(w, mask) for w in ("biases", "channels", "dilations", "paddings")}
    weights_map = transformer.get_kernel_features("weights", mask)

    for rep, params in ((0, transformer.parameter), (1, transformer.parameter1)):
        dilations = np.asarray(params[-3], dtype=int)
        npd = np.asarray(params[-2], dtype=int)
        for dilation_index in range(len(dilations)):
            # First/middle/last kernels cover both parities of the crop rule.
            for kernel_index in (0, 1, 42, 83):
                for quantile in range(int(npd[dilation_index])):
                    f = int(84 * npd[:dilation_index].sum()) + kernel_index * int(npd[dilation_index]) + quantile
                    col = rep * 4 * n_rep + f

                    bias = float(maps["biases"][col])
                    dilation = int(maps["dilations"][col])
                    padding = int(maps["paddings"][col])
                    member = np.flatnonzero(maps["channels"][col] == 1.0)
                    assert dilation == dilations[dilation_index]
                    assert member.size > 0

                    parity = (dilation_index % 2 + kernel_index) % 2
                    for i in range(4):
                        series = X[i].astype(np.float64)
                        if rep:
                            series = np.diff(series, axis=-1)
                        C = _manual_conv(series[member].sum(axis=0), weights_map[col], dilation, is_diff=bool(rep))
                        C_used = C[padding:-padding] if parity == 1 else C
                        expected = _manual_poolings(C_used, bias)
                        actual = np.array([features[i, rep * 4 * n_rep + block * n_rep + f] for block in range(4)])
                        assert np.allclose(expected, actual, rtol=1e-4, atol=1e-4), (
                            f"Feature map mismatch at rep={rep} dilation_index={dilation_index} "
                            f"kernel={kernel_index} quantile={quantile}: expected {expected}, got {actual}"
                        )


def test_channels_map_zeroing_invariance(fitted):
    """Zeroing the channels a feature does NOT use must leave it unchanged;
    zeroing a channel it does use must change it."""
    transformer, X, features = fitted
    mask = _full_mask(features)
    channels_map = transformer.get_kernel_features("channels", mask)
    n_rep = features.shape[1] // 8

    # Sample PPV-block features (base and diff) with a strict channel subset.
    candidates = [
        (rep, f) for rep in (0, 1) for f in range(n_rep) if 0 < channels_map[rep * 4 * n_rep + f].sum() < N_CHANNELS
    ]
    candidates = candidates[:: max(1, len(candidates) // 20)]
    assert candidates, "Expected some kernels to use a strict subset of channels"

    for rep, f in candidates:
        col = rep * 4 * n_rep + f
        member = channels_map[col] == 1.0
        cols_all_poolings = [rep * 4 * n_rep + block * n_rep + f for block in range(4)]

        X_no_others = X.copy()
        X_no_others[:, ~member, :] = 0.0
        features_no_others = np.asarray(transformer.transform(X_no_others))
        assert np.allclose(features_no_others[:, cols_all_poolings], features[:, cols_all_poolings], atol=1e-6), (
            f"Column {col} changed when zeroing channels its kernel does not use"
        )

        X_no_member = X.copy()
        X_no_member[:, np.flatnonzero(member)[0], :] = 0.0
        features_no_member = np.asarray(transformer.transform(X_no_member))
        assert not np.allclose(features_no_member[:, cols_all_poolings], features[:, cols_all_poolings]), (
            f"Column {col} did not change when zeroing a channel its kernel uses"
        )


# -- Structural map checks -----------------------------------------------------


def test_channels_map_uses_base_combinations_for_both_halves(fitted):
    """aeon's transform applies the BASE channel combinations to the diff
    half as well (parameter1's combinations only shape bias fitting)."""
    transformer, _, features = fitted
    mask = _full_mask(features)
    channels_map = transformer.get_kernel_features("channels", mask)
    n_rep = features.shape[1] // 8

    n_cpc, ch_idx = transformer.parameter[0], transformer.parameter[1]
    offsets = np.concatenate([[0], np.cumsum(n_cpc)])

    for rep, params in ((0, transformer.parameter), (1, transformer.parameter1)):
        npd = np.asarray(params[-2], dtype=int)
        col = rep * 4 * n_rep
        for dilation_index in range(len(npd)):
            for kernel_index in range(84):
                combination = dilation_index * 84 + kernel_index
                expected = np.zeros(N_CHANNELS)
                expected[ch_idx[offsets[combination] : offsets[combination + 1]]] = 1.0
                for _ in range(int(npd[dilation_index])):
                    assert np.array_equal(channels_map[col], expected)
                    col += 1


def test_masked_rows_are_nan(fitted):
    transformer, _, features = fitted
    mask = _full_mask(features)
    mask[::3] = False
    for which in WHICH_ALL:
        out = transformer.get_kernel_features(which, mask)
        assert np.all(np.isnan(out[::3]))
        kept = out[mask]
        assert not np.any(np.isnan(kept))


def test_scalar_maps_tile_parameters(fitted):
    """Biases/dilations/paddings repeat each representation's parameter
    arrays across its 4 pooling blocks; poolings/representations label the
    blocks themselves."""
    transformer, _, features = fitted
    mask = _full_mask(features)
    n_rep = features.shape[1] // 8

    for rep, params in ((0, transformer.parameter), (1, transformer.parameter1)):
        dilations = np.asarray(params[-3], dtype=float)
        npd = np.asarray(params[-2], dtype=int)
        biases = np.asarray(params[-1], dtype=float)
        half = slice(rep * 4 * n_rep, (rep + 1) * 4 * n_rep)

        assert np.allclose(transformer.get_kernel_features("biases", mask)[half], np.tile(biases, 4))
        expected_dilations = np.repeat(dilations, 84 * npd)
        assert np.array_equal(transformer.get_kernel_features("dilations", mask)[half], np.tile(expected_dilations, 4))
        expected_paddings = np.repeat((9 - 1) * dilations // 2, 84 * npd)
        assert np.array_equal(transformer.get_kernel_features("paddings", mask)[half], np.tile(expected_paddings, 4))
        assert np.array_equal(
            transformer.get_kernel_features("poolings", mask)[half], np.repeat([0.0, 1.0, 2.0, 3.0], n_rep)
        )
        assert np.all(transformer.get_kernel_features("representations", mask)[half] == float(rep))

    assert AeonMultiRocket.POOLING_NAMES == ("ppv", "lspv", "mpv", "mipv")


def test_weights_map_is_minirocket_kernel(fitted):
    """Each row has 6 taps at -1 and 3 taps at +2, matching the kernel index."""
    transformer, _, features = fitted
    weights = transformer.get_kernel_features("weights", _full_mask(features))
    assert weights.shape == (features.shape[1], 9)
    assert np.all(np.sort(weights, axis=1) == np.array([-1.0] * 6 + [2.0] * 3))


def test_univariate_channels_all_ones():
    """Univariate fits draw no channel combinations; channel 0 is used everywhere."""
    X = _make_X(seed=5, n_channels=1)
    transformer = AeonMultiRocket(n_kernels=N_KERNELS_AEON, random_state=2)
    features = np.asarray(transformer.fit_transform(X))
    assert len(transformer.parameter) == 3

    channels = transformer.get_kernel_features("channels", _full_mask(features))
    assert channels.shape == (features.shape[1], 1)
    assert np.all(channels == 1.0)


# -- Validation and reproducibility -------------------------------------------


def test_invalid_arguments_raise(fitted):
    transformer, _, features = fitted
    mask = _full_mask(features)

    with pytest.raises(ValueError, match="not recognized"):
        transformer.get_kernel_features("quantiles", mask)
    with pytest.raises(ValueError, match="shape"):
        transformer.get_kernel_features("biases", mask[:-1])

    unfitted = AeonMultiRocket(n_kernels=N_KERNELS_AEON)
    with pytest.raises(ValueError, match="fitted"):
        unfitted.get_kernel_features("biases", mask)


def test_short_series_and_pooling_count_raise():
    X = _make_X()
    with pytest.raises(ValueError, match="n_timepoints >= 10"):
        AeonMultiRocket(n_kernels=N_KERNELS_AEON).fit(X[:, :, :9])
    with pytest.raises(ValueError, match="n_features_per_kernel=4"):
        AeonMultiRocket(n_kernels=N_KERNELS_AEON, n_features_per_kernel=3).fit(X)


def test_random_state_reproducibility():
    """Same seed -> identical features; different seed -> different features."""
    X = _make_X(seed=8)
    features_1 = np.asarray(AeonMultiRocket(n_kernels=N_KERNELS_AEON, random_state=7).fit_transform(X))
    features_2 = np.asarray(AeonMultiRocket(n_kernels=N_KERNELS_AEON, random_state=7).fit_transform(X))
    features_3 = np.asarray(AeonMultiRocket(n_kernels=N_KERNELS_AEON, random_state=8).fit_transform(X))

    assert np.array_equal(features_1, features_2)
    assert not np.array_equal(features_1, features_3)

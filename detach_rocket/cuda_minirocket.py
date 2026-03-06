"""
CUDA (CuPy) implementation of MiniRocket for multivariate time series.

Drop-in API replacement for PytorchMiniRocketMultivariate, but:
  - No PyTorch dependency
  - Uses CuPy RawKernel + NumPy
  - Returns NumPy arrays from forward/transform

Implements the same:
  - 84 fixed MiniRocket kernels (length 9, three +2 weights, rest -1)
  - dilation/padding schedule
  - random channel combinations per dilation (max 9 channels)
  - bias definition (quantiles of diagonal-sampled convolution traces)
  - feature ordering (parity split per dilation, with cropped second half)

Requirements:
  - cupy (CUDA)
  - numpy
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List

import numpy as np

try:
    import cupy as cp
except Exception as e:
    cp = None

try:
    import torch
    import torch.nn as nn
except Exception as e:
    torch = None
    nn = None

ArrayLike = Union[np.ndarray, "cp.ndarray"]

@dataclass
class _PackedChannels:
    chan_idx: "cp.ndarray"  # (84, 9) int32, padded with -1
    chan_cnt: "cp.ndarray"  # (84,) int32


class CudaMiniRocketMultivariate:
    """CuPy/CUDA implementation of MiniRocket for multivariate time series.

    Plug-and-play replacement (API-level) for PytorchMiniRocketMultivariate:
      - __init__(num_features=10_000, max_dilations_per_kernel=32, device=None)
      - fit(X, chunksize=128)
      - forward(x)
      - transform(o, chunksize=128)
      - fit_transform(o)
      - get_kernel_features(which, where)

    Notes:
      - Outputs are NumPy arrays (float32).
      - Inputs can be NumPy or CuPy; internally everything runs on GPU (CuPy).
      - `device` is accepted for signature compatibility but is not used; CuPy
        selects the current CUDA device context (use cp.cuda.Device if needed).
    """

    kernel_size, num_kernels = 9, 84

    def __init__(self, num_features: int = 10_000, max_dilations_per_kernel: int = 32, device=None):
        if cp is None:
            raise ImportError("cupy is required for CudaMiniRocketMultivariate.")

        self.num_features = int(num_features)
        self.max_dilations_per_kernel = int(max_dilations_per_kernel)
        self.device = device  # kept only for signature parity; not used

        # learned/fit state
        self.c_in: Optional[int] = None
        self.seq_len: Optional[int] = None

        self.num_dilations: Optional[int] = None
        self.dilations: Optional[np.ndarray] = None          # (D,) int32
        self.padding: Optional[List[int]] = None             # len D

        self.num_features_per_dilation: Optional[np.ndarray] = None  # (D,) int32
        self.prefit: bool = False

        # parameters stored for inspection / parity with PyTorch class
        self.base_kernels_np: Optional[np.ndarray] = None     # (84,9) float32
        self.kernels: Optional[np.ndarray] = None             # (c_in*84,1,9) float32, repeated as in torch
        self.channel_combinations: Dict[int, np.ndarray] = {} # i -> (1,C,84,1) float32 (numpy)
        self.biases: Dict[int, np.ndarray] = {}               # i -> (84, q_i) float32 (numpy)

        # indices for get_kernel_features (numpy int32)
        self.kernel_indices: Dict[int, np.ndarray] = {}       # i -> (84*q_i,) int32
        self.bias_indices: Dict[int, np.ndarray] = {}         # i -> (84*q_i,) int32

        # packed channel indices for CUDA
        self._packed_channels: Dict[int, _PackedChannels] = {}

        # CUDA compilation caches
        self._cuda_bias_kernels: Dict[Tuple[int, int, int, int], "cp.RawKernel"] = {}
        self._cuda_ppv_kernels: Dict[Tuple[int, int, int, int, int], "cp.RawKernel"] = {}

        # cached GPU constants
        self._base_kernels_cp: Optional["cp.ndarray"] = None  # (84,9) float32

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def fit(self, X: ArrayLike, chunksize: int = 128):
        """
        Fit computes:
          - dilation/padding schedule
          - random channel combinations per dilation
          - biases per dilation (84 x q_i)
          - feature index mappings for get_kernel_features

        `chunksize` is accepted for signature parity; bias computation in MiniRocket
        does not require chunking (it samples only 84 instances per dilation).
        """
        X_np_shape = self._shape_of(X)
        if len(X_np_shape) != 3:
            raise ValueError(f"Expected X with shape (N, C, L), got {X_np_shape}")

        N, C, L = map(int, X_np_shape)
        self.c_in, self.seq_len = C, L

        # round features down to multiple of 84
        self.num_features = (self.num_features // self.num_kernels) * self.num_kernels

        # build the fixed kernel bank (84,9)
        self.base_kernels_np = self._build_minirocket_kernels_np()  # float32
        self.kernels = np.repeat(self.base_kernels_np[:, None, :], repeats=self.c_in, axis=0).astype(np.float32)
        # shape is (c_in*84, 1, 9) as in torch after repeat(c_in,1,1)

        # dilations/padding schedule
        self._set_dilations(self.seq_len)

        # random channel combinations per dilation
        if self.c_in > 1:
            self._set_channel_combinations(self.c_in)
        else:
            self.channel_combinations = {}

        # upload base kernels to GPU once
        self._base_kernels_cp = cp.asarray(self.base_kernels_np, dtype=cp.float32)

        # upload data to GPU (float32, contiguous)
        X_cp = self._to_cupy_float32(X)

        # compute biases per dilation
        for i, (dilation, padding) in enumerate(zip(self.dilations.tolist(), self.padding)):
            q_i = int(self.num_features_per_dilation[i])

            # pack channel combinations for CUDA access
            packed = self._pack_channels_for_dilation(i)
            self._packed_channels[i] = packed

            # sample one instance per kernel (84 indices)
            if N >= self.num_kernels:
                idxs = np.random.choice(N, self.num_kernels, replace=False).astype(np.int32)
            else:
                idxs = np.random.choice(N, self.num_kernels, replace=True).astype(np.int32)

            biases_i_cp = self._compute_biases_for_dilation_cuda(
                X_cp=X_cp,
                dilation=int(dilation),
                padding=int(padding),
                q=q_i,
                idxs=idxs,
                packed_channels=packed,
            )
            self.biases[i] = cp.asnumpy(biases_i_cp)  # (84,q_i) float32

        # indices for get_kernel_features
        self._set_parameter_indices()

        self.prefit = True
        return self

    def forward(self, x: ArrayLike) -> np.ndarray:
        """Compute MiniRocket features. Returns NumPy array float32 (B, num_features)."""
        if not self.prefit:
            raise RuntimeError("CudaMiniRocketMultivariate must be fit() before forward().")

        x_cp = self._to_cupy_float32(x)
        B, C, L = map(int, x_cp.shape)
        if C != self.c_in or L != self.seq_len:
            raise ValueError(f"Input shape mismatch: expected (*,{self.c_in},{self.seq_len}) but got {(B,C,L)}")

        outputs = []
        for i, (dilation, padding) in enumerate(zip(self.dilations.tolist(), self.padding)):
            q_i = int(self.num_features_per_dilation[i])
            parity = int(i & 1)

            biases_i_cp = cp.asarray(self.biases[i], dtype=cp.float32)  # (84,q_i)

            packed = self._packed_channels[i]

            # kernel order matches torch concatenation:
            # first k with k%2 == parity, then k%2 != parity
            order = np.array(
                [k for k in range(self.num_kernels) if (k & 1) == parity]
                + [k for k in range(self.num_kernels) if (k & 1) != parity],
                dtype=np.int32,
            )
            order_cp = cp.asarray(order, dtype=cp.int32)

            out_i = cp.empty((B, self.num_kernels * q_i), dtype=cp.float32)
            self._ppv_transform_cuda(
                X_cp=x_cp,
                W_cp=self._base_kernels_cp,
                packed_channels=packed,
                BIAS_cp=biases_i_cp,
                dilation=int(dilation),
                padding=int(padding),
                q=q_i,
                parity=parity,
                korder_cp=order_cp,
                OUT_cp=out_i,
            )
            outputs.append(out_i)

        feats = cp.concatenate(outputs, axis=1)  # (B, num_features)
        return cp.asnumpy(feats)

    def transform(self, o: ArrayLike, chunksize: int = 128) -> np.ndarray:
        """Chunked transform. Returns NumPy float32 array (N, num_features)."""
        if not self.prefit:
            raise RuntimeError("CudaMiniRocketMultivariate must be fit() before transform().")

        o_np_shape = self._shape_of(o)
        if len(o_np_shape) != 3:
            raise ValueError(f"Expected o with shape (N,C,L), got {o_np_shape}")

        N = int(o_np_shape[0])
        if chunksize is None:
            chunksize = N
        chunksize = int(chunksize)

        out_chunks = []
        for start in range(0, N, chunksize):
            stop = min(N, start + chunksize)
            chunk = self._slice_first_axis(o, start, stop)
            out_chunks.append(self.forward(chunk))
        return np.concatenate(out_chunks, axis=0)

    def fit_transform(self, o: ArrayLike) -> np.ndarray:
        return self.fit(o).transform(o)

    def get_kernel_features(self, which: str, where: np.ndarray) -> np.ndarray:
        """
        Match the PyTorch helper. Returns np.where(where, full_features, np.nan).

        which in {"biases", "channels", "weights", "dilations", "paddings"}.
        where: boolean mask over flattened feature index space.
        """
        if not self.prefit:
            raise RuntimeError("Must fit() before calling get_kernel_features().")

        full_features: np.ndarray

        if which == "channels":
            full_features = np.empty(shape=(0, self.c_in), dtype=float)
            where = where[:, np.newaxis]
            where = np.repeat(where, self.c_in, axis=1)
        elif which == "weights":
            full_features = np.empty(shape=(0, self.kernel_size), dtype=float)
            where = where[:, np.newaxis]
            where = np.repeat(where, self.kernel_size, axis=1)
        else:
            full_features = np.empty(shape=(0,), dtype=float)

        for i, (dilation, padding) in enumerate(zip(self.dilations.tolist(), self.padding)):
            biases_i = self.biases[i]  # (84, q_i)
            q_i = int(biases_i.shape[1])

            kernel_idx = self.kernel_indices[i]  # (84*q_i,)
            bias_idx = self.bias_indices[i]      # (84*q_i,)

            if which == "biases":
                sorted_biases = biases_i.reshape(-1)[bias_idx]
                full_features = np.append(full_features, sorted_biases.astype(float, copy=False))

            elif which == "channels":
                if self.c_in == 1:
                    # univariate: always channel 0
                    channel_combinations_q = np.ones((self.num_kernels, 1), dtype=float)
                    for _ in range(q_i):
                        full_features = np.append(full_features, channel_combinations_q, axis=0)
                else:
                    cc = self.channel_combinations[i]  # (1,C,84,1)
                    cc2 = cc.squeeze(0).squeeze(-1)    # (C,84)
                    # replicate the PyTorch logic: for each quantile, select kernels in order and append (84,C)
                    for q in range(q_i):
                        selected = kernel_idx[q * self.num_kernels: q * self.num_kernels + self.num_kernels]
                        cc_sel = cc2[:, selected]              # (C,84)
                        cc_sel = np.transpose(cc_sel, (1, 0))  # (84,C)
                        full_features = np.append(full_features, cc_sel.astype(float, copy=False), axis=0)

            elif which == "weights":
                # kernels are equal for all channels; pick base weights (84,9)
                weights = self.base_kernels_np.astype(float, copy=False)
                for q in range(q_i):
                    selected = kernel_idx[q * self.num_kernels: q * self.num_kernels + self.num_kernels]
                    full_features = np.append(full_features, weights[selected], axis=0)

            elif which == "dilations":
                expanded = np.repeat(int(dilation), self.num_kernels * q_i).astype(float)
                full_features = np.append(full_features, expanded)

            elif which == "paddings":
                expanded = np.repeat(int(padding), self.num_kernels * q_i).astype(float)
                full_features = np.append(full_features, expanded)

            else:
                raise ValueError(
                    f'"{which}" is not recognized. Use "biases", "channels", "weights", "dilations" or "paddings".'
                )

        return np.where(where, full_features, np.nan)

    # ---------------------------------------------------------------------
    # Internals: exact logic parity with PyTorch reference
    # ---------------------------------------------------------------------

    def _set_parameter_indices(self):
        """
        Numpy re-implementation of the reference _set_parameter_indices, producing
        flattened kernel and bias indices matching the feature ordering.
        """
        for i, (dilation, padding) in enumerate(zip(self.dilations.tolist(), self.padding)):
            _padding1 = i & 1

            biases_i = self.biases[i]  # (84,q)
            num_kernels, num_quantiles = biases_i.shape

            bias_indices = np.arange(num_kernels * num_quantiles, dtype=np.int32).reshape(num_quantiles, num_kernels).T
            kernel_indices = np.repeat(np.arange(num_kernels, dtype=np.int32)[:, None], repeats=num_quantiles, axis=1)

            # even group in PyTorch corresponds to [_padding1::2]
            C_even = kernel_indices[_padding1::2, :]          # (42,q)
            B_even = bias_indices[_padding1::2, :]            # (42,q)
            C_even = C_even.reshape(-1)                       # (42*q,)
            B_even = B_even.reshape(-1)                       # (42*q,)

            C_odd = kernel_indices[1 - _padding1 :: 2, :]      # (42,q)
            B_odd = bias_indices[1 - _padding1 :: 2, :]
            C_odd = C_odd.reshape(-1)
            B_odd = B_odd.reshape(-1)

            C_full = np.concatenate([C_even, C_odd], axis=0)   # (84*q,)
            B_full = np.concatenate([B_even, B_odd], axis=0)

            self.kernel_indices[i] = C_full
            self.bias_indices[i] = B_full

    def _set_dilations(self, input_length: int):
        num_features_per_kernel = self.num_features // self.num_kernels
        true_max_dilations_per_kernel = min(num_features_per_kernel, self.max_dilations_per_kernel)
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel

        max_exponent = np.log2((input_length - 1) / (self.kernel_size - 1))
        dilations, counts = np.unique(
            np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
            return_counts=True,
        )
        num_features_per_dilation = (counts * multiplier).astype(np.int32)

        remainder = int(num_features_per_kernel - num_features_per_dilation.sum())
        j = 0
        while remainder > 0:
            num_features_per_dilation[j] += 1
            remainder -= 1
            j = (j + 1) % len(num_features_per_dilation)

        self.num_features_per_dilation = num_features_per_dilation
        self.num_dilations = int(len(dilations))
        self.dilations = dilations.astype(np.int32)
        self.padding = [int(((self.kernel_size - 1) * int(d)) // 2) for d in self.dilations]

    def _set_channel_combinations(self, num_channels: int):
        """
        Matches the PyTorch construction:
          - number of combinations = 84 * num_dilations
          - for each combination, choose k ~ 2^U(0, log2(max_num_channels+1))
          - set those channels to 1 without replacement
          - split into per-dilation tensors of shape (1,C,84,1)
        """
        num_combinations = self.num_kernels * self.num_dilations
        max_num_channels = min(num_channels, 9)
        max_exponent_channels = np.log2(max_num_channels + 1)

        num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent_channels, num_combinations)).astype(
            np.int32
        )

        cc = np.zeros((1, num_channels, num_combinations, 1), dtype=np.float32)
        for i in range(num_combinations):
            k = int(num_channels_per_combination[i])
            if k <= 0:
                k = 1
            chosen = np.random.choice(num_channels, k, replace=False)
            cc[:, chosen, i, :] = 1.0

        # split by dilation: each slice is (1,C,84,1)
        self.channel_combinations = {}
        for d in range(self.num_dilations):
            start = d * self.num_kernels
            stop = (d + 1) * self.num_kernels
            self.channel_combinations[d] = cc[:, :, start:stop, :].copy()

    def _get_quantiles_np(self, n: int) -> np.ndarray:
        phi = (math.sqrt(5.0) + 1.0) / 2.0
        return np.array([((k * phi) % 1.0) for k in range(1, n + 1)], dtype=np.float32)

    # ---------------------------------------------------------------------
    # CUDA kernels: bias sampling + fused PPV transform
    # ---------------------------------------------------------------------

    def _pack_channels_for_dilation(self, dilation_index: int) -> _PackedChannels:
        """
        Pack per-kernel channel selections into fixed-size arrays for CUDA:

          chan_idx[k, :]  = up to 9 channel indices used by kernel k (padded with -1)
          chan_cnt[k]     = number of active channels for kernel k

        For univariate, always channel 0.
        """
        if self.c_in == 1:
            chan_idx = np.full((self.num_kernels, 9), -1, dtype=np.int32)
            chan_idx[:, 0] = 0
            chan_cnt = np.ones((self.num_kernels,), dtype=np.int32)
            return _PackedChannels(
                chan_idx=cp.asarray(chan_idx, dtype=cp.int32),
                chan_cnt=cp.asarray(chan_cnt, dtype=cp.int32),
            )

        cc = self.channel_combinations[dilation_index]        # (1,C,84,1) float32
        cc2 = cc.squeeze(0).squeeze(-1)                       # (C,84)

        chan_idx = np.full((self.num_kernels, 9), -1, dtype=np.int32)
        chan_cnt = np.zeros((self.num_kernels,), dtype=np.int32)

        for k in range(self.num_kernels):
            active = np.flatnonzero(cc2[:, k] > 0.5).astype(np.int32)
            if active.size == 0:
                active = np.array([0], dtype=np.int32)
            if active.size > 9:
                active = active[:9]
            chan_cnt[k] = active.size
            chan_idx[k, : active.size] = active

        return _PackedChannels(
            chan_idx=cp.asarray(chan_idx, dtype=cp.int32),
            chan_cnt=cp.asarray(chan_cnt, dtype=cp.int32),
        )

    def _compile_bias_kernel(self, L: int, C: int, dilation: int, padding: int) -> "cp.RawKernel":
        key = (L, C, dilation, padding)
        if key in self._cuda_bias_kernels:
            return self._cuda_bias_kernels[key]

        calcs_full = L + 2 * padding - dilation * (self.kernel_size - 1)

        src = rf"""
        extern "C" __global__
        void bias_samples(const float* __restrict__ X,          // (N,C,L)
                          const float* __restrict__ W,          // (84,9)
                          const int*   __restrict__ chan_idx,   // (84,9)
                          const int*   __restrict__ chan_cnt,   // (84,)
                          const int*   __restrict__ idxs,       // (84,)
                          float* __restrict__ samples)          // (84, calcs_full)
        {{
            const int k = (int)blockIdx.x;      // kernel id [0..83]
            const int tid = (int)threadIdx.x;

            const int inst = idxs[k];
            const int calcs = {calcs_full};

            for (int t = tid; t < calcs; t += blockDim.x) {{
                float sum = 0.0f;

                const int cc = chan_cnt[k];
                #pragma unroll
                for (int jj = 0; jj < 9; jj++) {{
                    if (jj >= cc) break;
                    const int c = chan_idx[k*9 + jj];
                    if (c < 0) continue;

                    #pragma unroll
                    for (int i = 0; i < 9; i++) {{
                        const int sidx = t - {padding} + {dilation} * i;
                        if ((unsigned)sidx < (unsigned){L}) {{
                            const int base = inst * ({C}*{L}) + c*{L} + sidx;
                            sum += W[k*9 + i] * X[base];
                        }}
                    }}
                }}

                samples[k*calcs + t] = sum;
            }}
        }}
        """
        kern = cp.RawKernel(src, "bias_samples", options=("--std=c++11",))
        self._cuda_bias_kernels[key] = kern
        return kern

    def _compile_ppv_kernel(self, L: int, C: int, dilation: int, padding: int, q: int) -> "cp.RawKernel":
        key = (L, C, dilation, padding, q)
        if key in self._cuda_ppv_kernels:
            return self._cuda_ppv_kernels[key]

        calcs_full = L + 2 * padding - dilation * (self.kernel_size - 1)
        calcs_crop = max(calcs_full - 2 * padding, 1)  # guard for padding==0

        # Each block computes one (kernel position in ordered list, instance).
        # It accumulates PPV counts for q biases in shared memory using atomicAdd.
        src = rf"""
        extern "C" __global__
        void ppv_transform(const float* __restrict__ X,          // (B,C,L)
                           const float* __restrict__ W,          // (84,9)
                           const int*   __restrict__ chan_idx,   // (84,9)
                           const int*   __restrict__ chan_cnt,   // (84,)
                           const float* __restrict__ BIAS,       // (84,q)
                           const int*   __restrict__ korder,     // (84,)
                           const int    parity,                 // i%2
                           float* __restrict__ OUT)              // (B, 84*q)
        {{
            const int pos = (int)blockIdx.x;   // position in ordered kernel list [0..83]
            const int b   = (int)blockIdx.y;   // instance index
            const int tid = (int)threadIdx.x;

            __shared__ float counts[{q}];
            if (tid < {q}) counts[tid] = 0.0f;
            __syncthreads();

            const int k = korder[pos];
            const int use_full = ((k & 1) == parity);

            const int start = use_full ? 0 : {padding};
            const int calcs = use_full ? {calcs_full} : {calcs_crop};

            for (int tt = tid; tt < calcs; tt += blockDim.x) {{
                const int t = tt + start;
                float sum = 0.0f;

                const int cc = chan_cnt[k];
                #pragma unroll
                for (int jj = 0; jj < 9; jj++) {{
                    if (jj >= cc) break;
                    const int c = chan_idx[k*9 + jj];
                    if (c < 0) continue;

                    #pragma unroll
                    for (int i = 0; i < 9; i++) {{
                        const int sidx = t - {padding} + {dilation} * i;
                        if ((unsigned)sidx < (unsigned){L}) {{
                            const int base = b * ({C}*{L}) + c*{L} + sidx;
                            sum += W[k*9 + i] * X[base];
                        }}
                    }}
                }}

                #pragma unroll
                for (int j = 0; j < {q}; j++) {{
                    const float bias = BIAS[k*{q} + j];
                    if (sum > bias) atomicAdd(&counts[j], 1.0f);
                }}
            }}

            __syncthreads();

            if (tid < {q}) {{
                const float inv = 1.0f / (float)calcs;
                OUT[b * (84*{q}) + pos*{q} + tid] = counts[tid] * inv;
            }}
        }}
        """
        kern = cp.RawKernel(src, "ppv_transform", options=("--std=c++11",))
        self._cuda_ppv_kernels[key] = kern
        return kern

    def _compute_biases_for_dilation_cuda(
        self,
        X_cp: "cp.ndarray",                 # (N,C,L) float32
        dilation: int,
        padding: int,
        q: int,
        idxs: np.ndarray,                   # (84,) int32
        packed_channels: _PackedChannels,
    ) -> "cp.ndarray":
        """
        Matches the PyTorch bias definition:
          idxs = random choice of N with size 84
          samples = C[idxs].diagonal().T  (conceptually)
          biases = quantile(samples, golden_ratio_quantiles(q), dim=1).T
        """
        N, C, L = map(int, X_cp.shape)
        calcs_full = L + 2 * padding - dilation * (self.kernel_size - 1)

        idxs_cp = cp.asarray(idxs, dtype=cp.int32)
        samples_cp = cp.empty((self.num_kernels, calcs_full), dtype=cp.float32)

        bias_kernel = self._compile_bias_kernel(L=L, C=C, dilation=dilation, padding=padding)
        grid = (self.num_kernels,)
        block = (256,)

        bias_kernel(
            grid,
            block,
            (
                X_cp,
                self._base_kernels_cp.reshape(-1),
                packed_channels.chan_idx.reshape(-1),
                packed_channels.chan_cnt,
                idxs_cp,
                samples_cp.reshape(-1),
            ),
        )
        cp.cuda.get_current_stream().synchronize()

        qs = self._get_quantiles_np(q)                # (q,) float32
        qs_cp = cp.asarray(qs, dtype=cp.float32)

        # cupy.quantile: output shape (q,84) when axis=1; transpose to (84,q)
        biases_qk = cp.quantile(samples_cp, qs_cp, axis=1)
        biases_kq = biases_qk.T.astype(cp.float32, copy=False)

        return biases_kq

    def _ppv_transform_cuda(
        self,
        X_cp: "cp.ndarray",                 # (B,C,L) float32
        W_cp: "cp.ndarray",                 # (84,9) float32
        packed_channels: _PackedChannels,
        BIAS_cp: "cp.ndarray",              # (84,q) float32
        dilation: int,
        padding: int,
        q: int,
        parity: int,
        korder_cp: "cp.ndarray",            # (84,) int32
        OUT_cp: "cp.ndarray",               # (B,84*q) float32
    ):
        B, C, L = map(int, X_cp.shape)
        ppv_kernel = self._compile_ppv_kernel(L=L, C=C, dilation=dilation, padding=padding, q=q)

        grid = (self.num_kernels, B)
        block = (256,)

        ppv_kernel(
            grid,
            block,
            (
                X_cp,
                W_cp.reshape(-1),
                packed_channels.chan_idx.reshape(-1),
                packed_channels.chan_cnt,
                BIAS_cp.reshape(-1),
                korder_cp,
                np.int32(parity),
                OUT_cp.reshape(-1),
            ),
        )
        cp.cuda.get_current_stream().synchronize()

    # ---------------------------------------------------------------------
    # Kernel construction utilities
    # ---------------------------------------------------------------------

    def _build_minirocket_kernels_np(self) -> np.ndarray:
        """
        Deterministic kernel bank matching:
          indices = torch.combinations(torch.arange(9), 3)
          kernels = (-1).scatter_(indices, +2)
        """
        combs = np.array(list(self._combinations_3(self.kernel_size)), dtype=np.int32)  # (84,3)
        kernels = -np.ones((self.num_kernels, self.kernel_size), dtype=np.float32)
        for k in range(self.num_kernels):
            kernels[k, combs[k]] = 2.0
        return kernels

    @staticmethod
    def _combinations_3(n: int):
        # matches torch.combinations order for r=3 on arange(n)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    yield (i, j, k)

    # ---------------------------------------------------------------------
    # Array helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _shape_of(x: ArrayLike) -> Tuple[int, ...]:
        if isinstance(x, np.ndarray):
            return x.shape
        return tuple(x.shape)

    @staticmethod
    def _slice_first_axis(x: ArrayLike, start: int, stop: int) -> ArrayLike:
        return x[start:stop]

    @staticmethod
    def _to_cupy_float32(x: ArrayLike) -> "cp.ndarray":
        if isinstance(x, np.ndarray):
            # Ensure contiguous float32 on host before upload
            x = np.ascontiguousarray(x, dtype=np.float32)
            return cp.asarray(x, dtype=cp.float32)
        # cupy input
        if x.dtype != cp.float32:
            x = x.astype(cp.float32)
        return cp.ascontiguousarray(x)



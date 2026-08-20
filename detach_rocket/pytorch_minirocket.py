"""PyTorch implementation of MiniRocket for multivariate time series."""

import contextlib

import numpy as np
import torch
import torch.nn as nn


class PytorchMiniRocketMultivariate(nn.Module):
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
        super().__init__()
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @contextlib.contextmanager
    def _manage_cpu_threads(self):
        """Context manager to prevent OpenMP CPU deadlocks."""
        if self.device.type == "cpu":
            original_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                yield
            finally:
                torch.set_num_threads(original_threads)
        else:
            yield

    def fit(self, X, chunksize=128):
        with self._manage_cpu_threads():
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
                self.register_buffer(f"biases_{i}", torch.empty((self.num_kernels, self.num_features_per_dilation[i])))
            self.register_buffer("prefit", torch.BoolTensor([False]))

            # Move the defined model to device before computation
            self.to(self.device)

            # Compute biases with the initial forward pass using a subset of samples
            num_samples = X.shape[0]
            if chunksize is None:
                chunksize = min(num_samples, self.num_dilations * self.num_kernels)  # Deterministic
            else:
                chunksize = min(num_samples, chunksize)  # Stochastic for chunksize < num_samples
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
        for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)):  # Max 32
            _padding1 = i % 2

            # Convolution randomly combining channels for MTS
            C = torch.nn.functional.conv1d(x, self.kernels, padding=padding, dilation=dilation, groups=self.c_in)
            if self.c_in > 1:
                C = C.reshape(x.shape[0], self.c_in, self.num_kernels, -1)
                channel_combination = getattr(self, f"channel_combinations_{i}")
                C = torch.mul(C, channel_combination)
                C = C.sum(1)

            # Draw the biases (and compute them if the model is fitting)
            if not self.prefit or self.fitting:
                num_features_this_dilation = self.num_features_per_dilation[i]
                bias_this_dilation = self._get_bias(C, num_features_this_dilation)
                setattr(self, f"biases_{i}", bias_this_dilation)
                if self.fitting:
                    if i < self.num_dilations - 1:
                        continue
                    else:
                        self.prefit = torch.BoolTensor([True])
                        return
                elif i == self.num_dilations - 1:
                    self.prefit = torch.BoolTensor([True])
            else:
                bias_this_dilation = getattr(self, f"biases_{i}")

            # Pool into PPVs with alternating padding
            _features.append(self._get_PPVs(C[:, _padding1::2], bias_this_dilation[_padding1::2]))
            _features.append(
                self._get_PPVs(C[:, 1 - _padding1 :: 2, padding:-padding], bias_this_dilation[1 - _padding1 :: 2])
            )
        return torch.cat(_features, dim=1)

    def _set_parameter_indices(self):
        # Simulate a forward pass but keep the indices that match a kernel and bias with a feature
        for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)):  # Max 32
            _padding1 = i % 2

            # Indices for the kernels / channel combinations & biases
            bias_this_dilation = getattr(self, f"biases_{i}")
            num_kernels, num_quantiles = bias_this_dilation.shape
            # bias_indices[k, j] must be the row-major flat index of biases_[k, j],
            # since get_kernel_features indexes biases_this_dilation.flatten() with it.
            bias_indices = torch.arange(num_kernels * num_quantiles, dtype=int).reshape(num_kernels, num_quantiles)

            kernel_indices = torch.arange(num_kernels, dtype=int)
            kernel_indices = torch.stack([kernel_indices] * num_quantiles, dim=-1)

            # Simulated feature maps for the "even" kernels
            C_even = kernel_indices[_padding1::2]  # (num kernels / 2, num quantiles)
            bias_this_dilation_even = bias_indices[_padding1::2]  # (num kernels / 2, num quantiles)

            # Simulate the PPV reshape
            C_even = C_even.flatten()  # replaces .mean(2).flatten(1) and removes placeholder dimensions
            bias_this_dilation_even = bias_this_dilation_even.flatten()

            # Do the same for "odd" kernels
            C_odd = kernel_indices[1 - _padding1 :: 2]  # (num kernels / 2, num quantiles)
            bias_this_dilation_odd = bias_indices[1 - _padding1 :: 2]  # (num kernels / 2, num quantiles)

            # Simulate the PPV reshape
            C_odd = C_odd.flatten()  # replaces .mean(2).flatten(1) and removes placeholder dimensions
            bias_this_dilation_odd = bias_this_dilation_odd.flatten()

            # Stack into flat arrays
            C_full = torch.cat((C_even, C_odd))
            bias_this_dilation_full = torch.cat((bias_this_dilation_even, bias_this_dilation_odd))

            setattr(self, f"kernel_indices_{i}", C_full)
            setattr(self, f"bias_indices_{i}", bias_this_dilation_full)

        return

    def get_kernel_features(self, which, where):
        # Get the "which" kernel parameters at "where" certain indices
        full_features = np.empty(shape=(0,), dtype=float)

        if which == "channels":
            full_features = np.empty(shape=(0, self.c_in), dtype=float)
            where = where[:, np.newaxis]
            where = np.repeat(where, self.c_in, axis=1)
        elif which == "weights":
            full_features = np.empty(shape=(0, self.kernel_size), dtype=float)
            where = where[:, np.newaxis]
            where = np.repeat(where, self.kernel_size, axis=1)

        for i, (dilation, padding) in enumerate(zip(self.dilations, self.padding)):
            biases_this_dilation = getattr(self, f"biases_{i}")
            num_quantiles = biases_this_dilation.shape[1]

            kernel_indices = getattr(self, f"kernel_indices_{i}")
            bias_indices = getattr(self, f"bias_indices_{i}")

            # Biases (=features): as many as num dilations * num kernels * num quantiles
            if which == "biases":
                sorted_biases = biases_this_dilation.flatten()[bias_indices]
                full_features = np.append(full_features, sorted_biases.cpu().numpy())

            # Channel combinations: num_dilations * num_kernel, where each combination has self.c_in indices
            elif which == "channels":
                channel_combinations = getattr(self, f"channel_combinations_{i}")

                for q in range(0, num_quantiles):
                    selected_kernels = (
                        kernel_indices[q * self.num_kernels : q * self.num_kernels + self.num_kernels].cpu().numpy()
                    )
                    channel_combinations_q = channel_combinations[:, :, selected_kernels]
                    channel_combinations_q = torch.transpose(channel_combinations_q.squeeze(), 0, 1).cpu().numpy()

                    full_features = np.append(full_features, channel_combinations_q, axis=0)

            # Weights: num dilations * num kernels, where each kernel has 9 weights
            elif which == "weights":
                weights = (
                    self.kernels.view(-1, self.num_kernels, self.kernel_size)[0].cpu().numpy()
                )  # Kernels are equal for all channels, pick the first one

                for q in range(0, num_quantiles):
                    selected_kernels = (
                        kernel_indices[q * self.num_kernels : q * self.num_kernels + self.num_kernels].cpu().numpy()
                    )
                    weights_q = weights[selected_kernels]

                    full_features = np.append(full_features, weights_q, axis=0)

            elif which == "dilations":
                expanded_dilations = np.repeat(dilation, self.num_kernels * num_quantiles, axis=0)
                full_features = np.append(full_features, expanded_dilations)

            elif which == "paddings":
                expanded_dilations = np.repeat(padding, self.num_kernels * num_quantiles, axis=0)
                full_features = np.append(full_features, expanded_dilations)

            else:
                raise ValueError(
                    f'"{which}" is not recognized as a feature. Possible features are "biases", "channels", "weights", "dilations" or "paddings"'
                )

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
        dilations, num_features_per_dilation = np.unique(
            np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32), return_counts=True
        )
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
            self.padding.append(((self.kernel_size - 1) * dilation) // 2)

    def _set_channel_combinations(self, num_channels):
        num_combinations = self.num_kernels * self.num_dilations
        max_num_channels = min(num_channels, 9)
        max_exponent_channels = np.log2(max_num_channels + 1)
        num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent_channels, num_combinations)).astype(
            np.int32
        )
        channel_combinations = torch.zeros((1, num_channels, num_combinations, 1))
        for i in range(num_combinations):
            channel_combinations[:, np.random.choice(num_channels, num_channels_per_combination[i], False), i] = (
                1  # From all the channels, set to 1 those that will be combined without repeating
            )
        channel_combinations = torch.split(channel_combinations, self.num_kernels, 2)  # split by dilation
        for i, channel_combination in enumerate(channel_combinations):
            self.register_buffer(f"channel_combinations_{i}", channel_combination)  # per dilation

    def _get_quantiles(self, n):
        return torch.tensor([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)]).float()

    def _get_bias(self, C, num_features_this_dilation):
        # Gets as many biases as features this dilation, from the quantiles of random samples
        idxs = np.random.choice(C.shape[0], self.num_kernels)
        samples = C[idxs].diagonal().T
        biases = torch.quantile(samples, self._get_quantiles(num_features_this_dilation).to(C.device), dim=1).T
        return biases

    def transform(self, o, chunksize=128):
        with self._manage_cpu_threads():
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

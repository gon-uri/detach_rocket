# Detach-ROCKET

<img align="right" src="logo/detach_logo.png" alt="Logo" width="150"/>

Official repository for:

- [Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels](https://link.springer.com/article/10.1007/s10618-024-01062-7)
- [Classification of raw MEG/EEG data with detach-rocket ensemble: an improved rocket algorithm for multivariate time series analysis](https://www.arxiv.org/abs/2408.02760)

## Overview

This repository contains Python implementations of Sequential Feature
Detachment (SFD) for feature selection and Detach-ROCKET for time-series
classification. Developed entirely in Python using primarily NumPy, PyTorch,
Scikit-Learn and Sktime libraries, the core functionalities are encapsulated
within the following classes:

- `DetachRocket`: Detach-ROCKET model class. It is constructed by pruning an initial ROCKET, MiniRocket or MultiROCKET model using SFD and selecting the optimal size.
  
- `DetachMatrix`: Class for applying Sequential Feature Detachment to any dataset matrix structured as (n_instances, n_features).

- `DetachEnsemble`: Detach-ROCKET Ensemble model class. It creates an ensemble of Detach models. We recommend using this class for multivariate time series, especially if they are high-dimensional. After training, these models are also able to provide channel relevance estimation and label probability.

For a detailed explanation of the models and methods please refer to the [Detach-ROCKET article](https://link.springer.com/article/10.1007/s10618-024-01062-7) and the [Detach-ROCKET Ensemble article](https://www.arxiv.org/abs/2408.02760).

## Core Modules

- `detach_rocket/sfd.py`: Sequential Feature Detachment core logic (`feature_detachment`).
- `detach_rocket/model_selection.py`: model-size selection and final retraining utilities.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/gon-uri/detach_rocket
```

With optional dependencies:

```bash
# GPU-accelerated ensemble (DetachEnsemble)
pip install "detach_rocket[torch] @ git+https://github.com/gon-uri/detach_rocket"

# Dataset download utilities
pip install "detach_rocket[datasets] @ git+https://github.com/gon-uri/detach_rocket"

# Everything
pip install "detach_rocket[all] @ git+https://github.com/gon-uri/detach_rocket"
```

For development:

```bash
git clone https://github.com/gon-uri/detach_rocket.git
cd detach_rocket
pip install -e ".[dev]"
```

## Usage - DetachRocket
The model usage is the same as in the scikit-learn library. 

```python
# Import model and transformer
from detach_rocket.detach_classes import DetachRocket
from sktime.transformations.panel.rocket import Rocket

# Instantiate model
rocket = Rocket(num_kernels=10_000)
detach_model = DetachRocket(transformer=rocket, trade_off=0.1)

# Train model (validation set required when set_percentage=None)
detach_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# Predict and score
y_pred = detach_model.predict(X_test)
test_acc = detach_model.score(X_test, y_test)
full_model_acc = detach_model.score_full(X_test, y_test)  # Optional baseline
summary = detach_model.get_summary()
```

If you prefer a fixed pruning level, pass `set_percentage` and fit without a validation set:

```python
rocket = Rocket(num_kernels=10_000)
detach_model = DetachRocket(transformer=rocket, set_percentage=50)
detach_model.fit(X_train, y_train)
```

For univariate time series, the shape of `X_train` should be (n_instances, n_timepoints).

For multivariate time series, the shape of `X_train` should be (n_instances, n_variables, n_timepoints).

## Usage - DetachRocket Ensemble
This model is more suitable for Multivariate Time Series with a large number of channels/dimensions.
The Ensemble API is currently being aligned with the `DetachRocket` API in this branch.

```python
# Import model
from detach_rocket.detach_classes import DetachEnsemble

# Work in progress: API may change in upcoming updates
ensemble = DetachEnsemble(model_type='pytorch_minirocket', num_kernels=10000)
```

## Notebook Examples

Detailed usage examples can be found in the included Jupyter notebooks in the [examples folder](/examples).

## Upcoming Features

- [x] Built-in support for multilabel classification. (DONE!)
- [x] Pytorch implementation of Detach-MiniRocket. (DONE!)
- [x] Add channel releavance for Detach-MiniRocket. (DONE!)
- [x] Implementation of Detach-ROCKET Ensemble. (DONE!)
- [x] Add channel releavance and label probability for Detach-ROCKET Ensemble. (DONE!)
- [x] CUDA implementations of Detach-MiniRocket. (DONE!)
- [x] Real pruning of ROCKET model for faster inference. (DONE!)

## Troubleshooting

If `pip install` fails while building `llvmlite` (a `numba` dependency), install `numba` via conda first:

```bash
conda install "numba>=0.58"
pip install detach_rocket  # or pip install git+https://github.com/gon-uri/detach_rocket
```

If after this you see an `Intel MKL WARNING` about SSE4.2/AVX (conda installs MKL by default), run:

```bash
conda install nomkl
```

## License

This project is licensed under the BSD-3-Clause License.

## Citation

If you find these methods useful in your research, please cite the following articles:

*APA*
```
Uribarri, G., Barone, F., Ansuini, A., & Fransén, E. (2024). Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels. Data Mining and Knowledge Discovery, 1-26.

Solana, A., Fransén, E., & Uribarri, G. (2024). Classification of raw MEG/EEG data with detach-rocket ensemble: an improved rocket algorithm for multivariate time series analysis. arXiv preprint arXiv:2408.02760.
```

*BIBTEX*
```bibtex
@article{uribarri2024detach,
  title={Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels},
  author={Uribarri, Gonzalo and Barone, Federico and Ansuini, Alessio and Frans{\'e}n, Erik},
  journal={Data Mining and Knowledge Discovery},
  pages={1--26},
  year={2024},
  publisher={Springer}
}

@article{solana2024classification,
  title={Classification of raw MEG/EEG data with detach-rocket ensemble: an improved rocket algorithm for multivariate time series analysis},
  author={Solana, Adri{\`a} and Frans{\'e}n, Erik and Uribarri, Gonzalo},
  journal={arXiv preprint arXiv:2408.02760},
  year={2024}
}
```

<img src="logo/detach_logo.png" align="centered"
     alt="repo logo" width="80" height="80">

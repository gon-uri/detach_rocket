<img align="right" src="logo/detach_logo.png" alt="Logo" width="150"/>
<div id="toc">
    <ul style="list-style: none;">
    <summary>
      <h1>Detach-ROCKET</h1>
    </summary>
  </ul>
</div>

Official repository for [Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels](https://link.springer.com/article/10.1007/s10618-024-01062-7) and [Classification of raw MEG/EEG data with detach-rocket ensemble: an improved rocket algorithm for multivariate time series analysis](https://www.arxiv.org/abs/2408.02760).

## Overview

This repository contains Python implementations of Sequential Feature Detachment (SFD) for feature selection and Detach-ROCKET for time-series classification. Developed primarily in Python and utilizing NumPy, scikit-learn, and sktime libraries, the core functionalities are encapsulated within the following classes:

- `DetachRocket`: Detach-ROCKET model class. It is constructed by pruning an initial ROCKET, MiniRocket or MultiROCKET model using SFD and selecting the optimal size.
  
- `DetachMatrix`: Class for applying Sequential Feature Detachment to any dataset matrix structured as (n_instances, n_features).

- `DetachEnsemble`: Detach-ROCKET Ensemble model class. It creates an ensemble of Detach models. We recommend using this class for multivariate time series, especially if they are high-dimensional. After training, these models are also able to provide channel relevance estimation and label probability.

For a detailed explanation of the models and methods please refer to the [Detach-ROCKET article](https://link.springer.com/article/10.1007/s10618-024-01062-7) and the [Detach-ROCKET Ensemble article](https://www.arxiv.org/abs/2408.02760).

## Installation

To install the required dependencies, execute:

```bash
pip install numpy scikit-learn pyts torch matplotlib sktime==0.30.0
pip install git+https://github.com/gon-uri/detach_rocket --quiet
```

## Usage - DetachRocket
The model usage is the same as in the scikit-learn library. 

```python
# Import Model
from detach_rocket.detach_classes import DetachRocket

# Instantiate Model
DetachRocketModel = DetachRocket('rocket', num_kernels=10000)

# Trian Model
DetachRocketModel.fit(X_train,y_train)

# Predict Test Set
y_pred = DetachRocketModel.predict(X_test)
```

For univariate time series, the shape of `X_train` should be (n_instances, n_timepoints).

For multivariate time series, the shape of `X_train` should be (n_instances, n_variables, n_timepoints).

## Usage - DetachRocket Ensemble
This model is more suitable for Multivariate Time Series with a large number of channels/dimensions.

```python
# Import Model
from detach_rocket.detach_classes import DetachEnsemble

# Instantiate Model
DetachRocketEnsemble = DetachEnsemble('pytorch_minirocket', num_kernels=10000)

# Trian Model
DetachRocketEnsemble.fit(X_train,y_train)

# Predict Test Set
y_pred = DetachRocketEnsemble.predict(X_test)
```

## Notebook Examples

Detailed usage examples can be found in the included Jupyter notebooks in the [examples folder](/examples).

## Upcoming Features

- [x] Built-in support for multilabel classification. (DONE!)
- [x] Pytorch implementations of Detach-MiniRocket. (DONE!)
- [x] Add channel releavance for Detach-MiniRocket. (DONE!)
- [x] Implementation of Detach-ROCKET Ensemble. (DONE!)
- [x] Add channel releavance and label probability for Detach-ROCKET Ensemble. (DONE!)
- [ ] Pytorch implementations of Detach-MultiRocket. (Coming soon...)
- [ ] Fully pytorch implementation of ROCKET with on-the-fly convolutions during training.
- [ ] Pytorch implementation of SFD for Multilayer Perceptrons.

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
```
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

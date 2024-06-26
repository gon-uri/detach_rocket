<img src="logo/detach_logo.png" align="right"
     alt="repo logo" width="180" height="180">
<br/><br/>
<br/><br/>
# README
Official repository for [Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels.](https://arxiv.org/abs/2309.14518)

## Overview

This repository contains Python implementations of Sequential Feature Detachment (SFD) for feature selection and Detach-ROCKET for time-series classification. Developed primarily in Python and utilizing NumPy, scikit-learn, and sktime libraries, the core functionalities are encapsulated within the following classes:

- `DetachRocket`: Detach-ROCKET model class. It is constructed by pruning an initial ROCKET, MiniRocket or MultiROCKET model using SFD and selecting the optimal size.
  
- `DetachMatrix`: Class for applying Sequential Feature Detachment to any dataset matrix structured as (n_instances, n_features).

For a detailed explanation of the model and methods please refer to the [article](https://arxiv.org/abs/2309.14518).

## Installation

To install the required dependencies, execute:

```bash
pip install numpy scikit-learn sktime pyts
pip install git+https://github.com/gon-uri/detach_rocket --quiet
```

## Usage
The model usage is the same as in the scikit-learn library. 

```python
# Instantiate Model
DetachRocketModel = DetachRocket('rocket', num_kernels=10000)

# Trian Model
DetachRocketModel.fit(X_train,y_train)

# Predict Test Set
y_pred = DetachRocketModel.predict(X_test)
```

For univariate time series, the shape of `X_train` should be (n_instances, n_timepoints).

For multivariate time series, the shape of `X_train` should be (n_instances, n_variables, n_timepoints).

## Notebook Examples

Detailed usage examples can be found in the included Jupyter notebooks in the [examples folder](/examples).

## Upcoming Features

- [x] Built-in support for multilabel classification (DONE!).
- [ ] Pytorch implementation of ROCKET, MiniRocket or MultiROCKET.

## License

This project is licensed under the BSD-3-Clause License.

## Citation

If you find these methods useful in your research, please cite the article:

*APA*
```
Uribarri, G., Barone, F., Ansuini, A., & Fransén, E. (2023). Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels. arXiv preprint arXiv:2309.14518.
```

*BIBTEX*
```
@article{uribarri2023detach,
  title={Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels},
  author={Uribarri, Gonzalo and Barone, Federico and Ansuini, Alessio and Frans{\'e}n, Erik},
  journal={arXiv preprint arXiv:2309.14518},
  year={2023}
}
```

<img src="logo/detach_logo.png" align="centered"
     alt="repo logo" width="80" height="80">

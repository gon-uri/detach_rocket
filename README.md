# README

Official repository of [Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels.](https://arxiv.org/abs/2309.14518)

## Overview

This repository contains Python implementations of Sequential Feature Detachment (SFD) for feature selection and DEtach-ROCKET for time-series classification. Developed primarily in Python and utilizing NumPy, scikit-learn, and sktime libraries, the core functionalities are encapsulated within the following classes:

- `DetachRocket`: Detach-ROCKET model class. It is constructed by pruning an initial ROCKET, MiniRocket or MultiROCKET model using SFD and selecting the optimal size.
  
- `SFD`: Class for applying Sequential Feature Detachment to any dataset matrix composed of instances x features (not only the one obtained by unsing ROCKET on time series).

For a detailed explanation of the model and methods please refer to the [article](https://arxiv.org/abs/2309.14518).

## Installation

To install the required dependencies, execute:

```bash
pip install numpy scikit-learn sktime pyts
pip install git+https://github.com/gon-uri/detach_rocket --quiet
```

## Usage


```python
# Instantiate model object
DetachRocketModel = DetachRocket('rocket', num_kernels=10000)

# Trian Model
DetachRocketModel.fit(X_train,y_train)

# Predict Test Set
y_pred = DetachRocketModel.predict(X_test)
```

## Notebook Examples

Detailed usage examples can be found in the included Jupyter notebooks in the [examples folder](/examples).

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

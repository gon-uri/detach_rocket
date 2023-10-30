# README

## Overview

This repository contains Python implementations of Sequential Feature Detachment (SFD) for feature selection and DEtach-ROCKET for time-series classification. Developed primarily in Python and utilizing NumPy, scikit-learn, and sktime libraries, the core functionalities are encapsulated within the following classes:

- `DetachRocket`: Detach-ROCKET model class.
  
- `SFD`: Class for applying Sequential Feature Detachment (SFD) to any type of data.

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

Detailed usage examples can be found in the accompanying Jupyter Notebooks at the [examples folder](/examples).

## License

This project is licensed under the MIT License.

## Citation

If you find these methods useful in your research, please cite this article.

APA
```
Uribarri, G., Barone, F., Ansuini, A., & Fransén, E. (2023). Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels. arXiv preprint arXiv:2309.14518.
```

BIBTEX
```
@article{uribarri2023detach,
  title={Detach-ROCKET: Sequential feature selection for time series classification with random convolutional kernels},
  author={Uribarri, Gonzalo and Barone, Federico and Ansuini, Alessio and Frans{\'e}n, Erik},
  journal={arXiv preprint arXiv:2309.14518},
  year={2023}
}
```

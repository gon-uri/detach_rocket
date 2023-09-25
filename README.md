# README

## Overview

This repository contains Python implementations of Sequential Feature Detachment (SFD) for feature selection and DEtach-ROCKET for time-series classification. Developed primarily in Python and utilizing libraries such as NumPy, scikit-learn, and sktime, the core functionalities are encapsulated within the following functions:

- `train_full_rocket_model`: For applying the ROCKET transformation on time series data and training a ridge classifier.
  
- `feature_detachment`: For applying Sequential Feature Detachment (SFD) to feature matrices.

- `select_optimal_model`: For determining the optimal model size post-SFD.
  
- `retrain_optimal_model`: For retraining a classifier with the optimal subset of features.

## Installation

To install the required dependencies, execute:

```bash
pip install numpy scikit-learn sktime
```

## Function Descriptions

### `train_full_rocket_model`

Transforms time-series data using ROCKET (RandOm Convolutional KErnel Transform) and fits a ridge classifier to it. Outputs include the trained classifier, transformation objects, and evaluation metrics.

Parameters:

- `X_train`, `X_test`: 3D NumPy arrays representing training and test time series.
- `y_train`, `y_test`: 1D NumPy arrays representing training and test labels.
- `num_kernels`: Integer specifying the number of kernels.
- `model_type`: String indicating the type of ROCKET model.
- `verbose`: Boolean for verbosity.

Returns:

- Transformation objects, trained classifier, feature matrices, and evaluation metrics.

### `feature_detachment`

Performs Sequential Feature Detachment (SFD) on a given feature matrix. Outputs feature importance and performance metrics.

Parameters:

- `classifier`: Pre-trained ridge classifier.
- `X_train`, `X_test`: 2D NumPy arrays for training and test feature matrices.
- `y_train`, `y_test`: Training and test labels.
- `drop_percentage`: Proportion of features to drop.
- `total_number_steps`: Total number of SFD steps.
- `verbose`: Verbosity flag.

Returns:

- Feature importance matrix, performance metrics, and model size vectors.

### `select_optimal_model`

Selects the optimal model size after an SFD process. Outputs include the optimal model index and size as a proportion of the original.

Parameters:

- `percentage_vector`: Model sizes at each detachment step.
- `acc_test`: Test accuracies at each step.
- `full_model_score_test`: Full model test accuracy.
- `acc_size_tradeoff_coef`: Tradeoff coefficient.
- `smoothing_points`: Smoothing level.
- `graphics`: Plotting flag.

Returns:

- Optimal model index and size.

### `retrain_optimal_model`

Retrains a classifier using the optimal feature subset selected by SFD. Outputs two classifiers, one with a newly computed alpha and another with the original alpha.

Parameters:

- `feature_importance_matrix`: Feature importance at each SFD step.
- `X_train_scaled_transform`, `X_test_scaled_transform`: Feature matrices.
- `y_train`, `y_test`: Labels.
- `full_model_score_train`, `full_model_score_test`: Full model accuracies.
- `original_best_alpha`: Original alpha of the ridge classifier.
- `max_index`: Optimal model index.
- `verbose`: Verbosity flag.

Returns:

- Two trained classifiers, along with their respective accuracies on the training and test sets.

## Usage Examples

Detailed usage examples can be found in the accompanying Jupyter Notebook.

## License

This project is licensed under the MIT License.

## Citation

If you find these methods useful in your research, consider citing this repository.

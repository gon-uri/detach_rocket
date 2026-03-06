"""
Utility functions for fetching UCR and UEA time series classification datasets.
"""
# Code copied (and modified) from pyts library code:
# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import os
import zipfile
from urllib.request import urlretrieve

import numpy as np
from pyts.datasets import ucr_dataset_list, uea_dataset_list
from scipy.io.arff import loadarff
from sklearn.utils import Bunch


def _correct_ucr_name_download(dataset):
    if dataset == "CinCECGtorso":
        return "CinCECGTorso"
    elif dataset == "MixedShapes":
        return "MixedShapesRegularTrain"
    elif dataset == "NonInvasiveFetalECGThorax1":
        return "NonInvasiveFatalECGThorax1"
    elif dataset == "NonInvasiveFetalECGThorax2":
        return "NonInvasiveFatalECGThorax2"
    elif dataset == "StarlightCurves":
        return "StarLightCurves"
    else:
        return dataset


def _correct_ucr_name_description(dataset):
    if dataset == "CinCECGTorso":
        return "CinCECGtorso"
    elif dataset == "MixedShapesRegularTrain":
        return "MixedShapes"
    elif dataset == "NonInvasiveFatalECGThorax1":
        return "NonInvasiveFetalECGThorax1"
    elif dataset == "NonInvasiveFatalECGThorax2":
        return "NonInvasiveFetalECGThorax2"
    elif dataset == "StarLightCurves":
        return "StarlightCurves"
    else:
        return dataset


def fetch_ucr_dataset(dataset, use_cache=True, data_home=None, return_X_y=False):
    r"""Fetch dataset from UCR TSC Archive by name.

    Fetched data sets are automatically saved in the
    ``pyts/datasets/_cached_datasets`` folder. To avoid
    downloading the same data set several times, it is
    highly recommended not to change the default values
    of ``use_cache`` and ``path``.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    use_cache : bool (default = True)
        If True, look if the data set has already been fetched
        and load the fetched version if it is the case. If False,
        download the data set from the UCR Time Series Classification
        Archive.

    data_home : None or str (default = None)
        The path of the folder containing the cached data set.
        If None, the ``pyts.datasets.cached_datasets/UCR/`` folder is
        used. If the data set is not found, it is downloaded and cached
        in this path.

    return_X_y : bool (default = False)
        If True, returns ``(data_train, data_test, target_train, target_test)``
        instead of a Bunch object. See below for more information about the
        `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array of integers
            The classification labels in the training set.
        target_test : array of integers
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    (data_train, data_test, target_train, target_test) : tuple if ``return_X_y`` is True

    Notes
    -----
    Missing values are represented as NaN's.

    References
    ----------
    .. [1] H. A. Dau et al, "The UCR Time Series Archive".
           arXiv:1810.07758 [cs, stat], 2018.

    .. [2] A. Bagnall et al, "The UEA & UCR Time Series Classification
           Repository", www.timeseriesclassification.com.

    """  # noqa: E501
    if dataset not in ucr_dataset_list():
        raise ValueError(
            f"{dataset} is not a valid name. The list of available names "
            "can be obtained with ``pyts.datasets.ucr_dataset_list()``"
        )
    if data_home is None:
        import pyts

        home = os.sep.join(pyts.__file__.split(os.sep)[:-2])
        path = os.path.join(home, "pyts", "datasets", "cached_datasets", "UCR")
    else:
        path = data_home
    if not os.path.exists(path):
        os.makedirs(path)

    correct_dataset = _correct_ucr_name_download(dataset)
    if use_cache and os.path.exists(os.path.join(path, correct_dataset)):
        bunch = _load_ucr_dataset(correct_dataset, path=path)
    else:
        url = f"https://www.timeseriesclassification.com/aeon-toolkit/{correct_dataset}.zip"
        filename = f"temp_{correct_dataset}"
        _ = urlretrieve(url, os.path.join(path, filename))
        zipfile.ZipFile(os.path.join(path, filename)).extractall(os.path.join(path, correct_dataset))
        os.remove(os.path.join(path, filename))
        bunch = _load_ucr_dataset(correct_dataset, path)

    if return_X_y:
        return (bunch.data_train, bunch.data_test, bunch.target_train, bunch.target_test)
    return bunch


def _load_ucr_dataset(dataset, path):
    """Load a UCR data set from a local folder.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    path : str
        The path of the folder containing the cached data set.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array
            The classification labels in the training set.
        target_test : array
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    Notes
    -----
    Padded values are represented as NaN's.

    """
    new_path = os.path.join(path, dataset)
    try:
        with open(os.path.join(new_path, dataset + ".txt"), encoding="utf-8") as f:
            description = f.read()
    except UnicodeDecodeError:
        with open(os.path.join(new_path, dataset + ".txt"), encoding="ISO-8859-1") as f:
            description = f.read()
    try:
        data_train = np.genfromtxt(os.path.join(new_path, dataset + "_TRAIN.txt"))
        data_test = np.genfromtxt(os.path.join(new_path, dataset + "_TEST.txt"))

        X_train, y_train = data_train[:, 1:], data_train[:, 0]
        X_test, y_test = data_test[:, 1:], data_test[:, 0]

    except IndexError:
        train = loadarff(os.path.join(new_path, dataset + "_TRAIN.arff"))
        test = loadarff(os.path.join(new_path, dataset + "_TEST.arff"))

        data_train = np.asarray([train[0][name] for name in train[1].names()])
        X_train = data_train[:-1].T.astype("float64")
        y_train = data_train[-1]

        data_test = np.asarray([test[0][name] for name in test[1].names()])
        X_test = data_test[:-1].T.astype("float64")
        y_test = data_test[-1]

    try:
        y_train = y_train.astype("float64").astype("int64")
        y_test = y_test.astype("float64").astype("int64")
    except ValueError:
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

    bunch = Bunch(
        data_train=X_train,
        target_train=y_train,
        data_test=X_test,
        target_test=y_test,
        DESCR=description,
        url=(f"https://www.timeseriesclassification.com/description.php?Dataset={dataset}"),
    )

    return bunch


# ---------------------------------------------------------------------------
# UEA multivariate time series classification archive
# ---------------------------------------------------------------------------


def _correct_uea_name_download(dataset):
    if dataset == "Ering":
        return "ERing"
    else:
        return dataset


def fetch_uea_dataset(dataset, use_cache=True, data_home=None, return_X_y=False):  # noqa 207
    """Fetch dataset from UEA TSC Archive by name.

    Fetched data sets are saved by default in the
    ``pyts/datasets/cached_datasets/UEA/`` folder. To avoid
    downloading the same data set several times, it is
    highly recommended not to change the default values
    of ``use_cache`` and ``path``.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    use_cache : bool (default = True)
        If True, look if the data set has already been fetched
        and load the fetched version if it is the case. If False,
        download the data set from the UCR Time Series Classification
        Archive.

    data_home : None or str (default = None)
        The path of the folder containing the cached data set.
        If None, the ``pyts.datasets.cached_datasets/UEA/`` folder is
        used. If the data set is not found, it is downloaded and cached
        in this path.

    return_X_y : bool (default = False)
        If True, returns ``(data_train, data_test, target_train, target_test)``
        instead of a Bunch object. See below for more information about the
        `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array of integers
            The classification labels in the training set.
        target_test : array of integers
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    (data_train, data_test, target_train, target_test) : tuple if \
``return_X_y`` is True

    Notes
    -----
    Missing values are represented as NaN's.

    References
    ----------
    .. [1] A. Bagnall et al, "The UEA multivariate time series
           classification archive, 2018". arXiv:1811.00075 [cs, stat],
           2018.

    .. [2] A. Bagnall et al, "The UEA & UCR Time Series Classification
           Repository", www.timeseriesclassification.com.

    """
    if dataset not in uea_dataset_list():
        raise ValueError(
            f"{dataset} is not a valid name. The list of available names "
            "can be obtained with ``pyts.datasets.uea_dataset_list()``"
        )
    if data_home is None:
        import pyts

        home = os.sep.join(pyts.__file__.split(os.sep)[:-2])
        path = os.path.join(home, "pyts", "datasets", "cached_datasets", "UEA")
    else:
        path = data_home
    if not os.path.exists(path):
        os.makedirs(path)

    correct_dataset = _correct_uea_name_download(dataset)
    if use_cache and os.path.exists(os.path.join(path, correct_dataset)):
        bunch = _load_uea_dataset(correct_dataset, path)
    else:
        url = f"https://www.timeseriesclassification.com/aeon-toolkit/{correct_dataset}.zip"
        filename = f"temp_{correct_dataset}"
        _ = urlretrieve(url, os.path.join(path, filename))
        zipfile.ZipFile(os.path.join(path, filename)).extractall(os.path.join(path, correct_dataset))
        os.remove(os.path.join(path, filename))
        bunch = _load_uea_dataset(correct_dataset, path)

    if return_X_y:
        return (bunch.data_train, bunch.data_test, bunch.target_train, bunch.target_test)
    return bunch


def _load_uea_dataset(dataset, path):
    """Load a UEA data set from a local folder.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    path : str
        The path of the folder containing the cached data set.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array
            The classification labels in the training set.
        target_test : array
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    Notes
    -----
    Missing values are represented as NaN's.

    """
    new_path = os.path.join(path, dataset)
    try:
        description_file = [
            file for file in os.listdir(new_path) if ("Description.txt" in file or f"{dataset}.txt" in file)
        ][0]
    except IndexError:
        description_file = None

    if description_file is not None:
        try:
            with open(os.path.join(new_path, description_file), encoding="utf-8") as f:
                description = f.read()
        except UnicodeDecodeError:
            with open(os.path.join(new_path, description_file), encoding="ISO-8859-1") as f:
                description = f.read()
    else:
        description = None

    data_train = loadarff(os.path.join(new_path, f"{dataset}_TRAIN.arff"))
    X_train, y_train = _parse_relational_arff(data_train)

    data_test = loadarff(os.path.join(new_path, f"{dataset}_TEST.arff"))
    X_test, y_test = _parse_relational_arff(data_test)

    bunch = Bunch(
        data_train=X_train,
        target_train=y_train,
        data_test=X_test,
        target_test=y_test,
        DESCR=description,
        url=(f"https://www.timeseriesclassification.com/description.php?Dataset={dataset}"),
    )

    return bunch


def _parse_relational_arff(data):
    X_data = np.asarray(data[0])
    n_samples = len(X_data)
    X, y = [], []

    if X_data[0][0].dtype.names is None:
        for i in range(n_samples):
            X_sample = np.asarray([X_data[i][name] for name in X_data[i].dtype.names])
            X.append(X_sample.T)
            y.append(X_data[i][1])
    else:
        for i in range(n_samples):
            X_sample = np.asarray([X_data[i][0][name] for name in X_data[i][0].dtype.names])
            X.append(X_sample.T)
            y.append(X_data[i][1])

    X = np.asarray(X).astype("float64")
    y = np.asarray(y)

    try:
        y = y.astype("float64").astype("int64")
    except ValueError:
        y = y.astype(str)

    return X, y

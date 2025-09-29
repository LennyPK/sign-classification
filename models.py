"""Module for training SVC models with hyperparameter tuning options."""

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def train_with_gridsearch(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """Train an SVC model using GridSearchCV for hyperparameter tuning."""
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": [0.0001, 0.001, 0.1],
        "kernel": ["linear", "rbf", "poly"],
    }

    svc = svm.SVC(verbose=True)
    print("The training of the model has started")
    model = GridSearchCV(svc, param_grid, verbose=2, n_jobs=-1)
    print(model)
    print("The model is being fitted")
    model.fit(x_train, y_train)
    print("The training of the model has ended")
    print("Best parameters found: ", model.best_params_)


# def train_with_custom_params(
#     x_train: np.ndarray,
#     y_train: np.ndarray,
#     C: float,  # pylint: disable=invalid-name
#     kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"],
#     gamma: float | Literal["scale", "auto"] = "scale",
# ) -> None:
#     """Train an SVC model with custom hyperparameters."""
#     model = svm.SVC(kernel=kernel, C=C, gamma=gamma, verbose=True)
#     print(model)
#     print("SVC Model configured")
#     model.fit(x_train, y_train)
#     print("SVC Model fitted")


def gridsearch_linear(x_train: np.ndarray, y_train: np.ndarray) -> None:
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear"],
    }

    svc = svm.SVC(verbose=True)
    print("The training of the model has started")
    model = GridSearchCV(svc, param_grid, verbose=2, n_jobs=-1)
    print(model)
    print("The model is being fitted")
    model.fit(x_train, y_train)
    print("The training of the model has ended")
    print("Best parameters found: ", model.best_params_)


def gridsearch_rbf(x_train: np.ndarray, y_train: np.ndarray) -> None:
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": [0.0001, 0.001, 0.1],
        "kernel": ["rbf"],
    }

    svc = svm.SVC(verbose=True)
    print("The training of the model has started")
    model = GridSearchCV(svc, param_grid, verbose=2, n_jobs=-1)
    print(model)
    print("The model is being fitted")
    model.fit(x_train, y_train)
    print("The training of the model has ended")
    print("Best parameters found: ", model.best_params_)


def gridsearch_poly(x_train: np.ndarray, y_train: np.ndarray) -> None:
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["poly"],
    }

    svc = svm.SVC(verbose=True)
    print("The training of the model has started")
    model = GridSearchCV(svc, param_grid, verbose=2, n_jobs=-1)
    print(model)
    print("The model is being fitted")
    model.fit(x_train, y_train)
    print("The training of the model has ended")
    print("Best parameters found: ", model.best_params_)


def linear_svc(x_train: np.ndarray, y_train: np.ndarray) -> svm.SVC:
    model = svm.SVC(kernel="linear", C=1, verbose=True)
    print(model)
    print("Linear SVC Model configured")
    model.fit(x_train, y_train)
    print("Linear SVC Model fitted")

    return model


def rbf_svc(x_train: np.ndarray, y_train: np.ndarray) -> svm.SVC:
    model = svm.SVC(kernel="rbf", C=10, gamma=0.001, verbose=True)
    print(model)
    print("RBF SVC Model configured")
    model.fit(x_train, y_train)
    print("RBF SVC Model fitted")

    return model


def poly_svc(x_train: np.ndarray, y_train: np.ndarray) -> svm.SVC:
    model = svm.SVC(kernel="poly", C=10, verbose=True)
    print(model)
    print("Poly SVC Model configured")
    model.fit(x_train, y_train)
    print("Poly SVC Model fitted")

    return model

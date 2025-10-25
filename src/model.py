"""Model training utilities for SVC and MLP.

This module provides two complementary workflows:
- Grid search utilities to discover good SVC hyperparameters (printing results only)
- Thin training helpers that fit final models (SVC or MLP) with chosen parameters

Typical flow:
1) Use the grid search helpers to identify good hyperparameters
2) Train the final classifier using `train_svc_with_params` or `train_with_mlp`
"""

from typing import Literal

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from config import SVC_PARAM_GRIDS


def train_with_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate_init: float = 0.0001,
    max_iter: int = 100000,
    random_state: int = 0,
    verbose: bool = True,
) -> MLPClassifier:
    """
    Train an MLP classifier with configurable hyperparameters.

    Args:
        x_train (np.ndarray): Training features (n_samples, n_features)
        y_train (np.ndarray): Training labels (n_samples,)
        learning_rate_init (float): Initial learning rate
        max_iter (int): Maximum number of training iterations
        random_state (int): Random seed for reproducibility
        verbose (bool): Whether to print configuration and training progress

    Returns:
        MLPClassifier: Trained MLP model
    """
    # Configure the MLP with provided hyperparameters
    model = MLPClassifier(
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        verbose=verbose,
    )

    if verbose:
        print(model)
        print("MLP model configured")

    # Fit the model on the training data
    model.fit(x_train, y_train)

    if verbose:
        print("MLP model fitted successfully")

    return model


def train_with_gridsearch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 2,
    n_jobs: int = -1,
    verbose: int = 2,
) -> None:
    """
    Run a comprehensive grid search across multiple SVC kernels.

    Uses `SVC_PARAM_GRIDS["comprehensive"]` to search over kernels and common
    hyperparameters. Prints the best parameters and CV score for reference.

    Args:
        x_train (np.ndarray): Training features (n_samples, n_features)
        y_train (np.ndarray): Training labels (n_samples,)
        cv (int): Number of cross-validation folds
        n_jobs (int): Number of parallel jobs (-1 uses all cores)
        verbose (int): GridSearchCV verbosity level

    Returns:
        None
    """

    svc = svm.SVC(verbose=True)
    print("Starting comprehensive grid search across all kernel types...")
    print("This may take a long time to complete...")

    model = GridSearchCV(
        svc, SVC_PARAM_GRIDS["comprehensive"], cv=cv, n_jobs=n_jobs, verbose=verbose
    )
    model.fit(x_train, y_train)

    print("Comprehensive grid search completed!")
    print(f"Best parameters found: {model.best_params_}")


def gridsearch_with_params(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel: Literal["linear", "polynomial", "rbf"],
    param_grid: dict,
    cv: int = 2,
    n_jobs: int = -1,
    verbose: int = 2,
) -> None:
    """
    Run a kernel-specific grid search over provided hyperparameters.

    Notes:
    - The `kernel` argument is used for display/logging only.
    - The estimator is created as `svm.SVC()`; `param_grid` should contain the
      hyperparameters you want to search (e.g., {"C": [...], "gamma": [...]}).
    - Prints the best parameters found; does not return the fitted model.

    Args:
        x_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        kernel (Literal): Kernel name for logging ('linear', 'polynomial', 'rbf')
        param_grid (dict): Parameter grid for GridSearchCV
        cv (int): Number of cross-validation folds
        n_jobs (int): Number of parallel jobs (-1 for all cores)
        verbose (int): Verbosity level for GridSearchCV

    Returns:
        None
    """
    svc = svm.SVC(verbose=True)
    print(f"Starting {kernel.upper()} SVC grid search...")

    model = GridSearchCV(svc, param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
    model.fit(x_train, y_train)

    print(f"{kernel.upper()} SVC grid search completed!")
    print(f"Best parameters found: {model.best_params_}")


def train_svc_with_params(
    x_train: np.ndarray,
    y_train: np.ndarray,
    kernel: Literal["linear", "poly", "rbf"],
    C: float = 1.0,  # pylint: disable=invalid-name
    gamma: float | Literal["scale", "auto"] = "scale",
    verbose: bool = True,
) -> svm.SVC:
    """
    Train a final SVC model with explicit hyperparameters.

    Use this after grid search: pass the selected kernel, C, and gamma
    to obtain a trained model suitable for evaluation.

    Args:
        x_train (np.ndarray): Training features (n_samples, n_features)
        y_train (np.ndarray): Training labels (n_samples,)
        kernel (Literal): One of {'linear', 'poly', 'rbf'}
        C (float): Regularization parameter
        gamma (float | Literal): Kernel coefficient ('scale', 'auto', or float)
        verbose (bool): Whether to print configuration and training progress

    Returns:
        svm.SVC: Trained SVC model
    """
    # Create model with specified parameters
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, verbose=verbose)

    if verbose:
        print(f"Training {kernel.upper()} SVC with parameters:")
        print(f"  C: {C}, gamma: {gamma}")
        print("Model configuration completed")

    # Train the model
    model.fit(x_train, y_train)

    if verbose:
        print(f"{kernel.upper()} SVC model fitted successfully")

    return model

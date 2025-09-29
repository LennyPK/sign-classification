"""Main script for traffic sign classification project.

This script orchestrates the complete machine learning pipeline:
1. Load and preprocess data
2. Split data into train/test sets
3. Train models with hyperparameter optimization
4. Evaluate model performance
5. Generate comprehensive reports
"""

import gc

import pandas as pd
from sklearn.metrics import accuracy_score

import data
from config import CV_FOLDS, N_JOBS, SVC_PARAM_GRIDS
from model import (
    gridsearch_with_params,
    train_svc_with_params,
    train_with_mlp,
)
from preprocessing import preprocessing_data


def main():
    """Main function to load and process traffic sign images."""

    label_map = data.load_labels()

    # For the purpose of printing, convert map to df
    label_map_df = pd.DataFrame(label_map.items(), columns=["ClassId", "Name"])
    print(label_map_df.head(10))

    flat_data, target = data.load_data(label_map)

    gc.collect()

    x_train, y_train, x_test, y_test = preprocessing_data(flat_data, target)

    print("=" * 100)
    print("TRAIN MODEL")
    print("=" * 100)

    # MLP model
    model = train_with_mlp(
        x_train,
        y_train,
        learning_rate_init=0.0001,
        max_iter=100000,
        random_state=0,
        verbose=True,
    )
    # Accuracy Score: 97.74%

    # SVC model
    gridsearch_with_params(
        x_train,
        y_train,
        kernel="linear",
        param_grid=SVC_PARAM_GRIDS["linear"],
        cv=CV_FOLDS,
        n_jobs=-N_JOBS,
        verbose=2,
    )
    # Best parameters found: {'C': 1, 'kernel': 'linear'}

    gridsearch_with_params(
        x_train,
        y_train,
        kernel="rbf",
        param_grid=SVC_PARAM_GRIDS["rbf"],
        cv=CV_FOLDS,
        n_jobs=-N_JOBS,
        verbose=2,
    )
    # Best parameters found: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

    gridsearch_with_params(
        x_train,
        y_train,
        kernel="polynomial",
        param_grid=SVC_PARAM_GRIDS["poly"],
        cv=CV_FOLDS,
        n_jobs=-N_JOBS,
        verbose=2,
    )
    # Best parameters found: {'C': 10, 'kernel': 'poly'}

    model = train_svc_with_params(x_train, y_train, kernel="linear", C=1, verbose=True)
    # SVC Accuracy Score: 98.14%

    model = train_svc_with_params(
        x_train, y_train, kernel="rbf", C=10, gamma=0.001, verbose=True
    )
    # SVC Accuracy Score: 95.08%

    model = train_svc_with_params(x_train, y_train, kernel="poly", C=10, verbose=True)
    # SVC Accuracy Score: 84.82%

    y_pred = model.predict(x_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"SVC Accuracy Score: {acc_score * 100:.2f}%")


if __name__ == "__main__":
    main()

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

import data
from evaluate import evaluate_multiclass_classification
from model import (
    # train_with_mlp,
    # train_with_gridsearch,
    # gridsearch_with_params,
    train_svc_with_params,
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
    # model = train_with_mlp(
    #     x_train,
    #     y_train,
    #     learning_rate_init=0.0001,
    #     max_iter=100000,
    #     random_state=0,
    #     verbose=True,
    # )
    # Accuracy Score: 97.74%

    # SVC Model
    # gridsearch_with_params(
    #     x_train,
    #     y_train,
    #     kernel="linear",
    #     param_grid=SVC_PARAM_GRIDS["linear"],
    #     cv=CV_FOLDS,
    #     n_jobs=-N_JOBS,
    #     verbose=2,
    # )
    # Best parameters found: {'C': 1, 'kernel': 'linear'}

    # gridsearch_with_params(
    #     x_train,
    #     y_train,
    #     kernel="rbf",
    #     param_grid=SVC_PARAM_GRIDS["rbf"],
    #     cv=CV_FOLDS,
    #     n_jobs=-N_JOBS,
    #     verbose=2,
    # )
    # Best parameters found: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

    # gridsearch_with_params(
    #     x_train,
    #     y_train,
    #     kernel="polynomial",
    #     param_grid=SVC_PARAM_GRIDS["poly"],
    #     cv=CV_FOLDS,
    #     n_jobs=-N_JOBS,
    #     verbose=2,
    # )
    # Best parameters found: {'C': 10, 'kernel': 'poly'}

    # model = train_svc_with_params(x_train, y_train, kernel="linear", C=1, verbose=True)
    # SVC Accuracy Score: 98.14%

    model = train_svc_with_params(
        x_train, y_train, kernel="rbf", C=10, gamma=0.01, verbose=True
    )
    # model = train_svc_with_params(
    #     x_train, y_train, kernel="rbf", C=10, gamma=0.001, verbose=True
    # )
    # SVC Accuracy Score: 95.08%

    # model = train_svc_with_params(x_train, y_train, kernel="poly", C=10, verbose=True)
    # SVC Accuracy Score: 84.82%

    y_pred = model.predict(x_test)

    # accuracy_score = metrics.accuracy_score(y_test, y_pred)
    # precision_score = metrics.precision_score(y_test, y_pred)
    # recall_score = metrics.recall_score(y_test, y_pred)
    # f1_score = metrics.f1_score(y_test, y_pred)
    # print(f"Accuracy Score: {accuracy_score * 100:.2f}%")
    # print(f"Precision Score: {precision_score * 100:.2f}%")
    # print(f"Recall Score: {recall_score * 100:.2f}%")
    # print(f"F1 Score: {f1_score * 100:.2f}%")

    # print(
    #     classification_report(
    #         y_test,
    #         y_pred,
    #     )
    # )
    evaluate_multiclass_classification(y_test, y_pred, list(label_map.values()))

    # Classification metrics: Accuracy, precision, recall, F1-score,Receiver Operating
    # Receiver Operating Characteristics (ROC), Area Under Curve (AUC),
    # Mathews Correlation Coefficient (MCC)
    # multi class Confusion matrix and classification report
    # Mean absolute error (MAE) and mean square error (MSE) root mean square error (RMSE)
    # R2 score (R²)
    # Precision is the ratio of correct positive predictions and total positives predictions
    # Recall (aka True Positive Rate, TPR) is the ratio of the correct positive predictions and all actual positive samples
    # Precision measures type I error (incorrect prediction), while recall measures type II error (incorrect rejection)
    # F1-score is a combined metric that takes both precision and recall into consideration by working out the harmonic mean of the two


if __name__ == "__main__":
    main()

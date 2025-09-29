from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


def evaluate_multiclass_classification(y_test, y_pred, class_names):
    """Comprehensive multiclass evaluation"""
    print("=" * 100)
    print("EVALUATION REPORT")
    print("=" * 100)

    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Macro averages (treats all classes equally)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Weighted averages (weighted by class frequency)
    precision_weighted = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Micro averages (global precision = recall = F1 = accuracy)
    precision_micro = precision_score(y_test, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_test, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix Shape: {cm.shape}")
    print(f"Total Predictions: {cm.sum()}")
    print(f"Correct Predictions: {np.trace(cm)}")
    print(f"Incorrect Predictions: {cm.sum() - np.trace(cm)}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.show()

    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    print(report)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(f"Precision (Macro): {precision_macro * 100:.2f}%")
    print(f"Recall (Macro): {recall_macro * 100:.2f}%")
    print(f"F1 (Macro): {f1_macro * 100:.2f}%")

    print(f"Precision (Weighted): {precision_weighted * 100:.2f}%")
    print(f"Recall (Weighted): {recall_weighted * 100:.2f}%")
    print(f"F1 (Weighted): {f1_weighted * 100:.2f}%")

    print(f"Precision (Micro): {precision_micro * 100:.2f}%")
    print(f"Recall (Micro): {recall_micro * 100:.2f}%")
    print(f"F1 (Micro): {f1_micro * 100:.2f}%")

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "confusion_matrix": cm,
    }


def evaluate_model(
    model: svm.SVC,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    detailed: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a trained model and return comprehensive metrics.

    Args:
        model (svm.SVC): Trained SVC model
        x_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        model_name (str): Name of the model for display
        detailed (bool): Whether to include detailed per-class metrics

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Calculate macro averages
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }

    if detailed:
        # Cross-validation scores
        cv_scores = cross_val_score(model, x_test, y_test, cv=3)
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()

        # Confusion matrix info
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix_shape"] = cm.shape
        metrics["correct_predictions"] = np.trace(cm)
        metrics["total_predictions"] = cm.sum()
        metrics["incorrect_predictions"] = cm.sum() - np.trace(cm)

    return metrics


def print_evaluation_report(
    model: svm.SVC,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> None:
    """
    Print a comprehensive evaluation report for a model.

    Args:
        model (svm.SVC): Trained SVC model
        x_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        model_name (str): Name of the model for display
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*60}")

    # Get evaluation metrics
    metrics = evaluate_model(model, x_test, y_test, model_name, detailed=True)

    # Print basic metrics
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")

    # Print cross-validation info
    print("\nCross-Validation (3-fold):")
    print(f"  Mean Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

    # Print confusion matrix summary
    print("\nConfusion Matrix Summary:")
    print(f"  Shape: {metrics['confusion_matrix_shape']}")
    print(f"  Correct Predictions: {metrics['correct_predictions']}")
    print(f"  Incorrect Predictions: {metrics['incorrect_predictions']}")
    print(f"  Total Predictions: {metrics['total_predictions']}")

    # Detailed classification report
    y_pred = model.predict(x_test)
    print("\nDetailed Classification Report:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, zero_division=0))


def compare_models(
    models: Dict[str, svm.SVC],
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Compare multiple trained models and return a comparison DataFrame.

    Args:
        models (Dict[str, svm.SVC]): Dictionary mapping model names to trained models
        x_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels

    Returns:
        pd.DataFrame: Comparison of model performance metrics
    """
    comparison_data = []

    for model_name, model in models.items():
        metrics = evaluate_model(model, x_test, y_test, model_name, detailed=False)
        comparison_data.append(metrics)

    return pd.DataFrame(comparison_data)


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Print a formatted comparison of model performance.

    Args:
        comparison_df (pd.DataFrame): DataFrame from compare_models function
    """
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")

    # Select and format columns for display
    display_columns = [
        "model_name",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    display_df = comparison_df[display_columns].copy()

    # Round to 4 decimal places
    numeric_columns = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    for col in numeric_columns:
        display_df[col] = display_df[col].round(4)

    # Rename columns for better display
    display_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]

    print(display_df.to_string(index=False))

    # Highlight best performing model
    best_model = display_df.loc[display_df["Accuracy"].idxmax()]
    print(
        f"\nBest Model: {best_model['Model']} with {best_model['Accuracy']:.4f} accuracy"
    )

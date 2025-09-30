from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from plot import plot_confusion_matrix


def evaluate_model(
    y_test: np.ndarray, y_pred: np.ndarray, class_names: list, model_name: str
):
    """Comprehensive multiclass evaluation"""
    print("=" * 100)
    print(f"EVALUATION REPORT: {model_name}")
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

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names, model_name=model_name)

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

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Args:
        results_dict: Dictionary mapping model names to their metrics dictionaries
                     Each metrics dict should contain: accuracy, precision_macro,
                     recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted

    Returns:
        pd.DataFrame: Comparison table with key metrics
    """
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    # Create comparison table
    comparison_data = []
    for model_name, metrics in results_dict.items():
        row = {"model_name": model_name, **metrics}
        comparison_data.append(row)

    # Convert to DataFrame for nice formatting
    df = pd.DataFrame(comparison_data)

    # Select key metrics for display
    display_cols = [
        "model_name",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]
    display_df = df[display_cols].copy()

    # Round to 4 decimal places
    for col in display_cols[1:]:
        display_df[col] = display_df[col].round(4)

    # Rename columns
    display_df.columns = [
        "Model",
        "Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1 (Macro)",
    ]

    print(display_df.to_string(index=False))

    # Find best model
    best_model_name = display_df.loc[display_df["Accuracy"].idxmax(), "Model"]
    best_accuracy = display_df.loc[display_df["Accuracy"].idxmax(), "Accuracy"]
    print(f"\nBest Model: {best_model_name} with {best_accuracy:.4f} accuracy")

    return df

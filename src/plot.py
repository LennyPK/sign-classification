import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix Shape: {cm.shape}")
    print(f"Total Predictions: {cm.sum()}")
    print(f"Correct Predictions: {np.trace(cm)}")
    print(f"Incorrect Predictions: {cm.sum() - np.trace(cm)}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(f'confusion_matrix_{model_name.lower().replace("-", "_")}.png', dpi=300)
    plt.show()


def plot_model_comparison(results_dict, metric="accuracy"):
    """Plot comparison of models for a specific metric"""
    models = list(results_dict.keys())
    values = [results_dict[model][metric] for model in models]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.title(f"Model Comparison - {metric.title()}")
    plt.ylabel(metric.title())
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

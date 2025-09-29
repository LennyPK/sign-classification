"""Data preprocessing utilities module.

This module provides utilities for data preprocessing, including train/test
splitting, data validation, and feature preparation for machine learning models.
"""

import gc
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE, VERBOSE


def preprocessing_data(
    flat_data: np.ndarray,
    target: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning by creating train/test splits.

    Args:
        flat_data (np.ndarray): Flattened image data with shape (n_samples, n_features)
        target (np.ndarray): Target labels with shape (n_samples,)
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 0)
        stratify (bool): Whether to stratify the split to maintain class distribution

    Returns:
        Tuple containing:
            - x_train (np.ndarray): Training features
            - y_train (np.ndarray): Training labels
            - x_test (np.ndarray): Test features
            - y_test (np.ndarray): Test labels
    """
    print("=" * 100)
    print("PREPROCESSING DATA")
    print("=" * 100)

    if VERBOSE:
        print("Preparing data for machine learning...")
        print(f"Original data shape: {flat_data.shape}")
        print(f"Target shape: {target.shape}")

    # Create DataFrame with features and target
    df = pd.DataFrame(flat_data)
    df["Target"] = target

    if VERBOSE:
        print(df.head())
        print(f"DataFrame shape: {df.shape}")
        print("Target distribution:")
        print(df["Target"].value_counts().sort_index().head(10))

    # Separate features and target
    x = df.iloc[:, :-1]  # All columns except the last (Target)
    y = df.iloc[:, -1]  # Last column (Target)

    if VERBOSE:
        print(f"Features shape: {x.shape}")
        print(f"Target shape: {y.shape}")

    # Perform train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if VERBOSE:
        print("Data split completed successfully")
        print("=" * 50)
        print("TRAIN DATA SHAPES")
        print("=" * 50)
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print("=" * 50)
        print("TEST DATA SHAPES")
        print("=" * 50)
        print(f"x_test shape: {x_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print("=" * 50)

        # Show class distribution in train/test sets
        print("Class distribution in training set:")
        train_dist = y_train.value_counts().sort_index()
        print(train_dist.head(10))

        print("\nClass distribution in test set:")
        test_dist = y_test.value_counts().sort_index()
        print(test_dist.head(10))

    # Force garbage collection to free memory
    gc.collect()

    return x_train, y_train, x_test, y_test

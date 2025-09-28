"""Data loading and preprocessing module for traffic sign classification.

This module provides functionality to load and preprocess traffic sign image data
from the dataset. It handles:

- Loading class labels from CSV files and caching them as pickle files
- Loading and preprocessing image data with batch processing for memory efficiency
- Resizing images to standard dimensions (32x32x3)
- Caching preprocessed data as pickle files for faster subsequent loading
- Memory-efficient batch processing for large datasets

The module uses a two-tier caching system:
1. Labels are cached as pickle files to avoid repeated CSV parsing
2. Preprocessed image data is cached in batches to enable fast loading of large datasets

Example:
    >>> from data import load_labels, load_data
    >>>
    >>> # Load class labels
    >>> label_map = load_labels()
    >>> print(f"Loaded {len(label_map)} classes")
    >>>
    >>> # Load preprocessed image data
    >>> X, y = load_data(label_map)
    >>> print(f"Loaded {X.shape[0]} images with {X.shape[1]} features each")
"""

import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize  # pylint: disable=no-name-in-module
from tqdm import tqdm, trange

from config import (
    BATCH_SIZE,
    DATA_META_PICKLE_PATH,
    DATA_PATH,
    DATA_PICKLE_PATH,
    IMAGE_SIZE,
    LABELS_CSV_PATH,
    LABELS_PICKLE_PATH,
    NUM_CATEGORIES,
)


def load_labels() -> Dict[int, str]:
    """
    Load or create label mapping from CSV file.

    This function implements a caching mechanism where labels are first loaded from
    a CSV file, converted to a dictionary mapping class IDs to class names, and
    then cached as a pickle file for faster subsequent access.

    The function checks if a cached pickle file exists first. If it does, the labels
    are loaded from the pickle file. If not, the labels are loaded from the CSV file,
    converted to a dictionary, and then saved as a pickle file for future use.

    Returns:
        Dict[int, str]: A dictionary mapping class IDs (integers) to class names (strings).
                       Keys are class IDs (0 to NUM_CATEGORIES-1), values are descriptive
                       class names (e.g., "Speed limit (30km/h)", "Stop", etc.).

    Example:
        >>> label_map = load_labels()
        >>> print(label_map[0])  # Should print something like "Speed limit (20km/h)"
        >>> print(len(label_map))  # Should print 43 (NUM_CATEGORIES)
    """
    if os.path.exists(LABELS_PICKLE_PATH):
        print("Loading labels from cached pickle file...")

        # Load label mapping from pickle file for faster access
        with open(LABELS_PICKLE_PATH, "rb") as f:
            label_map = pickle.load(f)

        print("Labels loaded successfully from cache")

    else:
        print("Loading labels from CSV file (first time or cache missing)...")

        # Load labels from CSV file and create mapping
        labels_df = pd.read_csv(LABELS_CSV_PATH)
        label_map = dict(zip(labels_df["ClassId"], labels_df["Name"]))

        print("Saving labels to pickle file for faster future access...")

        # Cache the label mapping as pickle file for faster future loading
        with open(LABELS_PICKLE_PATH, "wb") as f:
            pickle.dump(label_map, f)

        print("Labels cached successfully")

    return label_map


def load_data(label_map: Dict[int, str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load or create preprocessed image data with intelligent caching.

    This function implements a sophisticated caching system for image data processing.
    It first checks if preprocessed data exists in pickle format. If found, it loads
    the data efficiently in batches. If not found, it processes raw images from the
    file system, resizes them to standard dimensions, and caches the results.

    The function processes images in batches to manage memory usage efficiently,
    especially important for large datasets. Images are resized to IMAGE_SIZE (32x32x3)
    and flattened into 1D arrays for machine learning compatibility.

    Args:
        label_map (Dict[int, str]): Mapping of class IDs to class names, typically
                                   obtained from load_labels().

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - flat_data (np.ndarray): Flattened image data with shape (n_samples, 3072)
                                     where 3072 = 32*32*3. Each row is a flattened
                                     image with pixel values normalized to [0,1].
            - target (np.ndarray): Class labels with shape (n_samples,) containing
                                  integer class IDs corresponding to the images.

    Note:
        - Images are automatically resized to 32x32 pixels with 3 color channels
        - Pixel values are normalized to the range [0, 1] during resizing
        - The function uses batch processing to handle large datasets efficiently
        - Cached data is stored in pickle format for fast subsequent loading

    Example:
        >>> label_map = load_labels()
        >>> X, y = load_data(label_map)
        >>> print(f"Loaded {X.shape[0]} images")
        >>> print(f"Image shape: {X.shape[1]} features (32x32x3 flattened)")
        >>> print(f"Classes: {len(np.unique(y))}")
    """
    # Initialize list of all category IDs (0 to NUM_CATEGORIES-1)
    categories = list(range(NUM_CATEGORIES))

    if os.path.exists(DATA_PICKLE_PATH):
        print("Loading preprocessed data from cached pickle file...")

        # Load metadata about the cached data
        with open(DATA_META_PICKLE_PATH, "rb") as f:
            meta = pickle.load(f)

        # Extract metadata for efficient loading
        n_samples = meta["n_samples"]
        batch_size = meta["batch_size"]
        n_batches = int(np.ceil(n_samples / batch_size))

        print(f"Metadata loaded: n_samples={n_samples}, batch_size={batch_size}")
        print(f"Loading data in {n_batches} batches for memory efficiency...")

        # Preallocate arrays with known dimensions for memory efficiency
        flat_data = np.empty((n_samples, np.prod(IMAGE_SIZE)), dtype=np.float32)
        target = np.empty((n_samples,), dtype=np.int32)

        idx = 0

        # Load data from pickle file in batches to manage memory usage
        with open(DATA_PICKLE_PATH, "rb") as f:
            for _ in trange(n_batches, desc="Loading cached data", unit="batch"):
                batch_flat_data, batch_targets = pickle.load(f)

                # Handle the last batch which might be smaller than batch_size
                batch_size_actual = batch_flat_data.shape[0]

                # Store batch data in preallocated arrays
                flat_data[idx : idx + batch_size_actual] = batch_flat_data
                target[idx : idx + batch_size_actual] = batch_targets

                idx += batch_size_actual

        print("Data loaded successfully from cache")

        return flat_data, target

    else:
        print("Processing raw image data (first time or cache missing)...")

        n_samples = 0

        # Process and cache image data in batches
        with open(DATA_PICKLE_PATH, "wb") as f:
            batch_flat_data, batch_targets = [], []

            for category in categories:
                print(f"Processing category [{category}]: {label_map[category]}")

                # Construct path to images for the current category
                img_path = os.path.join(DATA_PATH, str(category))
                img_files = os.listdir(img_path)

                # Process each image in the current category
                for img_name in tqdm(
                    img_files, desc=f"Category {category}", unit="img"
                ):
                    # Read image from file system
                    img_array = imread(os.path.join(img_path, img_name))

                    # Remove extra singleton dimensions (handles grayscale images)
                    # This prevents errors when concatenating arrays of different shapes
                    img_array: np.ndarray = np.squeeze(img_array)

                    # Resize image to standard size (32x32x3) and normalize to [0,1]
                    img_resized = np.asarray(resize(img_array, IMAGE_SIZE))

                    # Flatten image to 1D array and add to current batch
                    batch_flat_data.append(img_resized.flatten())
                    batch_targets.append(category)

                    # Save batch when it reaches BATCH_SIZE to manage memory usage
                    if len(batch_flat_data) == BATCH_SIZE:
                        batch_np_flat_data = np.array(batch_flat_data, dtype=np.float32)
                        batch_np_targets = np.array(batch_targets, dtype=np.int32)
                        pickle.dump((batch_np_flat_data, batch_np_targets), f)

                        # Reset batch lists for next batch
                        batch_flat_data, batch_targets = [], []

                    # Increment total sample count
                    n_samples += 1

                print(f"Completed category [{category}]: {label_map[category]}")
                print(f"Total samples processed so far: {n_samples}")

            # Save any remaining data that didn't fill a complete batch
            if batch_flat_data:
                batch_np_flat_data = np.array(batch_flat_data, dtype=np.float32)
                batch_np_targets = np.array(batch_targets, dtype=np.int32)
                pickle.dump((batch_np_flat_data, batch_np_targets), f)

        # Save metadata about the processed dataset for future loading
        meta = {"n_samples": n_samples, "batch_size": BATCH_SIZE}
        with open(DATA_META_PICKLE_PATH, "wb") as f:
            pickle.dump(meta, f)

        print(f"Data processing complete. Saved {n_samples} samples to cache.")

        # Recursively call this function to load the newly cached data
        # This ensures consistent return format and validates the caching worked
        return load_data(label_map)

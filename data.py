"""Data loading and preprocessing module."""

import os
import pickle

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


def load_labels():
    """Load or create label mapping."""
    if os.path.exists(LABELS_PICKLE_PATH):
        print("loading labels from labels pickle file")

        # Load label mapping from pickle
        with open(LABELS_PICKLE_PATH, "rb") as f:
            label_map = pickle.load(f)

        print("labels loaded successfully")

    else:
        print("loading labels from labels csv file")

        # Load labels from CSV
        labels_df = pd.read_csv(LABELS_CSV_PATH)
        label_map = dict(zip(labels_df["ClassId"], labels_df["Name"]))

        print("saving labels to labels pickle file")

        # Save label mapping to pickle
        with open(LABELS_PICKLE_PATH, "wb") as f:
            pickle.dump(label_map, f)

        print("labels saved successfully")

    # Return label map
    return label_map


def load_data(label_map: dict[int, str]):
    """
    Load or create image data.

    Args:
        label_map (dict): Mapping of class IDs to class names.

    Returns:
        flat_data (np.ndarray): Flattened image data, shape (n_samples, np.prod(IMAGE_SIZE)).
        target (np.ndarray): Labels, shape (n_samples,).
    """

    # Initialise categories
    categories = list(range(NUM_CATEGORIES))

    if os.path.exists(DATA_PICKLE_PATH):
        print("loading data from data pickle file")

        # Load data meta information
        with open(DATA_META_PICKLE_PATH, "rb") as f:
            meta = pickle.load(f)

        # Extract meta information
        n_samples = meta["n_samples"]
        batch_size = meta["batch_size"]
        n_batches = int(np.ceil(n_samples / batch_size))

        print(f"Metadata loaded: n_samples={n_samples}, batch_size={batch_size}")
        print(f"Loading data in {n_batches} batches...")

        # Preallocate arrays
        flat_data = np.empty((n_samples, np.prod(IMAGE_SIZE)), dtype=np.float32)
        target = np.empty((n_samples,), dtype=np.int32)

        idx = 0

        # Load data from pickle file in batches
        with open(DATA_PICKLE_PATH, "rb") as f:
            for _ in trange(n_batches, desc="Loading", unit="batch"):
                batch_flat_data, batch_targets = pickle.load(f)

                # Handle last batch which might be smaller than batch_size
                batch_size_actual = batch_flat_data.shape[0]

                # Store batch data in preallocated arrays
                flat_data[idx : idx + batch_size_actual] = batch_flat_data
                target[idx : idx + batch_size_actual] = batch_targets

                idx += batch_size_actual

                # flat_data_arr.extend(batch_flat_data)
                # target_arr.extend(batch_targets)

        print("data loaded successfully")

        return flat_data, target

    else:
        print("loading data from data image files")

        n_samples = 0

        with open(DATA_PICKLE_PATH, "wb") as f:
            batch_flat_data, batch_targets = [], []

            for category in categories:
                print(f"loading... category : [{category}] {label_map[category]}")

                # Construct path to images for the current category
                img_path = os.path.join(DATA_PATH, str(category))
                img_files = os.listdir(img_path)

                # Process each image in the category
                for img_name in tqdm(
                    img_files, desc=f"Category {category}", unit="img"
                ):
                    # Read image
                    img_array = imread(os.path.join(img_path, img_name))

                    # Remove extra singleton dimensions, otherwise cat140 throws error
                    img_array: np.ndarray = np.squeeze(img_array)

                    # Resize image to standard size
                    img_resized = np.asarray(resize(img_array, IMAGE_SIZE))

                    # Flatten image and append to batch data
                    batch_flat_data.append(img_resized.flatten())
                    batch_targets.append(category)

                    # Convert to np array and save batch when it reaches BATCH_SIZE
                    if len(batch_flat_data) == BATCH_SIZE:
                        batch_np_flat_data = np.array(batch_flat_data, dtype=np.float32)
                        batch_np_targets = np.array(batch_targets, dtype=np.int32)
                        pickle.dump((batch_np_flat_data, batch_np_targets), f)

                        # Reset batch lists
                        batch_flat_data, batch_targets = [], []

                    # Increment sample count
                    n_samples += 1

                print(
                    f"loaded category : [{category}] {label_map[category]} successfully"
                )
                print(f"n_samples so far: {n_samples}")

            # Conver to np array and save any remaining data
            if batch_flat_data:
                batch_np_flat_data = np.array(batch_flat_data, dtype=np.float32)
                batch_np_targets = np.array(batch_targets, dtype=np.int32)
                pickle.dump((batch_np_flat_data, batch_np_targets), f)

        # Save data meta information
        meta = {"n_samples": n_samples, "batch_size": BATCH_SIZE}
        with open(DATA_META_PICKLE_PATH, "wb") as f:
            pickle.dump(meta, f)

        print(f"Saved {n_samples}")

        # Reload to get NumPy arrays
        return load_data(label_map)

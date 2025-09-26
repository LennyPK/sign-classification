"""main.py"""

import gc
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import data

# Project paths
PROJECT_ROOT = Path(__file__).parent
# LABELS_CSV_PATH = PROJECT_ROOT / "labels.csv"
# DATA_PATH = PROJECT_ROOT / "myData"
# DATA_PICKLE_PATH = PROJECT_ROOT / "data.pickle"z
# LABELS_PICKLE_PATH = PROJECT_ROOT / "labels.pickle"

# Image processing parameters
IMAGE_SIZE = (150, 150, 3)
NUM_CATEGORIES = 43

# Data processing parameters
BATCH_SIZE = 1000  # For processing large datasets in chunks
VERBOSE = True

LABELS_CSV_PATH = "labels.csv"
DATA_PATH = "myData"
DATA_PICKLE_PATH = "data.pickle"
DATA_META_PICKLE_PATH = "data_meta.pickle"
LABELS_PICKLE_PATH = "labels.pickle"


def main():
    """Main function to load and process traffic sign images."""

    label_map = data.load_labels()

    # For the purpose of printing, convert map to df
    label_map_df = pd.DataFrame(label_map.items(), columns=["ClassId", "Name"])
    print(label_map_df.head(10))

    flat_data, target = data.load_data(label_map)

    gc.collect()

    df = pd.DataFrame(flat_data)
    df["Target"] = target
    print(df.head())
    print(df.shape)

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y
    )
    print("Data split successfully")
    print("X_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("\n")
    print("X_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    model = MLPClassifier(random_state=0, learning_rate_init=0.001, max_iter=100)
    print(model)
    print("Model configured")
    model.fit(x_train, y_train)
    print("Model fitted")
    y_pred = model.predict(x_test)

    acc_score = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {acc_score * 100:.2f}%")


if __name__ == "__main__":
    main()

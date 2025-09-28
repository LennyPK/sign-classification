"""Configuration module for traffic sign classification project.

This module contains all configuration constants used throughout the project, including file paths,
image processing parameters, and logging settings. Centralizing configuration makes the codebase
more maintainableand allows for easy parameter tuning.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
# All file and directory paths are defined here to ensure consistency
# across the entire project and make path changes easier to manage

# Root directory of the project
PROJECT_ROOT = Path(__file__).parent
# CSV file containing class labels and descriptions
LABELS_CSV_PATH = PROJECT_ROOT / "labels.csv"
# Directory containing training/test image data organized by class
DATA_PATH = PROJECT_ROOT / "myData"
# Serialized preprocessed image data for faster loading
DATA_PICKLE_PATH = PROJECT_ROOT / "data.pickle"
# Serialized label data for faster access
LABELS_PICKLE_PATH = PROJECT_ROOT / "labels.pickle"
# Metadata about the dataset (counts, statistics)
DATA_META_PICKLE_PATH = PROJECT_ROOT / "data_meta.pickle"

# =============================================================================
# IMAGE PROCESSING PARAMETERS
# =============================================================================
# These parameters define how images are processed and what the model expects

# Target image dimensions: 32x32 pixels with 3 color channels (RGB)
IMAGE_SIZE = (
    32,
    32,
    3,
)
# Number of traffic sign classes in the dataset
NUM_CATEGORIES = 43

# =============================================================================
# DATA PROCESSING PARAMETERS
# =============================================================================
# Parameters that control how data is loaded and processed in batches

BATCH_SIZE = 1000  # Number of images to process in each batch to manage memory usage

"""Configuration module for traffic sign classification project."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
LABELS_CSV_PATH = PROJECT_ROOT / "labels.csv"
DATA_PATH = PROJECT_ROOT / "myData"
DATA_PICKLE_PATH = PROJECT_ROOT / "data.pickle"
LABELS_PICKLE_PATH = PROJECT_ROOT / "labels.pickle"
DATA_META_PICKLE_PATH = PROJECT_ROOT / "data_meta.pickle"

# Image processing parameters
IMAGE_SIZE = (150, 150, 3)
NUM_CATEGORIES = 43

# Data processing parameters
BATCH_SIZE = 1000  # For processing large datasets in chunks
VERBOSE = True

# File extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

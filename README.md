# Traffic Sign Classification Project

A machine learning project for classifying traffic signs using Support Vector Machines (SVM) and Multilayer Perceptron (MLP) neural networks.

## Project Overview

This project implements a complete machine learning pipeline for traffic sign classification, including data preprocessing, model training, hyperparameter optimization, and comprehensive evaluation. The system can classify 43 different types of traffic signs with high accuracy.

## Project Structure

### Core Python Files

#### `main.py`

- **Function**: Entry point for the entire project
- **Key Features**:
  - Loads and preprocesses traffic sign data
  - Trains multiple models (MLP, SVC Linear, SVC RBF, SVC Poly)
  - Evaluates model performance on test data
  - Generates comparison reports and visualizations
- **Usage**: Run `python main.py` to execute the complete pipeline

#### `data.py`

**Purpose**: Data loading and preprocessing module

- **Function**: Handles all data-related operations
- **Key Features**:
  - Loads traffic sign images from `myData/` directory
  - Resizes images to 32×32×3 pixels
  - Normalizes pixel values to [0,1] range
  - Implements caching system using pickle files for efficiency
  - Loads class label mappings from `labels.csv`
- **Dependencies**: `labels.csv`, `myData/` directory structure

#### `model.py`

**Purpose**: Model training and hyperparameter optimization

- **Function**: Contains all machine learning model implementations
- **Key Features**:
  - **SVC Models**: Linear, RBF, and Polynomial kernel implementations
  - **MLP Model**: Multilayer Perceptron neural network
  - **Grid Search**: Automated hyperparameter optimization using GridSearchCV
  - **Training Functions**: Individual model training with specified parameters
- **Models Supported**:
  - Support Vector Classifier (SVC) with multiple kernels
  - Multilayer Perceptron (MLP) classifier

#### `evaluate.py`

**Purpose**: Model evaluation and performance metrics

- **Function**: Comprehensive evaluation and comparison utilities
- **Key Features**:
  - **Metrics Calculation**: Accuracy, precision, recall, F1-score
  - **Confusion Matrix**: Detailed classification analysis
  - **Model Comparison**: Side-by-side performance comparison
  - **Per-Class Analysis**: Individual class performance breakdown
- **Output**: Detailed evaluation reports and comparison tables

#### `preprocessing.py`

**Purpose**: Data preprocessing and validation utilities

- **Function**: Handles data preparation and quality checks
- **Key Features**:
  - **Train/Test Split**: Stratified splitting with configurable test size
  - **Feature Preparation**: Ensures data is ready for model training
- **Dependencies**: Uses configuration from `config.py`

#### `config.py`

**Purpose**: Centralized configuration management

- **Function**: Stores all project parameters and hyperparameters
- **Key Features**:
  - **Data Paths**: File and directory locations
  - **Model Parameters**: Hyperparameter grids for grid search
  - **Training Settings**: Cross-validation folds, random seeds, job counts
  - **Image Processing**: Resize dimensions, batch sizes
- **Benefits**: Easy parameter tuning without code modification

#### `plot.py`

**Purpose**: Visualization and plotting utilities

- **Function**: Generates charts and graphs for model analysis
- **Key Features**:
  - **Model Comparison Plots**: Bar charts comparing model performance
  - **Confusion Matrix Visualization**: Heatmap plots for classification analysis
  - **Performance Charts**: Various metrics visualization
- **Dependencies**: Matplotlib, Seaborn

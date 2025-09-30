"""Main script for traffic sign classification project.

This script orchestrates the complete machine learning pipeline:
1. Load and preprocess data
2. Split data into train/test sets
3. Train models with hyperparameter optimization
4. Evaluate model performance
5. Generate comprehensive reports

The script provides three main functions:
- main(): Complete pipeline with model training and evaluation
- find_best_model(): Comprehensive grid search across all models
- find_best_params_per_model(): Individual grid search per model type
"""

import gc  # For memory management during data processing

import pandas as pd

# Import custom modules
import data
from config import CV_FOLDS, N_JOBS, SVC_PARAM_GRIDS, VERBOSE_LEVEL
from evaluate import compare_models, evaluate_model
from model import (
    gridsearch_with_params,
    train_svc_with_params,
    train_with_gridsearch,
    train_with_mlp,
)
from plot import plot_model_comparison
from preprocessing import preprocessing_data


def find_best_model():
    """
    Perform comprehensive grid search across all model types to find optimal hyperparameters.
    
    This function:
    1. Loads traffic sign data and labels
    2. Preprocesses the data (resize, normalize, split)
    3. Runs grid search across all SVC kernels and MLP configurations
    4. Prints the best parameters for each model type
    
    Note: This is computationally expensive and may take several hours to complete.
    """
    # Load class label mappings (ClassId -> Name)
    label_map = data.load_labels()

    # Display sample of class labels for verification
    label_map_df = pd.DataFrame(label_map.items(), columns=["ClassId", "Name"])
    print("Sample of traffic sign classes:")
    print(label_map_df.head(10))

    # Load and preprocess image data
    flat_data, target = data.load_data(label_map)

    # Force garbage collection to free memory after data loading
    gc.collect()

    # Split data into training and testing sets
    x_train, y_train, x_test, y_test = preprocessing_data(flat_data, target)

    print("=" * 100)
    print("COMPREHENSIVE GRID SEARCH - ALL MODELS")
    print("=" * 100)

    # Run comprehensive grid search across all model configurations
    # This will test all SVC kernels (linear, RBF, poly) and MLP configurations
    train_with_gridsearch(
        x_train=x_train,
        y_train=y_train,
        cv=CV_FOLDS,  # Number of cross-validation folds
        n_jobs=N_JOBS,  # Number of parallel jobs
        verbose=VERBOSE_LEVEL,  # Verbosity level for output
    )


def find_best_params_per_model():
    """
    Perform individual grid search for each model type to find optimal hyperparameters.
    
    This function:
    1. Loads traffic sign data and labels
    2. Preprocesses the data (resize, normalize, split)
    3. Runs separate grid searches for each SVC kernel type
    4. Prints the best parameters for each individual model
    
    This approach is more efficient than comprehensive search as it focuses on one model at a time.
    """
    # Load class label mappings (ClassId -> Name)
    label_map = data.load_labels()

    # Display sample of class labels for verification
    label_map_df = pd.DataFrame(label_map.items(), columns=["ClassId", "Name"])
    print("Sample of traffic sign classes:")
    print(label_map_df.head(10))

    # Load and preprocess image data
    flat_data, target = data.load_data(label_map)

    # Force garbage collection to free memory after data loading
    gc.collect()

    # Split data into training and testing sets
    x_train, y_train, x_test, y_test = preprocessing_data(flat_data, target)

    print("=" * 100)
    print("INDIVIDUAL GRID SEARCH - PER MODEL TYPE")
    print("=" * 100)

    # SVC Linear Kernel Grid Search
    print("\n--- SVC Linear Kernel Grid Search ---")
    gridsearch_with_params(
        x_train,
        y_train,
        kernel="linear",
        param_grid=SVC_PARAM_GRIDS["linear"],
        cv=CV_FOLDS,
        n_jobs=-N_JOBS,  # Use negative value for all available cores
        verbose=VERBOSE_LEVEL,
    )
    # Best parameters found: {'C': 1, 'kernel': 'linear'}

    # SVC RBF Kernel Grid Search
    print("\n--- SVC RBF Kernel Grid Search ---")
    gridsearch_with_params(
        x_train,
        y_train,
        kernel="rbf",
        param_grid=SVC_PARAM_GRIDS["rbf"],
        cv=CV_FOLDS,
        n_jobs=-N_JOBS,
        verbose=VERBOSE_LEVEL,
    )
    # Best parameters found: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

    # SVC Polynomial Kernel Grid Search
    print("\n--- SVC Polynomial Kernel Grid Search ---")
    gridsearch_with_params(
        x_train,
        y_train,
        kernel="polynomial",
        param_grid=SVC_PARAM_GRIDS["poly"],
        cv=CV_FOLDS,
        n_jobs=-N_JOBS,
        verbose=VERBOSE_LEVEL,
    )
    # Best parameters found: {'C': 10, 'kernel': 'poly'}


def main():
    """
    Main function to execute the complete traffic sign classification pipeline.
    
    This function:
    1. Loads traffic sign data and labels
    2. Preprocesses the data (resize, normalize, split)
    3. Trains multiple models with optimized hyperparameters
    4. Evaluates each model on the test set
    5. Compares model performance and generates visualizations
    
    Models trained:
    - MLP Classifier (Multilayer Perceptron)
    - SVC Linear (Support Vector Classifier with linear kernel)
    - SVC RBF (Support Vector Classifier with RBF kernel)
    - SVC Poly (Support Vector Classifier with polynomial kernel)
    """

    # Load class label mappings (ClassId -> Name)
    label_map = data.load_labels()
    results = {}  # Dictionary to store evaluation results for each model

    # Display sample of class labels for verification
    label_map_df = pd.DataFrame(label_map.items(), columns=["ClassId", "Name"])
    print("Sample of traffic sign classes:")
    print(label_map_df.head(10))

    # Load and preprocess image data
    flat_data, target = data.load_data(label_map)

    # Force garbage collection to free memory after data loading
    gc.collect()

    # Split data into training and testing sets
    x_train, y_train, x_test, y_test = preprocessing_data(flat_data, target)

    print("=" * 100)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 100)

    # =============================================================================
    # MLP CLASSIFIER
    # =============================================================================
    print("\n--- Training MLP Classifier ---")
    model = train_with_mlp(
        x_train,
        y_train,
        learning_rate_init=0.0001,  # Initial learning rate
        max_iter=100000,  # Maximum number of iterations
        random_state=0,  # Random seed for reproducibility
        verbose=True,
    )
    y_pred = model.predict(x_test)
    metrics = evaluate_model(y_test, y_pred, list(label_map.values()), "MLP Classifier")
    results["MLP Classifier"] = metrics
    # Accuracy Score: 97.74%

    # =============================================================================
    # SVC LINEAR KERNEL
    # =============================================================================
    print("\n--- Training SVC Linear ---")
    model = train_svc_with_params(x_train, y_train, kernel="linear", C=1, verbose=True)
    y_pred = model.predict(x_test)
    metrics = evaluate_model(y_test, y_pred, list(label_map.values()), "SVC Linear")
    results["SVC Linear"] = metrics
    # SVC Accuracy Score: 98.14%

    # =============================================================================
    # SVC RBF KERNEL (with gamma=0.01)
    # =============================================================================
    print("\n--- Training SVC RBF (gamma=0.01) ---")
    model = train_svc_with_params(
        x_train, y_train, kernel="rbf", C=10, gamma=0.01, verbose=True
    )
    y_pred = model.predict(x_test)
    metrics = evaluate_model(y_test, y_pred, list(label_map.values()), "SVC RBF (gamma=0.01)")
    results["SVC RBF (gamma=0.01)"] = metrics
    # SVC Accuracy Score: 99.00%

    # =============================================================================
    # SVC RBF KERNEL (with gamma=0.001)
    # =============================================================================
    print("\n--- Training SVC RBF (gamma=0.001) ---")
    model = train_svc_with_params(
        x_train, y_train, kernel="rbf", C=10, gamma=0.001, verbose=True
    )
    y_pred = model.predict(x_test)
    metrics = evaluate_model(y_test, y_pred, list(label_map.values()), "SVC RBF (gamma=0.001)")
    results["SVC RBF (gamma=0.001)"] = metrics
    # Expected Accuracy: ~95.08%

    # =============================================================================
    # SVC POLYNOMIAL KERNEL
    # =============================================================================
    print("\n--- Training SVC Polynomial ---")
    model = train_svc_with_params(x_train, y_train, kernel="poly", C=10, verbose=True)
    y_pred = model.predict(x_test)
    metrics = evaluate_model(y_test, y_pred, list(label_map.values()), "SVC Poly")
    results["SVC Poly"] = metrics
    # SVC Accuracy Score: 84.82%

    # =============================================================================
    # MODEL COMPARISON AND VISUALIZATION
    # =============================================================================
    print("\n" + "=" * 100)
    print("MODEL COMPARISON RESULTS")
    print("=" * 100)
    
    # Generate comparison table and identify best performing model
    compare_models(results)
    
    # Generate visualization plots for model comparison
    plot_model_comparison(results)


if __name__ == "__main__":
    """
    Entry point for the script.
    
    Uncomment the desired function to run:
    - main(): Complete pipeline with model training and evaluation (default)
    - find_best_model(): Comprehensive grid search across all models
    - find_best_params_per_model(): Individual grid search per model type
    """
    main()  # Run the complete pipeline by default
    # find_best_model()  # Uncomment for comprehensive grid search
    # find_best_params_per_model()  # Uncomment for individual model grid search

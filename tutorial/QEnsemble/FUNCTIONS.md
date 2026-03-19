# QEnsemble Functions Reference

This document provides a quick reference for all functions in the QEnsemble tutorial modules.

## modeling.py

### Core Quantum Classifiers

- **`cos_classifier_OLD(train, test, label_train, printing=False)`** - Legacy single-qubit cosine classifier
- **`cos_classifier(train, test, label_train, printing=False)`** - Multi-qubit cosine classifier
- **`state_prep(x)`** - Prepare quantum state from classical data using unitary simulation
- **`quantum_cosine_classifier(train, test, label_train)`** - Cosine classifier with unitary state preparation

### Ensemble Methods

- **`ensemble_OLD(X_data, Y_data, x_test, n_swap=1, d=2, balanced=True)`** - Legacy single-qubit ensemble
- **`ensemble(X_data, Y_data, x_test, n_swap=1, d=2, mode="balanced", barriers=False)`** - Main quantum ensemble with modes: balanced, unbalanced, pair_sample
- **`ensemble_fixed_U(X_data, Y_data, x_test, d=2)`** - Ensemble with fixed unitary operations
- **`ensemble_random_swap(X_data, Y_data, x_test, d=2)`** - Ensemble using ancilla qubits

### Execution

- **`exec_simulator(qc, n_shots=8192, device='CPU')`** - Execute quantum circuit on Aer simulator

## modeling_random_unitary.py

### Workflow

- **`run_ensemble(d, n_train, seed, n_swap, X_train, X_test, y_train, y_test, mode="balanced", n_shots=8192)`** - Complete ensemble workflow with evaluation

### Quantum Classifiers (Random Unitary Variants)

- **`cos_classifier_OLD(train, test, label_train, printing=False)`** - Legacy classifier
- **`cos_classifier(train, test, label_train, printing=False)`** - Multi-qubit classifier
- **`state_prep(x)`** - State preparation
- **`quantum_cosine_classifier(train, test, label_train)`** - Unitary-based classifier

### Random Unitary Ensemble

- **`ensemble(X_data, Y_data, x_test, n_swap=1, d=2, mode="balanced", barriers=False)`** - Ensemble using Haar-random unitaries
- **`ensemble_fixed_U(X_data, Y_data, x_test, d=2)`** - Fixed unitary variant
- **`ensemble_random_swap(X_data, Y_data, x_test, d=2)`** - Ancilla-based variant

### Utilities

- **`exec_simulator(qc, n_shots=1024)`** - Execute on simulator
- **`label_to_array(y)`** - Convert binary labels to one-hot encoding
- **`evaluation_metrics(predictions, y_test, save=True)`** - Calculate accuracy and Brier score

## Utils.py

### Prediction Analysis

- **`calculate_predicted_classes(preds)`** - Convert probabilities to class labels
- **`calculate_number_predicted_classes(preds)`** - Count unique predicted classes
- **`calculate_f1(preds, y_test)`** - Calculate weighted F1 score

### Visualization

- **`plot_figures(results_df_agg, dataset_name, method, figsize=(12,3))`** - Create performance comparison plots
- **`plot_cls(predictions, title='Test point classification', file=None)`** - Plot classification probabilities
- **`multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5)`** - Create multivariate grid plots
- **`quantum_cos_random_data(x, P0, P1, err)`** - Plot quantum cosine classifier behavior with error bands
- **`avg_vs_ensemble(avg, ens, ens_real=None)`** - Compare ensemble vs average predictions

### Quantum Ensemble Workflows

- **`run_quantum_ensemble(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, ds, n_swaps, n_features, n_trains, n_shots, ...)`** - Run quantum ensemble experiments across parameter configurations
- **`run_quantum_cosine(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, n_features, n_trains, n_shots, ...)`** - Run quantum cosine classifier experiments
- **`run_ensemble(d, n_train, seed, n_swap, X_train, X_test, y_train, y_test, mode="pair_sample", n_shots=8192, ...)`** - Execute single ensemble run with IBM Quantum support

### Classical Baselines

- **`run_random_forest(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, select_features=[], params={}, ...)`** - Run Random Forest with hyperparameter tuning
- **`run_xgboost(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, select_features=[], params={}, ...)`** - Run XGBoost with hyperparameter tuning
- **`run_lazy_predict(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, select_features=[])`** - Run LazyPredict for quick baseline comparison

### Data Processing

- **`normalize_custom_OLD(x, C=1)`** - Legacy normalization for 2D data
- **`normalize_custom(x, C=1)`** - Normalize data for quantum encoding
- **`training_set_OLD(X, Y, n=4, seed=123)`** - Legacy training set selection
- **`training_set(X, Y, n=4, seed=123, selectRandom=True)`** - Select balanced training subset
- **`label_to_array(y)`** - Convert binary labels to one-hot encoding
- **`load_data_custom(X_data=None, Y_data=None, x_test=None, normalize=True)`** - Load custom data with normalization
- **`load_data(n=100, centers=[[1,.3],[.3,1]], std=.20, seed=123, plot=True, save=True)`** - Generate blob datasets

### Evaluation

- **`evaluation_metrics(predictions, y_test, save=True)`** - Calculate accuracy and Brier score
- **`post_process_ensemble_results_df(df)`** - Add confusion matrix, precision, recall, F1 to results
- **`retrieve_proba(r)`** - Extract probabilities from measurement counts
- **`predict_cos(M)`** - Predict from cosine similarity measurements
- **`cosine_classifier(x, y)`** - Classical cosine similarity classifier

### Utilities

- **`create_dir(path)`** - Create directory if it doesn't exist
- **`save_dict(d, name='dict')`** - Save dictionary to CSV
- **`add_label(d, label='0')`** - Add label to dictionary if missing
- **`pdf(url)`** - Display PDF in Jupyter notebook

## Parameter Glossary

### Common Parameters

- **`d`** - Number of control qubits (ensemble depth), creates 2^d ensemble members
- **`n_swap`** - Number of swap operations per control qubit
- **`n_train`** - Number of training samples (must be power of 2 for quantum encoding)
- **`n_shots`** - Number of measurement shots for probability estimation
- **`mode`** - Ensemble sampling strategy:
  - `"balanced"` - Class-balanced sampling
  - `"unbalanced"` - Random sampling
  - `"pair_sample"` - All pairwise swaps
- **`seed`** - Random seed for reproducibility
- **`device`** - Execution device: 'CPU', 'GPU', or IBM device name
- **`barriers`** - Add barrier gates for visualization
- **`pca_embed`** - Use PCA for dimensionality reduction
- **`umap_embed`** - Use UMAP for dimensionality reduction
- **`select_features`** - List of specific features to use

### Return Types

- Most functions return **DataFrames** with columns: accuracy, brier_score, predictions, y_test
- Quantum circuit functions return **QuantumCircuit** objects
- Execution functions return **dict** of measurement counts

## Usage Tips

1. **Start with small datasets** - Use n_train=4, d=2 for initial testing
2. **Monitor qubit count** - Total qubits = d + 2*n_train + 1 + log2(n_features)*n_train
3. **Tune parameters** - Optimal d and n_swap depend on dataset characteristics
4. **Use embeddings** - PCA/UMAP can reduce dimensionality for complex datasets
5. **Save results** - All run functions automatically save to pickle files
6. **Compare methods** - Use plot_figures() to visualize performance across methods

## See Also

- **README.md** - Comprehensive tutorial guide
- **QEnsemble_example_blobs.ipynb** - Complete working example
- Module docstrings in each .py file for detailed parameter descriptions
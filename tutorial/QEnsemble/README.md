# Quantum Ensemble Learning Tutorial

This tutorial demonstrates quantum ensemble learning methods for classification tasks using Qiskit.

## Overview

Quantum ensemble learning combines multiple quantum classifiers to improve prediction accuracy and robustness. This implementation uses controlled swap operations and quantum superposition to create ensembles of training data arrangements. The tutorial includes both fixed swap-based ensembles and random unitary-based ensembles for more general transformations.

**Key Innovation**: By leveraging quantum superposition, the ensemble evaluates multiple training set configurations simultaneously, potentially offering advantages over classical ensemble methods.

## Files

### Core Modules

- **`modeling.py`**: Main quantum ensemble implementation with fixed swap operations
  - `cos_classifier__legacy()`: Legacy single-qubit cosine classifier using SWAP test
  - `cos_classifier()`: Multi-qubit quantum cosine similarity classifier
  - `state_prep()`: Extract unitary matrix for quantum state preparation
  - `quantum_cosine_classifier()`: Cosine classifier with explicit unitary gates
  - `ensemble()`: Main quantum ensemble with three modes:
    - **balanced**: Class-balanced sampling for balanced datasets
    - **unbalanced**: Random sampling for imbalanced datasets
    - **pair_sample**: Comprehensive pairwise swaps
  - `exec_simulator()`: Execute circuits on Aer simulator (CPU/GPU support)

- **`modeling_random_unitary.py`**: Quantum ensemble using random unitary transformations
  - Implements Haar-random unitaries sampled from the unitary group
  - More general transformation approach than fixed swaps
  - `run_ensemble()`: Complete workflow with training set selection and evaluation
  - `ensemble()`: Random unitary-based ensemble circuit construction
  - `state_prep()`: Unitary extraction for state preparation
  - `label_to_array()`: Convert binary labels to one-hot encoding
  - `evaluation_metrics()`: Calculate accuracy and Brier score

- **`Utils.py`**: Comprehensive utility functions for workflows and analysis
  - **Data Processing**: `normalize_custom()`, `training_set()`, `label_to_array()`
  - **Quantum Workflows**: `run_quantum_ensemble()`, `run_quantum_cosine()`, `run_ensemble()`
  - **Classical Baselines**: `run_random_forest()`, `run_xgboost()`, `run_lazy_predict()`
  - **Evaluation**: `evaluation_metrics()`, `calculate_f1()`, `retrieve_proba()`
  - **Visualization**: `plot_figures()`, `post_process_results()`
  - **IBM Quantum Support**: Hardware execution with error mitigation and dynamic decoupling

### Notebooks

- **`QEnsemble_example_blobs.ipynb`**: Complete tutorial demonstrating quantum ensemble learning on synthetic blob datasets

### Documentation

- **`README.md`**: This file - comprehensive tutorial guide
- **`FUNCTIONS.md`**: Quick reference for all functions and parameters

## Key Concepts

### Quantum Cosine Classifier

The quantum cosine classifier measures similarity between quantum states using the controlled-SWAP test (also known as the SWAP test):

**Algorithm Steps**:
1. **State Preparation**: Encode training data, test data, and labels as quantum states
2. **Superposition**: Apply Hadamard gate to test label qubit
3. **Controlled-SWAP**: Swap training and test data qubits controlled by test label
4. **Interference**: Apply Hadamard gate to test label qubit
5. **Label Integration**: CNOT from training label to test label
6. **Measurement**: Measure test label qubit

**Mathematical Foundation**: The measurement probability P(0) = 1/2 + 1/2 * |⟨train|test⟩|² encodes the squared cosine similarity between quantum states, providing a natural similarity metric for classification.

### Quantum Ensemble Methods

#### Balanced Mode
- Samples training data pairs from each class separately
- Maintains class balance in superposition states
- Best for balanced datasets

#### Unbalanced Mode
- Randomly samples training data pairs without class constraints
- Simpler circuit structure
- Suitable for imbalanced datasets

#### Pair Sample Mode
- Creates all possible pairwise swaps
- More comprehensive exploration of data arrangements
- Higher circuit depth

#### Random Unitary Mode
- Uses random unitaries sampled from U(N) instead of fixed swaps
- Applies controlled random unitaries to combined data+label registers
- More general transformation providing uniform coverage of unitary group
- Potentially better generalization but computationally more expensive
- Unitary dimension: 2^(n_obs_qubits + n_obs)

## Usage Example

```python
from modeling import ensemble, exec_simulator
from Utils import training_set, normalize_custom, evaluation_metrics
import numpy as np

# Prepare data
X_train, y_train = ...  # Your training data
X_test, y_test = ...    # Your test data

# Select training subset
X_data, Y_data = training_set(X_train, y_train, n=4, seed=42)

# Run ensemble for each test point
predictions = []
for x_test in X_test:
    x_test_norm = normalize_custom(x_test)
    
    # Create quantum circuit
    qc = ensemble(X_data, Y_data, x_test_norm, 
                  n_swap=1, d=2, mode="balanced")
    
    # Execute on simulator
    result = exec_simulator(qc, n_shots=8192)
    
    # Extract prediction
    p0 = result['0'] / (result['0'] + result['1'])
    p1 = 1 - p0
    predictions.append([p0, p1])

# Evaluate
accuracy, brier = evaluation_metrics(predictions, y_test)
print(f"Accuracy: {accuracy:.3f}, Brier Score: {brier:.3f}")
```

## Parameters

### Ensemble Parameters

- **`d`**: Number of control qubits (ensemble depth)
  - Creates 2^d ensemble members in superposition
  - Higher values = deeper ensembles with more diversity
  - Typical range: 1-3 (d=3 creates 8 ensemble members)
  - Constraint: n_train > d (need more samples than control qubits)

- **`n_swap`**: Number of swap/unitary operations per control qubit
  - Controls diversity of ensemble members
  - Typical range: 1-5
  - More operations = more diverse data rearrangements
  - In random unitary mode: number of Haar-random unitaries applied

- **`n_train`**: Number of training samples
  - Must be even for balanced mode (n/2 per class)
  - Typical values: 4, 8, 16
  - Limited by qubit count: log2(n_features) * n_train qubits needed
  - Constraint: n_train > d

- **`mode`**: Ensemble sampling strategy
  - `"balanced"`: Class-balanced sampling (recommended for balanced datasets)
  - `"unbalanced"`: Random sampling (suitable for imbalanced data)
  - `"pair_sample"`: All pairwise swaps (most comprehensive, highest depth)

- **`n_shots`**: Number of measurement shots
  - Higher = more accurate probability estimates
  - Typical range: 1024-8192
  - Trade-off: accuracy vs. execution time
  - IBM hardware: typically 4096 due to cost constraints

- **`device`**: Execution device
  - `'CPU'`: Local CPU simulation (default)
  - `'GPU'`: GPU-accelerated simulation (requires qiskit-aer-gpu)
  - `'ibm_*'`: IBM Quantum hardware (e.g., 'ibm_kyoto')

## Performance Considerations


### Optimization Tips

- Start with small configurations: d=2, n_train=4, n_features=2
- Use PCA/UMAP for dimensionality reduction on high-dimensional data
- For hardware: enable error mitigation and dynamic decoupling
- Monitor qubit count before execution to avoid memory issues

## References

### Primary References
- **This Tutorial**: Part of QBioCode framework for quantum bioinformatics
- **Quantum Ensemble Paper**: Macaluso et al., "A variational algorithm for quantum ensemble learning", IET Quantum Communication (2023)
- **Original Implementation**: [GitHub Repository](https://github.com/amacaluso/Quantum-algorithm-for-ensemble-using-bagging)
- **QBioCode Documentation**: See main repository for comprehensive guides

# Quantum Ensemble Learning Tutorial

This tutorial demonstrates quantum ensemble learning methods for classification tasks using Qiskit.

## Overview

Quantum ensemble learning combines multiple quantum classifiers to improve prediction accuracy and robustness. This implementation uses controlled swap operations and quantum superposition to create ensembles of training data arrangements.

## Files

### Core Modules

- **`modeling.py`**: Main quantum ensemble implementation with various ensemble strategies
  - `cos_classifier()`: Quantum cosine similarity classifier
  - `ensemble()`: Main quantum ensemble function with multiple modes (balanced, unbalanced, pair_sample)
  - `ensemble_fixed_U()`: Ensemble with fixed unitary operations
  - `ensemble_random_swap()`: Ensemble using ancilla qubits for complex swap patterns
  - `exec_simulator()`: Execute circuits on Aer simulator

- **`modeling_random_unitary.py`**: Quantum ensemble using random unitary transformations
  - Implements Haar-random unitaries for more general ensemble construction
  - `run_ensemble()`: Complete ensemble workflow with evaluation
  - `ensemble()`: Random unitary-based ensemble circuit construction

- **`Utils.py`**: Utility functions for data processing, evaluation, and visualization
  - Data normalization and preprocessing
  - Training set selection
  - Evaluation metrics (accuracy, Brier score, F1)
  - Visualization functions
  - Classical baseline methods (Random Forest, XGBoost)

### Notebooks

- **`QEnsemble_example_blobs.ipynb`**: Complete tutorial demonstrating quantum ensemble learning on blob datasets

## Key Concepts

### Quantum Cosine Classifier

The quantum cosine classifier measures similarity between quantum states using:
1. State preparation of training and test data
2. Controlled-SWAP test for similarity measurement
3. Hadamard gates for interference
4. Measurement to extract classification probability

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
- Uses Haar-random unitaries instead of fixed swaps
- More general transformation of training data
- Potentially better generalization

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
  - Higher values create deeper ensembles
  - Typical range: 1-3
  - More qubits = exponentially more ensemble members

- **`n_swap`**: Number of swap operations per control qubit
  - Controls diversity of ensemble members
  - Typical range: 1-5
  - More swaps = more data rearrangements

- **`n_train`**: Number of training samples
  - Must be power of 2 for quantum encoding
  - Typical values: 4, 8, 16
  - Limited by qubit count

- **`mode`**: Ensemble sampling strategy
  - `"balanced"`: Class-balanced sampling
  - `"unbalanced"`: Random sampling
  - `"pair_sample"`: All pairwise swaps

- **`n_shots`**: Number of measurement shots
  - Higher = more accurate probability estimates
  - Typical range: 1024-8192
  - Trade-off with execution time

## Requirements

- qiskit >= 0.40.0
- qiskit-aer >= 0.11.0
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Performance Considerations

1. **Qubit Count**: Total qubits = d + 2*n_train + 1 + log2(n_features)*n_train
2. **Circuit Depth**: Increases with d and n_swap
3. **Simulation Time**: Exponential in qubit count
4. **Practical Limits**: ~30-36 qubits for classical simulation

## References

For more information on quantum machine learning and ensemble methods, see the QBioCode documentation.

## Notes

- All data must be normalized before encoding
- Feature count must be a power of 2
- Training set size must be even for balanced mode
- Results are probabilistic - multiple runs recommended
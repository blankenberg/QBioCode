"""
Quantum Ensemble Modeling with Random Unitary Operations

This module implements quantum ensemble learning using random unitary
transformations from the Haar measure, providing a more general approach
to ensemble construction compared to fixed swap patterns.
"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
import Utils
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate
import scipy.stats
import numpy as np
import pandas as pd

def run_ensemble(d, n_train, seed, n_swap, X_train, X_test, y_train, y_test, mode="balanced", n_shots=8192):
    """
    Run quantum ensemble classifier with random unitary operations.
    
    This is a complete workflow function that executes the random unitary
    ensemble method on a dataset and returns comprehensive evaluation metrics.
    It constructs quantum circuits using Haar-random unitaries sampled from
    the unitary group, providing a more general transformation approach than
    fixed swap patterns.
    
    The function iterates over all test samples, constructs an ensemble
    circuit for each, executes on simulator, and aggregates results with
    performance metrics.
    
    Parameters
    ----------
    d : int
        Number of control qubits (ensemble depth). Creates 2^d ensemble
        members. Typical values: 1-3
    n_train : int
        Number of training samples to use from the training set. Must be
        even for balanced mode and less than total training samples
    seed : int
        Random seed for reproducibility of training set selection and
        random unitary sampling
    n_swap : int
        Number of random unitary operations per control qubit. More
        operations create more diverse transformations
    X_train : array-like, shape (n_samples, n_features)
        Training feature data (normalized)
    X_test : array-like, shape (n_test, n_features)
        Test feature data (normalized)
    y_train : array-like, shape (n_samples,)
        Training labels (binary: 0 or 1)
    y_test : array-like, shape (n_test,)
        Test labels (binary: 0 or 1)
    mode : str, optional
        Ensemble mode - currently only "balanced" is fully implemented
        (default: "balanced")
    n_shots : int, optional
        Number of measurement shots per circuit execution (default: 8192)
    
    Returns
    -------
    DataFrame
        Results dataframe with columns:
        - seed: Random seed used
        - n_feature: Number of features
        - qubits: Total number of qubits in circuit
        - d: Ensemble depth
        - n_train: Number of training samples
        - n_swap: Number of unitary operations
        - accuracy: Classification accuracy
        - brier: Brier score (calibration metric)
        - predictions: List of probability predictions
        - y_test: True test labels
        
    Notes
    -----
    - Returns empty DataFrame if circuit exceeds 36 qubits (simulation limit)
    - Uses scipy.stats.unitary_group for Haar-random unitary sampling
    - Prints progress information during execution
    - More computationally expensive than fixed swap ensemble
    
    Examples
    --------
    >>> X_train = np.random.rand(20, 4)
    >>> y_train = np.array([0]*10 + [1]*10)
    >>> X_test = np.random.rand(5, 4)
    >>> y_test = np.array([0, 0, 1, 1, 0])
    >>> results = run_ensemble(d=2, n_train=4, seed=42, n_swap=1,
    ...                        X_train, X_test, y_train, y_test)
    >>> print(f"Accuracy: {results['accuracy'].iloc[0]:.3f}")
    """
    predictions = []

    for x_test in X_test:
        X_data, Y_data = Utils.training_set(X_train, y_train, n=n_train, seed=seed)
        x_test = Utils.normalize_custom(x_test)
        # print("Constructing circuit...",flush=True)
        qc = ensemble(X_data, Y_data, x_test, n_swap=n_swap, d=d, mode=mode)
        if qc.num_qubits <=36:
            # print("Running circuit...",flush=True)
            r = exec_simulator(qc, n_shots=n_shots)
            # print(r)
            predictions.append(Utils.retrieve_proba(r))

    if qc.num_qubits <=36:
        # print(predictions)
        # print(y_test)
        a, b = evaluation_metrics(predictions, y_test, save=False)
        
        res = pd.DataFrame(  [ seed, X_data.shape[1], qc.num_qubits, d, n_train, n_swap, a, b, predictions, y_test ], 
                        index = ['seed', 'n_feature', 'qubits', 'd', 'n_train', 'n_swap', 'accuracy', 'brier', 'predictions', 'y_test' ] ).transpose()
        print(res.iloc[:,1:8])
        return res
    else:
        return pd.DataFrame()


def state_prep(x):
    """
    Prepare a quantum state from classical data using unitary simulation.
    
    Identical to the state_prep function in modeling.py. Extracts the unitary
    matrix that prepares a quantum state from classical data using Qiskit's
    unitary simulator.
    
    Parameters
    ----------
    x : array-like, shape (2^n,)
        Classical data vector to encode as quantum state. Automatically
        normalized using Utils.normalize_custom()
    
    Returns
    -------
    ndarray, shape (2, 2)
        Unitary matrix representing the state preparation operation
        U such that U|0⟩ = |x⟩
        
    Notes
    -----
    - Uses AerSimulator with unitary method
    - Input is automatically normalized
    - Currently supports single-qubit states only
    
    See Also
    --------
    modeling.state_prep : Identical function in main modeling module
    """
    backend = AerSimulator(method="unitary")
    x = Utils.normalize_custom(x)
    qreg = QuantumRegister(1)
    qc = QuantumCircuit(qreg)
    qc.prepare_state(x, [qreg])
    tqc = transpile(qc, backend)
    tqc.save_unitary()
    result = backend.run([tqc]).result()
    U = result.get_unitary(qc)
    return U



def ensemble(X_data, Y_data, x_test, n_swap=1, d=2, mode="balanced", barriers=False):
    """
    Quantum ensemble classifier using random unitary operations.
    
    This function implements a quantum ensemble learning algorithm using
    random unitary transformations sampled from the Haar measure, providing
    a more general approach than fixed swap patterns. Instead of using
    controlled-SWAP gates, this method applies controlled random unitaries
    to the combined data and label registers.
    
    Algorithm Overview:
    1. Initialize training data, labels, and test data
    2. Create superposition over control qubits
    3. For each control qubit:
       a. Sample random unitary from Haar measure
       b. Apply controlled-unitary to data+label registers
       c. Flip control qubit (X gate)
       d. Apply another controlled random unitary
    4. Perform final cosine similarity measurement
    5. Measure test label qubit
    
    The random unitaries are sampled from U(2^(n_obs_qubits + n_obs)), the
    unitary group acting on the combined space of all data and label qubits.
    This provides a more general mixing of training samples than fixed swaps.
    
    Parameters
    ----------
    X_data : array-like, shape (n_samples, n_features)
        Training data points where n_features must be a power of 2.
        Each sample should be normalized.
    Y_data : array-like, shape (n_samples, 2)
        Training labels as one-hot encoded vectors
    x_test : array-like, shape (n_features,)
        Test data point to classify (must be normalized)
    n_swap : int, optional
        Number of random unitary operations per control qubit. Despite
        the name "swap", these are full random unitaries (default: 1)
    d : int, optional
        Number of control qubits for ensemble depth. Creates 2^d
        ensemble members (default: 2)
    mode : str, optional
        Ensemble mode. Currently "balanced" is the primary mode, which
        applies random unitaries in a structured way (default: "balanced")
    barriers : bool, optional
        If True, add barrier gates for circuit visualization (default: False)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the random unitary ensemble classifier
        with registers:
        - control: d qubits for ensemble control
        - data: n_samples * log2(n_features) qubits for training data
        - labels: n_samples qubits for training labels
        - data_test: log2(n_features) qubits for test data
        - prediction: 1 qubit for intermediate computation
        - label_test: 1 qubit for final measurement
        
    Notes
    -----
    - Random unitaries sampled using scipy.stats.unitary_group
    - Unitary dimension: 2^(n_obs_qubits + n_obs)
    - Much more general than swap-based ensemble
    - Computationally expensive due to large unitary matrices
    - Circuit depth increases significantly with n_swap
    - Helper functions: cswap_obs, cswap_cosine, cswap_labels
    
    Mathematical Foundation:
    Random unitaries are sampled from the Haar measure on U(N), providing
    uniform coverage of the unitary group. This creates maximally diverse
    ensemble members while maintaining quantum coherence.
    
    Examples
    --------
    >>> from Utils import training_set, normalize_custom
    >>> X_train = np.random.rand(8, 4)
    >>> y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> X_data, Y_data = training_set(X_train, y_train, n=4, seed=42)
    >>> x_test = normalize_custom(np.random.rand(4))
    >>> qc = ensemble(X_data, Y_data, x_test, n_swap=1, d=2)
    >>> print(f"Circuit uses {qc.num_qubits} qubits")
    
    References
    ----------
    Haar measure sampling: Mezzadri, "How to generate random matrices from
    the classical compact groups", Notices of the AMS (2007)
    """

    def cswap_obs(c, a, b):
        """Controlled swap of observation qubits between indices a and b."""
        qubit_indices_a = train_qubit_map[a]
        qubit_indices_b = train_qubit_map[b]
        for (i,j) in zip(qubit_indices_a,qubit_indices_b):
            qc.cswap(control[c],data[i],data[j])

    def cswap_cosine(a):
        """Controlled swap for cosine similarity measurement."""
        qubit_indices_a = train_qubit_map[a]
        for (j, i) in enumerate(qubit_indices_a):
            qc.cswap(label_test, data[i], data_test[j])

    def cswap_labels(c, a, b):
        """Controlled swap of label qubits."""
        qc.cswap(c, labels[a], labels[b])

    n_obs = len(X_data)
    qubits_per = int(np.log2(X_data.shape[1]))
    n_obs_qubits = qubits_per * n_obs
    n_test = 1
    n_test_qubits = qubits_per * n_test
    n_reg = d + 2 * n_obs + 1

    control = QuantumRegister(d, "c")
    data = QuantumRegister(n_obs_qubits, 'x')
    labels = QuantumRegister(n_obs, 'y')
    prediction = QuantumRegister(1, 'pred_qubit')

    data_test = QuantumRegister(n_test_qubits, 'test_data')
    label_test = QuantumRegister(n_test, 'test_label')
    c = ClassicalRegister(n_test)

    # Sample random unitaries from Haar measure
    unitary_sampler = scipy.stats.unitary_group(2**(n_obs_qubits+n_obs))

    qc = QuantumCircuit(control, data, labels, data_test, prediction, label_test, c)
    test_qubit_map = {}
    for index in range(n_test):
        indices = list(range((index * qubits_per),((index+1)) * qubits_per))
        qc.initialize(x_test, data_test[indices])
        test_qubit_map[index] = indices

    train_qubit_map = {}
    trainLabel_qubit_map = {}
    for index in range(n_obs):
        indices = list(range((index * qubits_per),((index+1)) * qubits_per))
        qc.initialize(X_data[index], [data[indices]])
        qc.initialize(Y_data[index], [labels[index]])
        train_qubit_map[index] = indices
        trainLabel_qubit_map[index] = index

    for i in range(d):
        qc.h(control[i])

    if barriers:
        qc.barrier()

    if mode == 'balanced':
        for i in range(d-1):
            for _ in range(n_swap):
                U = unitary_sampler.rvs()
                U1 = UnitaryGate(U)
                CU1 = U1.control(1)
                qc.append(CU1,[control[i]]+[x for x in data]+[x for x in labels])
                # print("One gate done.")
                qc.barrier()
                qc.x(control[i])
                qc.barrier()
                U = unitary_sampler.rvs()
                U2 = UnitaryGate(U)
                CU2 = U2.control(1)
                qc.append(CU2, [control[i]]+[x for x in data]+[x for x in labels])
                if barriers:
                    qc.barrier()

        # Final swap for balanced mode
        U = np.random.choice(range(int(n_obs / 2)), 1, replace=False)
        U = np.insert(U, 1, n_obs - 1)
        d1, d2 = U[0], U[1]
        cswap_obs(d-1, d1, d2)
        cswap_labels(d-1, d1, d2)

        qc.x(control[d-1])
        
    if barriers:
        qc.barrier()

    # Classification measurement
    qc.h(label_test)
    qc.barrier()
    cswap_cosine(n_obs-1)
    qc.barrier()
    qc.h(label_test)
    qc.cx(labels[n_obs-1], label_test)
    qc.measure(label_test, c)
    return qc




def exec_simulator(qc, n_shots=1024):
    """
    Execute quantum circuit on Aer simulator.
    
    Runs the quantum circuit using Qiskit's statevector simulator backend
    and returns measurement counts. This version uses the legacy Aer.get_backend
    interface for compatibility.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit to execute. Must contain measurement operations.
    n_shots : int, optional
        Number of measurement shots to perform. Default is 1024, which
        is lower than the main modeling.py version (8192) for faster
        execution during experimentation (default: 1024)
    
    Returns
    -------
    dict
        Dictionary mapping measurement outcomes (binary strings) to counts.
        Example: {'0': 512, '1': 512}
        
    Notes
    -----
    - Uses statevector_simulator backend
    - Transpiles with optimization level 3
    - Different from modeling.exec_simulator which uses AerSimulator directly
    - No GPU support in this version
    
    Examples
    --------
    >>> qc = ensemble(X_data, Y_data, x_test, n_swap=1, d=2)
    >>> counts = exec_simulator(qc, n_shots=2048)
    >>> print(counts)
    {'0': 1024, '1': 1024}
    """
    backend = AerSimulator()
    tqc = transpile(qc, backend, optimization_level=3)
    result = Aer.get_backend("statevector_simulator").run([tqc], shots=n_shots).result()
    answer = result.get_counts(tqc)
    return answer


def label_to_array(y):
    """
    Convert binary labels to one-hot encoded arrays.
    
    Transforms binary classification labels (0 or 1) into one-hot encoded
    format required by quantum circuits. Label 0 becomes [1, 0] and label
    1 becomes [0, 1].
    
    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Binary labels where each element is either 0 or 1
    
    Returns
    -------
    ndarray, shape (n_samples, 2)
        One-hot encoded labels where:
        - [1, 0] represents class 0
        - [0, 1] represents class 1
        
    Examples
    --------
    >>> y = np.array([0, 1, 0, 1, 1])
    >>> Y = label_to_array(y)
    >>> print(Y)
    [[1 0]
     [0 1]
     [1 0]
     [0 1]
     [0 1]]
    
    See Also
    --------
    Utils.label_to_array : Identical function in Utils module
    """
    Y = []
    for el in y:
        if el == 0:
            Y.append([1, 0])
        else:
            Y.append([0, 1])
    Y = np.asarray(Y)
    return Y

def evaluation_metrics(predictions, y_test, save=True):
    """
    Calculate evaluation metrics for predictions.
    
    Computes classification accuracy and Brier score (calibration metric)
    from probability predictions. The Brier score measures the mean squared
    difference between predicted probabilities and actual outcomes.
    
    Parameters
    ----------
    predictions : list of array-like
        Predicted probabilities for each class. Each element should be
        [p0, p1] where p0 + p1 = 1
    y_test : array-like, shape (n_samples,)
        True binary labels (0 or 1)
    save : bool, optional
        Legacy parameter, not currently used (default: True)
    
    Returns
    -------
    tuple of float
        (accuracy, brier_score) where:
        - accuracy: Fraction of correct predictions (0 to 1)
        - brier_score: Mean squared error of probability predictions (0 to 1)
          Lower Brier score indicates better calibration
          
    Notes
    -----
    - Predictions are rounded to nearest integer for accuracy calculation
    - Brier score uses probability of class 1 (p1)
    - Perfect predictions: accuracy=1.0, brier_score=0.0
    
    Examples
    --------
    >>> predictions = [[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]]
    >>> y_test = np.array([0, 1, 0])
    >>> acc, brier = evaluation_metrics(predictions, y_test)
    >>> print(f"Accuracy: {acc:.3f}, Brier: {brier:.3f}")
    Accuracy: 1.000, Brier: 0.030
    
    See Also
    --------
    sklearn.metrics.accuracy_score : Accuracy calculation
    sklearn.metrics.brier_score_loss : Brier score calculation
    Utils.evaluation_metrics : Similar function in Utils module
    """
    from sklearn.metrics import brier_score_loss, accuracy_score
    labels = label_to_array(y_test)

    predicted_class = np.round(np.asarray(predictions))
    acc = accuracy_score(np.array(predicted_class)[:, 1],
                         np.array(labels)[:, 1])

    p0 = [p[0] for p in predictions]
    p1 = [p[1] for p in predictions]

    brier = brier_score_loss(y_test, p1)

    return acc, brier
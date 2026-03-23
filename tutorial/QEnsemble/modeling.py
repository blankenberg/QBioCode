"""
Quantum Ensemble Modeling Module

This module provides quantum circuit implementations for ensemble learning,
including cosine classifiers and quantum ensemble methods using controlled
swap operations.
"""
import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
import Utils


def cos_classifier_OLD(train, test, label_train, printing=False):
    """
    Legacy cosine classifier for single qubit data.
    
    Parameters
    ----------
    train : array-like
        Training data point (normalized vector)
    test : array-like
        Test data point (normalized vector)
    label_train : array-like
        Training label (normalized vector)
    printing : bool, optional
        If True, print the quantum circuit (default: False)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the cosine classifier
    """
    c = ClassicalRegister(1, 'c')
    x_train = QuantumRegister(1, 'x^{(i)}')
    x_test = QuantumRegister(1, 'x^{(test)}')
    y_train = QuantumRegister(1, 'y^{(i)}')
    y_test = QuantumRegister(1, 'y^{(test)}')
    qc = QuantumCircuit(x_train, x_test, y_train, y_test, c)
    qc.initialize(train, [x_train[0]])
    qc.initialize(test, [x_test[0]])
    qc.initialize(label_train, [y_train[0]])
    qc.barrier()
    qc.h(y_test)
    qc.cswap(y_test, x_train, x_test)
    qc.h(y_test)
    qc.barrier()
    qc.cx(y_train, y_test)
    qc.measure(y_test, c)
    if printing:
        print(qc)
    return qc 

def cos_classifier(train, test, label_train, printing=False):
    """
    Cosine classifier for multi-qubit data.
    
    Implements a quantum cosine similarity classifier using controlled swap
    operations and Hadamard gates to measure similarity between training
    and test data points.
    
    Parameters
    ----------
    train : array-like
        Training data point (normalized vector)
    test : array-like
        Test data point (normalized vector)
    label_train : array-like
        Training label (normalized vector)
    printing : bool, optional
        If True, print the quantum circuit (default: False)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the cosine classifier
    """
    n_obs = len(train)
    qubits_per = int(np.log2(len(train)))
    n_obs_qubits = qubits_per * n_obs
    n_test = len(test)
    n_test_qubits = qubits_per * n_test

    c = ClassicalRegister(1, 'c')
    x_train = QuantumRegister(qubits_per, 'x_{b}')
    x_test = QuantumRegister(qubits_per, 'x^{(test)}')
    y_train = QuantumRegister(1, 'y_{b}')
    y_test = QuantumRegister(1, 'y^{(test)}')
    qc = QuantumCircuit(x_train, x_test, y_train, y_test, c)
    
    qc.initialize(train, [x_train])
    qc.initialize(test, [x_test])
    qc.initialize(label_train, [y_train])
    qc.barrier()
    qc.h(y_test)
    qc.cswap(y_test, x_train, x_test)
    qc.h(y_test)
    qc.barrier()
    qc.cx(y_train, y_test)
    qc.measure(y_test, c)
    if printing:
        print(qc)
    return qc


def state_prep(x):
    """
    Prepare a quantum state from classical data using unitary simulation.
    
    Parameters
    ----------
    x : array-like
        Classical data vector to encode as quantum state
    
    Returns
    -------
    ndarray
        Unitary matrix representing the state preparation operation
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


def quantum_cosine_classifier(train, test, label_train):
    """
    Quantum cosine classifier using unitary state preparation.
    
    This function creates a quantum circuit that implements cosine similarity
    classification using unitary gates for state preparation instead of
    direct initialization.
    
    Parameters
    ----------
    train : array-like
        Training data points (2D array)
    test : array-like
        Test data points (2D array)
    label_train : array-like
        Training labels
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the classifier with unitary gates
    """
    n_obs = len(train)
    qubits_per = int(np.log2(train.shape[1]))
    n_obs_qubits = qubits_per * n_obs
    n_test = len(test)
    n_test_qubits = qubits_per * n_test

    c = ClassicalRegister(n_test, 'c')
    x_train = QuantumRegister(n_obs_qubits, 'x_{b}')
    x_test = QuantumRegister(n_test_qubits, 'x^{(test)}')
    y_train = QuantumRegister(n_obs, 'y_{b}')
    y_test = QuantumRegister(n_test, 'y^{(test)}')
    qc = QuantumCircuit(x_train, x_test, y_train, y_test, c)
    
    S1 = state_prep(train)
    qc.unitary(S1, [0], label='$S_{x}$')

    S2 = state_prep(test)
    qc.unitary(S2, [1], label='$S_{x}$')
    
    S3 = state_prep(label_train)
    qc.unitary(S3, [2], label='$S_{y}$')

    qc.barrier()
    qc.h(y_test)
    qc.cswap(y_test, x_train, x_test)
    qc.h(y_test)
    qc.barrier()
    qc.cx(y_train, y_test)
    qc.measure(y_test, c)
    return qc
 

def ensemble(X_data, Y_data, x_test, n_swap=1, d=2, mode="balanced", barriers=False):
    """
    Quantum ensemble classifier for multi-qubit data.
    
    This function implements a quantum ensemble learning algorithm using
    controlled swap operations to create superpositions of different
    training data arrangements.
    
    Parameters
    ----------
    X_data : array-like, shape (n_samples, n_features)
        Training data points (must be power of 2 features)
    Y_data : array-like, shape (n_samples,)
        Training labels
    x_test : array-like
        Test data point
    n_swap : int, optional
        Number of swap operations per control qubit (default: 1)
    d : int, optional
        Number of control qubits for ensemble depth (default: 2)
    mode : str, optional
        Sampling mode: "balanced", "unbalanced", or "pair_sample" (default: "balanced")
    barriers : bool, optional
        If True, add barrier gates for visualization (default: False)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the ensemble classifier
    """
    n_obs = len(X_data)
    qubits_per = int(np.log2(X_data.shape[1]))
    n_obs_qubits = qubits_per * n_obs
    n_test = 1
    n_test_qubits = qubits_per * n_test
    n_reg = d + 2 * n_obs + 1

    control = QuantumRegister(d)
    data = QuantumRegister(n_obs_qubits, 'x')
    labels = QuantumRegister(n_obs, 'y')
    data_test = QuantumRegister(n_test_qubits, 'test_data')
    label_test = QuantumRegister(n_test, 'test_label')
    c = ClassicalRegister(n_test)


    qc = QuantumCircuit(control, data, labels, data_test, label_test, c)
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
            for j in range(n_swap):
                U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                U_b = np.random.choice( range(len(train_qubit_map[U[0]])), 1 )[0]
                d1 = train_qubit_map[U[0]][U_b]
                d2 = train_qubit_map[U[1]][U_b]
                qc.cswap(control[i], data[d1], data[d2])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                U_b = np.random.choice( range(len(train_qubit_map[U[0]])), 1 )[0]
                d1 = train_qubit_map[U[0]][U_b]
                d2 = train_qubit_map[U[1]][U_b]
                qc.cswap(control[i], data[d1], data[d2])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

            qc.x(control[i])

            for j in range(n_swap):
                U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                U_b = np.random.choice( range(len(train_qubit_map[U[0]])), 1 )[0]
                d1 = train_qubit_map[U[0]][U_b]
                d2 = train_qubit_map[U[1]][U_b]
                qc.cswap(control[i], data[d1], data[d2])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                U_b = np.random.choice( range(len(train_qubit_map[U[0]])), 1 )[0]
                d1 = train_qubit_map[U[0]][U_b]
                d2 = train_qubit_map[U[1]][U_b]
                qc.cswap(control[i], data[d1], data[d2])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                if barriers:
                    qc.barrier()
        
        qc.x(control[d-1])
        
        U = np.random.choice(range(int(n_obs / 2)), 1, replace=False)
        U = np.insert(U, 1, n_obs - 1)
        U_b = np.random.choice(range(len(train_qubit_map[U[0]])), 1)[0]
        d1 = train_qubit_map[U[0]][U_b]
        d2 = train_qubit_map[U[1]][U_b]
        qc.cswap(control[d-1], data[d1], data[d2])
        qc.cswap(control[d-1], labels[int(U[0])], labels[int(U[1])])

    elif mode == "unbalanced":
        for i in range(d):
            for j in range(n_swap):
                U = np.random.choice(range(n_obs), 2, replace=False)
                U_b = np.random.choice( range(len(train_qubit_map[U[0]])), 1 )[0]
                d1 = train_qubit_map[U[0]][U_b]
                d2 = train_qubit_map[U[1]][U_b]
                qc.cswap(control[i], data[d1], data[d2])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

            qc.x(control[i])

            for j in range(n_swap):
                U = np.random.choice(range(n_obs), 2, replace=False)
                U_b = np.random.choice( range(len(train_qubit_map[U[0]])), 1 )[0]
                d1 = train_qubit_map[U[0]][U_b]
                d2 = train_qubit_map[U[1]][U_b]
                qc.cswap(control[i], data[d1], data[d2])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])
    elif mode == "pair_sample":
        for i in range(d):
            for j in range(n_swap):
                pairs = np.random.choice(range(n_obs), n_obs, replace=False)
                for U in pairs.reshape(int(len(pairs)/2),2):
                    U_b = np.random.choice( range(len(train_qubit_map[U[0]])), 1 )[0]
                    d1 = train_qubit_map[U[0]][U_b]
                    d2 = train_qubit_map[U[1]][U_b]
                    qc.cswap(control[i], data[d1], data[d2])
                    qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

            qc.x(control[i])
            if barriers:
                qc.barrier()  

            for j in range(n_swap):
                pairs = np.random.choice(range(n_obs), n_obs, replace=False)
                for U in pairs.reshape(int(len(pairs)/2),2):
                    U_b = np.random.choice( range(len(train_qubit_map[U[0]])), 1 )[0]
                    d1 = train_qubit_map[U[0]][U_b]
                    d2 = train_qubit_map[U[1]][U_b]
                    qc.cswap(control[i], data[d1], data[d2])
                    qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])        
            
            if barriers:
                qc.barrier()

    # Final classification step
    ix_cls = n_obs - 1
    d1 = train_qubit_map[ix_cls][0]

    qc.h(label_test[0])
    qc.cswap(label_test[0], data[d1], data_test[0])
    qc.h(label_test[0])
    qc.cx(labels[ix_cls], label_test[0])
    qc.measure(label_test[0], c)
    return qc


def exec_simulator(qc, n_shots=8192, device='CPU'):
    """
    Execute quantum circuit on Aer simulator.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit to execute
    n_shots : int, optional
        Number of measurement shots (default: 8192)
    device : str, optional
        Device type for simulation: 'CPU' or 'GPU' (default: 'CPU')
    
    Returns
    -------
    dict
        Dictionary of measurement counts
    """
    backend = AerSimulator(method='statevector', device=device, statevector_parallel_threshold=50)
    tqc = transpile(qc, backend, optimization_level=3)
    result = backend.run([tqc]).result()
    answer = result.get_counts(tqc)
    return answer

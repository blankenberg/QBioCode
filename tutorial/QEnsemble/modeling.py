"""
Quantum Ensemble Modeling Module

This module provides quantum circuit implementations for ensemble learning,
including cosine classifiers and quantum ensemble methods using controlled
swap operations.
"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.compiler import transpile
from qiskit.visualization import *
from qiskit_aer import AerSimulator
from Utils import *
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, InstructionProperties
from qiskit.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, XGate, YGate


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
    x = normalize_custom(x)
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
 


def ensemble_OLD(X_data, Y_data, x_test, n_swap=1, d=2, balanced=True):
    """
    Legacy quantum ensemble classifier for single-qubit data.
    
    Parameters
    ----------
    X_data : array-like
        Training data points
    Y_data : array-like
        Training labels
    x_test : array-like
        Test data point
    n_swap : int, optional
        Number of swap operations per control qubit (default: 1)
    d : int, optional
        Number of control qubits (default: 2)
    balanced : bool, optional
        If True, use balanced sampling strategy (default: True)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the ensemble classifier
    """
    n_obs = len(X_data)
    n_reg = d + 2 * n_obs + 1

    control = QuantumRegister(d)
    data = QuantumRegister(n_obs, 'x')
    labels = QuantumRegister(n_obs, 'y')
    data_test = QuantumRegister(1, 'test_data')
    label_test = QuantumRegister(1, 'test_label')
    c = ClassicalRegister(1)

    qc = QuantumCircuit(control, data, labels, data_test, label_test, c)

    qc.initialize(x_test, [data_test[0]])

    for index in range(n_obs):
        qc.initialize(X_data[index], [data[index]])
        qc.initialize(Y_data[index], [labels[index]])

    for i in range(d):
        qc.h(control[i])

    if balanced:
        for i in range(d-1):
            for j in range(n_swap):
                U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

            qc.x(control[i])

            for j in range(n_swap):
                U = np.random.choice(range(int(n_obs / 2)), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                U = np.random.choice(range(int(n_obs / 2), n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

                qc.barrier()
        U = np.random.choice(range(int(n_obs / 2)), 1, replace=False)
        U = np.insert(U, 1, n_obs - 1)
        qc.cswap(control[d-1], data[int(U[0])], data[int(U[1])])
        qc.cswap(control[d-1], labels[int(U[0])], labels[int(U[1])])

        qc.x(control[d-1])
    else:
        for i in range(d):
            for j in range(n_swap):
                U = np.random.choice(range(n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

            qc.x(control[i])

            for j in range(n_swap):
                U = np.random.choice(range(n_obs), 2, replace=False)
                qc.cswap(control[i], data[int(U[0])], data[int(U[1])])
                qc.cswap(control[i], labels[int(U[0])], labels[int(U[1])])

    qc.barrier()
    ix_cls = n_obs - 1
    qc.h(label_test[0])
    qc.cswap(label_test[0], data[ix_cls], data_test[0])
    qc.h(label_test[0])
    qc.cx(labels[ix_cls], label_test[0])
    qc.measure(label_test[0], c)
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


def ensemble_fixed_U(X_data, Y_data, x_test, d=2):
    """
    Quantum ensemble with fixed unitary operations.
    
    This variant uses predefined swap patterns instead of random sampling,
    and employs unitary state preparation for data encoding.
    
    Parameters
    ----------
    X_data : array-like
        Training data points
    Y_data : array-like
        Training labels
    x_test : array-like
        Test data point
    d : int, optional
        Number of control qubits (default: 2)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit with fixed unitary ensemble operations
    """
    n_obs = len(X_data)

    if n_obs != len(Y_data):
        return print('Error: in the input size')

    n_reg = d + 2 * n_obs + 1

    control = QuantumRegister(d, 'd')
    data = QuantumRegister(n_obs, 'x')
    labels = QuantumRegister(n_obs, 'y')
    data_test = QuantumRegister(1, 'x^{(test)}')
    label_test = QuantumRegister(1, 'y^{(test)}')
    c = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(control, data, labels, data_test, label_test, c)

    # Prepare training data and labels using unitary gates
    for index in range(n_obs):
        Sx = state_prep(X_data[index])
        qc.unitary(Sx, [2+index], label='$S_{x}$')
        
        Sy = state_prep(Y_data[index])
        qc.unitary(Sy, [6+index], label='$S_{y}$')

    # Initialize control qubits
    for i in range(d):
        qc.h(control[i])

    # Fixed swap patterns
    U1 = [0, 2]
    U2 = [1, 3]
    U4 = [2, 3]

    qc.barrier()

    # Apply controlled swaps with fixed patterns
    qc.cswap(control[0], data[int(U1[0])], data[int(U1[1])])
    qc.cswap(control[0], labels[int(U1[0])], labels[int(U1[1])])

    qc.x(control[0])

    qc.cswap(control[0], data[int(U2[0])], data[int(U2[1])])
    qc.cswap(control[0], labels[int(U2[0])], labels[int(U2[1])])

    qc.barrier()

    qc.x(control[1])

    qc.cswap(control[1], data[int(U4[0])], data[int(U4[1])])
    qc.cswap(control[1], labels[int(U4[0])], labels[int(U4[1])])

    qc.barrier()
    
    # Prepare test data
    Sx = state_prep(x_test)
    qc.unitary(Sx, [10], label='$S_{x}$')

    # Classification step
    ix_cls = 3
    qc.barrier()
    qc.h(label_test[0])
    qc.cswap(label_test[0], data[ix_cls], data_test[0])
    qc.h(label_test[0])
    qc.cx(labels[ix_cls], label_test[0])
    qc.measure(label_test[0], c)
    return qc


def ensemble_random_swap(X_data, Y_data, x_test, d=2):
    """
    Quantum ensemble using ancilla qubits for random swap operations.
    
    This variant uses ancilla qubits to implement more complex swap patterns
    that can create richer superpositions of training data arrangements.
    
    Parameters
    ----------
    X_data : array-like
        Training data points
    Y_data : array-like
        Training labels
    x_test : array-like
        Test data point
    d : int, optional
        Number of control qubits (default: 2)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit with ancilla-based random swap ensemble
    """
    n_obs = len(X_data)
    n_reg = d + 4 * n_obs + 1

    control = QuantumRegister(d, 'control')
    data = QuantumRegister(n_obs, 'x')
    labels = QuantumRegister(n_obs, 'y')
    ancilla_x = QuantumRegister(n_obs, 'ancilla_x')
    ancilla_y = QuantumRegister(n_obs, 'ancilla_y')
    ancilla_test = QuantumRegister(2, 'ancilla_test')
    data_test = QuantumRegister(1, 'x test')
    label_test = QuantumRegister(1, 'y test')
    c = ClassicalRegister(1)

    qc = QuantumCircuit(control, data, labels, data_test, label_test, ancilla_x, ancilla_y, ancilla_test, c)

    # Initialize training data and test data
    for index in range(n_obs):
        qc.initialize(X_data[index], [data[index]])
        qc.initialize(Y_data[index], [labels[index]])

    qc.initialize(x_test, [data_test[0]])

    # Initialize control qubits in superposition
    for i in range(d):
        qc.h(control[i])

    # Random swap patterns
    U1 = np.random.choice(range(n_obs), 2, replace=False)
    U2 = np.random.choice(range(n_obs), 2, replace=False)
    U3 = np.random.choice(range(n_obs), 2, replace=False)
    U4 = np.random.choice(range(n_obs), 2, replace=False)

    qc.barrier()

    # First layer: swap data/labels with ancilla based on control[0]
    for i in range(n_obs):
        qc.cswap(control[0], data[i], ancilla_x[i])
        qc.cswap(control[0], labels[i], ancilla_y[i])
    qc.cswap(control[0], data_test[0], ancilla_test[0])
    qc.cswap(control[0], label_test[0], ancilla_test[1])

    qc.barrier()

    # Apply swaps in main registers
    qc.swap(data[int(U1[0])], data[int(U1[1])])
    qc.swap(labels[int(U1[0])], labels[int(U1[1])])

    # Apply swaps in ancilla registers
    qc.swap(ancilla_x[int(U2[0])], ancilla_x[int(U2[1])])
    qc.swap(ancilla_y[int(U2[0])], ancilla_x[int(U2[1])])

    qc.barrier()

    # Swap back from ancilla
    for i in range(n_obs):
        qc.cswap(control[0], data[i], ancilla_x[i])
        qc.cswap(control[0], labels[i], ancilla_y[i])
    qc.cswap(control[0], data_test[0], ancilla_test[0])
    qc.cswap(control[0], label_test[0], ancilla_test[1])

    qc.barrier()

    # Second layer with control[1]
    for i in range(n_obs):
        qc.cswap(control[1], data[i], ancilla_x[i])
        qc.cswap(control[1], labels[i], ancilla_y[i])
    qc.cswap(control[1], data_test[0], ancilla_test[0])
    qc.cswap(control[1], label_test[0], ancilla_test[1])

    qc.barrier()

    # Apply more swaps
    qc.swap(data[int(U3[0])], data[int(U3[1])])
    qc.swap(labels[int(U3[0])], labels[int(U3[1])])

    qc.swap(ancilla_x[int(U4[0])], ancilla_x[int(U4[1])])
    qc.swap(ancilla_y[int(U4[0])], ancilla_y[int(U4[1])])

    qc.barrier()

    # Swap back from ancilla
    for i in range(n_obs):
        qc.cswap(control[1], data[i], ancilla_x[i])
        qc.cswap(control[1], labels[i], ancilla_y[i])
    qc.cswap(control[1], data_test[0], ancilla_test[0])
    qc.cswap(control[1], label_test[0], ancilla_test[1])

    qc.barrier()

    # Final classification
    ix_cls = 3
    qc.h(label_test[0])
    qc.cswap(label_test[0], data[ix_cls], data_test[0])
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

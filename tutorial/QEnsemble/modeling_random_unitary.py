"""
Quantum Ensemble Modeling with Random Unitary Operations

This module implements quantum ensemble learning using random unitary
transformations from the Haar measure, providing a more general approach
to ensemble construction compared to fixed swap patterns.
"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.compiler import transpile
from qiskit.visualization import *
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from Utils import *
import Utils
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager, InstructionProperties
from qiskit.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
from qiskit_aer import Aer
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RXGate, RYGate, RZGate, UnitaryGate
import scipy.stats


def run_ensemble(d, n_train, seed, n_swap, X_train, X_test, y_train, y_test, mode="balanced", n_shots=8192):
    """
    Run quantum ensemble classifier with random unitary operations.
    
    Parameters
    ----------
    d : int
        Number of control qubits (ensemble depth)
    n_train : int
        Number of training samples to use
    seed : int
        Random seed for reproducibility
    n_swap : int
        Number of swap operations per control qubit
    X_train : array-like
        Training feature data
    X_test : array-like
        Test feature data
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    mode : str, optional
        Ensemble mode (default: "balanced")
    n_shots : int, optional
        Number of measurement shots (default: 8192)
    
    Returns
    -------
    DataFrame
        Results containing accuracy, Brier score, and predictions
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
    Quantum ensemble classifier using random unitary operations.
    
    This function implements a quantum ensemble learning algorithm using
    random unitary transformations sampled from the Haar measure, providing
    a more general approach than fixed swap patterns.
    
    Parameters
    ----------
    X_data : array-like, shape (n_samples, n_features)
        Training data points (must be power of 2 features)
    Y_data : array-like, shape (n_samples,)
        Training labels
    x_test : array-like
        Test data point
    n_swap : int, optional
        Number of unitary operations per control qubit (default: 1)
    d : int, optional
        Number of control qubits for ensemble depth (default: 2)
    mode : str, optional
        Ensemble mode: "balanced" or other (default: "balanced")
    barriers : bool, optional
        If True, add barrier gates for visualization (default: False)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the random unitary ensemble classifier
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


def ensemble_fixed_U(X_data, Y_data, x_test, d=2):
    """
    Quantum ensemble with fixed unitary operations.
    
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

    n_reg = d + 2 * n_obs + 1  # total number of registers

    control = QuantumRegister(d, 'd')
    data = QuantumRegister(n_obs, 'x')
    labels = QuantumRegister(n_obs, 'y')
    data_test = QuantumRegister(1, 'x^{(test)}')
    label_test = QuantumRegister(1, 'y^{(test)}')
    c = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(control, data, labels, data_test, label_test, c)
    
    
    for index in range(n_obs):

        Sx = state_prep(X_data[index])
        qc.unitary(Sx, [2+index], label='$S_{x}$')
        
        Sy = state_prep(Y_data[index])
        qc.unitary(Sy, [6+index], label='$S_{y}$')


    for i in range(d):
        qc.h(control[i])

    U1 = [0, 2]  # np.random.choice(range(4), 2, replace=False)
    U2 = [1, 3]  # np.random.choice(range(4), 2, replace=False)
    U4 = [2,3]  # np.random.choice(range(4), 2, replace=False)

    qc.barrier()

    # U1
    qc.cswap(control[0], data[int(U1[0])], data[int(U1[1])])
    qc.cswap(control[0], labels[int(U1[0])], labels[int(U1[1])])

    qc.x(control[0])

    # U2
    qc.cswap(control[0], data[int(U2[0])], data[int(U2[1])])
    qc.cswap(control[0], labels[int(U2[0])], labels[int(U2[1])])

    qc.barrier()



    qc.x(control[1])

    # U4
    qc.cswap(control[1], data[int(U4[0])], data[int(U4[1])])
    qc.cswap(control[1], labels[int(U4[0])], labels[int(U4[1])])

    qc.barrier()
    Sx = state_prep(x_test)
    qc.unitary(Sx, [10], label='$S_{x}$')

    # qc.barrier()

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

    for index in range(n_obs):
        qc.initialize(X_data[index], [data[index]])
        qc.initialize(Y_data[index], [labels[index]])


    qc.initialize(x_test, [data_test[0]])


    for i in range(d):
        qc.h(control[i])

    U1 = np.random.choice(range(n_obs), 2, replace=False)
    U2 = np.random.choice(range(n_obs), 2, replace=False)
    U3 = np.random.choice(range(n_obs), 2, replace=False)
    U4 = np.random.choice(range(n_obs), 2, replace=False)

    qc.barrier()

    for i in range(n_obs):
        qc.cswap(control[0], data[i], ancilla_x[i])
        qc.cswap(control[0], labels[i], ancilla_y[i])
    qc.cswap(control[0], data_test[0], ancilla_test[0])
    qc.cswap(control[0], label_test[0], ancilla_test[1])

    qc.barrier()

    # U1
    qc.swap(data[int(U1[0])], data[int(U1[1])])
    qc.swap(labels[int(U1[0])], labels[int(U1[1])])


    # U2
    qc.swap(ancilla_x[int(U2[0])], ancilla_x[int(U2[1])])
    qc.swap(ancilla_y[int(U2[0])], ancilla_x[int(U2[1])])

    qc.barrier()

    for i in range(n_obs):
        qc.cswap(control[0], data[i], ancilla_x[i])
        qc.cswap(control[0], labels[i], ancilla_y[i])
    qc.cswap(control[0], data_test[0], ancilla_test[0])
    qc.cswap(control[0], label_test[0], ancilla_test[1])

    qc.barrier()

    for i in range(n_obs):
        qc.cswap(control[1], data[i], ancilla_x[i])
        qc.cswap(control[1], labels[i], ancilla_y[i])
    qc.cswap(control[1], data_test[0], ancilla_test[0])
    qc.cswap(control[1], label_test[0], ancilla_test[1])

    qc.barrier()

    # U3
    qc.swap(data[int(U3[0])], data[int(U3[1])])
    qc.swap(labels[int(U3[0])], labels[int(U3[1])])

    # U4
    qc.swap(ancilla_x[int(U4[0])], ancilla_x[int(U4[1])])
    qc.swap(ancilla_y[int(U4[0])], ancilla_y[int(U4[1])])

    qc.barrier()

    for i in range(n_obs):
        qc.cswap(control[1], data[i], ancilla_x[i])
        qc.cswap(control[1], labels[i], ancilla_y[i])
    qc.cswap(control[1], data_test[0], ancilla_test[0])
    qc.cswap(control[1], label_test[0], ancilla_test[1])

    qc.barrier()

    # C
    ix_cls = 3

    qc.h(label_test[0])
    qc.cswap(label_test[0], data[ix_cls], data_test[0])
    qc.h(label_test[0])
    qc.cx(labels[ix_cls], label_test[0])
    qc.measure(label_test[0], c)
    return qc


def exec_simulator(qc, n_shots=1024):
    """
    Execute quantum circuit on Aer simulator.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit to execute
    n_shots : int, optional
        Number of measurement shots (default: 1024)
    
    Returns
    -------
    dict
        Dictionary of measurement counts
    """
    backend = AerSimulator()
    tqc = transpile(qc, backend, optimization_level=3)
    result = Aer.get_backend("statevector_simulator").run([tqc], shots=n_shots).result()
    answer = result.get_counts(tqc)
    return answer


def label_to_array(y):
    """
    Convert binary labels to one-hot encoded arrays.
    
    Parameters
    ----------
    y : array-like
        Binary labels (0 or 1)
    
    Returns
    -------
    ndarray
        One-hot encoded labels, shape (n_samples, 2)
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
    
    Parameters
    ----------
    predictions : list of array-like
        Predicted probabilities for each class
    y_test : array-like
        True labels
    save : bool, optional
        Legacy parameter (not used, default: True)
    
    Returns
    -------
    tuple
        (accuracy, brier_score) - Classification accuracy and Brier score
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
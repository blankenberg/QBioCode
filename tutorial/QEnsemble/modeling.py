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
    
    This function implements a quantum cosine similarity classifier using the
    controlled-SWAP test (also known as the SWAP test). The circuit measures
    the overlap between quantum states encoding the training and test data,
    which corresponds to their cosine similarity.
    
    Algorithm:
    1. Initialize training data, test data, and training label in separate qubits
    2. Apply Hadamard gate to test label qubit (creates superposition)
    3. Perform controlled-SWAP between training and test data qubits
    4. Apply Hadamard gate to test label qubit (interference step)
    5. Apply CNOT gate controlled by training label
    6. Measure test label qubit
    
    The measurement probability P(0) = 1/2 + 1/2 * |<train|test>|^2, which
    encodes the cosine similarity between the data points.
    
    Parameters
    ----------
    train : array-like, shape (2,)
        Training data point as normalized 2D vector [x, y]
    test : array-like, shape (2,)
        Test data point as normalized 2D vector [x, y]
    label_train : array-like, shape (2,)
        Training label as normalized vector [p0, p1] where p0 + p1 = 1
    printing : bool, optional
        If True, print the quantum circuit diagram (default: False)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit with 4 qubits and 1 classical bit implementing
        the cosine classifier
        
    Notes
    -----
    This is a legacy function for 2D data only. For multi-dimensional data,
    use `cos_classifier()` instead.
    
    Examples
    --------
    >>> train = np.array([1.0, 0.0])
    >>> test = np.array([0.707, 0.707])
    >>> label = np.array([1.0, 0.0])  # Class 0
    >>> qc = cos_classifier_OLD(train, test, label)
    >>> result = exec_simulator(qc, n_shots=1024)
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
    and test data points. This is the generalized version that works with
    data of arbitrary dimensionality (must be power of 2).
    
    Algorithm:
    1. Encode training data, test data, and label into quantum states
    2. Apply Hadamard to test label qubit (superposition)
    3. Controlled-SWAP between all training and test data qubits
    4. Apply Hadamard to test label qubit (interference)
    5. CNOT from training label to test label
    6. Measure test label qubit
    
    The circuit computes P(0) = 1/2 + 1/2 * |<train|test>|^2, encoding
    the squared cosine similarity between the quantum states.
    
    Parameters
    ----------
    train : array-like, shape (2^n,)
        Training data point as normalized vector where length is power of 2
    test : array-like, shape (2^n,)
        Test data point as normalized vector where length is power of 2
    label_train : array-like, shape (2,)
        Training label as normalized probability vector [p0, p1]
    printing : bool, optional
        If True, print the quantum circuit diagram (default: False)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the multi-qubit cosine classifier
        
    Notes
    -----
    - Data dimensionality must be a power of 2 (2, 4, 8, 16, ...)
    - All input vectors must be normalized (L2 norm = 1)
    - Uses log2(len(train)) qubits per data point
    
    Examples
    --------
    >>> train = np.array([0.5, 0.5, 0.5, 0.5])  # 4D normalized data
    >>> test = np.array([0.7, 0.3, 0.5, 0.4])
    >>> test = test / np.linalg.norm(test)  # Normalize
    >>> label = np.array([1.0, 0.0])  # Class 0
    >>> qc = cos_classifier(train, test, label)
    >>> result = exec_simulator(qc, n_shots=8192)
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
    
    This function extracts the unitary matrix that prepares a quantum state
    from classical data. It uses Qiskit's unitary simulator to obtain the
    exact unitary transformation that would initialize the given state.
    
    The unitary can be used in circuits where explicit state initialization
    is not available or when the unitary representation is needed for
    further manipulation.
    
    Parameters
    ----------
    x : array-like, shape (2^n,)
        Classical data vector to encode as quantum state. Will be normalized
        internally if not already normalized.
    
    Returns
    -------
    ndarray, shape (2, 2)
        Unitary matrix (2x2 for single qubit) representing the state
        preparation operation U such that U|0⟩ = |x⟩
        
    Notes
    -----
    - Uses AerSimulator with unitary method
    - Input is automatically normalized using Utils.normalize_custom()
    - Currently supports single-qubit states only
    
    Examples
    --------
    >>> x = np.array([0.6, 0.8])
    >>> U = state_prep(x)
    >>> print(U.shape)
    (2, 2)
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
    direct initialization. This approach is useful when working with hardware
    that doesn't support arbitrary state initialization or when the unitary
    representation is preferred.
    
    The circuit uses the state_prep() function to extract unitaries for
    encoding data and labels, then applies the same SWAP test algorithm
    as the standard cosine classifier.
    
    Parameters
    ----------
    train : array-like, shape (n_samples, n_features)
        Training data points as 2D array where each row is a sample
    test : array-like, shape (n_samples, n_features)
        Test data points as 2D array where each row is a sample
    label_train : array-like, shape (n_samples,)
        Training labels corresponding to training data
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the classifier with explicit unitary
        gates for state preparation
        
    Notes
    -----
    - Uses unitary gates labeled as $S_x$ for data and $S_y$ for labels
    - More hardware-compatible than direct state initialization
    - May have different gate decomposition than initialize() method
    
    Examples
    --------
    >>> train = np.array([[0.7, 0.7], [0.6, 0.8]])
    >>> test = np.array([[0.8, 0.6]])
    >>> labels = np.array([0, 1])
    >>> qc = quantum_cosine_classifier(train, test, labels)
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
    training data arrangements. The ensemble leverages quantum superposition
    to evaluate multiple training set configurations simultaneously.
    
    Algorithm Overview:
    1. Initialize training data, labels, and test data in quantum registers
    2. Create superposition over control qubits (2^d ensemble members)
    3. Apply controlled-SWAP operations to rearrange training data
    4. Perform final cosine similarity measurement
    5. Measure to obtain ensemble prediction
    
    The circuit creates 2^d different arrangements of the training data,
    each weighted equally in superposition. The final measurement collapses
    to a prediction that incorporates information from all arrangements.
    
    Parameters
    ----------
    X_data : array-like, shape (n_samples, n_features)
        Training data points where n_features must be a power of 2
        (e.g., 2, 4, 8, 16). Each sample should be normalized.
    Y_data : array-like, shape (n_samples, 2)
        Training labels as one-hot encoded vectors [[1,0] or [0,1]]
    x_test : array-like, shape (n_features,)
        Test data point to classify (must be normalized)
    n_swap : int, optional
        Number of swap operations per control qubit. More swaps create
        more diverse ensemble members (default: 1)
    d : int, optional
        Number of control qubits determining ensemble depth. Creates
        2^d ensemble members. Typical values: 1-3 (default: 2)
    mode : str, optional
        Sampling strategy for swap operations:
        - "balanced": Swaps within each class separately, maintains balance
        - "unbalanced": Random swaps across all training samples
        - "pair_sample": All pairwise swaps for comprehensive coverage
        (default: "balanced")
    barriers : bool, optional
        If True, add barrier gates for circuit visualization (default: False)
    
    Returns
    -------
    QuantumCircuit
        Quantum circuit implementing the ensemble classifier with:
        - d control qubits
        - n_samples * log2(n_features) data qubits
        - n_samples label qubits
        - log2(n_features) test data qubits
        - 1 test label qubit
        - 1 classical bit for measurement
        
    Notes
    -----
    - Total qubits: d + 2*n_samples*log2(n_features) + n_samples + 1
    - Circuit depth increases with d and n_swap
    - Balanced mode requires even number of samples per class
    - Simulation limited to ~30-36 qubits on classical hardware
    
    Mathematical Foundation:
    The ensemble creates a superposition state:
    |ψ⟩ = (1/√(2^d)) Σᵢ |i⟩|Dᵢ⟩|Lᵢ⟩
    where |Dᵢ⟩ and |Lᵢ⟩ are different arrangements of training data and labels.
    
    Examples
    --------
    >>> from Utils import training_set, normalize_custom
    >>> X_train = np.random.rand(8, 4)  # 8 samples, 4 features
    >>> y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> X_data, Y_data = training_set(X_train, y_train, n=4, seed=42)
    >>> x_test = normalize_custom(np.random.rand(4))
    >>> qc = ensemble(X_data, Y_data, x_test, n_swap=2, d=2, mode="balanced")
    >>> print(f"Circuit has {qc.num_qubits} qubits and depth {qc.depth()}")
    
    References
    ----------
    Macaluso et al., "A variational algorithm for quantum ensemble learning"
    IET Quantum Communication (2023)
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
    
    Runs the quantum circuit using Qiskit's AerSimulator with statevector
    method and returns measurement counts. Supports both CPU and GPU
    execution for accelerated simulation of larger circuits.
    
    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit to execute. Must contain measurement operations.
    n_shots : int, optional
        Number of measurement shots to perform. Higher values give more
        accurate probability estimates but take longer to execute.
        Typical values: 1024-8192 (default: 8192)
    device : str, optional
        Device type for simulation:
        - 'CPU': Use CPU for simulation (default)
        - 'GPU': Use GPU acceleration if available (requires qiskit-aer-gpu)
    
    Returns
    -------
    dict
        Dictionary mapping measurement outcomes (as binary strings) to
        their counts. For example: {'0': 4123, '1': 4069}
        
    Notes
    -----
    - Uses statevector method with optimization level 3
    - Parallel threshold set to 50 qubits for multi-threading
    - GPU support requires CUDA-enabled GPU and qiskit-aer-gpu package
    - Memory requirements grow exponentially with qubit count
    
    Examples
    --------
    >>> from modeling import cos_classifier
    >>> train = np.array([1.0, 0.0])
    >>> test = np.array([0.707, 0.707])
    >>> label = np.array([1.0, 0.0])
    >>> qc = cos_classifier(train, test, label)
    >>> counts = exec_simulator(qc, n_shots=1024, device='CPU')
    >>> print(counts)
    {'0': 512, '1': 512}
    
    >>> # For GPU acceleration (if available)
    >>> counts_gpu = exec_simulator(qc, n_shots=8192, device='GPU')
    """
    backend = AerSimulator(method='statevector', device=device, statevector_parallel_threshold=50)
    tqc = transpile(qc, backend, optimization_level=3)
    result = backend.run([tqc]).result()
    answer = result.get_counts(tqc)
    return answer

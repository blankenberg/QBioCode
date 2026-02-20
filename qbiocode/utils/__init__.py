"""
Utilities Module for QBioCode
=============================

This module provides helper functions and utilities for data preprocessing,
model management, IBM Quantum account handling, and result analysis.

Available Functions
------------------
- scaler_fn: Data scaling and normalization
- feature_encoding: Encode features for quantum circuits
- qml_winner: Identify best performing quantum model
- checkpoint_restart: Save and load model checkpoints
- track_progress: Track progress of dataset processing
- combine_results: Combine evaluation results from multiple runs
- find_duplicate_files: Find duplicate entries in datasets
- find_string_in_files: Search for strings in files
- generate_qml_experiment_configs: Generate config files for QML grid search
- get_creds: Get IBM Quantum credentials
- instantiate_runtime_service: Instantiate Qiskit Runtime Service
- get_backend_session: Get backend session for quantum execution
- get_sampler: Get sampler primitive
- get_estimator: Get estimator primitive
- get_ansatz: Get quantum ansatz circuit
- get_feature_map: Get quantum feature map
- get_optimizer: Get classical optimizer

Usage
-----
>>> from qbiocode.utils import scaler_fn, feature_encoding
>>> # Scale data
>>> X_scaled = scaler_fn(X, scaling='StandardScaler')
>>> # Encode features for quantum circuits
>>> X_encoded = feature_encoding(X, feature_encoding='OneHotEncoder')
"""

from .helper_fn import scaler_fn, feature_encoding
from .qc_winner_finder import qml_winner
from .dataset_checkpoint import checkpoint_restart
from .combine_evals_results import track_progress, combine_results
from .find_duplicates import find_duplicate_files
from .find_string import find_string_in_files
from .generate_qml_configs import generate_qml_experiment_configs
from .ibm_account import get_creds, instantiate_runtime_service
from .qutils import (
    get_backend_session,
    get_sampler,
    get_estimator,
    get_ansatz,
    get_feature_map,
    get_optimizer,
)

__all__ = [
    # Data preprocessing
    'scaler_fn',
    'feature_encoding',
    
    # Model management
    'qml_winner',
    'checkpoint_restart',
    
    # Results management
    'track_progress',
    'combine_results',
    
    # Configuration generation
    'generate_qml_experiment_configs',
    
    # File utilities
    'find_duplicate_files',
    'find_string_in_files',
    
    # IBM Quantum utilities
    'get_creds',
    'instantiate_runtime_service',
    
    # Quantum utilities
    'get_backend_session',
    'get_sampler',
    'get_estimator',
    'get_ansatz',
    'get_feature_map',
    'get_optimizer',
]

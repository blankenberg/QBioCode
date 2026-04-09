"""
Machine Learning Module for QBioCode
====================================

This module provides implementations of classical and quantum machine learning
algorithms for classification tasks. Each algorithm includes both standard and
optimized versions (where applicable) with hyperparameter tuning.

Classical Algorithms
-------------------
- Decision Tree (DT)
- Logistic Regression (LR)
- Multi-Layer Perceptron (MLP)
- Naive Bayes (NB)
- Random Forest (RF)
- Support Vector Classifier (SVC)
- XGBoost (XGB)

Quantum Algorithms
-----------------
- Quantum Neural Network (QNN)
- Quantum Support Vector Classifier (QSVC)
- Variational Quantum Classifier (VQC)
- Projected Quantum Kernel (PQK)

Usage
-----
>>> from qbiocode.learning import compute_rf, compute_qsvc
>>> # Train classical model
>>> results = compute_rf(X_train, y_train, X_test, y_test)
>>> # Train quantum model
>>> qresults = compute_qsvc(X_train, y_train, X_test, y_test)
"""

# Classical ML algorithms
from .compute_dt import compute_dt, compute_dt_opt
from .compute_lr import compute_lr, compute_lr_opt
from .compute_mlp import compute_mlp, compute_mlp_opt
from .compute_nb import compute_nb, compute_nb_opt
from .compute_rf import compute_rf, compute_rf_opt
from .compute_svc import compute_svc, compute_svc_opt

try:
    from .compute_xgb import compute_xgb, compute_xgb_opt
except Exception:
    # XGBoost not available (e.g., OpenMP not installed on macOS)
    compute_xgb = None  # type: ignore
    compute_xgb_opt = None  # type: ignore

from .compute_pqk import compute_pqk
# Quantum ML algorithms
from .compute_qnn import compute_qnn
from .compute_qsvc import compute_qsvc
from .compute_vqc import compute_vqc

__all__ = [
    # Classical algorithms
    "compute_dt",
    "compute_dt_opt",
    "compute_lr",
    "compute_lr_opt",
    "compute_mlp",
    "compute_mlp_opt",
    "compute_nb",
    "compute_nb_opt",
    "compute_rf",
    "compute_rf_opt",
    "compute_svc",
    "compute_svc_opt",
    "compute_xgb",
    "compute_xgb_opt",
    # Quantum algorithms
    "compute_qnn",
    "compute_qsvc",
    "compute_vqc",
    "compute_pqk",
]

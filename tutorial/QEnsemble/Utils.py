"""
Utility Functions for Quantum Ensemble Learning

This module provides comprehensive utility functions for:
- Data preprocessing and normalization
- Model evaluation and metrics calculation
- Visualization of results
- Classical baseline methods (Random Forest, XGBoost, LazyPredict)
- Quantum ensemble execution workflows
- Result post-processing and analysis
"""
import os
import re
import pickle
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier
import umap

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import XGate

import modeling_random_unitary
from modeling import *

warnings.filterwarnings('ignore')


def calculate_predicted_classes(preds):
    """
    Convert probability predictions to class labels.
    
    Takes probability predictions in the form [p0, p1] and converts them
    to binary class labels by selecting the class with higher probability.
    
    Parameters
    ----------
    preds : list of array-like
        Predicted probabilities for each class. Each element should be
        [p0, p1] where p0 is probability of class 0 and p1 is probability
        of class 1
    
    Returns
    -------
    list of int
        Predicted class labels (0 or 1) based on argmax of probabilities
        
    Examples
    --------
    >>> preds = [[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]]
    >>> calculate_predicted_classes(preds)
    [0, 1, 1]
    """
    return [0 if x[0] > x[1] else 1 for x in preds]


def calculate_number_predicted_classes(preds):
    """
    Count the number of unique predicted classes.
    
    Parameters
    ----------
    preds : list of array-like
        Predicted probabilities for each class
    
    Returns
    -------
    int
        Number of unique predicted classes
    """
    return len(set(calculate_predicted_classes(preds)))


def plot_figures(results_df_agg, dataset_name, method, figsize=(12, 3)):
    """
    Create visualization plots for model performance comparison.
    
    Generates four plots:
    - Top performing models by accuracy (>= 0.5)
    - Worst performing models by accuracy (<= 0.5)
    - Top performing models by F1 score (>= 0.5)
    - Worst performing models by F1 score (<= 0.5)
    
    Parameters
    ----------
    results_df_agg : DataFrame
        Aggregated results with columns: Model, Accuracy, F1 Score, dataset
    dataset_name : str
        Name of the dataset for plot titles
    method : str
        Method name for plot titles
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 3))
    """
    
    results_df_agg_top = results_df_agg[results_df_agg['Accuracy'] >= 0.5]
    results_df_agg_top.sort_values('Accuracy', ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(data=results_df_agg_top,
                x='Model',
                y='Accuracy',
                hue='dataset')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title('Top Performing for ' + dataset_name + ' with ' + method)
    plt.show()
    plt.close()

    results_df_agg_worst = results_df_agg[results_df_agg['Accuracy'] <= 0.5]
    results_df_agg_worst.sort_values('Accuracy', ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(data=results_df_agg_worst,
                x='Model',
                y='Accuracy',
                hue='dataset')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title('Worst Performing for ' + dataset_name + ' with ' + method)
    plt.show()
    plt.close()

    results_df_agg_top = results_df_agg[results_df_agg['F1 Score'] >= 0.5]
    results_df_agg_top.sort_values('F1 Score', ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(data = results_df_agg_top,
                x = 'Model',
                y = 'F1 Score',
                hue = 'dataset')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title( 'Top Performing for ' + dataset_name + ' with ' + method )
    plt.show()
    plt.close()

    results_df_agg_worst = results_df_agg[ results_df_agg['F1 Score'] <= 0.5 ]
    results_df_agg_worst.sort_values('F1 Score', ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(data = results_df_agg_worst,
                x = 'Model',
                y = 'F1 Score',
                hue = 'dataset')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title( 'Worst Performing for ' + dataset_name + ' with ' + method )
    plt.show()
    plt.close()

def calculate_f1(preds, y_test):
    """
    Calculate weighted F1 score from probability predictions.
    
    Converts probability predictions to class labels and computes the
    weighted F1 score, which is the harmonic mean of precision and recall
    weighted by class support.
    
    Parameters
    ----------
    preds : list of array-like
        Predicted probabilities for each class. Each element should be
        [p0, p1] where p0 + p1 ≈ 1
    y_test : array-like, shape (n_samples,)
        True binary labels (0 or 1)
    
    Returns
    -------
    float
        Weighted F1 score in range [0, 1]. Higher is better.
        F1 = 2 * (precision * recall) / (precision + recall)
        
    Notes
    -----
    Uses sklearn's f1_score with average='weighted' to account for
    class imbalance.
    
    Examples
    --------
    >>> preds = [[0.9, 0.1], [0.3, 0.7], [0.8, 0.2]]
    >>> y_test = np.array([0, 1, 0])
    >>> f1 = calculate_f1(preds, y_test)
    >>> print(f"F1 Score: {f1:.3f}")
    F1 Score: 1.000
    """
    preds = [1 if p[1] > p[0] else 0 for p in preds]
    return f1_score(y_pred=preds, y_true=y_test, average='weighted')


def run_quantum_ensemble(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, ds, n_swaps, n_features,
                         n_trains, n_shots, pca_embed=False, umap_embed=False, device='CPU', instance='', random_unitary=False, select_features=[]):
    """
    Run quantum ensemble experiments across multiple parameter configurations.
    
    This is a comprehensive workflow function that performs a grid search over
    quantum ensemble hyperparameters (d, n_swap, n_features, n_train) and
    evaluates performance on test data. Results are automatically saved to
    disk after each configuration.
    
    The function handles:
    - Data preprocessing (scaling, dimensionality reduction)
    - Feature selection (variance-based or PCA/UMAP)
    - Circuit construction and execution
    - Performance evaluation
    - Result aggregation and persistence
    
    Parameters
    ----------
    predictions : dict
        Dictionary to store prediction results. Structure:
        predictions[dataset_name][method] = DataFrame of results
    dataset : dict
        Dictionary containing dataset splits. Structure:
        dataset[dataset_name][params] = (X_train, X_test, y_train, y_test)
    method : str
        Method name for results tracking (e.g., 'qensemble', 'qensemble_random_unitary')
    dataset_name : str
        Name of the dataset being processed
    seed : int
        Random seed for reproducibility of training set selection
    test_size : float
        Fraction of data for testing (informational, not used directly)
    file_predictions : str
        Path to pickle file for saving predictions dictionary
    ds : list of int
        List of ensemble depths (control qubits) to test. Example: [1, 2, 3]
    n_swaps : list of int
        List of swap operation counts to test. Example: [1, 2, 3]
    n_features : list of int
        List of feature counts to test. Must be powers of 2. Example: [2, 4, 8]
    n_trains : list of int
        List of training sample sizes to test. Example: [4, 8, 16]
    n_shots : int
        Number of measurement shots per circuit execution
    pca_embed : bool, optional
        If True, use PCA for dimensionality reduction (default: False)
    umap_embed : bool, optional
        If True, use UMAP for dimensionality reduction (default: False)
    device : str, optional
        Execution device:
        - 'CPU': Local CPU simulation
        - 'GPU': Local GPU simulation (requires qiskit-aer-gpu)
        - 'ibm_*': IBM Quantum device name (requires instance)
        (default: 'CPU')
    instance : str, optional
        IBM Quantum instance string (e.g., 'ibm-q/open/main') required
        when device is an IBM backend (default: '')
    random_unitary : bool, optional
        If True, use random unitary ensemble variant from
        modeling_random_unitary module (default: False)
    select_features : list, optional
        List of specific feature names to select. If empty, uses
        variance-based selection (default: [])
    
    Returns
    -------
    dict
        Updated predictions dictionary with new results appended
        
    Notes
    -----
    - Automatically skips configurations that have already been run
    - Uses MinMaxScaler for data normalization
    - Feature selection: variance-based (default), PCA, or UMAP
    - Results saved after each configuration for fault tolerance
    - Skips configurations where n_train <= d (insufficient samples)
    
    Examples
    --------
    >>> predictions = {}
    >>> dataset = {'blobs': {(100, 2, 2): (X_train, X_test, y_train, y_test)}}
    >>> predictions = run_quantum_ensemble(
    ...     predictions, dataset, 'qensemble', 'blobs', seed=42,
    ...     test_size=0.2, file_predictions='results.pkl',
    ...     ds=[2], n_swaps=[1], n_features=[2], n_trains=[4],
    ...     n_shots=8192, device='CPU'
    ... )
    """

    if (dataset_name not in predictions.keys()) or (method not in predictions[dataset_name].keys()):
        results_df = pd.DataFrame()
        predictions[dataset_name][method] = results_df
    else:
        results_df = predictions[dataset_name][method]
        
    for k,v in dataset[dataset_name].items():
        for f in n_features:
            (X_train_orig, X_test_orig, y_train, y_test) = v

            if len( select_features ) > 0: 
                X_train = X_train.loc[:,select_features]
                X_test = X_test.loc[:,select_features]
            
            scaler = MinMaxScaler()
            X_train = pd.DataFrame( scaler.fit_transform(X_train_orig), index=X_train_orig.index, columns=X_train_orig.columns )
            X_test = pd.DataFrame( scaler.transform(X_test_orig), index=X_test_orig.index, columns=X_test_orig.columns )
            embed = 'none'

            if pca_embed:
                embed = 'pca'
                embedder = PCA(f)
                X_train = embedder.fit_transform(X_train)
                X_test = embedder.transform(X_test)
            elif umap_embed:
                embed = 'umap'
                reducer = umap.UMAP(f)
                X_train = reducer.fit_transform(X_train)
                X_test = reducer.transform(X_test)
            else:
                vr = X_train.apply(np.var, axis = 0)
                vr = vr.sort_values(ascending=False)
                X_train = X_train[ list(vr[0:f].index) ].to_numpy()
                X_test = X_test[ list(vr[0:f].index) ].to_numpy()

            for d in ds:                
                for n_train in n_trains:
                    if n_train > d:
                        for n_swap in n_swaps:
                            if (len(results_df) == 0) or (len(results_df[(results_df['dataset'] == dataset_name) &
                                                                        (results_df['method'] == method) &
                                                                        (results_df['dataset_params'] == k) &
                                                                        (results_df['n_feature'] == f ) &
                                                                        (results_df['n_swap'] == n_swap ) &
                                                                        (results_df['n_train'] == n_train ) &
                                                                        (results_df['embed'] == embed ) &
                                                                        (results_df['select_features'] == ','.join(select_features) ) &
                                                                        (results_df['d'] == d)
                                                                        ]) == 0):
                                if random_unitary:
                                    res = modeling_random_unitary.run_ensemble(d, n_train, seed, n_swap, X_train, X_test, y_train, y_test, n_shots = n_shots)
                                else:
                                    res = run_ensemble(d, n_train, seed, n_swap, X_train, X_test, y_train, y_test, n_shots = n_shots, device = device, instance = instance)

                                res['dataset'] = [dataset_name]*len(res)
                                res['method'] = [method]*len(res)
                                res['dataset_params'] = [k]*len(res)
                                res['embed'] = [embed]*len(res)
                                res['select_features'] = [','.join(select_features)]*len(res)
                                results_df = pd.concat( [results_df, res] )
        
                                predictions[dataset_name][method] = results_df
                                        
                                # save
                                pickle.dump( predictions, open( file_predictions, 'wb') ) 
    
    return predictions


def run_quantum_cosine(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, n_features,
                       n_trains, n_shots, pca_embed=False, umap_embed=False, select_features=[]):
    """
    Run quantum cosine classifier experiments.
    
    Parameters
    ----------
    predictions : dict
        Dictionary to store prediction results
    dataset : dict
        Dictionary containing dataset splits
    method : str
        Method name for results tracking
    dataset_name : str
        Name of the dataset
    seed : int
        Random seed for reproducibility
    test_size : float
        Fraction of data for testing
    file_predictions : str
        Path to save predictions
    n_features : list of int
        List of feature counts to test
    n_trains : list of int
        List of training sample sizes to test
    n_shots : int
        Number of measurement shots
    pca_embed : bool, optional
        Use PCA for dimensionality reduction (default: False)
    umap_embed : bool, optional
        Use UMAP for dimensionality reduction (default: False)
    select_features : list, optional
        Specific features to select (default: [])
    
    Returns
    -------
    dict
        Updated predictions dictionary with results
    """
    epsilon = 1e-15
    if (dataset_name not in predictions.keys()) or (method not in predictions[dataset_name].keys()):
        results_df = pd.DataFrame()
        predictions[dataset_name][method] = results_df
    else:
        results_df = predictions[dataset_name][method]
        
    for k,v in dataset[dataset_name].items():
        for f in n_features:
            (X_train, X_test, y_train, y_test) = v
            if len( select_features ) > 0: 
                X_train = X_train.loc[:,select_features]
                X_test = X_test.loc[:,select_features]
                
            Y_vector_train = label_to_array(y_train)
            Y_vector_test = label_to_array(y_test)
            test_size = Y_vector_test.shape[0]
            train_size = Y_vector_train.shape[0]
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            embed = 'none'

            # adding epsilon to avoid zero division
            X_train = X_train + epsilon
            X_test = X_test + epsilon

            if pca_embed:
                embed = 'pca'
                embedder = PCA(f)
                X_train = embedder.fit_transform(X_train)
                X_test = embedder.transform(X_test)
            elif umap_embed:
                embed = 'umap'
                reducer = umap.UMAP(f)
                X_train = reducer.fit_transform(X_train)
                X_test = reducer.transform(X_test)
                    
            for n_train in n_trains:
                preds = []
                
                if (len(results_df) == 0) or (len(results_df[(results_df['dataset'] == dataset_name) &
                                                             (results_df['method'] == method) &
                                                             (results_df['dataset_params'] == k) &
                                                             (results_df['n_feature'] == f ) &
                                                             (results_df['embed'] == embed ) &
                                                             (results_df['select_features'] == ','.join(select_features) ) &
                                                             (results_df['n_train'] == n_train )
                                                            ]) == 0):

                    for x_test, y_ts in zip(X_test, Y_vector_test):
                        ix = np.random.choice(train_size, n_train)[0]
                        x_train: Any | ndarray[Any, dtype[float64]] = X_train[ix]
                        x_tr = normalize_custom_legacy(x_train)
                        y_tr = Y_vector_train[ix]
                        x_ts = normalize_custom_legacy(x_test)
                        qc = cos_classifier_legacy(x_tr, x_ts, y_tr)

                        r = exec_simulator(qc, n_shots=n_shots)

                        if '0' not in r.keys():
                            r['0'] = 0
                        elif '1' not in r.keys():
                            r['1'] = 0

                        preds.append(retrieve_proba(r,))

                    a, b = evaluation_metrics(preds, y_test, save=False)    
                    res = pd.DataFrame( [dataset_name, method, k, seed, X_train.shape[1], qc.num_qubits, n_train, embed, ','.join(select_features), a, b, preds, y_test ],
                                 index = ['dataset', 'method', 'dataset_params', 'seed', 'n_feature', 'qubits', 'n_train', 'embed', 'select_features', 'accuracy', 'brier', 'predictions', 'y_test']).transpose()
                    results_df = pd.concat( [results_df, res])
                    
                    predictions[dataset_name][method] = results_df
                    
                    # save
                    pickle.dump( predictions, open( file_predictions, 'wb') )
    
    return predictions



def run_random_forest(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, select_features = [],
                            params = {}, pca_embed = False, umap_embed = False, n_features = 0):

    if (dataset_name not in predictions.keys()) or (method not in predictions[dataset_name].keys()):
        results_df = pd.DataFrame()
        predictions[dataset_name][method] = results_df
    else:
        results_df = predictions[dataset_name][method]


    # Define the hyperparameter grid
    param_distributions = {
        'n_estimators': np.arange(100, 1000, 100),
        'max_depth': np.arange(5, 20),
        'min_samples_split': np.arange(2, 10),
        'min_samples_leaf': np.arange(1, 5),
        'max_features': ['sqrt', 'log2']
    }

    for k,v in dataset[dataset_name].items():
        (X_train, X_test, y_train, y_test) = v
        if len( select_features ) > 0: 
            X_train = X_train.loc[:,select_features]
            X_test = X_test.loc[:,select_features]
            
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        embed = 'none'

        if pca_embed:
            embed = 'pca'
            embedder = PCA(n_features)
            X_train = embedder.fit_transform(X_train)
            X_test = embedder.transform(X_test)
        elif umap_embed:
            embed = 'umap'
            reducer = umap.UMAP(n_features)
            X_train = reducer.fit_transform(X_train)
            X_test = reducer.transform(X_test)

        if len(params) > 0:
            # Initialize the Random Forest Classifier
            rf = RandomForestClassifier(random_state=seed, 
                                        n_estimators=params['n_estimators'],
                                        min_samples_split=params['min_samples_split'],
                                        max_features=params['max_features'],
                                        max_depth=params['max_depth'],
                                        min_samples_leaf=params['min_samples_leaf']
                                        )

            rf.fit(X_train, y_train)
            preds = rf.predict_proba(X_test)
        else:
            # Initialize the Random Forest Classifier
            rf = RandomForestClassifier(random_state=seed)

            # Initialize RandomizedSearchCV
            rf_random = RandomizedSearchCV(estimator=rf, 
                                        param_distributions=param_distributions, 
                                        n_iter=10, 
                                        cv=3, 
                                        random_state=seed,
                                        n_jobs=-1)
            rf_random.fit(X_train, y_train)
            preds = rf_random.predict_proba(X_test)
            params = rf_random.best_params_

        a, b = evaluation_metrics(preds, y_test, save=False)    

        res = pd.DataFrame( [dataset_name, method, k, seed, X_train.shape[1], ','.join(select_features), a, b, preds, y_test, params ],
                        index = ['dataset', 'method', 'dataset_params', 'seed', 'n_feature', 'select_features', 'accuracy', 'brier', 'predictions', 'y_test', 'best_params']).transpose()
        results_df = pd.concat( [results_df, res])
            
        predictions[dataset_name][method] = results_df
        
        # save
        pickle.dump( predictions, open( file_predictions, 'wb') )
        

    return predictions


def run_xgboost(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, select_features = [],
                            params = {}, pca_embed = False, umap_embed = False, n_features = 0):

    if (dataset_name not in predictions.keys()) or (method not in predictions[dataset_name].keys()):
        results_df = pd.DataFrame()
        predictions[dataset_name][method] = results_df
    else:
        results_df = predictions[dataset_name][method]

    for k,v in dataset[dataset_name].items():
        (X_train, X_test, y_train, y_test) = v
        if len( select_features ) > 0: 
            X_train = X_train.loc[:,select_features]
            X_test = X_test.loc[:,select_features]
            
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        embed = 'none'

        if pca_embed:
            embed = 'pca'
            embedder = PCA(n_features)
            X_train = embedder.fit_transform(X_train)
            X_test = embedder.transform(X_test)
        elif umap_embed:
            embed = 'umap'
            reducer = umap.UMAP(n_features)
            X_train = reducer.fit_transform(X_train)
            X_test = reducer.transform(X_test)

 
        ##XGB
        if len(params) > 0:
            # Initialize XGB
            xgb = XGBClassifier(
                random_state=seed,
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=params['min_child_weight'],
                eval_metric='logloss'
            )
            xgb.fit(X_train, y_train)
            preds = xgb.predict_proba(X_test)
        else:
            xgb = XGBClassifier(
                random_state=seed,
                eval_metric='logloss'
            )

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                    
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'min_child_weight': [1, 3, 5]
            }
                    
            xgb_grid = GridSearchCV(
                estimator=xgb,
                param_grid=param_grid,
                scoring='f1_weighted',
                n_jobs=-1,
                cv=cv,
                verbose=1
            )

            xgb_grid.fit(X_train, y_train)
            preds = xgb_grid.predict_proba(X_test)
            params = xgb_grid.best_params_

        a, b = evaluation_metrics(preds, y_test, save=False)    

        res = pd.DataFrame( [dataset_name, method, k, seed, X_train.shape[1], ','.join(select_features), a, b, preds, y_test, params ],
                        index = ['dataset', 'method', 'dataset_params', 'seed', 'n_feature', 'select_features', 'accuracy', 'brier', 'predictions', 'y_test', 'best_params']).transpose()
        results_df = pd.concat( [results_df, res])
            
        predictions[dataset_name][method] = results_df
        
        # save
        pickle.dump( predictions, open( file_predictions, 'wb') )
        

    return predictions


def run_lazy_predict(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, select_features = []):
    predictions[dataset_name][method] = {}

    for k,v in dataset[dataset_name].items():
        (X_train, X_test, y_train, y_test) = v
        if len( select_features ) > 0: 
            X_train = X_train.loc[:,select_features]
            X_test = X_test.loc[:,select_features]
            
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        models,preds = clf.fit(X_train, X_test, y_train, y_test)
        
        predictions[dataset_name][method][k] = {}
        predictions[dataset_name][method][k]['seed'] = seed
        predictions[dataset_name][method][k]['models'] = models
        predictions[dataset_name][method][k]['preds'] = preds
        predictions[dataset_name][method][k]['y_test'] = y_test
        predictions[dataset_name][method][k]['select_features'] = select_features

    # save
    pickle.dump( predictions, open( file_predictions, 'wb') )
    
    return predictions

        
def run_ensemble(d, n_train, seed, n_swap, X_train, X_test, y_train, y_test,
                 mode="pair_sample", n_shots=8192, selectRandom=True, device='CPU',
                 barriers=False, instance='', resilience_level=1, optimization_level=3,
                 nthreads=20, dynamicDecoupling=False):
    """
    Execute quantum ensemble classifier with comprehensive hardware support.
    
    This function provides a complete workflow for running quantum ensemble
    classification with support for both simulation and IBM Quantum hardware.
    It handles training set selection, circuit construction, transpilation,
    execution, and evaluation.
    
    Key features:
    - IBM Quantum hardware execution with error mitigation
    - Dynamic decoupling for noise reduction
    - Parallel transpilation
    - Automatic result aggregation
    
    Parameters
    ----------
    d : int
        Number of control qubits (ensemble depth)
    n_train : int
        Number of training samples to use
    seed : int
        Random seed for training set selection
    n_swap : int
        Number of swap operations per control qubit
    X_train : array-like, shape (n_samples, n_features)
        Training feature data
    X_test : array-like, shape (n_test, n_features)
        Test feature data
    y_train : array-like, shape (n_samples,)
        Training labels
    y_test : array-like, shape (n_test,)
        Test labels
    mode : str, optional
        Ensemble mode: "pair_sample", "balanced", or "unbalanced"
        (default: "pair_sample")
    n_shots : int, optional
        Number of measurement shots (default: 8192)
    selectRandom : bool, optional
        If True, randomly select training subset; if False, use all
        training data (default: True)
    device : str, optional
        Execution device: 'CPU', 'GPU', or IBM device name like 'ibm_kyoto'
        (default: 'CPU')
    barriers : bool, optional
        Add barrier gates for visualization (default: False)
    instance : str, optional
        IBM Quantum instance string (default: '')
    resilience_level : int, optional
        IBM error mitigation level (0-3). Higher = more mitigation
        (default: 1)
    optimization_level : int, optional
        Transpilation optimization level (0-3) (default: 3)
    nthreads : int, optional
        Number of threads for parallel transpilation (default: 20)
    dynamicDecoupling : bool, optional
        Enable dynamic decoupling (XY4 sequence) for IBM hardware
        (default: False)
    
    Returns
    -------
    DataFrame
        Results with columns: seed, n_feature, qubits, d, n_train, n_swap,
        accuracy, brier, predictions, y_test. Returns empty DataFrame if
        circuit exceeds 36 qubits.
        
    Notes
    -----
    IBM Quantum Features:
    - Uses SamplerV2 with twirling enabled
    - Supports dynamic decoupling with XY4 sequence
    - Parallel circuit transpilation
    - Automatic gate duration handling
    
    Simulation Features:
    - CPU/GPU support via AerSimulator
    - Statevector method with optimization
    - Limited to ~36 qubits
    
    Examples
    --------
    >>> # Local simulation
    >>> results = run_ensemble(d=2, n_train=4, seed=42, n_swap=1,
    ...                        X_train, X_test, y_train, y_test,
    ...                        device='CPU', n_shots=8192)
    
    >>> # IBM Quantum hardware
    >>> results = run_ensemble(d=2, n_train=4, seed=42, n_swap=1,
    ...                        X_train, X_test, y_train, y_test,
    ...                        device='ibm_kyoto', instance='ibm-q/open/main',
    ...                        dynamicDecoupling=True, n_shots=4096)
    """

    predictions = []
    qc_list = []

    if 'ibm' in device:
        service = QiskitRuntimeService(instance=instance)
        backend = service.backend(device)
        sampler = Sampler(mode=backend, options={"default_shots": n_shots})

        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.num_randomizations = "auto"
        sampler.options.twirling.enable_measure = True


        # Set Sampler Options
        if dynamicDecoupling:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = 'XY4'

        
        # Get gate durations so the transpiler knows how long each operation takes
        durations = backend.target.durations()

        # This is the sequence we'll apply to idling qubits
        dd_sequence = [XGate(), XGate()]

        for x_test in X_test:
            x_test = normalize_custom(x_test)
            X_data, Y_data = training_set(X_train, y_train, n=n_train, seed=seed)

            qc_orig = ensemble(X_data, Y_data, x_test, n_swap=n_swap, d=d, mode=mode, barriers = barriers)
            pass_manager = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
            qc = pass_manager.run(qc_orig, num_processes=nthreads)
            qc_list.append(qc)



        for i in range(len(qc_list)):
            result = sampler.run([qc_list[i]]).result()
            r = result[0].join_data().get_counts()
            predictions.append(retrieve_proba(r))
    else:
        for x_test in X_test:
            X_data, Y_data = training_set(X_train, y_train, n=n_train, seed=seed, selectRandom = selectRandom)
            x_test = normalize_custom(x_test)
            
            qc = qc_orig = ensemble(X_data, Y_data, x_test, n_swap=n_swap, d=d, mode=mode, barriers = barriers)
            if qc.num_qubits <=36:
                r = exec_simulator(qc, n_shots=n_shots, device = device)
                predictions.append(retrieve_proba(r))

    if ('ibm' in device) or (qc.num_qubits <=36):
        a, b = evaluation_metrics(predictions, y_test, save=False)
        
        res = pd.DataFrame(  [ seed, X_data.shape[1], qc_orig.num_qubits, d, n_train, n_swap, a, b, predictions, y_test ], 
                        index = ['seed', 'n_feature', 'qubits', 'd', 'n_train', 'n_swap', 'accuracy', 'brier', 'predictions', 'y_test' ] ).transpose()
        print(res.iloc[:,1:8])
        return res
    else:
        return pd.DataFrame()
    
def post_process_ensemble_results_df( df ):
    cmatrices = []
    precisions = []
    recalls = []
    f1s = []
    label_all = []
    pred_all = []
    correct_all = []
    incorrect_all = []

    for idx, row in df.iterrows():
        labels = [int(x) for x in re.sub( '\[', '', re.sub('\]', '', row['y_test'] ) ).split(' ') ]
        preds = [float(x) for x in re.sub( '\[', '', re.sub('\]', '', row['predictions'] ) ).split(', ') ]
        preds_bin = []
        for i in np.arange(0,len(preds),2):
            if preds[i] > preds[i+1]:
                preds_bin.append(0)
            else:
                preds_bin.append(1)
                
        correct = [ 1 if labels[i] == preds_bin[i] else  0 for i in range(len(labels))]
        correct_all.append(correct)
        incorrect_all.append( [np.abs(x-1) for x in correct ])
        label_all.append(labels)
        pred_all.append(preds_bin)
        cmatrices.append(confusion_matrix( labels, preds_bin))
        precisions.append(precision_score(labels, preds_bin,average='weighted'))
        recalls.append(recall_score(labels, preds_bin,average='weighted'))
        f1s.append(f1_score(labels, preds_bin,average='weighted'))
        
    df['confusion_matrix'] = cmatrices
    df['precision'] = precisions
    df['recall'] = recalls
    df['f1'] = f1s
    df['labels'] = label_all
    df['preds'] = pred_all
    df['correct'] = correct_all
    df['incorrect'] = incorrect_all

    return(df)




def create_dir(path):
    if not os.path.exists(path):
        print('The directory', path, 'does not exist and will be created')
        os.makedirs(path)
    else:
        print('The directory', path, ' already exists')


def save_dict(d, name='dict'):
    df = pd.DataFrame(list(d.items()))
    name = name + '_' + str(np.random.randint(10 ** 6)) + '.csv'
    df.to_csv(name)


def normalize_custom_legacy(x, C=1):
    M = x[0] ** 2 + x[1] ** 2
    x_normed = [
        1 / np.sqrt(M * C) * complex(x[0], 0),  # 00
        1 / np.sqrt(M * C) * complex(x[1], 0),  # 01
    ]
    return x_normed


def normalize_custom(x, C=1):
    """
    Normalize data vector for quantum state encoding.
    
    Normalizes a classical data vector to unit L2 norm and converts to
    complex amplitudes suitable for quantum state initialization. This
    ensures the encoded quantum state is properly normalized.
    
    Parameters
    ----------
    x : array-like, shape (n,)
        Classical data vector to normalize
    C : float, optional
        Scaling constant (default: 1)
    
    Returns
    -------
    list of complex
        Normalized vector as list of complex numbers with zero imaginary part
        
    Notes
    -----
    - Computes M = Σᵢ xᵢ²
    - Returns [x₀/√(MC), x₁/√(MC), ..., xₙ/√(MC)] as complex numbers
    - Ensures Σᵢ |xᵢ|² = 1 for valid quantum state
    
    Examples
    --------
    >>> x = np.array([3.0, 4.0])
    >>> x_norm = normalize_custom(x)
    >>> print([abs(xi) for xi in x_norm])
    [0.6, 0.8]
    >>> print(sum([abs(xi)**2 for xi in x_norm]))
    1.0
    """
    M = sum( x**2 )
    x_normed = [ 1/np.sqrt(M*C)*complex(i,0) for i in x ]
    return x_normed


def add_label(d, label='0'):
    try:
        d[label]
        print('Label', label, 'exists')
    except:
        d[label] = 0
    return d

def label_to_array(y):
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
    Calculate accuracy and Brier score for binary classification.
    
    Computes two key metrics for evaluating probabilistic binary classifiers:
    accuracy (correctness) and Brier score (calibration quality).
    
    Parameters
    ----------
    predictions : list of array-like
        Predicted probabilities. Each element should be [p0, p1] where
        p0 + p1 = 1
    y_test : array-like, shape (n_samples,)
        True binary labels (0 or 1)
    save : bool, optional
        Legacy parameter, not currently used (default: True)
    
    Returns
    -------
    tuple of float
        (accuracy, brier_score) where:
        - accuracy: Fraction of correct predictions (0 to 1)
        - brier_score: Mean squared error of probabilities (0 to 1)
          Lower Brier score indicates better calibration
          
    Notes
    -----
    - Predictions rounded to nearest integer for accuracy
    - Brier score computed using probability of class 1
    - Perfect predictions: accuracy=1.0, brier_score=0.0
    - Uses sklearn.metrics for computation
    
    Examples
    --------
    >>> predictions = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]
    >>> y_test = np.array([0, 1, 0])
    >>> acc, brier = evaluation_metrics(predictions, y_test)
    >>> print(f"Accuracy: {acc:.3f}, Brier: {brier:.3f}")
    Accuracy: 1.000, Brier: 0.030
    
    See Also
    --------
    sklearn.metrics.accuracy_score : Accuracy calculation
    sklearn.metrics.brier_score_loss : Brier score calculation
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



def training_set(X, Y, n=4, seed=123, selectRandom=True):
    """
    Select and prepare balanced training subset for quantum ensemble.
    
    Creates a balanced training set by selecting equal numbers of samples
    from each class, normalizing them for quantum encoding, and converting
    labels to one-hot format.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training feature data
    Y : array-like, shape (n_samples,)
        Training labels (binary: 0 or 1)
    n : int, optional
        Total number of training samples to select. Must be even for
        balanced selection (default: 4)
    seed : int, optional
        Random seed for reproducible selection (default: 123)
    selectRandom : bool, optional
        If True, randomly select n/2 samples from each class.
        If False, use all training data (default: True)
    
    Returns
    -------
    tuple of (ndarray, ndarray)
        X_data : array of normalized training samples, shape (n, n_features)
                 Each sample is normalized using normalize_custom()
        Y_data : one-hot encoded labels, shape (n, 2)
                 [[1,0] for class 0, [0,1] for class 1]
                 
    Notes
    -----
    - Ensures class balance by selecting n/2 samples from each class
    - All data vectors are normalized to unit L2 norm
    - Labels converted to quantum-compatible one-hot encoding
    - Random selection uses numpy's random.choice without replacement
    
    Examples
    --------
    >>> X = np.random.rand(20, 4)
    >>> Y = np.array([0]*10 + [1]*10)
    >>> X_data, Y_data = training_set(X, Y, n=4, seed=42)
    >>> print(X_data.shape, Y_data.shape)
    (4, 4) (4, 2)
    >>> print(Y_data)
    [[0 1]
     [0 1]
     [1 0]
     [1 0]]
    """
    np.random.seed(seed)

    X_data = X.copy()
    Y_data = Y_vector = label_to_array(Y)

    if selectRandom:
        ix_y1 = np.random.choice(np.where(Y == 1)[0], int(n / 2), replace=False)
        ix_y0 = np.random.choice(np.where(Y == 0)[0], int(n / 2), replace=False)

        X_data = np.concatenate([X[ix_y1], X[ix_y0]])
        Y_data = np.concatenate([Y_vector[ix_y1], Y_vector[ix_y0]])

    X_data_new = []

    for i in range(len(X_data)):
        X_data_new.append(normalize_custom(X_data[i]))

    X_data_new = np.array(X_data_new)

    return X_data_new, Y_data



# Define the cosine classifier
def cosine_classifier(x,y):
    return 1/2 + (cosine_similarity([x], [y])**2)/2

def retrieve_proba(r):
    """
    Extract probability predictions from measurement counts.
    
    Converts raw measurement counts from quantum circuit execution into
    probability predictions for binary classification. Handles edge cases
    where only one outcome is observed.
    
    Parameters
    ----------
    r : dict
        Dictionary of measurement counts with keys '0' and/or '1'
        Example: {'0': 4123, '1': 4069}
    
    Returns
    -------
    list of float
        [p0, p1] where p0 is probability of class 0 and p1 is probability
        of class 1. Always sums to 1.0.
        
    Notes
    -----
    - Handles missing keys gracefully (assigns probability 0 or 1)
    - If only '0' observed: returns [1.0, 0.0]
    - If only '1' observed: returns [0.0, 1.0]
    - If both observed: returns normalized probabilities
    
    Examples
    --------
    >>> counts = {'0': 6000, '1': 2000}
    >>> probs = retrieve_proba(counts)
    >>> print(probs)
    [0.75, 0.25]
    
    >>> counts = {'0': 8192}  # Only one outcome
    >>> probs = retrieve_proba(counts)
    >>> print(probs)
    [1.0, 0.0]
    """
    state_zero = '0'
    state_one = '1'
    p0 = 0
    p1 = 0
    try:
        p0 = r[state_zero] / (r[state_zero] + r[state_one])
        p1 = 1 - p0
    except:
        if list(r.keys())[0] == state_zero:
            p0 = 1
            p1 = 0
        elif list(r.keys())[0] == state_one:
            p0 = 0
            p1 = 1
    return [p0, p1]


def post_process_results( predictions, dir_output, datasets, metrics = ['Accuracy', 'F1 Score', 'brier'] ):

    total_results = pd.DataFrame()
    for dataset_name in datasets:
        print(f"Dataset: {dataset_name}")
        methods = list( predictions[dataset_name].keys() )

        all_results = pd.DataFrame()

        for method in methods:
            if method not in ['random_forest_gs', 'xgb_gs']: # Ignore the parameter search experiments for the random forest and xgb
                print(f"Method: {method}")
                results_df = predictions[dataset_name][method]
                results_df = results_df.reset_index(drop=True)

                results_df['num_pred_classes'] = [calculate_number_predicted_classes(x) for x in results_df['predictions']]
            
                results_df.columns = [ re.sub( 'dataset_params', 'split', re.sub( 'accuracy', 'Accuracy', re.sub('method', 'Model', x ) ) ) for x in results_df.columns]
                results_df['F1 Score'] = [ calculate_f1( row.predictions, row.y_test) for idx, row in results_df.iterrows() ]

                if method == 'qcosine':
                    results_df = results_df[ (results_df['n_train']==1) & (results_df['n_feature']==2)]
                    results_df['Model'] = [ ':'.join( [row['Model'],str(row['n_train']),str(row['n_feature']), row['embed'] ]) for idx, row in results_df.iterrows() ]
                elif method in ['random_forest', 'xgb']:
                    results_df['Model'] = [ ':'.join( [row['Model'],str(row['n_feature']) ]) for idx, row in results_df.iterrows() ]
                else:
                    results_df['Model'] = [ ':'.join( [row['Model'],str(row['d']),str(row['n_train']),str(row['n_swap']),str(row['n_feature']),row['embed'] ]) for idx, row in results_df.iterrows() ]
                    
                all_results = pd.concat( [all_results, results_df] )
                
                if method in ['random_forest', 'xgb']:
                    results_df = results_df.drop([ 'predictions', 'y_test', 'best_params', 'select_features'], axis =1)
                else:
                    results_df = results_df.drop([ 'predictions', 'y_test', 'embed', 'select_features'], axis =1)

        total_results = pd.concat( [total_results, all_results] )
    total_results_full = total_results.reset_index().copy()


    from typing import Any
    import scipy.stats as stats

    metrics = ['Accuracy', 'F1 Score', 'brier']
    methods = ['random_forest', 'xgb', 'qcosine', 'qensemble', 'qensemble_random_unitary']
    methods_cmap = dict(zip( methods[0:2], sns.color_palette('Greys', n_colors=2)))
    methods_cmap.update( dict(zip(methods[2:], sns.color_palette(n_colors=len(methods[2:]))))  )


    total_results_full['key'] = [ '-'.join( [row['Model'], row['dataset']] ) for idx,row in total_results_full.iterrows() ]
    total_results_full['method'] = [ re.sub( ':.*', '', x ) for x in total_results['Model'] ]

    total_results = total_results_full[['Model', 'Accuracy', 'F1 Score', 'brier', 'dataset', 'split', 'num_pred_classes']]
    total_results['split'] = [ x[0] for x in total_results['split'] ]

    total_results['key'] = [ '-'.join( [row['Model'], row['dataset']] ) for idx,row in total_results.iterrows() ]
    total_results['method'] = [ re.sub( ':.*', '', x ) for x in total_results['Model'] ]


    total_results = total_results.groupby ( ['Model', 'key', 'method', 'dataset']).median()
    total_results = total_results.reset_index()
    total_results = total_results[ total_results.method.isin(methods)]


    blob_names = list(set(total_results['dataset']))

    sig_bestmethods: list[Any] = []

    for metric in metrics:
        max_df = []
        for method in methods:
            for bn in blob_names:
                b = total_results[ total_results['dataset'] == bn ]
                m = b[b['method'] == method]
                if len(m) > 0:
                    if method == 'brier':
                        mm = m[ m[metric] == min(m[metric])].sort_values('Model')
                    else:
                        mm = m[ m[metric] == max(m[metric])].sort_values('Model')
                max_df.append(total_results_full[total_results_full['key']==mm['key'].iloc[len(mm)-1]])
        max_df = pd.concat(max_df)

        for method in methods:
            for d in blob_names:
                a = max_df[ (max_df.dataset == d) & (max_df.method == method) ][metric].apply(float)
                b_r = max_df[ (max_df.dataset == d) & (max_df.method == 'random_forest') ][metric].apply(float)
                b_x = max_df[ (max_df.dataset == d) & (max_df.method == 'xgb') ][metric].apply(float)
            
                if metric != 'brier':
                    if len(a) > 0:
                        t_statistic, p_value = stats.ttest_ind(a, b_r, alternative='greater')  # one-tailed test)
                        sig_bestmethods.append( [method, 'random_forest', d, metric, round(float(a.median()),3), round(float(b_r.median()),3), 
                        round(float(a.std()), 3), round(float(b_r.std()), 3), round(float(t_statistic),3), round(float(p_value),3) ] )
                        if p_value < 0.05:                                             
                            print(f"RF: {d} : {method} (n={max_df[ (max_df.dataset == d) & (max_df.method == method) ].shape[0]}) : t={round( t_statistic, 3 )}; p={round( p_value, 3 ) }" )

                        t_statistic, p_value = stats.ttest_ind(a, b_x, alternative='greater')  # one-tailed test)
                        sig_bestmethods.append( [method, 'xgb', d, metric, round(float(a.median()),3), round(float(b_x.median()),3), 
                        round(float(a.std()), 3), round(float(b_x.std()), 3), round(float(t_statistic),3), round(float(p_value),3) ] )
                        if p_value < 0.05:
                            print(f"XGB: {d} : {method} (n={max_df[ (max_df.dataset == d) & (max_df.method == method) ].shape[0]}) : t={round( t_statistic, 3 )}; p={round( p_value, 3 ) }" )
                else:

                    if len(a) > 0:
                        t_statistic, p_value = stats.ttest_ind(a, b_r, alternative='less')  # one-tailed test)
                        sig_bestmethods.append( [method, 'random_forest', d, metric, round(float(a.median()),3), round(float(b_r.median()),3), 
                        round(float(a.std()), 3), round(float(b_r.std()), 3), round(float(t_statistic),3), round(float(p_value),3) ] )
                        if p_value < 0.05:                                             
                            print(f"RF: {d} : {method} (n={max_df[ (max_df.dataset == d) & (max_df.method == method) ].shape[0]}) : t={round( t_statistic, 3 )}; p={round( p_value, 3 ) }" )

                        t_statistic, p_value = stats.ttest_ind(a, b_x, alternative='less')  # one-tailed test)
                        sig_bestmethods.append( [method, 'xgb', d, metric, round(float(a.median()),3), round(float(b_x.median()),3), 
                        round(float(a.std()), 3), round(float(b_x.std()), 3), round(float(t_statistic),3), round(float(p_value),3) ] )
                        if p_value < 0.05:
                            print(f"XGB: {d} : {method} (n={max_df[ (max_df.dataset == d) & (max_df.method == method) ].shape[0]}) : t={round( t_statistic, 3 )}; p={round( p_value, 3 ) }" )

        def reformat_dataset(x):
            xs = [ str(i) for i in x]
            new_x = [ xs[1], '(' + xs[2] + ',' + xs[3] + ')', '(' + xs[3] + ',' + xs[2] + ')' ]
            return ' | '.join( new_x)

        max_df['Blob Config'] = [ reformat_dataset(x) for x in max_df['split'] ]

        max_df = max_df[ ['Blob Config', 'method'] + metrics ]
        max_df = max_df.drop_duplicates()
        
        max_df = max_df.sort_values('method')
        max_df = max_df.sort_values('Blob Config')

        plt.figure(figsize=(7,5))
        sns.barplot( data = max_df, y = 'Blob Config', x = metric, hue = 'method', hue_order=methods, errorbar='se', palette=methods_cmap.values() )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()
        plt.tight_layout()
        plt.savefig( os.path.join( dir_output, 'Blob_max_median_'+ re.sub( '\ ', '_', metric ) + '.pdf' ) )
        plt.show()
        plt.close()

    sig_bestmethods_df = pd.DataFrame(sig_bestmethods, columns = ['Method', 'Baseline', 'Dataset', 'Metric', 'Median method', 'Median baseline', 
        'Std. dev. method', 'Std. dev. baseline', 't statistic', 'p-value'])
    sig_bestmethods_df = sig_bestmethods_df[ sig_bestmethods_df['Method'] != sig_bestmethods_df['Baseline']]
    sig_bestmethods_df.to_csv( os.path.join( dir_output, 'Blobs_best_stats.csv' ), index = False ) 

    return total_results_full, sig_bestmethods_df
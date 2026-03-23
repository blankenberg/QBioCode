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

from numpy import dtype, float64, ndarray
import os, re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier

from sklearn.decomposition import PCA
import umap
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from lazypredict.Supervised import LazyClassifier
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import modeling_random_unitary
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import XGate
from qiskit_ibm_runtime import SamplerV2 as Sampler
warnings.filterwarnings('ignore')

from modeling import *


def calculate_predicted_classes(preds):
    """
    Convert probability predictions to class labels.
    
    Parameters
    ----------
    preds : list of array-like
        Predicted probabilities for each class [p0, p1]
    
    Returns
    -------
    list
        Predicted class labels (0 or 1)
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
    
    Parameters
    ----------
    preds : list of array-like
        Predicted probabilities for each class [p0, p1]
    y_test : array-like
        True labels
    
    Returns
    -------
    float
        Weighted F1 score
    """
    preds = [1 if p[1] > p[0] else 0 for p in preds]
    return f1_score(y_pred=preds, y_true=y_test, average='weighted')


def run_quantum_ensemble(predictions, dataset, method, dataset_name, seed, test_size, file_predictions, ds, n_swaps, n_features,
                         n_trains, n_shots, pca_embed=False, umap_embed=False, device='CPU', instance='', random_unitary=False, select_features=[]):
    """
    Run quantum ensemble experiments across multiple parameter configurations.
    
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
    ds : list of int
        List of ensemble depths (control qubits) to test
    n_swaps : list of int
        List of swap counts to test
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
    device : str, optional
        Device for simulation: 'CPU', 'GPU', or IBM device name (default: 'CPU')
    instance : str, optional
        IBM Quantum instance string (default: '')
    random_unitary : bool, optional
        Use random unitary ensemble variant (default: False)
    select_features : list, optional
        Specific features to select (default: [])
    
    Returns
    -------
    dict
        Updated predictions dictionary with results
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
                        x_tr = normalize_custom_OLD(x_train)
                        y_tr = Y_vector_train[ix]
                        x_ts = normalize_custom_OLD(x_test)
                        qc = cos_classifier_OLD(x_tr, x_ts, y_tr)

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

        
def run_ensemble(d, n_train, seed, n_swap, X_train, X_test,  y_train, y_test, 
                 mode = "pair_sample", n_shots = 8192, selectRandom = True, device = 'CPU',
                 barriers = False, instance = '', resilience_level = 1, optimization_level = 3,
                 nthreads = 20, dynamicDecoupling = False):

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


def normalize_custom_OLD(x, C=1):
    M = x[0] ** 2 + x[1] ** 2
    x_normed = [
        1 / np.sqrt(M * C) * complex(x[0], 0),  # 00
        1 / np.sqrt(M * C) * complex(x[1], 0),  # 01
    ]
    return x_normed


def normalize_custom(x, C=1):
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
    from sklearn.metrics import brier_score_loss, accuracy_score
    labels = label_to_array(y_test)

    predicted_class = np.round(np.asarray(predictions))
    acc = accuracy_score(np.array(predicted_class)[:, 1],
                         np.array(labels)[:, 1])

    p0 = [p[0] for p in predictions]
    p1 = [p[1] for p in predictions]

    brier = brier_score_loss(y_test, p1)

    return acc, brier



def training_set_OLD(X, Y, n=4, seed=123):
    np.random.seed(seed)
    ix_y1 = np.random.choice(np.where(Y == 1)[0], int(n / 2), replace=False)
    ix_y0 = np.random.choice(np.where(Y == 0)[0], int(n / 2), replace=False)

    X_data = np.concatenate([X[ix_y1], X[ix_y0]])
    X_data_new = []

    for i in range(len(X_data)):
        X_data_new.append(normalize_custom_OLD(X_data[i]))

    X_data_new = np.array(X_data_new)
    
    Y_vector = label_to_array(Y)
    Y_data = np.concatenate([Y_vector[ix_y1], Y_vector[ix_y0]])

    return X_data_new, Y_data


def training_set(X, Y, n=4, seed=123, selectRandom = True):
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


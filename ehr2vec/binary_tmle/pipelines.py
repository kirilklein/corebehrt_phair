
from ehr2vec.binary_tmle.simulate_data import simulate_binary_data, compute_ATE_theoretical_from_data
from joblib import Parallel, delayed
import numpy as np


# Helper function for a single bootstrap iteration
def single_bootstrap_iteration(model, n, ate_th_data, estimators, estimator_args={}):
    data = simulate_binary_data(n, **model, seed=None)
    results = {}
    for estimator in estimators:
        args = estimator_args.get(estimator.__name__, {})
        ate, _ = estimator(data, **args)
        results[estimator.__name__] = ate_th_data - ate
    return results

def compute_and_store_results_bootstrap(model_name: str, model: dict,  n: int,  
                              ate_th_data: float, diffs: dict, stds: dict,
                              estimators: list, n_bootstrap: int, estimator_args: dict={}):
    """
    Compute and store diffs and stds for a given model and the list of estimators using bootstrapping.
    We simulate the population n_bootstrap times and compute the difference between the theoretical ATE and the estimated ATE for each estimator.
    Args:
        model_name: name of the model
        model: dictionary of model parameters
        n: number of patients in the simulated dataset
        ate_th_data: theoretical ATE computed from the data
        diffs: dictionary to store the differences
        stds: dictionary to store the standard deviations
        estimators: list of estimators to use
        n_bootstrap: number of bootstraps to use
    """
    results = Parallel(n_jobs=-1)(delayed(single_bootstrap_iteration)(model, n, ate_th_data, estimators, estimator_args) for _ in range(n_bootstrap))
    
    # Initialize temporary storage for differences
    diff_temp = {estimator.__name__: [] for estimator in estimators}

    for result in results:
        for estimator in estimators:
            diff_temp[estimator.__name__].append(result[estimator.__name__])
    
    for estimator in estimators:
        diff_arr = np.array(diff_temp[estimator.__name__])
        diffs[estimator.__name__][model_name].append(np.nanmean(diff_arr))
        stds[estimator.__name__][model_name].append(np.nanstd(diff_arr))

    # Helper function to compute and store results
def compute_and_store_results(model_name: str, model: dict,  n: int,  
                              ate_th_data: float, diffs: dict, stds: dict,
                              estimators: list, estimator_args: dict={}):
    """
    Compute and store diffs and stds for a given model and the list of estimators.
    Args:
        model_name: name of the model
        model: dictionary of model parameters
        n: number of patients in the simulated dataset
        ate_th_data: theoretical ATE computed from the data
        diffs: dictionary to store the differences
        stds: dictionary to store the standard deviations
        estimators: list of estimators to use
    """
    data = simulate_binary_data(n, **model, seed=42)
    
    for estimator in estimators:
        args = estimator_args.get(estimator.__name__, {})
        ate, ate_std = estimator(data, **args)
        diffs[estimator.__name__][model_name].append(ate_th_data - ate)
        stds[estimator.__name__][model_name].append(ate_std)

def convert_to_numpy(diffs: dict, stds: dict)->tuple:
    """Convert nested dictionaries to numpy arrays."""
    for estimator in diffs.keys():
        for model_name in diffs[estimator].keys():
            diffs[estimator][model_name] = np.array(diffs[estimator][model_name])
            stds[estimator][model_name] = np.array(stds[estimator][model_name])
    return diffs, stds

def get_scores_for_models_and_estimators(patient_numbers: int, models: dict, estimators: list, n_bootstraps: int = 1, estimator_args: dict={})->tuple:
    """
    Compute differences between theoretical and estimated ATEs for different models and estimators.
    Args:
        patient_numbers: list of integers, number of patients in each simulated dataset
        models: dictionary of models to simulate data from
        estimators: list of estimators to use
        n_bootstraps: number of bootstraps to use for computing standard deviations, if 1, no bootstrapping is used and the sample standard deviation is computed
    """
    def _init_empty_model_dict(models):
        return {model_name: [] for model_name in models.keys()}
    diffs = {estimator.__name__: _init_empty_model_dict(models) for estimator in estimators}
    stds = {estimator.__name__:  _init_empty_model_dict(models) for estimator in estimators}

    for model_name, model in models.items():
        print('\n',model_name)
        ate_th_data = compute_ATE_theoretical_from_data(simulate_binary_data(10000, **model), model['beta'])
        
        for n in patient_numbers:
            print(f"n={n}", end=' ')
            if n_bootstraps>1:
                compute_and_store_results_bootstrap(model_name, model, n, ate_th_data, diffs, stds, estimators, n_bootstraps, estimator_args)
            else:
                compute_and_store_results(model_name, model, n, ate_th_data, diffs, stds, estimators, estimator_args)
    diffs, stds = convert_to_numpy(diffs, stds)
    return diffs, stds

    
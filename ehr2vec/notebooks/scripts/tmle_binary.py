# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: patbert
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import importlib
import sys

if '..' not in sys.path:
   sys.path.append('..')

from ehr2vec.binary_tmle import estimators, simulate_data, pipelines, vis
importlib.reload(estimators)
importlib.reload(simulate_data)
importlib.reload(pipelines)
importlib.reload(vis)

from ehr2vec.binary_tmle.estimators import IPW_estimator, AIPW_estimator, TMLE_estimator
from ehr2vec.binary_tmle.vis import display_results
from ehr2vec.binary_tmle.simulate_data import (
    compute_ATE_theoretical_from_data, simulate_binary_data, print_basic_stats)
from ehr2vec.binary_tmle.pipelines import get_scores_for_models_and_estimators

# %% [markdown]
# ## Simulate binary data and compute theoretical ATE

# %%
ALPHA = [0, 0.5, -0.5, 0]
BETA = [-1, 2, 1, -1, 0]

data = simulate_binary_data(10000, alpha=ALPHA, beta=BETA)
ate_th_data = compute_ATE_theoretical_from_data(simulate_binary_data(10000, alpha=ALPHA, beta=BETA), BETA)
# ate_th_model = compute_ATE_theoretical_from_model()
print(f"ATE theoretical from data: {round(ate_th_data, 4)}")
# print(f"from model: {round(ate_th_model, 4)}")

# %% [markdown]
# ### Make sure the model is appropriate

# %%
print_basic_stats(simulate_binary_data(100, alpha=ALPHA, beta=BETA))

# %%
# check this models too
models = [{'alpha': [0, 0.5, -0.5, 0], 'beta': [-1, 2, 1, -1, 0]},
          {'alpha': [0.2, 1, -1, 0], 'beta': [-1, 2, 1, -1, 0]},
          {'alpha': [0, 0.5, -0.5, 0], 'beta': [1, 5, .1, -2, 0]},
          ]
for model in models:
    data = simulate_binary_data(100, alpha=model['alpha'], beta=model['beta'])
    ate_th_data = compute_ATE_theoretical_from_data(data, model['beta'])
    print(f"ATE theoretical from data: {round(ate_th_data, 4)}")
    print_basic_stats(data)

# %% [markdown]
# ## IPTW estimator of ATE

# %%
data = simulate_binary_data(1000, alpha=ALPHA, beta=BETA)
ate_iptw, ate_iptw_std = IPW_estimator(data)    
print(f"ATE IPTW: {ate_iptw}")
print("Difference between theoretical and IPTW ATE:", round(ate_th_data - ate_iptw, 4))

# %% [markdown]
# ## TMLE estimator of ATE

# %%
data = simulate_binary_data(1000, alpha=ALPHA, beta=BETA, seed=44)
print('ATE from TMLE', TMLE_estimator(data))
print("Difference between theoretical and TMLE ATE:", round(ate_th_data - TMLE_estimator(data)[0], 4))

# %% [markdown]
# ## Compare estimators for different N and different models

# %%
#patient_numbers = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
estimators = [IPW_estimator, 
              TMLE_estimator, 
              AIPW_estimator]
patient_numbers = [100, 400, 800, 1600, 3200, 6400]# 3200, 6400, 12800]

models = {'base model':{'alpha': [0, 0.5, -0.5, 0], 'beta': [-1, 2, 1, -1, 0]},
          'change alpha':{'alpha': [-0.2, 1, -1, 0], 'beta': [-1, 2, 1, -1, 0]},
          'reverse treatment effect':{'alpha': [0, 0.5, -0.5, 0], 'beta': [-1, -2, 1, -1, 0]},
          'increase treatment effect ':{'alpha': [0, 0.5, -0.5, 0], 'beta': [1, 5, 1, -1, 0]},
          'small treatment effect':{'alpha': [0, 0.5, -0.5, 0], 'beta': [1, .5, 1, -1, 0]},
          'nonlinear treatment':{'alpha': [0, 0.5, -0.5, 1], 'beta': [1, -2, 1, -1, 0]},
          'nonlinear outcome': {'alpha': [0, 0.5, -0.5, 0], 'beta': [1, -2, 1, -1, 1]},
          'both nonlinear': {'alpha': [0, 0.5, -0.5, 1], 'beta': [1, -2, 1, -1, 1]},
          }

diffs, stds = get_scores_for_models_and_estimators(patient_numbers, models, estimators, n_bootstraps=10)

# %%
display_results(patient_numbers, models, diffs, stds, 'simulation_new.png')

# %% [markdown]
# ## tests

# %%
#patient_numbers = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
estimators = [IPW_estimator, 
               TMLE_estimator, ]
              # AIPW_estimator]
patient_numbers = [100, 400, ]#1600, 3200, 6400, 12800]# 3200, 6400, 12800]
estimator_args = {k.__name__: {'cv':True} for k in estimators}
models = {'base model':{'alpha': [0, 0.5, -0.5, 0], 'beta': [-1, 2, 1, -1, 0]},
          #'change alpha':{'alpha': [-0.2, 1, -1, 0], 'beta': [-1, 2, 1, -1, 0]},
        #   'reverse beta 1':{'alpha': [0, 0.5, -0.5, 0], 'beta': [-1, -2, 1, -1, 0]},
        #   'increase beta 1 ':{'alpha': [0, 0.5, -0.5, 0], 'beta': [1, 5, 1, -1, 0]},
           'nonlinear treatment':{'alpha': [0, 0.5, -0.5, 1], 'beta': [1, -2, 1, -1, 0]},
        #   'nonlinear outcome': {'alpha': [0, 0.5, -0.5, 0], 'beta': [1, -2, 1, -1, 1]},
        #   'both nonlinear': {'alpha': [0, 0.5, -0.5, 1], 'beta': [1, -2, 1, -1, 1]},
          }

diffs, stds = get_scores_for_models_and_estimators(patient_numbers, models, estimators, n_bootstraps=1, estimator_args=estimator_args)

# %%
display_results(patient_numbers, models, diffs, stds, 'nonlin_treat_cv.png')

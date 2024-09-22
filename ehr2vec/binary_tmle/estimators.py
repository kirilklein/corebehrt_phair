import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from statsmodels.api import add_constant
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold


def IPW_estimator(data, **kwargs):
    """Estimate the average treatment effect using the inverse probability of treatment weighting (IPTW) method."""
    # Estimate propensity scores
    A = data['A']
    Y = data['Y']
    data = estimate_ps(data, ['X1', 'X2'], **kwargs)
    # Compute weighted outcomes
    ate_ipw = compute_ate_ipw(A, Y, data['propensity'])
    sample_std_iptw = sample_std(ate_ipw)
    # Estimate ATE using IPTW
    return ate_ipw.mean(), sample_std_iptw

def TMLE_estimator(data, **kwargs):
    """Estimate the average treatment effect using the targeted maximum likelihood estimation (TMLE) method."""
    data = tmle_initial_estimates(data, **kwargs)
    epsilon, H = estimate_fluctuation_parameter(data)
    Q_star_1, Q_star_0 = update_Q_star(data, epsilon)
    ate_tmle = (Q_star_1 - Q_star_0).mean()
    data['updated_outcome'] = data['A'] * Q_star_1 + (1 - data['A']) * Q_star_0
    st_error = compute_standard_error(data, H, Q_star_1, Q_star_0, ate_tmle)
    return ate_tmle, st_error

def AIPW_estimator(data, **kwargs):
    """Augmented Inverse Probability of Treatment Weighting (AIPW) estimator as described in
        Glynn, Adam N., and Kevin M. Quinn. 
        "An introduction to the augmented inverse propensity weighted estimator." 
        Political analysis 18.1 (2010): 36-56.
    """
    A = data['A']
    Y = data['Y']
    data = estimate_ps(data, ['X1', 'X2'], **kwargs)
    data = estimate_outcome(data, ['X1', 'X2'], **kwargs)
    g = data['propensity']
    Q1 = data['outcome_1']
    Q0 = data['outcome_0']
    ATE_IPW = compute_ate_ipw(A, Y, g)
    AIPW = ATE_IPW - (A-g)/(g*(1-g)) * ( (1-g)*Q1 + g*Q0)
    ATE_std = sample_std(AIPW)    
    return AIPW.mean(), ATE_std

def compute_ate_ipw(A, Y, ps):
    Y1_weighted = A * Y / ps
    Y0_weighted = (1 - A) * Y / (1 - ps)
    return Y1_weighted - Y0_weighted

def sample_std(estimates):
    """Compute the sample standard error of the estimate."""
    I = estimates - estimates.mean() 
    return np.sqrt((I**2).sum() / len(estimates)**2)

def estimate_ps(data: pd.DataFrame, covariates: list, model: object = None, cv: bool = False, **kwargs):
    """
    Estimate propensity scores for a binary treatment variable using logistic regression.
        data : DataFrame containing the treatment variable 'A' and covariates.
        covariates : List of column names used as predictors in the logistic regression model.
        model (optional): A scikit-learn model implementing `fit` and `predict_proba`. Defaults to `LogisticRegression`.
        cv (optional) : If `True`, perform cross-validation to estimate propensity scores.
    Returns: DataFrame with an additional 'propensity' column containing the estimated propensity scores.
    """
    if model is None:
        model = LogisticRegression(penalty=None)
    if cv:
        data['propensity'] = cross_val_predict(
            model, data[covariates], data['A'], cv=5, method='predict_proba'
        )[:, 1]
    else:
        treatment_model = model.fit(data[covariates], data['A'])
        data['propensity'] = treatment_model.predict_proba(data[covariates])[:, 1]
    
    return data    

def estimate_outcome(data: pd.DataFrame, covariates: list, model: object = None, cv: bool = False, **kwargs):
    """Estimate the outcome and use fitted model to estimate outcomes under counterfactual treatments.."""
    if cv:
        data = cross_validated_outcome_estimation(data, covariates, model, cv_folds=5)
    else:
        if model is None:
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        data = train_and_estimate_outcomes(data, covariates, model)
    
    return data

def cross_validated_outcome_estimation(data: pd.DataFrame, covariates: list, model: object, cv_folds: int):
    skf = StratifiedKFold(n_splits=cv_folds)
    data['outcome'] = np.nan
    data['outcome_1'] = np.nan
    data['outcome_0'] = np.nan

    for train_index, val_index in skf.split(data, data['A']):
        if model is None:
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]
        
        outcome_model = train_model(train_data, covariates, model)
        
        data = estimate_all_outcomes(data, val_data, covariates, outcome_model, val_index)
    
    return data

def train_and_estimate_outcomes(data: pd.DataFrame, covariates: list, model: object):
    outcome_model = train_model(data, covariates, model)
    data['outcome'] = outcome_model.predict_proba(data[covariates + ['A']])[:, 1]
    data = estimate_counterfactual_outcome(data, covariates, outcome_model, 1)
    data = estimate_counterfactual_outcome(data, covariates, outcome_model, 0)
    return data

def train_model(data: pd.DataFrame, covariates: list, model: object):
    X = data[covariates + ['A']]
    y = data['Y']
    return model.fit(X, y)

def estimate_all_outcomes(data: pd.DataFrame, val_data: pd.DataFrame, covariates: list, outcome_model: object, val_index: np.ndarray):
    X_val = val_data[covariates + ['A']]
    data.loc[val_index, 'outcome'] = outcome_model.predict_proba(X_val)[:, 1]
    
    val_data = estimate_counterfactual_outcome(val_data, covariates, outcome_model, 1)
    val_data = estimate_counterfactual_outcome(val_data, covariates, outcome_model, 0)
    
    data.loc[val_index, 'outcome_1'] = val_data['outcome_1']
    data.loc[val_index, 'outcome_0'] = val_data['outcome_0']
    
    return data

def estimate_counterfactual_outcome(data: pd.DataFrame, covariates: list, fitted_model: object, treatment: int):
    """Estimate the counterfactual outcome e.g., all patients are treated or all patients are untreated."""
    data = data.copy()
    data['A_temp'] = treatment
    X = data[covariates + ['A_temp']].rename(columns={'A_temp': 'A'})
    data[f'outcome_{treatment}'] = fitted_model.predict_proba(X)[:, 1]
    del data['A_temp']
    return data

def estimate_fluctuation_parameter(data: pd.DataFrame)->float:
    """"Estimate the fluctuation parameter epsilon using a logistic regression model."""
    ps = data['propensity']
    A = data['A']
    H = A / ps - (1 - A) / (1 - ps)  
    
    # Use logit of the current outcome as offset
    offset = logit(data['outcome'])
    
    # Fit the model with offset
    model = GLM(data['Y'], add_constant(H), family=Binomial(), offset=offset).fit()
    return model.params[0], H


def tmle_initial_estimates(data, **kwargs):
    data = estimate_ps(data, ['X1', 'X2'], **kwargs)
    data = estimate_outcome(data, ['X1', 'X2'], **kwargs)
    return data

def update_Q_star(data, epsilon):
    H_1 = 1 / data['propensity']
    Q_star_1 = expit(logit(data['outcome_1']) + epsilon * H_1)
    
    H_0 = 1 / (1 - data['propensity'])
    Q_star_0 = expit(logit(data['outcome_0']) - epsilon * H_0)
    
    return Q_star_1, Q_star_0

def compute_standard_error(data, H, Q_star_1, Q_star_0, ate_tmle):
    """Compute the standard error of the average treatment effect estimate."""
    y_diff = data['Y'] - data['updated_outcome']
    IF = y_diff * H + Q_star_1 - Q_star_0 - ate_tmle
    var_IF = (IF ** 2).mean()
    return np.sqrt(var_IF/len(data))


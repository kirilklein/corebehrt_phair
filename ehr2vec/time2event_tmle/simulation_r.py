"""Simulation from R code from appendix in the paper 
    'One-step targeted maximum likelihood estimation for time-to-event outcomes' 
    !The formulas provided in the paper differ from that using a log-normal for T"""
import numpy as np
# Load the data
X_MIN = 0
X_MAX = 1.5
A_OFFSET = 0.4
A_MULT = 0.5
A_MIN_X = 0.75
T_OFFSET = 1
T_X_MULT = 0.7
T_A_MULT = 0.8
C_OFFSET = 1
C_MULT = 0.5
C_SCALE = 75

def simulate_covariates(n, min=X_MIN, max=X_MAX):
    return np.random.uniform(min, max, n)

def simulate_treatment(n, X, 
                       offset=A_OFFSET, 
                       multiplier=A_MULT, 
                       min_X=A_MIN_X):
    prob = offset + multiplier * (X > min_X)
    return np.random.binomial(1, prob, n)

def compute_exponential_rate(X, A, rate_offset=T_OFFSET, rate_X_mult=T_X_MULT, rate_A_mult=T_A_MULT):
    return rate_offset + rate_X_mult * X**2 - rate_A_mult * A

def simulate_outcome_time(n, X, A, 
                          rate_offset=T_OFFSET,
                          rate_X_mult=T_X_MULT,
                           rate_A_mult=T_A_MULT):
    rate = compute_exponential_rate(X, A, rate_offset, rate_X_mult, rate_A_mult)
    Trexp = np.random.exponential(1/rate, n)
    return np.round(Trexp * 2)

def simulate_censoring(n, X, 
                       shape_offset=C_OFFSET,  
                       shape_multiplier=C_MULT, 
                       scale=C_SCALE):
    shape = shape_offset + shape_multiplier * X
    Cweib = np.random.weibull(shape, n) * scale
    return np.round(Cweib * 2)

def theoretical_survival_function(t, X, A, 
                                  rate_offset=T_OFFSET, 
                                  rate_X_mult=T_X_MULT, 
                                  rate_A_mult=T_A_MULT):
    # Compute the rate parameter lambda
    rate = compute_exponential_rate(X, A, rate_offset, rate_X_mult, rate_A_mult)
    rate = np.asarray(rate)
    # Compute the survival probability
    survival_prob = np.exp(-rate[:, np.newaxis] * t / 2)
    # Average the survival probabilities over the population
    return np.mean(survival_prob, axis=0)

def simulate_data(n, A=None, 
                  covariate_kwargs={},
                  treatment_kwargs={}, 
                  censoring_kwargs={}, 
                  outcome_kwargs={}):
    """
    Simulate Data using model from appendix (R code) in the paper
    One-step targeted maximum likelihood estimation for time-to-event outcomes
    by Mark J. van der Laan
    """
    np.random.seed(42)
    # Covariates and Treatment
    X = simulate_covariates(n, **covariate_kwargs)
    if A is None:
        A = simulate_treatment(n, X, **treatment_kwargs)
    else:
        assert (A == 0) | (A == 1)
        A = np.ones(n) * A
    # Time to event
    T = simulate_outcome_time(n, X, A, **outcome_kwargs)
    # Censoring time
    C = simulate_censoring(n, X, **censoring_kwargs)
    # Observed time and event indicator
    T_observed = np.minimum(T, C)
    Y = (T <= C).astype(int)
    data = {'X': X, 'A': A, 'T': T, 'C': C, 'T_observed': T_observed, 'Y': Y}
    return data
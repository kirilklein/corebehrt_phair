import numpy as np
from scipy.stats import norm

# Load the data
def simulate_covariates(n, min=0, max=1.5):
    return np.random.uniform(min, max, n)

def simulate_treatment(n, X, offset=0.4, multiplier=0.5, min_X=0.75):
    """Parameters used from vdl paper."""
    return np.random.binomial(1, offset + multiplier * (X > min_X), n)

def simulate_outcome_time(n, X, A, sigma=0.01):
    mu = 2 - X + A
    return np.random.lognormal(mu, sigma, n)

def simulate_censoring(n, X, shape_offset=1,  shape_multiplier=0.5, scale=75):
    shape = shape_offset + shape_multiplier * X
    return np.random.weibull(shape, n) * scale

def calculate_theoretical_survival_curve(t, X, A, sigma=0.01):
    """
    Calculate the true survival probability at times t for each individual, 
    given a log-normal distribution with mean mu and standard deviation sigma.
    Returns a 2D array where each row corresponds to an individual's survival curve.
    
    Parameters:
    - t: An array of times at which to calculate the survival probability.
    - X: An array of covariates for each individual.
    - A: An array of treatment effects for each individual.
    - sigma: Standard deviation of the log-normal distribution (default 0.01).
    
    Returns:
    - A 2D array of survival probabilities with dimensions len(X) x len(t).
    """
    mu = 2 - X + A
    mu = mu[:, np.newaxis]  # Convert mu to a column vector for broadcasting
    survival_curves = 1 - norm.cdf((np.log(t) - mu) / sigma)
    return np.mean(survival_curves, axis=0)

def simulate_data(n, A=None, 
                  covariate_kwargs={},
                  treatment_kwargs={}, 
                  censoring_kwargs={}, 
                  outcome_kwargs={}):
    """
    Simulate Data using model from 
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
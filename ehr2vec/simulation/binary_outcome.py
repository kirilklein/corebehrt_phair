from scipy.special import expit as sigmoid
from scipy.stats import bernoulli


def tbehrt(ps, exposure, a: float, b: float, c: float):
    """Simulate binary outcome as presented in
    Rao, Shishir, et al.
    "Targeted-BEHRT: deep learning for observational causal inference on longitudinal electronic health records."
    IEEE Transactions on Neural Networks and Learning Systems 35.4 (2022): 5027-5038.
    ."""
    probability = sigmoid(a * exposure + b * (ps + c))
    return bernoulli.rvs(probability)

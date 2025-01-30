"""
This script contains the implementation of the CV-TMLE algorithm for estimating
the causal effect of a treatment on a binary outcome. The intended use is to compare
with estimates from the CausalEstimate package.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


class CVTMLE:
    def __init__(self, q_t0=None, q_t1=None, g=None, t=None, y=None):
        """
        Targeted BEHRT implementation
        CVTMLE as conceived by Levi, 2018:
        Levy, Jonathan. "An easy implementation of CV-TMLE." arXiv preprint arXiv:1811.04573 (2018).

        :param q_t0: initial estimate with control exposure
        :param q_t1: initial estimate with treatment/non-control exposure
        :param g: prediction of propensity score
        :param t: treatment label
        :param y: factual outcome
        :param truncate_level: truncation for propensity scores (0.05 default means that only patients with estimates between 0.05 and 0.95 will be considered)
        """

        self.q_t0 = q_t0
        self.q_t1 = q_t1
        self.g = g
        self.t = t
        self.y = y

    def _perturbed_model_bin_outcome(self, q_t0, q_t1, g, t, eps):
        """
        Helper for psi_tmle_bin_outcome

        Returns q_\eps (t,x) and the h term
        (i.e., value of perturbed predictor at t, eps, x; where q_t0, q_t1, g are all evaluated at x
        """
        h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
        full_lq = (1.0 - t) * logit(q_t0) + t * logit(
            q_t1
        )  # logit predictions from unperturbed model
        logit_perturb = full_lq + eps * h
        return expit(logit_perturb), h

    def run_tmle_binary(self):
        """
        This is for CV-TMLE on binary outcomes yielding risk ratio with 95% CI. Read Levi et al for methodological details.
        Influence curves coded from Gruber S, van der Laan, MJ. (2011).

        """

        print("running CV-TMLE for binary outcomes...")
        q_t0, q_t1, g, t, y = (
            np.copy(self.q_t0),
            np.copy(self.q_t1),
            np.copy(self.g),
            np.copy(self.t),
            np.copy(self.y),
        )

        eps_hat = minimize(
            lambda eps: self.cross_entropy(
                y, self._perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps)[0]
            ),
            0.0,
            method="Nelder-Mead",
        )
        eps_hat = eps_hat.x[0]

        def q1(t_cf):
            return self._perturbed_model_bin_outcome(q_t0, q_t1, g, t_cf, eps_hat)

        qall = ((1.0 - t) * (q_t0)) + (
            t * (q_t1)
        )  # full predictions from unperturbed model

        qq1, h1 = q1(np.ones_like(t))
        qq0, h0 = q1(np.zeros_like(t))
        rr = np.mean(qq1) - np.mean(qq0)

        ic = 1 / np.mean(qq1) * (h1 * (y - qall) + qq1 - np.mean(qq1)) - (
            1 / np.mean(qq0)
        ) * (-1 * h0 * (y - qall) + qq0 - np.mean(qq0))
        psi_tmle_std = 1.96 * np.sqrt(np.var(ic) / (t.shape[0]))

        return [
            rr,
            np.exp(np.log(rr) - psi_tmle_std),
            np.exp(np.log(rr) + psi_tmle_std),
        ]

    def cross_entropy(self, y, p):
        return -np.mean((y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

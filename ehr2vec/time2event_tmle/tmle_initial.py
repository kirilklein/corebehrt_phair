from sklearn.linear_model import LogisticRegression
import pandas as pd
from scipy.special import expit, logit
import numpy as np

"""Here we follow the steps from the supplementary material of ONe step TMLE for time2event outcomes."""


def compute_clever_covariate(patient, cls_patient, t, k, A):
    """
    Compute clever covariate, see
    Application of Time-to-Event Methods in the Assessment of Safety in Clinical Trials, Moore et al. 2009
    page 5+6

    Args:
        patient (pd.Series): The patient data.
        cls_patient (pd.DataFrame): The temporal patient data
        t (int): The time point to estimate the survival probability.
        k (int): Second time index running to T_observed.
        A (int): Which curve to estimate, 0 or 1.
    """
    if patient["A"] != A:
        return 0

    if k > t:
        return 0
    if patient["A"] == 1:
        g = patient["propensity"]
    else:
        g = 1 - patient["propensity"]

    C_surv_prob_k_ = cls_patient.loc[cls_patient["t"] == k, "S_C"].values[0]
    E_surv_t = cls_patient.loc[cls_patient["t"] == t, "S_E"].values[0]
    E_surv_k = cls_patient.loc[cls_patient["t"] == k, "S_E"].values[0]

    clever_covariate = -E_surv_t / (E_surv_k * C_surv_prob_k_ * g)
    return float(clever_covariate)


def compute_lambda_E(patient, t, failure_model, A):
    lambda_input = pd.DataFrame({"t": [t], "A": [A], "X": [float(patient["X"])]})
    lambda_E = failure_model.predict_proba(lambda_input)[:, 1]
    return float(lambda_E)


def check_if_event_happened_for_positive_patient(patient, k):
    if (patient["T_observed"] == k) and (patient["Y"] == 1):
        return 1
    return 0


def estimate_survival_probability_for_all_t(
    cls_data: pd.DataFrame,
    name: str = None,
    failure_model: object = None,
    failure_column: str = None,
) -> pd.DataFrame:
    """
    Estimate the survival probability for each entry in the dataframe.
    Args:
        data (pd.DataFrame): The input data containing t, A, X.
        model (object): The fitted censoring/failure model.

    Returns:
        cls_data: data with an additional columns 'S_{name}' surival probability and 'F_{name}' failure_probability.
    """
    cls_data = cls_data.copy()
    if failure_model is None:
        if failure_column is None:
            raise ValueError("Please provide a failure column name.")
    else:
        if name is None:
            raise ValueError("Please provide a column name e.g. E or C.")
        failure_column = f"F_{name}"
        cls_data.loc[:, failure_column] = failure_model.predict_proba(
            cls_data[["t", "A", "X"]]
        )[:, 1]
    cls_data.loc[:, f"S_{name}"] = (
        (1 - cls_data[failure_column]).groupby(cls_data["pid"]).transform("cumprod")
    )
    return cls_data


def compute_TMLE(
    data: pd.DataFrame,
    cls_data: pd.DataFrame,
    failure_model: object,
    censoring_model: object,
):
    # Estimate Curve TMLE for the treated A = 1
    TREATMENT = 1
    cls_data = estimate_survival_probability_for_all_t(
        cls_data,
        "E",
        failure_model,
    )
    cls_data = estimate_survival_probability_for_all_t(
        cls_data,
        "C",
        censoring_model,
    )

    Psi = []
    for t in range(1, 6):  # set to t_max for the full range
        print("t", t)
        # print('t', t)
        j = 0  # iterations
        epsilon = 1
        while (epsilon > 1e-3) and (j < 20):
            h_arr = []
            N_arr = []
            lambda_E_arr = []
            start_inds = []
            indices = []
            #  we will concatenate along i and k to form a vector which we run regression on
            for i, patient in data.iterrows():  #
                # print('i', i, end=' ')
                # print('T_observed', patient['T_observed'], end=' ')

                cls_patient = cls_data[cls_data.pid == i]
                # if i>10: # for testing purposes
                #    break
                start_inds.append(len(h_arr))
                if t > patient["T_observed"]:
                    continue
                for k in range(0, int(patient["T_observed"]) + 1):
                    # print('k', k, end=' ')

                    N = check_if_event_happened_for_positive_patient(patient, k)
                    # ! TODO: we can add ps to cls_data or simply pass the ps model for consistency.
                    ht = compute_clever_covariate(
                        patient, cls_patient, t, k, A=TREATMENT
                    )
                    lambda_E = compute_lambda_E(patient, k, failure_model, A=TREATMENT)

                    h_arr.append(ht)
                    N_arr.append(N)
                    lambda_E_arr.append(lambda_E)
                    indices.append(int(cls_patient[cls_patient["t"] == k].index[0]))

            h_arr = np.array(h_arr)
            N_arr = np.array(N_arr)
            lambda_E_arr = np.array(lambda_E_arr)
            indices = np.array(indices)

            # get epsilon by running a logisitc regression on logit(N) = logit(lambda) + epsilon*h
            logit_lambda = logit(
                lambda_E_arr.clip(1e-10, 1 - 1e-10)
            )  # Avoid log(0) by clipping values
            # Prepare the data for logistic regression
            X = pd.DataFrame({"logit_lambda": logit_lambda, "h": h_arr})

            logistic_regression = LogisticRegression(
                fit_intercept=False
            )  # No intercept as we model logit(N) = logit(lambda) + epsilon*h
            logistic_regression.fit(X, N_arr)

            # Get the epsilon coefficient
            epsilon = logistic_regression.coef_[0][1]
            new_lambda = expit(logit_lambda + epsilon * h_arr)

            cls_data.loc[indices, "F_E"] = new_lambda
            cls_data = estimate_survival_probability_for_all_t(
                cls_data, "E", failure_column="F_E"
            )

            print("epsilon:", round(epsilon, 3), end=" ")

            j += 1  # increase the iteration counter
            Psi.append(cls_data.loc[cls_data["t"] == t, "S_E"].mean())

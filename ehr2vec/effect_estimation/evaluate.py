import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from ehr2vec.common.default_args import (
    CF_CONTROL_COL,
    CF_TREATED_COL,
    OUTCOME_COL,
    OUTCOME_CONTROL_COL,
    OUTCOME_PROBABILITY_COL,
    OUTCOME_TREATED_COL,
    PS_COL,
    TREATMENT_COL,
)


def evaluate_binary_outcome_models(df):
    """
    Given a DataFrame 'df' with columns:
        - 'treatment': actual treatment indicator A (0 or 1)
        - 'ps': predicted propensity scores p(A=1|X)   [already handled elsewhere]
        - 'outcome': observed outcome (binary) under the realized treatment
        - 'Y_hat': predicted probability for the realized treatment (optional)
        - 'Y0_hat': predicted prob of outcome if A=0
        - 'Y1_hat': predicted prob of outcome if A=1
        - 'Y0': true outcome (binary) if A=0
        - 'Y1': true outcome (binary) if A=1
    This function computes classification metrics (AUC, average precision, etc.)
    separately for:
        - Y0 vs. Y0_hat (the "control" potential outcome)
        - Y1 vs. Y1_hat (the "treatment" potential outcome)
        - on-policy (observed outcome vs. Y_hat), if 'Y_hat' is present
    Returns a dictionary of metric dictionaries.
    """

    metrics = {}

    # Helper function to compute a set of binary classification metrics
    def compute_classif_metrics(y_true, y_pred_proba, label):
        """
        y_true: array of {0,1}
        y_pred_proba: array of predicted probabilities in [0,1]
        label: string name for logging
        """
        # We'll collect metrics in a dict
        out = {}

        # 1) ROC AUC
        #    (catch ValueError if y_true is all 0 or all 1)
        try:
            out["auc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            out["auc"] = np.nan

        # 4) Brier score
        #    (mean squared difference between predicted probability and actual outcome)
        out["brier"] = brier_score_loss(y_true, y_pred_proba)

        return out

    metrics[PS_COL] = compute_classif_metrics(
        df[TREATMENT_COL].values, df[PS_COL].values, label="ps"
    )
    # Evaluate the "control" potential outcome (Y0 vs Y0_hat)
    y0_true = df[OUTCOME_CONTROL_COL].values
    y0_pred = df[CF_CONTROL_COL].values
    metrics["control"] = compute_classif_metrics(
        y0_true, y0_pred, label=OUTCOME_CONTROL_COL
    )

    # Evaluate the "treatment" potential outcome (Y1 vs Y1_hat)
    y1_true = df[OUTCOME_TREATED_COL].values
    y1_pred = df[CF_TREATED_COL].values
    metrics["treatment"] = compute_classif_metrics(
        y1_true, y1_pred, label=OUTCOME_TREATED_COL
    )

    # (Optional) Evaluate the on-policy predictions vs. observed outcome
    if OUTCOME_PROBABILITY_COL in df.columns:
        y_obs_true = df[OUTCOME_COL].values
        y_obs_pred = df[OUTCOME_PROBABILITY_COL].values
        metrics["on_policy"] = compute_classif_metrics(
            y_obs_true, y_obs_pred, label="on_policy"
        )
    else:
        metrics["on_policy"] = None

    # Print a summary
    def pretty_print(label, m):
        if m is None:
            print(f"{label}: N/A")
            return
        print(f"{label}:")
        print(
            f"  AUC:             {m['auc']:.3f}"
            if not np.isnan(m["auc"])
            else "  AUC: N/A"
        )
        print(f"  BrierScore:      {m['brier']:.3f}")
        print()

    pretty_print("PS", metrics[PS_COL])
    print("=== Binary Outcome Model Evaluation ===")
    pretty_print("Control Potential Outcome (Y0)", metrics["control"])
    pretty_print("Treatment Potential Outcome (Y1)", metrics["treatment"])
    pretty_print("On-Policy Outcome (Observed)", metrics["on_policy"])
    print("========================================")

    return metrics

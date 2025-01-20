DEFAULT_BLOBSTORE = "CINF"
DEFAULT_N_SPLITS = 5  # You can change this to desired value
DEFAULT_VAL_SPLIT = 0.2

TREATMENT_COL = "treatment"
OUTCOME_COL = "outcome"
PS_COL = "ps"

CF_TREATED_COL = "Y1_hat"
CF_CONTROL_COL = "Y0_hat"
OUTCOME_PROBABILITY_COL = "Y_hat"

ORG_PID_COL = "pid"
PID_COL = "PID"

PROBA_COL = "proba"
TARGET_COL = "target"

TIMESTAMP_COL = "TIMESTAMP"

OUTCOME_TREATED_COL = "Y1"
OUTCOME_CONTROL_COL = "Y0"


XGBOOST_RANDOM_SEARCH_PARAM_GRID = {
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [1e-3, 1e-2, 1e-1],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "min_child_weight": [1, 2, 3],
    "gamma": [0, 0.1, 0.3, 0.5],
}

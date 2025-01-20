import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


def tune_xgboost_hyperparams(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    n_iter: int = 10,
    cv: int = 5,
) -> dict:
    """
    Search over hyperparameters using inner CV on the training set only.

    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        param_grid (dict): Hyperparameter grid, e.g.:
            {
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
                ...
            }
        n_iter (int): Number of iterations for random search.

    Returns:
        dict: The best hyperparams found.
    """
    # Create a base XGBClassifier
    xgb_estimator = xgb.XGBClassifier(
        tree_method="hist", eval_metric="logloss"  # faster on GPUs/large datasets
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",  # or your metric of choice
        cv=cv,  # internal cross-validation splits
        verbose=1,
        random_state=42,
        n_jobs=-1,  # use multiple CPUs if available
    )

    random_search.fit(x_train, y_train)
    return random_search.best_params_

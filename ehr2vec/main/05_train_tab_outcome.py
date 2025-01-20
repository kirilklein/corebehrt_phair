"""
This script is used to train a tabular model on the patient vectors produced by the exposure model.
The goal is to restrict the features of the outcome model to the features of the exposure model (covariates).
"""

import logging
import os
from os.path import abspath, dirname, join
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from tqdm import tqdm

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.cli import override_config_from_cli
from ehr2vec.common.default_args import (
    DEFAULT_BLOBSTORE,
    ORG_PID_COL,
    PID_COL,
    PROBA_COL,
    TARGET_COL,
    XGBOOST_RANDOM_SEARCH_PARAM_GRID,
)
from ehr2vec.common.initialize import Initializer
from ehr2vec.common.loader import load_config
from ehr2vec.common.setup import DirectoryPreparer, get_args
from ehr2vec.downstream_tasks.outcomes import OutcomeHandler
from ehr2vec.downstream_tasks.tabular import load_tabular_data
from ehr2vec.downstream_tasks.tabular_training import tune_xgboost_hyperparams
from ehr2vec.evaluation.calibration import calibrate_data, train_calibrator

DEFAULT_CONFIG_NAME = "example_configs/05_train_tab_outcome.yaml"
logger = logging.getLogger(__name__)


def get_binary_outcomes(
    finetune_folder: str, outcome_csv_path: str, n_hours_start_follow_up: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Load and process binary outcome data.

    Args:
        finetune_folder (str): Path to folder containing finetuning results with
            'index_dates.pt' and 'pids.pt'.
        outcome_csv_path (str): Path to CSV file containing outcome data.
        n_hours_start_follow_up (int): Number of hours after index date to start
            follow-up period.

    Returns:
        Tuple[pd.Series, pd.Series]:
            - A Series of outcome indicators (1 for outcome, 0 for no outcome),
              indexed by patient ID.
            - A Series of patient IDs that had outcomes before the follow-up start.
    """
    # Load index dates with weights_only=False since numpy dtype is needed
    index_dates = torch.load(
        join(finetune_folder, "index_dates.pt"), weights_only=False
    )
    pids = torch.load(join(finetune_folder, "pids.pt"), weights_only=True)
    index_dates_series = pd.Series(index_dates, index=pids)
    index_dates_series.index.name = PID_COL

    outcome_df = pd.read_csv(outcome_csv_path)
    outcomes, outcome_pre_followup_pids = OutcomeHandler.get_first_outcome_in_follow_up(
        outcome_df, index_dates_series, n_hours_start_follow_up
    )

    # Convert found outcomes to 1
    outcomes = pd.Series(1, index=outcomes.index)
    outcomes.index.name = PID_COL

    # Create missing outcomes (0) for PIDs without events
    missing_pids = set(index_dates_series.index) - set(outcomes.index)
    if missing_pids:
        missing_outcomes = pd.Series(0, index=list(missing_pids))
        missing_outcomes.index.name = PID_COL
        outcomes = pd.concat([outcomes, missing_outcomes])

    return outcomes, outcome_pre_followup_pids


def train_xgboost_on_fold(
    config, fold_dir: str, outcomes: pd.Series, exposures: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train an XGBoost model on a single fold and produce both raw and calibrated
    predictions for the validation set.

    Args:
        config: Configuration object.
        fold_dir (str): Path to the specific fold directory.
        outcomes (pd.Series): Outcome values indexed by PID.
        exposures (pd.Series): Exposure values indexed by PID.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Uncalibrated validation predictions DataFrame.
            - Calibrated validation predictions DataFrame.
    """
    # Load data
    train_data = load_tabular_data(fold_dir, "train", outcomes, exposures)
    val_data = load_tabular_data(fold_dir, "val", outcomes, exposures)

    # Prepare features
    x_train = train_data.vectors.numpy()
    x_val = val_data.vectors.numpy()

    if train_data.exposures is not None:
        x_train = np.column_stack([x_train, train_data.exposures])
        x_val = np.column_stack([x_val, val_data.exposures])

    # Tune hyperparameters
    param_grid = config.model.random_search.get(
        "param_grid", XGBOOST_RANDOM_SEARCH_PARAM_GRID
    )
    n_iter = config.model.random_search.n_iter

    best_params = tune_xgboost_hyperparams(
        x_train,
        train_data.targets,
        param_grid,
        n_iter,
        cv=config.model.random_search.get("cv", 5),
    )
    logger.info(f"Best params for fold {fold_dir}: {best_params}")

    # Initialize and train XGBoost model
    model = xgb.XGBClassifier(
        **best_params,
        verbosity=config.model.get("verbosity", 1),
        early_stopping_rounds=config.model.get("early_stopping_rounds", 10),
        random_state=config.model.get("random_state", 42),
    )
    model.fit(x_train, train_data.targets, eval_set=[(x_val, val_data.targets)])

    # Predictions
    train_probs = model.predict_proba(x_train)[:, 1]
    val_probs = model.predict_proba(x_val)[:, 1]

    # Wrap train predictions in a DataFrame
    train_df = pd.DataFrame(
        {
            PID_COL: train_data.pids,
            PROBA_COL: train_probs,
            TARGET_COL: train_data.targets,
        }
    )

    # Train calibrator
    calibrator = train_calibrator(train_df, config.calibration)

    # Validation DataFrame
    val_df = pd.DataFrame(
        {
            PID_COL: val_data.pids,
            PROBA_COL: val_probs,
            TARGET_COL: val_data.targets,
        }
    )

    # Calibrate validation probabilities
    calibrated_val_df = calibrate_data(calibrator, val_df)

    return val_df, calibrated_val_df


def process_outcomes(config, xgboost_output_dir):
    """Process and save outcomes data."""
    outcomes, outcome_pre_followup_pids = get_binary_outcomes(
        finetune_folder=config.paths.model_path,
        outcome_csv_path=config.paths.outcome,
        n_hours_start_follow_up=config.outcome.n_hours_start_follow_up,
    )

    torch.save(
        outcome_pre_followup_pids,
        join(xgboost_output_dir, "outcome_pre_followup_pids.pt"),
    )
    outcomes.to_csv(join(xgboost_output_dir, "binary_outcomes.csv"), index=False)
    return outcomes


def load_exposures(config):
    """Load and process exposures data."""
    exposures_df = pd.read_csv(
        join(config.paths.model_path, "predictions_and_targets.csv")
    )
    exposures_df = exposures_df.rename(columns={ORG_PID_COL: PID_COL})[
        [PID_COL, TARGET_COL]
    ]
    exposures_df = exposures_df.set_index(PID_COL)
    return exposures_df[TARGET_COL]


def save_results(results_df, output_dir, filename):
    """Save results to CSV with proper column renaming."""
    results_df = results_df.rename(columns={PID_COL: ORG_PID_COL})
    results_df.to_csv(join(output_dir, filename), index=False)


def setup_environment():
    args = get_args(DEFAULT_CONFIG_NAME)
    config_path = join(dirname(dirname(abspath(__file__))), args.config_path)
    config = load_config(config_path)
    override_config_from_cli(config)

    # Initialize config and environment
    config, _, mount_context = Initializer.initialize_configuration_tabular(
        config, dataset_name=config.get("project", DEFAULT_BLOBSTORE)
    )
    return config, mount_context


def main():
    # Initialize configuration and environment
    config, mount_context = setup_environment()
    # Prepare run folder and logger
    logger, xgboost_output_dir = DirectoryPreparer.setup_run_folder(config)
    config.save_to_yaml(join(xgboost_output_dir, "xgboost_config.yaml"))

    # Process outcomes and exposures
    outcomes = process_outcomes(config, xgboost_output_dir)
    exposures = load_exposures(config)

    # Collect fold directories
    finetune_folder = config.paths.model_path
    fold_dirs = [d for d in os.listdir(finetune_folder) if d.startswith("fold_")]
    num_folds = len(fold_dirs)

    # Train for each fold
    uncalibrated_results = []
    calibrated_results = []

    for fold_dir_name in tqdm(fold_dirs, desc="Training folds"):
        fold_index = int(fold_dir_name.split("_")[1])
        logger.info(f"Training fold {fold_index}/{num_folds}")
        fold_dir_path = join(finetune_folder, fold_dir_name)

        val_df, cal_val_df = train_xgboost_on_fold(
            config, fold_dir_path, outcomes, exposures
        )
        uncalibrated_results.append(val_df)
        calibrated_results.append(cal_val_df)

    # Save results
    save_results(
        pd.concat(uncalibrated_results),
        xgboost_output_dir,
        "predictions_and_targets.csv",
    )
    save_results(
        pd.concat(calibrated_results),
        xgboost_output_dir,
        f"predictions_and_targets_calibrated_{config.calibration}.csv",
    )

    # Save to Azure if needed
    if config.env == "azure":
        save_to_blobstore(
            local_path=config.paths.run_name,
            remote_path=join(
                config.get("project", DEFAULT_BLOBSTORE), config.paths.run_name
            ),
        )
        mount_context.stop()

    logger.info("Done")


if __name__ == "__main__":
    main()

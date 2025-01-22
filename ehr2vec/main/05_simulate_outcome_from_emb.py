"""This script uses a model to simulate a binary outcome"""

import os
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.checks import check_pids
from ehr2vec.common.cli import override_config_from_cli
from ehr2vec.common.config import Config
from ehr2vec.common.default_args import (
    DEFAULT_BLOBSTORE,
    INDEX_DATE,
    OUTCOME_CONTROL_COL,
    OUTCOME_TREATED_COL,
    PID_COL,
    TARGET_COL,
    TIMESTAMP_COL,
)
from ehr2vec.common.loader import load_binary_outcomes, load_config, load_index_dates
from ehr2vec.common.setup import (
    get_args,
    initialize_configuration_effect_estimation,
    setup_logger,
)
from ehr2vec.simulation.longitudinal_outcome import simulate_abspos_from_binary_outcome
from ehr2vec.simulation.save import (
    save_counterfactual_probas_and_targets,
    save_probas_and_targets,
)
from ehr2vec.simulation.binary_outcome import simulate_outcome_from_embeddings
import torch

DEFAULT_CONFIG_NAME = "example_configs/05_simulate_outcome_from_emb.yaml"


args = get_args(DEFAULT_CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)


def load_validation_patient_embeddings(model_dir_path: str) -> pd.DataFrame:
    """Load patient embeddings and IDs from validation folds and combine them.

    Args:
        model_dir_path: Path to the directory containing fold subdirectories

    Returns:
        DataFrame containing patient embeddings with patient ID as index
    """
    validation_fold_dfs = []

    # Iterate through fold directories
    for fold_name in sorted(os.listdir(model_dir_path)):
        if not fold_name.startswith("fold_"):
            continue

        fold_path = join(model_dir_path, fold_name)
        # Load validation patient IDs and embeddings
        patient_ids = torch.load(join(fold_path, "val_pids.pt"))
        patient_embeddings = torch.load(join(fold_path, "val_patient_vectors.pt"))

        # Create dataframe with embeddings and patient IDs
        fold_embeddings_df = pd.DataFrame(patient_embeddings)
        fold_embeddings_df[PID_COL] = patient_ids
        validation_fold_dfs.append(fold_embeddings_df)

    # Concatenate all validation folds
    combined_embeddings_df = pd.concat(validation_fold_dfs)
    return combined_embeddings_df


def main(config_path: str) -> None:

    # 1) Load and set up configuration
    cfg: Config = load_config(config_path)
    output_path = cfg.path.output_path
    override_config_from_cli(cfg)
    cfg, _, mount_context, _ = initialize_configuration_effect_estimation(
        cfg, dataset_name=cfg.get("project", DEFAULT_BLOBSTORE)
    )

    # 2) Prepare output folder and logging
    simulation_folder = cfg.paths.output_path
    if cfg.env == "azure":
        simulation_folder = join(simulation_folder, cfg.paths.run_name)

    os.makedirs(simulation_folder, exist_ok=True)
    logger = setup_logger(simulation_folder)

    # Save the final, possibly CLI-overridden, config
    cfg.save_to_yaml(join(simulation_folder, "simulation_config.yaml"))

    # 3) Load model predictions and index dates
    ps_model_path = cfg.paths.ps_model_path
    logger.info(
        "Load outcomes, index dates, and patient embeddings from %s", ps_model_path
    )
    df_treatment = load_binary_outcomes(ps_model_path)
    df_index_dates = load_index_dates(ps_model_path)
    df_patient_vectors = load_validation_patient_embeddings(ps_model_path)
    check_pids(df_patient_vectors, df_treatment)

    # 4) Merge exposure status, probas, and index dates
    logger.info("Merge predictions and index dates on %s", PID_COL)
    # Note: TARGET_COL here represents actual treatment assignment (0 or 1).
    df = pd.merge(df_patient_vectors, df_index_dates, on=PID_COL)

    df = pd.merge(df, df_treatment, on=PID_COL, how="inner")

    feature_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    features = df[feature_cols].values
    exposure = df[TARGET_COL].values

    # 5) Simulate outcomes in three scenarios
    logger.info("Simulating outcome for actual treatment assignment")
    outcome_actual, probas_actual = simulate_outcome_from_embeddings(
        features, exposure, **cfg.simulation
    )
    logger.info("Simulating outcome under TREATMENT for all (treated scenario).")
    outcome_treated, probas_treated = simulate_outcome_from_embeddings(
        features, np.ones(len(features)), **cfg.simulation
    )
    logger.info("Simulating outcome under CONTROL for all (untreated scenario).")
    outcome_control, probas_control = simulate_outcome_from_embeddings(
        features, np.zeros(len(features)), **cfg.simulation
    )

    # 6) Save counterfactual-based data (everyone treated vs. everyone untreated)
    # The "counterfactual" file is for potential future effect estimation.
    save_counterfactual_probas_and_targets(
        df[PID_COL],
        df[TARGET_COL],
        outcome_treated,
        outcome_control,
        probas_treated,
        probas_control,
        join(simulation_folder, "counterfactual_probas_and_targets.csv"),
    )

    # 7) Generate "actual" outcomes by combining each patient's relevant scenario:
    #    If a patient is truly treated, use 'outcome_treated'; if untreated, use 'outcome_control'.
    logger.info("Combine treated vs. untreated outcomes for the ACTUAL scenario")
    outcome_actual = np.where(
        df[TARGET_COL] == 1,
        outcome_treated,  # use the "treated" simulation for actually treated
        outcome_control,  # use the "untreated" simulation for actually untreated
    )

    # Same approach for predicted probabilities
    probas_actual = np.where(
        df[TARGET_COL] == 1,
        probas_treated,
        probas_control,
    )

    # 8) Save "actual" scenario probabilities and outcomes
    logger.info("Saving probabilities and outcomes for the ACTUAL scenario.")
    save_probas_and_targets(
        df[PID_COL],
        outcome_actual,
        probas_actual,
        join(simulation_folder, "probas_and_targets.csv"),
    )

    # 9) Simulate absolute position (timestamp) for patients with outcome=1 in the actual scenario
    logger.info("Simulating absolute position for the ACTUAL scenario.")
    abspos_outcome_actual = simulate_abspos_from_binary_outcome(
        outcome_actual,
        df[INDEX_DATE],
        cfg.get("max_years", 3),
        cfg.get("days_offset", 0),
    )
    result_df = pd.DataFrame(
        {PID_COL: df[PID_COL], TIMESTAMP_COL: abspos_outcome_actual}
    )
    # Save the actual scenario timestamps to SIMULATED.csv
    logger.info("Saving simulated outcome to %s", simulation_folder)
    os.makedirs(simulation_folder, exist_ok=True)
    # Rows with null timestamps are those with outcome=0
    result_df.dropna().to_csv(join(simulation_folder, "SIMULATED.csv"), index=False)

    # 10) Also save the explicit treated/control outcomes to a separate file
    # This is for potential future effect estimation.
    logger.info("Saving explicit TREATED vs. CONTROL outcomes to COUNTERFACTUAL.csv.")
    counterfactual_df = pd.DataFrame(
        {
            PID_COL: df[PID_COL],
            OUTCOME_TREATED_COL: outcome_treated,
            OUTCOME_CONTROL_COL: outcome_control,
        }
    )
    counterfactual_df.to_csv(join(simulation_folder, "COUNTERFACTUAL.csv"), index=False)

    # 11) If running in Azure, also store results to blob
    if cfg.env == "azure":
        save_to_blobstore(
            local_path="",
            remote_path=join(cfg.get("project", DEFAULT_BLOBSTORE), output_path),
            overwrite=False,
        )
        mount_context.stop()
    logger.info("Done")


if __name__ == "__main__":
    main(config_path)

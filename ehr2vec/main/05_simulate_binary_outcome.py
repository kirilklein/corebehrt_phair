"""This script uses a model to simulate a binary outcome"""

import os
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.config import Config, get_function
from ehr2vec.common.default_args import DEFAULT_BLOBSTORE
from ehr2vec.common.loader import load_config, load_index_dates
from ehr2vec.common.cli import override_config_from_cli
from ehr2vec.common.setup import (
    get_args,
    initialize_configuration_finetune,
    setup_logger,
)
from ehr2vec.simulation.longitudinal_outcome import simulate_abspos_from_binary_outcome

DEFAULT_CONFIG_NAME = "example_configs/05_simulate_binary_outcome.yaml"


args = get_args(DEFAULT_CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)


def main(config_path: str) -> None:
    cfg: Config = load_config(config_path)
    override_config_from_cli(cfg)
    cfg, run, mount_context, pretrain_model_path = initialize_configuration_finetune(
        cfg, dataset_name=cfg.get("project", DEFAULT_BLOBSTORE)
    )
    simulation_folder = cfg.paths.output
    os.makedirs(simulation_folder, exist_ok=True)
    logger = setup_logger(simulation_folder)

    cfg.save_to_yaml(join(simulation_folder, "simulation_config.yaml"))
    logger.info("Load predictions from %s", cfg.paths.model_path)
    df_predictions = pd.read_csv(join(cfg.paths.model_path, cfg.predictions_file))

    logger.info("Load index dates from %s", cfg.paths.model_path)
    df_index_dates = load_index_dates(cfg.paths.model_path)
    logger.info("Merge predictions and index dates")
    df_merged = pd.merge(df_predictions, df_index_dates, on="pid")
    logger.info("Simulate outcome")
    binary_outcome = simulate_outcome(
        df_merged["proba"], df_merged["target"], cfg.simulation
    )
    logger.info("Simulate outcome under treatment")
    binary_outcome_exp = simulate_outcome(
        df_merged["proba"], np.ones(len(df_merged)), cfg.simulation
    )
    logger.info("Simulate outcome under control")
    binary_outcome_ctrl = simulate_outcome(
        df_merged["proba"], np.zeros(len(df_merged)), cfg.simulation
    )
    logger.info("Simulate absolute position")
    abspos_outcome = simulate_abspos_from_binary_outcome(
        binary_outcome,
        df_merged["index_date"],
        cfg.get("max_years", 3),
        cfg.get("days_offset", 0),
    )
    result_df = pd.DataFrame({"PID": df_merged["pid"], "TIMESTAMP": abspos_outcome})
    logger.info("Save simulated outcome to %s", simulation_folder)
    os.makedirs(simulation_folder, exist_ok=True)
    result_df.dropna().to_csv(join(simulation_folder, "SIMULATED.csv"), index=False)
    counterfactual_df = pd.DataFrame(
        {"PID": df_merged["pid"], "Y1": binary_outcome_exp, "Y0": binary_outcome_ctrl}
    )
    counterfactual_df.to_csv(join(simulation_folder, "COUNTERFACTUAL.csv"), index=False)

    if cfg.env == "azure":
        save_path = (
            pretrain_model_path
            if cfg.paths.get("save_folder_path", None) is None
            else cfg.paths.save_folder_path
        )
        save_to_blobstore(
            local_path=cfg.paths.run_name,
            remote_path=join(
                cfg.get("project", DEFAULT_BLOBSTORE), save_path, cfg.paths.run_name
            ),
        )
        mount_context.stop()
    logger.info("Done")


def simulate_outcome(proba, target, simulation_cfg):
    return get_function(simulation_cfg)(proba, target, **simulation_cfg.params)


if __name__ == "__main__":
    main(config_path)

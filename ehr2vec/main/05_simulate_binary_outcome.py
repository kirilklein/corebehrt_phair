"""This script uses a model to simulate a binary outcome"""

import os
from os.path import abspath, dirname, join
import numpy as np
import pandas as pd

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.config import get_function
from ehr2vec.common.loader import load_index_dates, load_predictions_from_finetune_dir
from ehr2vec.common.setup import (
    DirectoryPreparer,
    get_args,
    initialize_configuration_finetune,
)
from ehr2vec.simulation.longitudinal_outcome import simulate_abspos_from_binary_outcome

DEFAULT_CONFIG_NAME = "example_configs/05_simulate_binary_outcome.yaml"
BLOBSTORE = "CINF"

args = get_args(DEFAULT_CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)


def main(config_path: str) -> None:
    cfg, run, mount_context, pretrain_model_path = initialize_configuration_finetune(
        config_path, dataset_name=BLOBSTORE
    )
    logger, simulation_folder = DirectoryPreparer.setup_run_folder(cfg)
    cfg.save_to_yaml(join(simulation_folder, "simulation_config.yaml"))

    df_predictions = load_predictions_from_finetune_dir(cfg.paths.model_path)
    df_index_dates = load_index_dates(cfg.paths.model_path)
    df_merged = pd.merge(df_predictions, df_index_dates, on="pid")

    binary_outcome = simulate_outcome(
        df_merged["proba"], df_merged["target"], cfg.simulation
    )
    binary_outcome_exp = simulate_outcome(
        df_merged["proba"], np.ones(len(df_merged)), cfg.simulation
    )
    binary_outcome_ctrl = simulate_outcome(
        df_merged["proba"], np.zeros(len(df_merged)), cfg.simulation
    )

    abspos_outcome = simulate_abspos_from_binary_outcome(
        binary_outcome,
        df_merged["index_date"],
        cfg.get("max_years", 3),
        cfg.get("days_offset", 0),
    )
    result_df = pd.DataFrame({"PID": df_merged["pid"], "TIMESTAMP": abspos_outcome})
    os.makedirs(cfg.paths.output, exist_ok=True)
    result_df.dropna().to_csv(join(cfg.paths.output, "SIMULATED.csv"), index=False)
    counterfactual_df = pd.DataFrame(
        {"PID": df_merged["pid"], "Y1": binary_outcome_exp, "Y0": binary_outcome_ctrl}
    )
    counterfactual_df.to_csv(join(cfg.paths.output, "COUNTERFACTUAL.csv"), index=False)

    if cfg.env == "azure":
        save_path = (
            pretrain_model_path
            if cfg.paths.get("save_folder_path", None) is None
            else cfg.paths.save_folder_path
        )
        save_to_blobstore(
            local_path=cfg.paths.run_name,
            remote_path=join(BLOBSTORE, save_path, cfg.paths.run_name),
        )
        mount_context.stop()
    logger.info("Done")


def simulate_outcome(proba, target, simulation_cfg):
    return get_function(simulation_cfg)(proba, target, **simulation_cfg.params)


if __name__ == "__main__":
    main(config_path)

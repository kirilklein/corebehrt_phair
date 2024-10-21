"""
This script is used to measure the effect of treatment on the outcome using propensity scores.
Inputs:
    - probas: the propensity scores
    - treatment: the treatment status
    - outcome: the binary outcome
Optional (for double robustness):
    - outcome probas: the predicted probabilities for the outcome
    - outcome under treatment: the potentially counterfactual outcome under treatment
    - outcome under no treatment: the potentially counterfactual outcome under no treatment

"""

import os
import numpy as np
from os.path import abspath, dirname, join, split

import pandas as pd
from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.stats.stats import compute_treatment_outcome_table
from CausalEstimate.filter.propensity import filter_common_support
from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.cli import override_config_from_cli
from ehr2vec.common.config import Config
from ehr2vec.common.default_args import (
    DEFAULT_BLOBSTORE,
    OUTCOME_COL,
    COUNTERFACTUAL_CONTROL_COL,
    COUNTERFACTUAL_TREATED_COL,
    PS_COL,
    TREATMENT_COL,
    OUTCOME_PREDICTIONS_COL,
)
from ehr2vec.common.loader import (
    load_config,
    load_counterfactual_outcomes,
    load_outcomes,
)
from ehr2vec.common.logger import log_config
from ehr2vec.common.setup import (
    fix_tmp_prefixes_for_azure_paths,
    get_args,
    initialize_configuration_effect_estimation,
    setup_logger,
)
from ehr2vec.common.wandb import finish_wandb, initialize_wandb
from ehr2vec.effect_estimation.counterfactual import compute_effect_from_counterfactuals
from ehr2vec.effect_estimation.data import (
    construct_data_for_effect_estimation,
    construct_data_to_estimate_effect_from_counterfactuals,
)
from ehr2vec.effect_estimation.utils import convert_effect_to_dataframe
from ehr2vec.common.wandb import log_dataframe

DEFAULT_CONFIG_NAME = "example_configs/06_estimate_effect_binary.yaml"
DOUBLE_ROBUST_METHODS = ["AIPW", "TMLE"]
args = get_args(DEFAULT_CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)


def main(config_path: str):
    cfg: Config = load_config(config_path)
    override_config_from_cli(cfg)
    if "wandb_kwargs" in cfg:
        cfg.wandb_kwargs.name = cfg.paths.run_name
    cfg, run, mount_context, azure_context = initialize_configuration_effect_estimation(
        cfg, dataset_name=cfg.get("project", DEFAULT_BLOBSTORE)
    )
    run = initialize_wandb(run, cfg, cfg.wandb_kwargs)
    # create test folder
    path_cfg: Config = cfg.paths
    exp_folder = join(path_cfg.output_path, f"experiment_{path_cfg.run_name}")
    os.makedirs(exp_folder, exist_ok=True)

    # later we will add outcome folder
    logger = setup_logger(exp_folder, "info.log")
    cfg.save_to_yaml(join(exp_folder, "config.yaml"))

    if path_cfg.get("outcome_predictions_counterfactual", None):
        counterfactual_predictions = (
            pd.read_csv(path_cfg.outcome_predictions_counterfactual)
            .rename(columns={"pid": "PID", "proba": OUTCOME_PREDICTIONS_COL})
            .set_index("PID")
        )
    else:
        counterfactual_predictions = None

    if path_cfg.get("outcome_predictions", None):
        outcome_predictions = (
            pd.read_csv(path_cfg.outcome_predictions)
            .rename(columns={"pid": "PID", "proba": OUTCOME_PREDICTIONS_COL})
            .set_index("PID")
        )
    else:
        outcome_predictions = None
    propensity_scores = (
        pd.read_csv(join(path_cfg.ps_model_path, cfg.ps_file))
        .rename(columns={"pid": "PID", "target": TREATMENT_COL, "proba": PS_COL})
        .set_index("PID")
    )
    outcomes = load_outcomes(path_cfg.outcome)

    df = construct_data_for_effect_estimation(
        propensity_scores, outcomes, outcome_predictions, counterfactual_predictions
    )

    num_patients = cfg.get("num_patients")
    if (num_patients is not None) and (num_patients < len(df)):
        logger.info(f"Sampling {cfg.num_patients} patients")
        df = df.sample(n=cfg.num_patients, replace=False)

    df_copy = df.copy(deep=True) # ! apply noise to the copy but use original to estimate true effect
    
    if cfg.get("ps_noise", 0) > 0:
        logger.info(f"Adding {cfg.get('ps_noise')} noise to propensity scores")
        df_copy[PS_COL] = df_copy[PS_COL] * (
            1 + np.random.uniform(-cfg.ps_noise, cfg.ps_noise, len(df_copy))
        )
    stats_table = compute_treatment_outcome_table(df, TREATMENT_COL, OUTCOME_COL)
    stats_table.index.name = "Treatment"
    stats_table.reset_index(inplace=True)
    log_dataframe(stats_table, "stats_table")

    logger.info("Estimating causal effect")
    estimator_cfg = cfg.get("estimator")
    estimator = Estimator(
        methods=estimator_cfg.methods,
        effect_type=estimator_cfg.effect_type,
    )
    # temporary fix for double robustness
    method_args = {
        method: {
            "predicted_outcome_treated_col": COUNTERFACTUAL_TREATED_COL,
            "predicted_outcome_control_col": COUNTERFACTUAL_CONTROL_COL,
            "predicted_outcome_col": OUTCOME_PREDICTIONS_COL,
        }
        for method in DOUBLE_ROBUST_METHODS
    }
    if cfg.estimator.get("method_args", None):
        method_args.update(cfg.estimator.method_args)

    common_support = (
        True if cfg.estimator.get("common_support_threshold", False) else False
    )
    common_support_threshold = cfg.estimator.get("common_support_threshold", None)

    effect = estimator.compute_effect(
        df_copy,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        ps_col=PS_COL,
        bootstrap=True if estimator_cfg.get("n_bootstrap", 0) > 1 else False,
        n_bootstraps=estimator_cfg.get("n_bootstrap", 0),
        method_args=method_args,
        apply_common_support=common_support,
        common_support_threshold=common_support_threshold,
    )
    if run is not None:
        run.log({"causal_effect": effect})
    effect_df = convert_effect_to_dataframe(effect)

    logger.info(f"Causal effect: {effect}")

    log_config(cfg, logger)
    path_cfg.run_name = split(exp_folder)[-1]

    if path_cfg.get("counterfactual_outcome", None):
        logger.info("Computing effect from counterfactual outcomes")
        counterfactuals = load_counterfactual_outcomes(path_cfg.counterfactual_outcome)

        df_counterfactual = construct_data_to_estimate_effect_from_counterfactuals(
            df, counterfactuals
        )
        if common_support:
            df_counterfactual = filter_common_support(
                df_counterfactual,
                ps_col=PS_COL,
                treatment_col=TREATMENT_COL,
                threshold=common_support_threshold,
            )
        effect_counterfactual = compute_effect_from_counterfactuals(
            df_counterfactual, estimator_cfg.effect_type
        )
        effect_df["effect_counterfactual"] = effect_counterfactual
        logger.info(f"Causal effect from counterfactuals: {effect_counterfactual}")
        if run is not None:
            run.log({"causal_effect_counterfactual (true)": effect_counterfactual})
    log_dataframe(effect_df, "effect_df")
    effect_df.to_csv(join(exp_folder, "effect.csv"), index=False)

    finish_wandb()
    if cfg.env == "azure":
        save_to_blobstore(
            local_path="",  # uses everything in 'outputs'
            remote_path=join(
                cfg.get("project", DEFAULT_BLOBSTORE),
                fix_tmp_prefixes_for_azure_paths(path_cfg.model_path),
            ),
        )
        mount_context.stop()
    logger.info("Done")


if __name__ == "__main__":
    main(config_path)

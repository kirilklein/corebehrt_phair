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
from os.path import abspath, dirname, join, split

from CausalEstimate.interface.estimator import Estimator

from ehr2vec.common.azure import save_to_blobstore

# from ehr2vec.common.calibration import calibrate_cv
from ehr2vec.common.loader import (
    load_counterfactual_outcomes,
    load_outcomes,
    load_propensities,
)
from ehr2vec.common.logger import log_config
from ehr2vec.common.setup import (
    fix_tmp_prefixes_for_azure_paths,
    get_args,
    initialize_configuration_effect_estimation,
    setup_logger,
)
from ehr2vec.effect_estimation.counterfactual import compute_effect_from_counterfactuals
from ehr2vec.effect_estimation.data import (
    construct_data_for_effect_estimation,
    construct_data_to_estimate_effect_from_counterfactuals,
)
from ehr2vec.effect_estimation.utils import convert_effect_to_dataframe

CONFIG_NAME = "example_configs/06_estimate_effect_binary.yaml"
BLOBSTORE = "CINF"

args = get_args(CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    cfg, run, mount_context, azure_context = initialize_configuration_effect_estimation(
        config_path, dataset_name=BLOBSTORE
    )

    # create test folder
    exp_folder = join(cfg.paths.output_path, f"experiment_{cfg.paths.run_name}")
    os.makedirs(exp_folder, exist_ok=True)

    # later we will add outcome folder
    logger = setup_logger(exp_folder, "info.log")

    if cfg.get("double_robust", False):
        # Here we will also load the counterfactual predictions necessary for double robustness
        # Should be automated, i.e. if method is double robust, e.g. TMLE, then load the necessary files
        raise NotImplementedError("Double robustness not implemented yet")
    propensity_scores = load_propensities(cfg.paths.get("ps_model_path"))
    outcomes = load_outcomes(cfg.paths.get("outcome"))
    df = construct_data_for_effect_estimation(propensity_scores, outcomes)

    logger.info("Estimating causal effect")
    estimator_cfg = cfg.get("estimator")
    estimator = Estimator(
        methods=estimator_cfg.methods, effect_type=estimator_cfg.effect_type
    )
    effect = estimator.compute_effect(
        df,
        treatment_col="treatment",
        outcome_col="outcome",
        ps_col="ps",
        bootstrap=True if estimator_cfg.n_bootstrap > 1 else False,
        n_bootstraps=estimator_cfg.n_bootstrap,
    )
    effect_df = convert_effect_to_dataframe(effect)

    logger.info(f"Causal effect: {effect}")

    log_config(cfg, logger)
    cfg.paths.run_name = split(exp_folder)[-1]

    if cfg.paths.get("counterfactual_outcome", None):
        logger.info("Computing effect from counterfactual outcomes")
        counterfactuals = load_counterfactual_outcomes(cfg.paths.counterfactual_outcome)
        df_counterfactual = construct_data_to_estimate_effect_from_counterfactuals(
            propensity_scores, counterfactuals
        )
        effect_counterfactual = compute_effect_from_counterfactuals(
            df_counterfactual, estimator_cfg.effect_type
        )
        effect_df["effect_counterfactual"] = effect_counterfactual
        logger.info(f"Causal effect from counterfactuals: {effect_counterfactual}")

    effect_df.to_csv(join(exp_folder, "effect.csv"), index=False)

    if cfg.env == "azure":
        save_to_blobstore(
            local_path="",  # uses everything in 'outputs'
            remote_path=join(
                BLOBSTORE, fix_tmp_prefixes_for_azure_paths(cfg.paths.model_path)
            ),
        )
        mount_context.stop()
    logger.info("Done")


if __name__ == "__main__":
    main()

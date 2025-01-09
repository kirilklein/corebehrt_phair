import os
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join
from dataclasses import dataclass
from typing import Optional, Any

from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.stats.stats import compute_treatment_outcome_table
from CausalEstimate.filter.propensity import filter_common_support
from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.cli import override_config_from_cli
from ehr2vec.common.config import Config
from ehr2vec.common.loader import (
    load_config,
    load_counterfactual_outcomes,
    load_outcomes,
)
from ehr2vec.common.setup import (
    get_args,
    initialize_configuration_effect_estimation,
    setup_logger,
)
from ehr2vec.common.wandb import finish_wandb, initialize_wandb, log_dataframe
from ehr2vec.effect_estimation.counterfactual import compute_effect_from_counterfactuals
from ehr2vec.effect_estimation.data import (
    construct_data_for_effect_estimation,
    construct_data_to_estimate_effect_from_counterfactuals,
)
from ehr2vec.effect_estimation.utils import convert_effect_to_dataframe
from ehr2vec.common.default_args import (
    DEFAULT_BLOBSTORE,
    OUTCOME_COL,
    COUNTERFACTUAL_CONTROL_COL,
    COUNTERFACTUAL_TREATED_COL,
    PS_COL,
    TREATMENT_COL,
    OUTCOME_PREDICTIONS_COL,
)


@dataclass
class EffectEstimator:
    cfg: Config
    logger: Any
    exp_folder: str
    mount_context: Any

    def run(self):
        df = self._load_data()
        self._log_basic_stats(df)

        df_noisy = self._add_noise(df)  # optional

        self.logger.info("Estimating causal effect")
        effect_df, common_support, threshold = self._compute_causal_effect(df_noisy)

        counterfactual_effect = self._compute_counterfactual_effect(
            df, common_support, threshold
        )
        if counterfactual_effect is not None:
            effect_df["effect_counterfactual"] = counterfactual_effect
            self.logger.info(
                f"Causal effect from counterfactuals: {counterfactual_effect}"
            )

        log_dataframe(effect_df, "effect_df")
        effect_df.to_csv(join(self.exp_folder, "effect.csv"), index=False)

        self._cleanup()

    @classmethod
    def from_config(cls, config_path: str) -> "EffectEstimator":
        cfg = load_config(config_path)
        override_config_from_cli(cfg)

        if "wandb_kwargs" in cfg:
            cfg.wandb_kwargs.name = cfg.paths.run_name

        cfg, run, mount_context, _ = initialize_configuration_effect_estimation(
            cfg, dataset_name=cfg.get("project", DEFAULT_BLOBSTORE)
        )
        run = initialize_wandb(run, cfg, cfg.get("wandb_kwargs", {}))

        exp_folder = join(cfg.paths.output_path, f"experiment_{cfg.paths.run_name}")
        os.makedirs(exp_folder, exist_ok=True)

        logger = setup_logger(exp_folder, "info.log")
        cfg.save_to_yaml(join(exp_folder, "config.yaml"))

        return cls(
            cfg=cfg,
            logger=logger,
            exp_folder=exp_folder,
            mount_context=mount_context,
        )

    def _load_data(self) -> pd.DataFrame:
        path_cfg = self.cfg.paths

        outcome_predictions = None
        if path_cfg.get("outcome_predictions", None):
            outcome_predictions = (
                pd.read_csv(path_cfg.outcome_predictions)
                .rename(columns={"pid": "PID", "proba": OUTCOME_PREDICTIONS_COL})
                .set_index("PID")
            )

        counterfactual_predictions = None
        if path_cfg.get("outcome_predictions_counterfactual", None):
            counterfactual_predictions = (
                pd.read_csv(path_cfg.outcome_predictions_counterfactual)
                .rename(columns={"pid": "PID", "proba": OUTCOME_PREDICTIONS_COL})
                .set_index("PID")
            )

        propensity_scores = (
            pd.read_csv(path_cfg.propensity_scores)
            .rename(columns={"pid": "PID", "target": TREATMENT_COL, "proba": PS_COL})
            .set_index("PID")
        )

        outcomes = load_outcomes(path_cfg.outcome)

        df = construct_data_for_effect_estimation(
            propensity_scores, outcomes, outcome_predictions, counterfactual_predictions
        )

        num_patients = self.cfg.get("num_patients")
        if num_patients and num_patients < len(df):
            self.logger.info(f"Sampling {num_patients} patients")
            df = df.sample(n=num_patients, replace=False)

        return df

    def _compute_causal_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        estimator_cfg = self.cfg.get("estimator")
        estimator = Estimator(
            methods=estimator_cfg.methods,
            effect_type=estimator_cfg.effect_type,
        )

        method_args = {
            method: {
                "predicted_outcome_treated_col": COUNTERFACTUAL_TREATED_COL,
                "predicted_outcome_control_col": COUNTERFACTUAL_CONTROL_COL,
                "predicted_outcome_col": OUTCOME_PREDICTIONS_COL,
            }
            for method in ["AIPW", "TMLE"]
        }

        if estimator_cfg.get("method_args"):
            method_args.update(estimator_cfg.method_args)

        common_support = bool(estimator_cfg.get("common_support_threshold", False))
        common_support_threshold = estimator_cfg.get("common_support_threshold")

        effect = estimator.compute_effect(
            df,
            treatment_col=TREATMENT_COL,
            outcome_col=OUTCOME_COL,
            ps_col=PS_COL,
            bootstrap=bool(estimator_cfg.get("n_bootstrap", 0) > 1),
            n_bootstraps=estimator_cfg.get("n_bootstrap", 0),
            method_args=method_args,
            apply_common_support=common_support,
            common_support_threshold=common_support_threshold,
        )

        return (
            convert_effect_to_dataframe(effect),
            common_support,
            common_support_threshold,
        )

    def _compute_counterfactual_effect(
        self, df: pd.DataFrame, common_support: bool, threshold: Optional[float]
    ) -> Optional[float]:
        """Compute causal effect using counterfactual outcomes if available.

        This method loads pre-computed counterfactual outcomes and uses them to estimate
        the causal effect. It optionally applies common support filtering to ensure
        comparable treatment and control groups.

        Args:
            df: DataFrame containing the original data with treatment assignments and outcomes
            common_support: Whether to apply common support filtering based on propensity scores
            threshold: Threshold value for common support filtering. Only used if common_support=True

        Returns:
            float: Estimated causal effect computed from counterfactuals if counterfactual
                  outcomes are available, None otherwise

        Note:
            The counterfactual outcomes must be pre-computed and specified in the config
            under paths.counterfactual_outcome
        """
        if not self.cfg.paths.get("counterfactual_outcome"):
            return None

        self.logger.info("Computing effect from counterfactual outcomes")
        counterfactuals = load_counterfactual_outcomes(
            self.cfg.paths.counterfactual_outcome
        )
        df_counterfactual = construct_data_to_estimate_effect_from_counterfactuals(
            df, counterfactuals
        )

        if common_support:
            df_counterfactual = filter_common_support(
                df_counterfactual,
                ps_col=PS_COL,
                treatment_col=TREATMENT_COL,
                threshold=threshold,
            )

        return compute_effect_from_counterfactuals(
            df_counterfactual, self.cfg.get("estimator").effect_type
        )

    def _log_basic_stats(self, df: pd.DataFrame) -> None:
        stats_table = compute_treatment_outcome_table(df, TREATMENT_COL, OUTCOME_COL)
        stats_table.index.name = "Treatment"
        stats_table.reset_index(inplace=True)
        log_dataframe(stats_table, "stats_table")

    def _add_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function is intended to test the robustness of the causal effect estimation.
        This adds noise to the propensity scores.
        """
        df_copy = df.copy(deep=True)
        noise = self.cfg.get("ps_noise", 0)

        if noise > 0:
            self.logger.info(f"Adding {noise} noise to propensity scores")
            df_copy[PS_COL] *= 1 + np.random.uniform(-noise, noise, len(df_copy))
            df_copy[PS_COL] = df_copy[PS_COL].clip(lower=1e-6, upper=1 - 1e-6)

        return df_copy

    def _cleanup(self):
        finish_wandb()
        if self.cfg.env == "azure":
            save_to_blobstore(
                local_path="",
                remote_path=join(
                    self.cfg.get("project", DEFAULT_BLOBSTORE),
                    "effect",
                    self.cfg.paths.run_name,
                ),
            )
        if self.mount_context is not None and hasattr(self.mount_context, "stop"):
            self.mount_context.stop()


def main(config_path: str):
    estimator = EffectEstimator.from_config(config_path)
    estimator.run()


if __name__ == "__main__":
    args = get_args("example_configs/06_estimate_effect_binary.yaml")
    config_path = join(dirname(dirname(abspath(__file__))), args.config_path)
    main(config_path)

import os
from dataclasses import dataclass
from os.path import join
from typing import Any, Optional

import numpy as np
import pandas as pd
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.stats.stats import compute_treatment_outcome_table
from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.cli import override_config_from_cli
from ehr2vec.common.config import Config
from ehr2vec.common.default_args import (
    CF_CONTROL_COL,
    CF_TREATED_COL,
    DEFAULT_BLOBSTORE,
    ORG_PID_COL,
    OUTCOME_COL,
    OUTCOME_PROBABILITY_COL,
    PID_COL,
    PROBA_COL,
    PS_COL,
    TARGET_COL,
    TREATMENT_COL,
)
from ehr2vec.common.loader import (
    load_config,
    load_counterfactual_outcomes,
    load_outcomes,
)
from ehr2vec.common.setup import (
    initialize_configuration_effect_estimation,
    setup_logger,
)
from ehr2vec.common.wandb import finish_wandb, initialize_wandb, log_dataframe
from ehr2vec.effect_estimation.counterfactual import compute_effect_from_counterfactuals
from ehr2vec.effect_estimation.data import (
    construct_from_counterfactuals,
    construct_from_observed_data,
)
from ehr2vec.effect_estimation.utils import convert_effect_to_dataframe
from scipy.special import logit, expit


@dataclass
class EffectEstimator:
    cfg: Config
    logger: Any
    exp_folder: str
    mount_context: Any

    def run(self):
        df = self._load_data()
        df.to_parquet(join(self.exp_folder, "data.parquet"), index=True)
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
            outcome_predictions = self._load_predictions(path_cfg.outcome_predictions)

        counterfactual_predictions = None
        if path_cfg.get("outcome_predictions_counterfactual", None):
            counterfactual_predictions = self._load_predictions(
                path_cfg.outcome_predictions_counterfactual
            )

        propensity_scores = self._load_propensity_scores(path_cfg.propensity_scores)
        outcomes = load_outcomes(path_cfg.outcome)

        df = construct_from_observed_data(
            propensity_scores=propensity_scores,
            outcomes=outcomes,
            outcome_predictions=outcome_predictions,
            counterfactual_predictions=counterfactual_predictions,
        )

        df = self._sample_patients(df)

        return df

    def _load_propensity_scores(self, path: str) -> pd.DataFrame:
        return (
            pd.read_csv(path)
            .rename(
                columns={
                    ORG_PID_COL: PID_COL,
                    TARGET_COL: TREATMENT_COL,
                    PROBA_COL: PS_COL,
                }
            )
            .set_index(PID_COL)
        )

    def _load_predictions(self, path: str) -> pd.DataFrame:
        """Load and format outcome predictions from a CSV file.

        Args:
            path: Path to CSV file containing predictions. Expected columns:
                - Original patient ID column (will be renamed to PID)
                - Original probability column (will be renamed to Y_hat)

        Returns:
            DataFrame with:
                - Index: Patient IDs (PID)
                - Y_hat: Predicted outcome probabilities

        Note:
            Renames columns to standardized names and sets patient ID as index
        """
        return (
            pd.read_csv(path)
            .rename(columns={ORG_PID_COL: PID_COL, PROBA_COL: OUTCOME_PROBABILITY_COL})
            .set_index(PID_COL)
        )

    def _compute_causal_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        estimator_cfg = self.cfg.get("estimator")
        estimator = Estimator(
            methods=estimator_cfg.methods,
            effect_type=estimator_cfg.effect_type,
        )

        method_args = {
            method: {
                "predicted_outcome_treated_col": CF_TREATED_COL,
                "predicted_outcome_control_col": CF_CONTROL_COL,
                "predicted_outcome_col": OUTCOME_PROBABILITY_COL,
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
        df_counterfactual = construct_from_counterfactuals(df, counterfactuals)
        df_counterfactual.to_parquet(
            join(self.exp_folder, "data_cf.parquet"), index=True
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

    def _sample_patients(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function is intended to test the robustness of the causal effect estimation.
        This samples a subset of patients from the data (optional).
        """
        num_patients = self.cfg.get("num_patients", False)
        if num_patients and num_patients < len(df):
            self.logger.info(f"Sampling {num_patients} patients")
            df = df.sample(n=num_patients, replace=False)

        return df

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


class EffectEstimator_with_bias(EffectEstimator):
    def run(self):
        df = self._load_data()
        df.to_parquet(join(self.exp_folder, "data.parquet"), index=True)
        self._log_basic_stats(df)

        # Get bias parameters from config
        bias_params = self.cfg.get("bias_params", [(0, 0)])  # Default to no bias case

        # Create lists to store results
        results = {
            "delta_ps": [],
            "delta_y": [],
        }

        os.makedirs(join(self.exp_folder, "effects"), exist_ok=True)

        for delta_ps, delta_y in bias_params:
            self.logger.info(
                f"Computing effect with bias parameters: delta_ps={delta_ps}, delta_y={delta_y}"
            )

            # Apply bias to the data
            df_biased = self._add_bias(df, delta_ps, delta_y)

            # Compute effect with biased data
            effect_df, common_support, threshold = self._compute_causal_effect(
                df_biased
            )

            # Add bias parameters to effect dataframe
            effect_df["delta_ps"] = delta_ps
            effect_df["delta_y"] = delta_y

            # Save individual effect dataframe
            filename = f"effect_ps{delta_ps}_y{delta_y}.csv"
            effect_df.to_csv(join(self.exp_folder, "effects", filename), index=False)
            log_dataframe(effect_df, f"effect_df_ps{delta_ps}_y{delta_y}")

            # Store results for consolidated dataframe
            results["delta_ps"].append(delta_ps)
            results["delta_y"].append(delta_y)

            # Add effect and std for each method
            for method in effect_df["method"].unique():
                method_data = effect_df[effect_df["method"] == method]
                effect_col = f"effect_{method}"
                std_col = f"effect_std_{method}"

                # Initialize lists for new methods
                if effect_col not in results:
                    results[effect_col] = []
                if std_col not in results:
                    results[std_col] = []
                # Store results
                results[effect_col].append(method_data["effect"].iloc[0])
                results[std_col].append(method_data["std_err"].iloc[0])

            # Compute counterfactual effect if available
            counterfactual_effect = self._compute_counterfactual_effect(
                df_biased, common_support, threshold
            )
            if counterfactual_effect is not None:
                effect_df["effect_counterfactual"] = counterfactual_effect
                # Initialize counterfactual list if first time
                if "effect_counterfactual" not in results:
                    results["effect_counterfactual"] = []
                results["effect_counterfactual"].append(counterfactual_effect)

                self.logger.info(
                    f"Causal effect from counterfactuals (bias δps={delta_ps}, δy={delta_y}): {counterfactual_effect}"
                )

        # Create and save consolidated results dataframe
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            join(self.exp_folder, "consolidated_results.csv"), index=False
        )
        log_dataframe(results_df, "consolidated_results")

        self._cleanup()

    @staticmethod
    def _add_bias(df: pd.DataFrame, delta_ps: float, delta_y: float) -> pd.DataFrame:
        """
        Add bias to propensity scores and outcomes for sensitivity analysis.
        This is a placeholder - implement your specific bias mechanism here.

        Args:
            df: Original dataframe
            delta_ps: Bias parameter for propensity scores
            delta_y: Bias parameter for outcomes

        Returns:
            DataFrame with added bias
        """
        df = df.copy(deep=True)
        # Implement your bias mechanism here
        # For example:
        df[PS_COL] = EffectEstimator_with_bias._transform_column(df[PS_COL], delta_ps)
        df[OUTCOME_PROBABILITY_COL] = EffectEstimator_with_bias._transform_column(
            df[OUTCOME_PROBABILITY_COL], delta_y
        )
        return df

    @staticmethod
    def _transform_column(column: pd.Series, delta: float) -> pd.Series:
        """Transform column to logit space, add bias, transform back to probability space.

        Args:
            column: Series of probabilities
            delta: Bias parameter to add in logit space

        Returns:
            Series of biased probabilities
        """
        # Convert to logit space using scipy's stable implementation
        logits = logit(column)
        # Add bias in logit space
        biased_logits = logits + delta
        # Convert back to probability space using scipy's stable implementation
        return expit(biased_logits)

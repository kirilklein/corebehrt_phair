import os
from dataclasses import dataclass
from os.path import join
from typing import Any, Optional, Callable, Tuple

import numpy as np
import pandas as pd
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.stats.stats import compute_treatment_outcome_table
from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.cli import override_config_from_cli
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
from ehr2vec.common.config import Config
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

        self.logger.info("Estimating causal effect")
        effect_df, common_support, threshold = self._compute_causal_effect(df)

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

    def _compute_causal_effect(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, bool, Optional[float]]:
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
                overwrite=True,
            )
        if self.mount_context is not None and hasattr(self.mount_context, "stop"):
            self.mount_context.stop()


CONSTANTS: str = "constants"
DELTA_PS: str = "dps"
DELTA_Y: str = "dy"
DELTA_CF: str = "dcf"


class EffectEstimator_with_transform(EffectEstimator):
    def run(self):
        df = self._load_data()
        df.to_parquet(join(self.exp_folder, "data.parquet"), index=True)
        self._log_basic_stats(df)

        os.makedirs(join(self.exp_folder, "effects"), exist_ok=True)

        transformation_params = self.get_transform_params()

        transform_function = self.get_transform_function()
        results = self._initialize_results(transformation_params[0])
        for param_combination in transformation_params:
            results = self._append_to_results(results, param_combination)
            self.logger.info(
                f"Computing effect with transformation parameters: {param_combination}"
            )
            df_transformed = self._transform_data(
                df, param_combination, transform_function
            )

            effect_df, common_support, threshold = self._compute_causal_effect(
                df_transformed
            )

            results = self._append_effect_to_results(results, effect_df)

            counterfactual_effect = self._compute_counterfactual_effect(
                df_transformed, common_support, threshold
            )
            if counterfactual_effect is not None:
                if "effect_counterfactual" not in results:
                    results["effect_counterfactual"] = []
                results["effect_counterfactual"].append(counterfactual_effect)

        # Create and save consolidated results dataframe
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            join(self.exp_folder, "consolidated_results.csv"), index=False
        )

        self._cleanup()

    @staticmethod
    def _transform_data(
        df: pd.DataFrame, params: dict, transform_function: Callable
    ) -> pd.DataFrame:
        df_transformed = df.copy()
        for col in [PS_COL, OUTCOME_PROBABILITY_COL, CF_TREATED_COL, CF_CONTROL_COL]:
            df_transformed[col] = transform_function(
                df_transformed[col], params[DELTA_PS], params[CONSTANTS]
            )
        return df_transformed

    @staticmethod
    def _add_bias(probas: pd.Series, delta: float, constants: dict) -> pd.Series:
        """Transform column to logit space, add bias, transform back to probability space.

        Args:
            column: Series of probabilities
            delta: Bias parameter to add in logit space

        Returns:
            Series of biased probabilities
        """
        if constants.get("sigma", 0) > 0:
            sigmas = np.random.normal(0, constants.get("sigma", 0), len(probas))
        else:
            sigmas = np.zeros(len(probas))
        # Convert to logit space using scipy's stable implementation
        logits = logit(probas)
        # Add bias in logit space
        biased_logits = logits + delta + sigmas
        # Convert back to probability space using scipy's stable implementation
        return expit(biased_logits)

    @staticmethod
    def _power_distortion(
        probas: pd.Series, alpha: float = 1.0, constants: dict = None
    ) -> pd.Series:
        """Apply power distortion to probabilities.

        Args:
            prob: Series of probabilities
            alpha: Power parameter (>1 for sharpening, <1 for flattening)
            constants: Dictionary of constants
        Returns:
            Series of transformed probabilities
        """
        p_alpha = probas**alpha
        q_alpha = (1 - probas) ** alpha
        return p_alpha / (p_alpha + q_alpha)

    def get_transform_function(self) -> Callable:
        """Get the transformation function from the config."""
        transform_function = self.cfg.get("transform_function", "add_bias")
        if transform_function == "add_bias":
            return self._add_bias
        elif transform_function == "power_distortion":
            return self._power_distortion
        else:
            raise ValueError(
                f"Unknown transform function: {transform_function}. Choose from: add_bias, power_distortion"
            )

    @staticmethod
    def _initialize_results(params: dict) -> dict:
        """Initialize the results dictionary."""
        results = {}
        # Add bias parameters to effect dataframe
        for k, v in params.items():
            if k != CONSTANTS:
                results[k] = []
            else:
                for kk, _ in v.items():
                    results[kk] = []
        return results

    @staticmethod
    def _append_to_results(results: dict, params: dict) -> dict:
        """Append parameters to results dictionary."""
        for k, v in params.items():
            if k != CONSTANTS:
                results[k].append(v)
            else:
                for kk, vv in v.items():
                    results[kk].append(vv)
        return results

    @staticmethod
    def _append_effect_to_results(results: dict, effect_df: pd.DataFrame) -> dict:
        """Append effect and standard error to results dictionary."""
        for method in effect_df["method"].unique():
            method_data = effect_df[effect_df["method"] == method]
            effect_col = f"effect_{method}"
            std_col = f"effect_std_{method}"

            if effect_col not in results:
                results[effect_col] = []
            if std_col not in results:
                results[std_col] = []

            results[effect_col].append(method_data["effect"].iloc[0])
            results[std_col].append(method_data["std_err"].iloc[0])
        return results

    def get_transform_params(self) -> list[dict]:
        transformation_config = self.cfg.get(
            "transform_params",
            {DELTA_PS: [0], DELTA_Y: [0], "cf_offset": 0, CONSTANTS: {}},
        )
        ps_values = transformation_config.get(DELTA_PS, [0])
        y_values = transformation_config.get(DELTA_Y, [0])
        cf_offset = transformation_config.get("cf_offset", 0)
        constants = transformation_config.get(CONSTANTS, {})
        return [
            {DELTA_PS: ps, DELTA_Y: y, DELTA_CF: y + cf_offset, CONSTANTS: constants}
            for ps in ps_values
            for y in y_values
        ]

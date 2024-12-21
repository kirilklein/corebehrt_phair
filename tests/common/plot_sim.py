from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.simulation.binary_simulation import simulate_binary_data
from CausalEstimate.vis.plotting import plot_propensity_score_dist
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from utils.predictions import predict_propensity_cv


def plot_causal_effect_estimation_comparison(
    ax: plt.Axes,
    true_effect: dict,
    estimated_effect: dict,
    std_effect: dict,
    methods: list,
    title: str,
    effect_type: str,
) -> plt.Axes:
    """Plot comparison of estimated vs true causal effects across different models and methods.

    Args:
        ax (plt.Axes): Matplotlib axes object to plot on
        true_effect (dict): Dictionary mapping model names to true causal effects
        estimated_effect (dict): Dictionary mapping model names to dictionaries of estimated effects by method
        std_effect (dict): Dictionary mapping model names to dictionaries of effect standard errors by method
        methods (list): List of causal estimation methods to plot
        title (str): Plot title
        effect_type (str): Type of causal effect being plotted ('ATE' or 'ATT')

    Returns:
        plt.Axes: The axes object with the plotted results

    The plot shows bars representing the difference between estimated and true effects for each
    model and method combination. Error bars show standard errors of the estimates. True effect
    values are displayed as text. A horizontal line at 0 indicates perfect estimation.
    """
    diffs_all = {}
    models = estimated_effect.keys()
    for model in models:
        diffs_all[model] = {
            method: estimated_effect[model][method] - true_effect[model]
            for method in methods
        }
    x = np.arange(len(models))
    width = 0.2  # Reduced width for less overlap
    colors = [
        "#2ecc71",
        "#3498db",
        "#e74c3c",
    ]  # Emerald green, Dodger blue, Alizarin red
    for i, method in enumerate(methods):
        diffs = [diffs_all[model][method] for model in models]
        stds = [std_effect[model][method] for model in models]
        ax.bar(
            x + (i - 1) * width, diffs, width, label=method, color=colors[i], alpha=0.8
        )
        # Add error bars using the standard deviations
        ax.errorbar(
            x + (i - 1) * width,
            diffs,
            yerr=stds,
            fmt="none",
            color="black",
            capsize=3,
            alpha=0.5,
        )

    # Add true ATE values as text
    for i, model in enumerate(models):
        # Get average y position of bars for this model
        y_pos = np.mean([diffs_all[model][method] for method in methods])
        # Place text above or below depending on if bars are positive or negative
        text_y = -0.005 if y_pos > 0 else 0.005
        ax.text(
            i,
            text_y,
            f"{true_effect[model]:.3f}",
            horizontalalignment="center",
            verticalalignment="top" if y_pos > 0 else "bottom",
            fontsize=12,
        )

    # Add horizontal line at 0 to show no difference from true ATE
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    ax.set_ylabel(f"Difference from True {effect_type}")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    return ax


def plot_propensity_score_dist_for_models(
    models: Dict[str, Dict[str, Any]],
    ps_model: BaseEstimator,
    ps_model_kwargs: Dict[str, Any],
    n_samples: int = 3000,
    calibration: bool = False,
    common_support_threshold: float = 0.05,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, pd.DataFrame]]:
    """Plot propensity score distributions for multiple simulation models.

    For each model, simulates data, estimates propensity scores using cross-validation,
    applies common support filtering, and plots the propensity score distributions for
    treated and control groups. Also displays ROC AUC scores.

    Args:
        models: Dictionary mapping model names to simulation parameters
        ps_model: Propensity score model estimator
        ps_model_kwargs: Keyword arguments for propensity score model
        n_samples: Number of samples to simulate per model
        calibration: Whether to calibrate propensity scores
        common_support_threshold: Threshold for common support filtering

    Returns:
        fig: Matplotlib figure with subplots
        axs: Array of subplot axes
        dfs: Dictionary mapping model names to their simulated/processed dataframes
    """
    fig, ax = plt.subplots(3, 3, figsize=(12, 6))
    axs = ax.flatten()
    dfs = {}
    for i, (model, params) in enumerate(models.items()):
        data = simulate_binary_data(
            n_samples, alpha=params["alpha"], beta=params["beta"], seed=44
        )
        data["ps"] = predict_propensity_cv(
            data,
            X_cols=["X1", "X2"],
            model=ps_model,
            calibrate=calibration,
            ps_model_kwargs=ps_model_kwargs,
        )

        data = filter_common_support(
            data, ps_col="ps", treatment_col="A", threshold=common_support_threshold
        )

        dfs[model] = data

        fig, ax = plot_propensity_score_dist(
            data, ps_col="ps", treatment_col="A", fig=fig, ax=axs[i]
        )
        roc_auc = roc_auc_score(data["A"], data["ps"])
        ax.text(
            0.05,
            0.95,
            f"ROC AUC: {roc_auc:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
        )
        ax.set_title(model, fontsize=8)
        ax.set_xlabel("", fontsize=8)

    fig.text(0.5, 0.02, "Propensity Score", ha="center")
    # increase distance between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    return fig, axs, dfs


def plot_calibration_curves(
    filtered_dataframes: Dict[str, pd.DataFrame]
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot calibration curves for multiple models.

    Args:
        filtered_dataframes: Dictionary mapping model names to their corresponding dataframes
            containing 'Y' (true labels) and 'ps' (predicted probabilities) columns

    Returns:
        fig: matplotlib figure object
        axs: array of matplotlib axes objects
    """
    fig, axs = plt.subplots(3, 3, figsize=(12, 6))
    axs = axs.flatten()
    for i, (model, data) in enumerate(filtered_dataframes.items()):
        # Plot calibration curve
        y_true = data["A"]
        y_pred = data["ps"]

        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=40, strategy="quantile"
        )

        # Plot perfect calibration line
        axs[i].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        # Plot calibration curve
        axs[i].plot(
            mean_predicted_value, fraction_of_positives, "s-", label="Calibration curve"
        )

        axs[i].set_xlabel("Mean predicted probability")
        axs[i].set_ylabel("Fraction of positives")
        axs[i].set_title(f"{model} Calibration Curve", fontsize=8)
        axs[i].legend(fontsize=6)
        axs[i].grid(True)

    plt.tight_layout()
    return fig, axs

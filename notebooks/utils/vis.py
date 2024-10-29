import os
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


def plot_effect_estimation(
    df: pd.DataFrame,
    x_var: str,
    x_label: str,
    y_label: str = "ATE",
    filename: str = "effect_estimation.png",
    true_effect_label: str = "True Effect",
    xticks: list[float] = None,
    yticks: list[float] = None,
    colors: list[str] = ["#ec008b", "#e88e2d", "#73bfe2"],
    baseline_color: str = "#696969",
    plot_baseline: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    set_plot_defaults()
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_methods(df, ax, x_var, colors)
    if plot_baseline:
        plot_true_effect(ax, df, x_var, baseline_color, true_effect_label)
    set_axes_labels(ax, x_label, y_label, xticks, yticks)
    customize_plot_appearance(ax, xticks, max(df[x_var]))
    add_legend(ax, len(df["method"].unique()))

    save_plot(fig, filename)
    return fig, ax


def plot_propensity_scores(
    ps: pd.DataFrame,
    save_path: str = "figures",
    filename: str = "propensity_scores.png",
    xticks: list[float] = None,
    yticks: list[float] = None,
    bins: int = 40,
    treated_color: str = "#ec008b",
    control_color: str = "#1696D2",
    treated_alpha: float = 0.5,
    control_alpha: float = 0.6,
) -> plt.Figure:
    treated, control = separate_treated_control(ps)
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, bins)

    control.proba.hist(
        bins=bins,
        density=False,
        color=control_color,
        alpha=control_alpha,
        label="Control",
        ax=ax,
    )
    treated.proba.hist(
        bins=bins,
        density=False,
        color=treated_color,
        alpha=treated_alpha,
        label="Treated",
        ax=ax,
    )

    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), frameon=False, ncol=2)

    if xticks is None:
        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]

    ax.set_xticks(xticks)

    hide_spines(ax)
    ax.spines["bottom"].set_bounds(min(xticks), max(xticks))

    os.makedirs(save_path, exist_ok=True)

    fig.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_expit_surface(
    a_range: tuple = (-5, 5),
    ps_range: tuple = (0, 1),
    b: float = -2,
    c: float = -2,
    n_points: int = 100,
    log_scale: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot expit(a*t + b*(ps + c)) for varying 'a' and 'ps' values, with separate subplots for t=0 and t=1.

    Args:
        a_range (tuple): Range of 'a' values (min, max).
        ps_range (tuple): Range of propensity score values (min, max).
        b (float): Fixed value for parameter b.
        c (float): Fixed value for parameter c.
        n_points (int): Number of points to evaluate for each dimension.
        log_scale (bool): Whether to use a logarithmic color scale.

    Returns:
        tuple: Matplotlib figure and axes objects.
    """
    # Create meshgrid for 'a' and 'ps' values
    a_values = np.linspace(a_range[0], a_range[1], n_points)
    ps_values = np.linspace(ps_range[0], ps_range[1], n_points)
    A, PS = np.meshgrid(a_values, ps_values)

    # Calculate expit values for t=0 and t=1
    Z_t0 = expit(b * (PS + c))
    Z_t1 = expit(A + b * (PS + c))

    # Determine vmin and vmax
    Z_min = min(Z_t0.min(), Z_t1.min())
    Z_max = max(Z_t0.max(), Z_t1.max())

    # Prepare plotting parameters based on log_scale
    if log_scale:
        # For log scale, vmin must be greater than 0
        vmin = max(Z_min, 1e-10)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=Z_max)
        kwargs = {"norm": norm}
    else:
        kwargs = {"vmin": Z_min, "vmax": Z_max}
        norm = None  # Not used, but kept for clarity

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for t=0
    im1 = ax1.pcolormesh(A, PS, Z_t0, shading="auto", cmap="viridis", **kwargs)
    ax1.set_title(f"Treatment = 0\nb={b}, c={c}")
    ax1.set_xlabel("a")
    ax1.set_ylabel("Propensity Score")

    # Plot for t=1
    im2 = ax2.pcolormesh(A, PS, Z_t1, shading="auto", cmap="viridis", **kwargs)
    ax2.set_title(f"Treatment = 1\nb={b}, c={c}")
    ax2.set_xlabel("a")
    ax2.set_ylabel("Propensity Score")

    # Add colorbars
    colorbar_label = "Probability (log scale)" if log_scale else "Probability"
    fig.colorbar(im1, ax=ax1, label=colorbar_label)
    fig.colorbar(im2, ax=ax2, label=colorbar_label)

    # Adjust layout
    plt.tight_layout()

    return fig, (ax1, ax2)


def hide_spines(ax: plt.Axes) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)


def separate_treated_control(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate treated and control groups, using the target column."""
    treated = df[df["target"] == 1]
    control = df[df["target"] == 0]
    return treated, control


def set_plot_defaults(
    size_default: int = 14,
    size_large: int = 16,
    font_family: str = "Roboto",
    font_weight: str = "normal",
) -> None:
    plt.rc("font", family=font_family)
    plt.rc("font", weight=font_weight)
    plt.rc("font", size=size_default)
    plt.rc("axes", titlesize=size_large)
    plt.rc("axes", labelsize=size_large)
    plt.rc("xtick", labelsize=size_default)
    plt.rc("ytick", labelsize=size_default)


def plot_methods(df: pd.DataFrame, ax: plt.Axes, x_var: str, colors: list[str]) -> None:
    """
    Plot the effect estimates for each method.

    Args:
        df (pd.DataFrame): The dataframe containing the effect estimates.
        ax (plt.Axes): The axes to plot on.
        x_var (str): The variable to plot on the x-axis.
        colors (list[str]): The colors to use for the methods.
    """
    df = df.sort_values(by=["method", x_var])
    for i, method in enumerate(df["method"].unique()):
        df_temp = df[df["method"] == method]
        x_values = df_temp[x_var]
        means = df_temp["effect"]
        stds = df_temp["std_err"]

        ax.plot(x_values, means, label=method, color=colors[i], marker="o")
        ax.fill_between(
            x_values, means - stds, means + stds, alpha=0.2, color=colors[i]
        )


def plot_true_effect(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_var: str,
    baseline_color: str,
    true_effect_label: str,
) -> None:
    """
    Use counterfactual effect as baseline.
    """
    true_effect = df.loc[0, "effect_counterfactual"]
    max_x = max(df[x_var])
    ax.plot(
        [min(df[x_var]), max_x],
        [true_effect, true_effect],
        label=true_effect_label,
        color=baseline_color,
        linestyle="--",
        linewidth=1,
    )


def set_axes_labels(
    ax: plt.Axes, x_label: str, y_label: str, xticks: list[float], yticks: list[float]
) -> None:
    """Set the labels for the axes."""
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if xticks:
        ax.set_xticks(xticks)
        ax.set_xlim(min(xticks), max(xticks))
    if yticks:
        ax.set_yticks(yticks)
        ax.set_ylim(min(yticks), max(yticks))


def customize_plot_appearance(ax: plt.Axes, xticks: list[float], max_x: float) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_bounds(min(xticks), max(xticks) if xticks else max_x)
    ax.tick_params(axis="y", which="both", length=0)


def add_legend(ax: plt.Axes, n_methods: int) -> None:
    ax.legend(
        ncol=n_methods + 1, loc="upper center", bbox_to_anchor=(0.5, 1.1), frameon=False
    )


def save_plot(fig: plt.Figure, filename: str, save_path: str = "figures") -> None:
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

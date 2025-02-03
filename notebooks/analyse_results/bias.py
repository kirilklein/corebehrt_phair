# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: ehr2vec_phair
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


# %% [markdown]
# ## Example Data
#

# %%
# Create meshgrid of delta_ps and delta_y
dps = np.linspace(-1, 1, 11)
dy = np.linspace(-1, 1, 11)
dps_grid, dy_grid = np.meshgrid(dps, dy)

# Create example results with all combinations
example_results = {
    "dps": dps_grid.flatten(),
    "dy": dy_grid.flatten(),
    "effect_TMLE": np.random.rand(11*11),
    "effect_std_TMLE": np.random.rand(11*11),
    "effect_IPW": np.random.rand(11*11),
    "effect_std_IPW": np.random.rand(11*11),
    "effect_counterfactual": np.ones(11*11)*0.1
}

df = pd.DataFrame(example_results)

# %% [markdown]
# ## Z scores

# %%
df['z_score_TMLE'] = (df['effect_TMLE'] - df['effect_counterfactual']) / df['effect_std_TMLE']
df['z_score_IPW'] = (df['effect_IPW'] - df['effect_counterfactual']) / df['effect_std_IPW']

# Reshape data into 2D grid for heatmap
z_score_grid_TMLE = df['z_score_TMLE'].values.reshape(11, 11)
z_score_grid_IPW = df['z_score_IPW'].values.reshape(11, 11)

# Plot heatmaps
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(z_score_grid_TMLE, annot=False, fmt=".2f", cmap='RdBu_r', center=0)
plt.title('Z-scores: (TMLE - Counterfactual) / std_TMLE')
plt.subplot(1, 2, 2)
sns.heatmap(z_score_grid_IPW, annot=False, fmt=".2f", cmap='RdBu_r', center=0)
plt.title('Z-scores: (IPW - Counterfactual) / std_IPW')
plt.show()


# %% [markdown]
# ## Difference

# %%
def get_diff_grids(df: pd.DataFrame, methods: list, dps_col: str, dy_col: str, clip_min: float = None, clip_max: float = None) -> Tuple[dict, np.ndarray, np.ndarray]:  
    """
    Create 2D grids of differences between each method and counterfactual for heatmap plotting.
    
    Args:
        df: DataFrame containing the results
        methods: List of method names to process
        dps_col: Name of the propensity score delta column
        dy_col: Name of the outcome delta column
    
    Returns:
        Tuple containing:
        - dict: Method-specific difference grids
        - np.ndarray: Sorted unique propensity score deltas
        - np.ndarray: Sorted unique outcome deltas
    """
    diff_grids = {}
    dps_sorted = np.sort(df[dps_col].unique())
    dy_sorted = np.sort(df[dy_col].unique())
    for method in methods:
        diffs_sorted = []
        for dy in dy_sorted:
            for dps in dps_sorted:
                diffs_sorted.append(
                    df.loc[(df[dy_col] == dy) & (df[dps_col] == dps), f"diff_{method}"].values[0]
                )
        effect_grid = np.array(diffs_sorted).reshape(len(dy_sorted), len(dps_sorted))
        if clip_min is not None or clip_max is not None:
            effect_grid = np.clip(effect_grid, clip_min, clip_max)
        diff_grids[method] = effect_grid
    return diff_grids, dps_sorted, dy_sorted
def get_ticklabels(sorted_values: np.ndarray, skip: int = 2) -> list:
    """
    Convert array of values to formatted strings for plot tick labels.
    
    Args:
        sorted_values: Array of values to convert to labels
        skip: Show every nth label (default=2 for every second label)
    """
    labels = [f'{x:.2f}' if abs(x - round(x, 1)) > 0.001 else f'{x:.1f}' for x in sorted_values]
    # Replace labels we want to hide with empty strings
    return [label if i % skip == 0 else '' for i, label in enumerate(labels)]

def get_vmin_vmax(diff_grids: dict, methods: list) -> Tuple[float, float]:
    """Calculate the global min and max values across all method differences."""
    all_values = np.concatenate([diff_grids[m].flatten() for m in methods])
    return all_values.min(), all_values.max()

def plot_method_differences(
    df: pd.DataFrame,
    methods: list,
    dps_col: str = "dps",
    dy_col: str = "dy",
    xlabel: str = "delta_ps",
    ylabel: str = "delta_y",
    title_prefix: str = "Difference:",
    figsize: tuple = (18, 6),
    clip_min: float = None,
    clip_max: float = None,
    tick_skip: int = 1,
) -> None:
    """
    Plot heatmaps comparing different methods against counterfactual results.
    
    Args:
        df: DataFrame containing the results
        methods: List of method names to process
        dps_col: Name of the propensity score delta column
        dy_col: Name of the outcome delta column
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title_prefix: Prefix for plot titles
        figsize: Figure size as (width, height)
    """
    # Compute the differences relative to the counterfactual
    for method in methods:
        df[f'diff_{method}'] = df[f'effect_{method}'] - df['effect_counterfactual']

    # Get grids and value ranges
    diff_grids, dps_sorted, dy_sorted = get_diff_grids(df, methods, dps_col, dy_col, clip_min, clip_max)
    vmin, vmax = get_vmin_vmax(diff_grids, methods)

    # Plot heatmaps
    plt.figure(figsize=figsize)
    
    for i, method in enumerate(methods, start=1):
        ax = plt.subplot(1, len(methods), i)
        sns.heatmap(
            diff_grids[method],
            annot=False,
            fmt=".2f",
            cmap='RdBu_r',
            center=0,
            xticklabels=get_ticklabels(dps_sorted, tick_skip),
            yticklabels=get_ticklabels(dy_sorted, tick_skip),
            vmin=vmin,
            vmax=vmax
        )

        # Keep only every nth tick
        ax.set_xticks(ax.get_xticks()[::tick_skip])
        ax.set_yticks(ax.get_yticks()[::tick_skip])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.title(f'{title_prefix} {method} - Counterfactual')

    plt.tight_layout()
    plt.show()



# %%
methods = ["TMLE", "IPW", ]
plot_method_differences(df, methods, "dps", "dy", clip_min=-1, clip_max=.6)

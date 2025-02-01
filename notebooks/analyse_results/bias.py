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


# %%
# Create meshgrid of delta_ps and delta_y
delta_ps = np.linspace(-1, 1, 11)
delta_y = np.linspace(-1, 1, 11)
delta_ps_grid, delta_y_grid = np.meshgrid(delta_ps, delta_y)

# Create example results with all combinations
example_results = {
    "delta_ps": delta_ps_grid.flatten(),
    "delta_y": delta_y_grid.flatten(),
    "effect_TMLE": np.random.rand(11*11),
    "effect_std_TMLE": np.random.rand(11*11),
    "effect_IPW": np.random.rand(11*11),
    "effect_std_IPW": np.random.rand(11*11),
    "effect_counterfactual": np.ones(11*11)*0.1
}

df = pd.DataFrame(example_results)

# Calculate z-scores
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

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create meshgrid of delta_ps and delta_y
delta_ps = np.linspace(-1, 1, 11)
delta_y = np.linspace(-1, 1, 11)
delta_ps_grid, delta_y_grid = np.meshgrid(delta_ps, delta_y)

# Create example results with all combinations
example_results = {
    "delta_ps": delta_ps_grid.flatten(),
    "delta_y": delta_y_grid.flatten(),
    "effect_TMLE": np.random.rand(11*11),
    "effect_std_TMLE": np.random.rand(11*11),
    "effect_IPW": np.random.rand(11*11),
    "effect_AIPW": np.random.rand(11*11),
    "effect_std_IPW": np.random.rand(11*11),
    "effect_std_AIPW": np.random.rand(11*11),
    "effect_counterfactual": np.ones(11*11) * 0.1
}

df = pd.DataFrame(example_results)

# Calculate z-scores
df['z_score_TMLE'] = (df['effect_TMLE'] - df['effect_counterfactual']) / df['effect_std_TMLE']
df['z_score_IPW'] = (df['effect_IPW'] - df['effect_counterfactual']) / df['effect_std_IPW']
df['z_score_AIPW'] = (df['effect_AIPW'] - df['effect_counterfactual']) / df['effect_std_AIPW']

# Reshape data into 2D grid for heatmap
z_score_grid_TMLE = df['z_score_TMLE'].values.reshape(11, 11)
z_score_grid_IPW = df['z_score_IPW'].values.reshape(11, 11)
z_score_grid_AIPW = df['z_score_AIPW'].values.reshape(11, 11)
# Plot heatmaps with actual delta values as tick labels
plt.figure(figsize=(12, 6))

# TMLE heatmap
plt.subplot(1, 2, 1)
ax1 = sns.heatmap(
    z_score_grid_TMLE,
    annot=False,
    fmt=".2f",
    cmap='RdBu_r',
    center=0,
    xticklabels=[f'{x:.1f}' for x in delta_ps],
    yticklabels=[f'{x:.1f}' for x in delta_y]
)
ax1.set_xlabel('delta_ps')
ax1.set_ylabel('delta_y')
plt.title('Z-scores: (TMLE - Counterfactual) / std_TMLE')

# IPW heatmap
plt.subplot(1, 2, 2)
ax2 = sns.heatmap(
    z_score_grid_IPW,
    annot=False,
    fmt=".2f",
    cmap='RdBu_r',
    center=0,
    xticklabels=[f'{x:.1f}' for x in delta_ps],
    yticklabels=[f'{x:.1f}' for x in delta_y]
)
ax2.set_xlabel('delta_ps')
ax2.set_ylabel('delta_y')
plt.title('Z-scores: (IPW - Counterfactual) / std_IPW')

plt.tight_layout()
plt.show()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppose 'df' is your DataFrame with columns:
# ["delta_ps", "delta_y", "effect_TMLE", "effect_AIPW", "effect_IPW", "effect_counterfactual"]

methods = ["TMLE", "IPW", "AIPW"]

# 1. Compute the differences relative to the counterfactual
for method in methods:
    df[f'diff_{method}'] = df[f'effect_{method}'] - df['effect_counterfactual']

# 2. Sort the unique delta values so the axes will be labeled in ascending order
delta_ps_sorted = np.sort(df["delta_ps"].unique())  # for the X-axis
delta_y_sorted = np.sort(df["delta_y"].unique())    # for the Y-axis

# 3. Create 2D arrays (grids) matching (# of delta_y) rows × (# of delta_ps) columns
diff_grids = {}
for method in methods:
    diffs_sorted = []
    for dy in delta_y_sorted:
        for dps in delta_ps_sorted:
            diffs_sorted.append(
                df.loc[(df["delta_y"] == dy) & (df["delta_ps"] == dps), f"diff_{method}"].values[0]
            )
    diff_grids[method] = np.array(diffs_sorted).reshape(len(delta_y_sorted), len(delta_ps_sorted))

# Determine global color scale limits (same for all heatmaps)
all_values = np.concatenate([diff_grids[m].flatten() for m in methods])
vmin, vmax = all_values.min(), all_values.max()

# 4. Plot heatmaps
plt.figure(figsize=(18, 6))

for i, method in enumerate(methods, start=1):
    ax = plt.subplot(1, 3, i)
    sns.heatmap(
        diff_grids[method],
        annot=False,
        fmt=".2f",
        cmap='RdBu_r',
        center=0,
        xticklabels=[f'{x:.1f}' for x in delta_ps_sorted],
        yticklabels=[f'{y:.1f}' for y in delta_y_sorted],
        vmin=vmin,  # Set fixed color range
        vmax=vmax
    )
    ax.set_xlabel('delta_ps')
    ax.set_ylabel('delta_y')
    plt.title(f'Difference: {method} - Counterfactual')

plt.tight_layout()
plt.show()


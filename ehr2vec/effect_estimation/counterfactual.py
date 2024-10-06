import pandas as pd


def compute_effect_from_counterfactuals(df: pd.DataFrame, effect_type: str) -> float:
    """
    Computes the effect from counterfactual outcomes.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'Y1', 'Y0', and 'treatment'.
        effect_type (str): The type of effect to compute. Options are 'ATE', 'ATT', 'ATC', 'RR', 'OR'.

    Returns:
        float: The computed effect.

    Raises:
        ValueError: If the effect type is not recognized.
    """
    y1_mean = df["Y1"].mean()
    y0_mean = df["Y0"].mean()

    if effect_type == "ATE":
        effect = y1_mean - y0_mean
    elif effect_type in ["ATT", "ATC"]:
        treated_flag = 1 if effect_type == "ATT" else 0
        subset = df[df["treatment"] == treated_flag]
        effect = subset["Y1"].mean() - subset["Y0"].mean()
    elif effect_type == "RR":
        effect = (y1_mean + 1) / (y0_mean + 1)
    elif effect_type == "OR":
        effect = (y1_mean / (1 - y1_mean)) / (y0_mean / (1 - y0_mean))
    else:
        raise ValueError(f"Effect type '{effect_type}' is not recognized.")

    return effect

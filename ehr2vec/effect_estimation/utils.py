import pandas as pd


def convert_effect_to_dataframe(effect: dict) -> pd.DataFrame:
    """
    Convert the effect estimates to a DataFrame.

    Args:
        effect (dict): The effect estimates.

    Returns:
        pd.DataFrame: The effect estimates as a DataFrame.
    """
    return (
        pd.DataFrame.from_dict(effect, orient="index")
        .reset_index()
        .rename(columns={"index": "method"})
    )

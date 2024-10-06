import pandas as pd
from typing import List
import os


def check_path(path: str, filename: str) -> None:
    """Check if a file exists in a path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {filename} found in {path}.")


def check_columns(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """Check if the dataframe has the expected columns."""
    if not all([col in df.columns for col in expected_columns]):
        raise ValueError(f"Expected columns {expected_columns} not found in dataframe.")

import os
from typing import List

import pandas as pd

from ehr2vec.common.default_args import PID_COL


def check_path(path: str, filename: str) -> None:
    """Check if a file exists in a path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No {filename} found in {path}.")


def check_columns(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """Check if the dataframe has the expected columns."""
    if not all([col in df.columns for col in expected_columns]):
        raise ValueError(f"Expected columns {expected_columns} not found in dataframe.")


def check_pids(df1: pd.DataFrame, df2: pd.DataFrame, pid_col: str = PID_COL) -> None:
    """Check that both dataframes have the same set of PIDs."""
    df1_pids = set(df1[pid_col])
    df2_pids = set(df2[pid_col])

    if df1_pids != df2_pids:
        raise ValueError(
            f"DataFrames have different PIDs. "
            f"Missing PIDs: {df1_pids - df2_pids}, "
            f"Extra PIDs: {df2_pids - df1_pids}"
        )

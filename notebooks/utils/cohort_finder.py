"""Utilities for analyzing patient cohorts in EHR data."""
from typing import List, Dict, Set
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Column names for easy reference
PATIENT_COL = "PATIENT"
CODE_COL = "CODE"
DESCRIPTION_COL = "DESCRIPTION"

class CohortUtils:
    """Utilities for analyzing patient cohorts by medical codes and descriptions."""

    @staticmethod
    def get_unique_patients_with_code(df: pd.DataFrame, code: str) -> Set[str]:
        """Return unique patients with a given medical code."""
        return set(df[df[CODE_COL] == code][PATIENT_COL].unique())

    @staticmethod
    def get_rows_matching_description_pattern(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
        """Return rows where the description matches the given pattern."""
        return df[df[DESCRIPTION_COL].str.contains(pattern, na=False)]

    @staticmethod
    def extract_all_codes_from_description_pattern(df: pd.DataFrame, pattern: str) -> List[str]:
        """Extract unique codes from descriptions matching the given pattern."""
        return CohortUtils.get_rows_matching_description_pattern(df, pattern)[CODE_COL].unique().tolist()

    @staticmethod
    def get_number_of_exposed(df: pd.DataFrame, pattern: str) -> int:
        """Count unique patients exposed to codes matching the pattern."""
        unique_codes = CohortUtils.extract_all_codes_from_description_pattern(df, pattern)
        unique_patients = {patient for code in unique_codes 
                           for patient in CohortUtils.get_unique_patients_with_code(df, code)}
        
        logger.info(f"Found {len(unique_patients)} patients matching pattern: {pattern}")
        return len(unique_patients)

    @staticmethod
    def break_down_by_code(df: pd.DataFrame, pattern: str) -> Dict[str, int]:
        """Return patient counts for each code matching the pattern."""
        unique_codes = CohortUtils.extract_all_codes_from_description_pattern(df, pattern)
        code_to_count = {code: len(CohortUtils.get_unique_patients_with_code(df, code)) for code in unique_codes}
        
        logger.info(f"Found {len(code_to_count)} unique codes matching pattern: {pattern}")
        return code_to_count

    @staticmethod
    def search_by_name(df: pd.DataFrame, name: str) -> Dict[str, str]:
        """
        Search for diagnostic descriptions matching a given name.
        Args:
            df: DataFrame containing DESCRIPTION and CODE columns
            name: String to search for in descriptions
        Returns:
            Dictionary mapping descriptions to their corresponding codes
        """
        mask = df[DESCRIPTION_COL].str.lower().str.contains(name.lower())
        # get a dictionary of unique descriptions to their codes
        masked_df = df.loc[mask, [DESCRIPTION_COL, CODE_COL]].drop_duplicates()
        return dict(zip(masked_df[DESCRIPTION_COL], masked_df[CODE_COL]))
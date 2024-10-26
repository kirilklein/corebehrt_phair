from typing import List, Dict
import pandas as pd

class CohortAnalyzer:
    """Analyzes EHR data to identify and characterize patient cohorts based on medical codes and descriptions."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the CohortAnalyzer with a DataFrame containing EHR data.
        
        Args:
            df: DataFrame with columns ['PATIENT', 'CODE', 'DESCRIPTION']
        """
        self._validate_dataframe(df)
        self.df = df
    
    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """Validate that the DataFrame has the required columns."""
        required_columns = {'PATIENT', 'CODE', 'DESCRIPTION'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    def get_patients_with_code(self, code: str) -> List[str]:
        """
        Get unique patients with a specific medical code.
        
        Args:
            code: Medical code to search for
            
        Returns:
            List of unique patient identifiers
        """
        return self.df[self.df["CODE"] == code]["PATIENT"].unique().tolist()

    def find_matching_descriptions(self, pattern: str) -> pd.DataFrame:
        """
        Find all rows where description matches a pattern.
        
        Args:
            pattern: Regular expression pattern to match in descriptions
            
        Returns:
            DataFrame with matching rows
        """
        return self.df[self.df["DESCRIPTION"].str.contains(pattern, na=False)]

    def get_codes_from_description(self, pattern: str) -> List[str]:
        """
        Extract all unique codes from descriptions matching a pattern.
        
        Args:
            pattern: Regular expression pattern to match in descriptions
            
        Returns:
            List of unique medical codes
        """
        matching_rows = self.find_matching_descriptions(pattern)
        return matching_rows["CODE"].unique().tolist()

    def count_exposed_patients(self, pattern: str) -> int:
        """
        Count unique patients with codes matching a description pattern.
        
        Args:
            pattern: Regular expression pattern to match in descriptions
            
        Returns:
            Number of unique patients
        """
        codes = self.get_codes_from_description(pattern)
        unique_patients = set()
        for code in codes:
            unique_patients.update(self.get_patients_with_code(code))
        return len(unique_patients)

    def get_code_distribution(self, pattern: str) -> Dict[str, int]:
        """
        Get distribution of patients across codes matching a description pattern.
        
        Args:
            pattern: Regular expression pattern to match in descriptions
            
        Returns:
            Dictionary mapping codes to patient counts
        """
        codes = self.get_codes_from_description(pattern)
        return {
            code: len(self.get_patients_with_code(code))
            for code in codes
        }

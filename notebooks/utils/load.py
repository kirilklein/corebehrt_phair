import os
import pandas as pd


def load_estimates(folder: str, is_noise: bool = False, is_patient_number: bool = False) -> pd.DataFrame:
    """
    Load estimates from the given folder.

    Args:
        folder (str): The path to the directory containing experiment folders.
        is_noise (bool): If True, loads noise-based estimates; otherwise, loads patient-number-based estimates.

    Returns:
        pd.DataFrame: Combined DataFrame of all experiment data with an additional column
                      indicating noise level or patient count, depending on `is_noise`.
    """
    dfs = []
    for path in os.listdir(folder):
        if is_noise and is_patient_number:
            raise ValueError("Cannot set both is_noise and is_patient_number to True")
        if is_valid_experiment_folder(path, is_noise, is_patient_number):
            file_path = os.path.join(folder, path, "effect.csv")
            df_temp = pd.read_csv(file_path)
            if is_noise:
                df_temp["noise_level"] = extract_noise_level(path)
            elif is_patient_number:
                df_temp["patient_number"] = extract_patient_number(path)
                
            dfs.append(df_temp)
    return pd.concat(dfs).reset_index(drop=True)

def is_valid_experiment_folder(path: str, is_noise: bool, is_patient_number: bool) -> bool:
    """
    Check if the folder matches the expected pattern.
    Args:
        path (str): The folder name to check
        is_noise (bool): If True, checks for noise experiment pattern
        is_patient_number (bool): If True, checks for patient number pattern

    Returns:
        bool: True if the folder matches the expected pattern

    Examples:
        >>> is_valid_experiment_folder("experiment_noise_0.1", True, False)
        True
        >>> is_valid_experiment_folder("experiment_n_100", False, True)
        True
     """
    if not isinstance(path, str):
        return False
    if is_noise:
        return path.startswith("experiment_noise_")
    elif is_patient_number:
        return path.startswith("experiment_n_")
    else:
        return True

def extract_noise_level(path: str) -> float:
    """
    Extract the noise level from the folder name.
    """
    return float(path.split("_")[-1])

def extract_patient_number(path: str) -> int:
    """
    Extract the patient number from the folder name.
    """
    return int(path.split("_")[-1])




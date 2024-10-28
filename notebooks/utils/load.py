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
        if is_valid_experiment_folder(path, is_noise, is_patient_number):
            file_path = os.path.join(folder, path, "effect.csv")
            df_temp = pd.read_csv(file_path)
            if is_noise:
                df_temp["noise"] = extract_noise_level(path)
            elif is_patient_number:
                df_temp["N"] = extract_patient_number(path)
                
            dfs.append(df_temp)
    return pd.concat(dfs).reset_index(drop=True)

def is_valid_experiment_folder(path: str, is_noise: bool, is_patient_number: bool) -> bool:
    """
    Check if the folder matches the expected pattern.
    """
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




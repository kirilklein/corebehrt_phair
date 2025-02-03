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
#     display_name: patbert
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from os.path import join
from utils.cohort_finder import CohortUtils
import json


# %%
from pathlib import Path


def remove_tmp_prefixes(path: str) -> Path:
    """Remove 'tmp' prefixes from a path."""
    path_parts = Path(path).parts
    start_index = next(
        (i for i, part in enumerate(path_parts) if not part.startswith("tmp")), 1
    )
    return Path(*path_parts[start_index:])


# %%
remove_tmp_prefixes("tmp/tmp223/sas")

# %%
with open("../ehr2vec/azure_run/credentials.json", "r") as f:
    credentials = json.load(f)
credentials


# %%
raw_data_dir = (
    r"C:\Users\fjn197\PhD\projects\PHAIR\pipelines\synthea\output\synthea5000\csv"
)
med = pd.read_csv(join(raw_data_dir, "medications.csv"))
diag = pd.read_csv(join(raw_data_dir, "conditions.csv"))


# %%
# print(CohortUtils.get_number_of_exposed(med, "amLODIPine"))
# print(CohortUtils.break_down_by_code(med, "amlodipine"))
# print(CohortUtils.break_down_by_code(med, "amLODIPine"))
print(diag.PATIENT.nunique(), "Patients with diagnoses")

print(CohortUtils.get_number_of_exposed(med, "Ibuprofen"))
print(CohortUtils.break_down_by_code(med, "Ibuprofen"))
# print(CohortUtils.get_number_of_exposed(med, "Paracetamol"))
# print(CohortUtils.break_down_by_code(med, "Paracetamol"))


# %%
heart_disease = CohortUtils.search_by_name(diag, "heart")
stroke = CohortUtils.search_by_name(diag, "stroke")
infarction = CohortUtils.search_by_name(diag, "infarction")
tvr = CohortUtils.search_by_name(diag, "revascularization")
cardiovascular = CohortUtils.search_by_name(diag, "cardio*")
print(cardiovascular)
print(heart_disease)


# %%
CohortUtils.search_by_name(med, "amLODIPine")

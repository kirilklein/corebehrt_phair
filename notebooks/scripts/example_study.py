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


# %%
raw_data_dir = (
    r"C:\Users\fjn197\PhD\projects\PHAIR\pipelines\synthea\output\synthea5000\csv"
)
med = pd.read_csv(join(raw_data_dir, "medications.csv"))
diag = pd.read_csv(join(raw_data_dir, "conditions.csv"))

print(CohortUtils.get_number_of_exposed(med, "amLODIPine"))
print(CohortUtils.break_down_by_code(med, "amLODIPine"))

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

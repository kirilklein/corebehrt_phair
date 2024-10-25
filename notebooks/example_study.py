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


data_dir = r"C:\Users\fjn197\PhD\projects\PHAIR\pipelines\corebehrt_phair\ehr2vec\example_data\synthea500"
diag = pd.read_csv(join(data_dir, "concept.diagnose.csv"))
med = pd.read_csv(join(data_dir, "concept.medication.csv"))


# %%
patient_concept_counts = (
    med.groupby("CONCEPT")["PID"].nunique().sort_values(ascending=False)
)

# %%
patient_concept_counts[:20]

# %%
diag_full = pd.read_csv(
    r"C:\Users\fjn197\PhD\projects\PHAIR\pipelines\synthea\output\synthea5000\csv\conditions.csv"
)


# %%
def search_by_name(df: pd.DataFrame, name: str) -> dict[str, str]:
    """
    Search for diagnostic descriptions matching a given name.
    Args:
        df: DataFrame containing DESCRIPTION and CODE columns
        name: String to search for in descriptions
    Returns:
        Dictionary mapping descriptions to their corresponding codes
    """
    mask = df.DESCRIPTION.str.lower().str.contains(name.lower())
    # get a dictionary of unique descriptions to their codes
    masked_df = df.loc[mask, ["DESCRIPTION", "CODE"]].drop_duplicates()
    return dict(zip(masked_df.DESCRIPTION, masked_df.CODE))


# %%
heart_disease = search_by_name(diag_full, "heart")
stroke = search_by_name(diag_full, "stroke")
infarction = search_by_name(diag_full, "infarction")
tvr = search_by_name(diag_full, "revascularization")
print(tvr)

# %% [markdown]
# ## Examine MACE outcomes

# %%
mace_codes = [
    22298006,
    230690007,
    4557003,
    84114007,
    414545008,
    230690007,
    401303003,
    401314000,
    22298006,
]
mace_codes = ["D" + str(x) for x in mace_codes]
mace_pids = []
for mace_code in mace_codes:
    print(mace_code, end=" ")
    mace_df = diag[diag["CONCEPT"] == mace_code]
    print("number of patients", mace_df.PID.nunique(), end=" ")
    print(
        "number of events",
        mace_df.shape[0],
    )
    mace_pids.extend(mace_df.PID.unique())

# %%
diag.CONCEPT.value_counts()[25:40]

# %% [markdown]
# ## Examine statin codes

# %%
med

# %%
statin_codes = [617312, 312961, 312962, 19821, 7597, 34482]
statin_codes = ["M" + str(x) for x in statin_codes]
for statin_code in statin_codes:
    print(statin_code)
    print(
        "number of patients",
        med[med["CONCEPT"].str.startswith(statin_code)].PID.nunique(),
    )
    print("number of events", med[med["CONCEPT"].str.startswith(statin_code)].shape[0])

# %% [markdown]
# We will take M312961 (Simvastatin) as an example since that is quite often used.

# %%
simvastatin_codes = [312961, 312962, 1189803, 484211, 1430892, 803516]
simvastatin_codes = ["M" + str(x) for x in simvastatin_codes]
simvastatin_pids = []
for statin_code in simvastatin_codes:
    print(statin_code)
    simvastatin_df = med[med["CONCEPT"] == statin_code]
    print("number of patients", simvastatin_df.PID.nunique())
    print("number of events", simvastatin_df.shape[0])
    simvastatin_pids.extend(simvastatin_df.PID.unique())

# %%
print("number of patients with MACE", len(set(mace_pids)))
print("number of patients with simvastatin", len(set(simvastatin_pids)))
print(
    "number of patients with both",
    len(set(mace_pids).intersection(set(simvastatin_pids))),
)

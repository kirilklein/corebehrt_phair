# CORE-BEHRT: A Carefully Optimized and Rigorously Evaluated BEHRT

## Overview
CORE-BEHRT aims to provide a robust framework for reproducing state-of-the-art results in EHR data analysis using a BEHRT-like model. Follow the steps below to reproduce the results.

## Prerequisites
Before you begin, ensure you have the necessary dependencies installed. This project may require:
- PyTorch
- transformers
- Numpy
- Pandas
- scikit_learn
- tqdm
- matplotlib
- (pyarrow if parquet files are used)

## Getting Started
1. Activate your virtual environment, e.g. conda activate myenv or source myenv/bin/activate for pip.
2. From the root directory, run `pip install -e .` to install the package.
3. Run the scripts in the following order to preprocess your data, train the model, and evaluate the results.
   1. **Data Preparation**: `main_create_data`
   2. **Model Pre-training**: `main_pretrain`
   3. **Data Preparation for Fine-tuning**: `main_create_outcomes`
   4. **Model Fine-tuning**: `main_finetune_cv`

Scripts can be run from the root directory using the following command: `python -m ehr2vec.scripts.{script_name}` 
By default example configs are used.  
To use your own config, pass the path (relative to configs directory) to the config file as an argument,  
e.g. `python -m ehr2vec.scripts.main_pretrain --config_path my_config.yaml` 

### Data Preparation
To correctly prepare your data for processing, execute the scripts in the following order. Ensure your data adheres to the specified format before starting:

1. **Data Format**
   - **Required Data Format:**
     - **Patient Data**: Ensure the file `patient_format.csv` contains columns for `PID`, `DATE_OF_BIRTH`, `DATE_OF_DEATH`, `RACE`, `GENDER`, and other relevant background features.
     - **Event Data**: The files `concept.{code_type}.csv` should include `TIMESTAMP`, `PID`, `ADMISSION_ID`, and `CONCEPT`.
   - Use the preprocessing tools available at [ehr_preprocess](https://github.com/kirilklein/ehr_preprocess.git) to convert your raw data into the required format.


### Evaluation and Visualization
To evaluate the model performance:
- `cv_get_stats`: Get basic statistics on patient sequences in train/val and test set.
- `finetune_cv_evaluate`: Test models trained in CV on a dataset of your choice.


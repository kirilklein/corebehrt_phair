# CORE-BEHRT: Carefully Optimized and Rigorously Evaluated BEHRT

[![Formatting using black](https://github.com/kirilklein/corebehrt_phair/actions/workflows/format.yml/badge.svg)](https://github.com/kirilklein/corebehrt_phair/actions/workflows/format.yml)
[![Lint using flake8](https://github.com/kirilklein/corebehrt_phair/actions/workflows/lint.yml/badge.svg)](https://github.com/kirilklein/corebehrt_phair/actions/workflows/lint.yml)
[![Pipeline test](https://github.com/kirilklein/corebehrt_phair/actions/workflows/pipeline.yml/badge.svg)](https://github.com/kirilklein/corebehrt_phair/actions/workflows/pipeline.yml)
[![Unittests](https://github.com/kirilklein/corebehrt_phair/actions/workflows/unittest.yml/badge.svg)](https://github.com/kirilklein/corebehrt_phair/actions/workflows/unittest.yml)

## Overview

CORE-BEHRT is an advanced framework for analyzing Electronic Health Records (EHR) using a BERT-like model optimized for healthcare data. This project aims to provide a robust, reproducible, and state-of-the-art solution for EHR data analysis.

## Features

- Efficient data preprocessing pipeline
- Customizable BEHRT model architecture
- Comprehensive training and evaluation scripts
- Cross-validation support for robust model assessment
- Visualization tools for result interpretation

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch 1.7+
- transformers 4.0+
- NumPy
- Pandas
- scikit-learn
- tqdm
- matplotlib
- pyarrow (for Parquet file support)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/core-behrt.git
   cd core-behrt
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the package:

   ```bash
   pip install -e .
   ```

## Usage

Follow these steps to preprocess your data, train the model, and evaluate the results:

1. **Data Preparation**:

   ```bash
   python -m ehr2vec.scripts.main_create_data
   ```

2. **Model Pre-training**:

   ```bash
   python -m ehr2vec.scripts.main_pretrain
   ```

3. **Prepare Fine-tuning Data**:

   ```bash
   python -m ehr2vec.scripts.main_create_outcomes
   ```

4. **Model Fine-tuning**:

   ```bash
   python -m ehr2vec.scripts.main_finetune_cv
   ```

To use a custom configuration, pass the path to your config file:

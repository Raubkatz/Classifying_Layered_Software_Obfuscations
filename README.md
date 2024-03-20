# Classifying_Layered_Software_Obfuscations

## Overview
This project aims to identify and classify obfuscation schemes in software metrics using machine learning models. It involves preprocessing data, splitting datasets for training and testing, applying data augmentation techniques, and training models for the classification task.

## File Descriptions

- `data_split/train_programs.txt`: List of programs used for training the models.
- `data_split/test_programs.txt`: List of programs reserved for testing the models.
- `original_data/results_2023_11_09.csv`: Results file containing output from initial experiments.
- `original_data/results_new.csv`: Most recent results file containing updated experiment outcomes.
- `01_fin_prepare_data.py`: Script for initial data preprocessing and preparation.
- `02_fin_train_test_split_ADASYN.py`: Script for splitting the data into training and test sets and applying the ADASYN technique for data augmentation.
- `03_fin_ML_ExtraTrees.py`: Script for training and evaluating the ExtraTrees classifier.
- `03_fin_ML_LGBM.py`: Script for training and evaluating the LightGBM model.
- `03_fin_ML_XGBoost.py`: Script for training and evaluating the XGBoost model.

## Usage

To run the scripts, use the following commands in your terminal:

```bash
python 01_fin_prepare_data.py
python 02_fin_train_test_split_ADASYN.py
python 03_fin_ML_ExtraTrees.py
python 03_fin_ML_LGBM.py
python 03_fin_ML_XGBoost.py


"""
This program is designed for preprocessing and preparing datasets for machine learning tasks, specifically focusing on handling imbalanced data through oversampling. It provides options for two types of splits (regular and program-based), applies standard scaling, and performs oversampling using ADASYN to address class imbalance. It operates on datasets named compiler_data_y2.csv, compiler_data_y3.csv, compiler_data_y4.csv, and compiler_data_y5.csv, which represent different sets of features and target variables for classification problems.

Key Functionalities:
- Set a consistent random seed for reproducibility across all operations that involve random processes.
- Load datasets based on user selection, dynamically choosing the target variable (`y3`, `y4`, or `y5`) while excluding the program name and the selected target variable from the feature set.
- Split data either through a conventional train-test split or based on predefined program names, facilitating experiments that require program-specific training and testing sets.
- Perform data scaling using StandardScaler to standardize features by removing the mean and scaling to unit variance.
- Apply oversampling with ADASYN (Adaptive Synthetic Sampling) to create synthetic samples of the minority classes, aiming to achieve a specified sample count for each class to balance the dataset.
- Save processed datasets (both training and testing sets) into specified directories, ensuring data is ready for subsequent modeling steps.

Parameters:
- split_option (int): Determines the type of split. `0` for a regular train-test split, `1` for a split based on predefined program names.
- oversampling (bool): Indicates whether to apply oversampling to address class imbalance.
- specific_sample_count (int): Target number of samples per class after oversampling.
- seed (int): Seed for random number generation, ensuring reproducibility.
- choose_y (int): Selector for the target variable. `0` for `y3`, `1` for `y4`, `2` for `y5`.

Outputs:
- Processed datasets (training and testing) are saved in a designated folder, ready for machine learning modeling. Oversampling is applied as per configuration to address class imbalance.

Usage:
Set the PARAMETERS section according to the experiment requirements. Ensure the data files are in the correct directory and named appropriately. Run the script to preprocess the data, apply oversampling if enabled, and save the processed datasets for further analysis or modeling.
"""
import random
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from collections import Counter

def set_seed(seed=42):
    """
    Sets a consistent seed for all random number generators for reproducibility.

    Parameters:
    - seed (int): The seed value to use for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_dataset(choose_y):
    """
    Loads the dataset corresponding to the choice of target variable (y3, y4, y5).

    Parameters:
    - choose_y (int): Selector for the target variable. 0 for y3, 1 for y4, 2 for y5.

    Returns:
    - df (DataFrame): The loaded dataset including features, target variable, and 'samplename'.
    """
    filename = f'./compiler_data/compiler_data_y{choose_y + 3}.csv'  # +3 offsets the index to match y naming convention
    df = pd.read_csv(filename)
    return df


def split_data(df, split_option, choose_y, train_programs=None, test_programs=None):
    """
    Splits the data into training and testing sets based on the split option.

    Parameters:
    - df (DataFrame): The complete dataset including 'samplename'.
    - split_option (int): 0 for regular train-test split, 1 for split based on program names.
    - choose_y (int): Selector for the target variable. 0 for y3, 1 for y4, 2 for y5.
    - train_programs (array-like, optional): List of program names for training. Required if split_option is 1.
    - test_programs (array-like, optional): List of program names for testing. Required if split_option is 1.

    Returns:
    - X_train, X_test, y_train, y_test (tuple): The split datasets.
    """
    y = df[f'y{choose_y + 3}']
    X = df.drop([f'y{choose_y + 3}'], axis=1)

    if split_option == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
    else:
        train_rows = df['samplename'].isin(train_programs)
        test_rows = df['samplename'].isin(test_programs)
        X_train, X_test, y_train, y_test = X[train_rows], X[test_rows], y[train_rows], y[test_rows]

    # Drop 'samplename' after splitting
    X_train = X_train.drop(['samplename'], axis=1)
    X_test = X_test.drop(['samplename'], axis=1)

    return X_train, X_test, y_train, y_test

def perform_oversampling(X_train, y_train, oversampling, specific_sample_count):
    """
    Applies oversampling to the training data using the ADASYN algorithm to mitigate class imbalance.

    Parameters:
    - X_train (DataFrame or ndarray): The training feature set.
    - y_train (Series or ndarray): The target values for the training set.
    - oversampling (bool): Flag indicating whether oversampling should be applied.
    - specific_sample_count (int): The target number of samples for each class after oversampling.

    Returns:
    - X_resampled, y_resampled (tuple): The resampled feature set and target values, respectively. If oversampling is not applied, returns the original datasets.
    """
    if not oversampling:
        return X_train, y_train
    oversampler = ADASYN(random_state=42, sampling_strategy={cls: specific_sample_count for cls in np.unique(y_train)})
    return oversampler.fit_resample(X_train, y_train)

def save_datasets(X_train, y_train, X_test, y_test, folder_name):
    """
    Saves the processed datasets (training and testing) to CSV files in a specified directory.

    Parameters:
    - X_train (DataFrame): The processed training feature set.
    - y_train (DataFrame): The target values for the training set.
    - X_test (DataFrame): The processed testing feature set.
    - y_test (DataFrame): The target values for the testing set.
    - folder_name (str): The directory where the CSV files will be saved.

    Outputs:
    - CSV files for each dataset are saved in the specified folder. Folder is created if it does not already exist.
    """
    os.makedirs(folder_name, exist_ok=True)
    paths = [os.path.join(folder_name, f'{kind}.csv') for kind in ('X_train', 'y_train', 'X_test', 'y_test')]
    for df, path in zip((X_train, y_train, X_test, y_test), paths):
        df.to_csv(path, index=False)
    print(f"Datasets saved in: {folder_name}")

########################################################################################################################
# PARAMETERS ###########################################################################################################
########################################################################################################################
split_option = 0  # Defines the type of data split. `0` for a regular train-test split, `1` for a program-based split.
oversampling = True # Determines whether to apply oversampling to balance the class distribution.
specific_sample_count = 2000 # The target number of instances per class after applying oversampling.
seed = 137 # Seed for random number generators to ensure reproducibility.
choose_y = 1  # Selector for the target variable: `0` for `y3`, `1` for `y4`, `2` for `y5`.
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Set the random seed across all random number generators for reproducibility.
set_seed(seed)

# Load the dataset selected by the user, including the target variable and features.
df = load_dataset(choose_y)

# Load predefined train and test program names if a program-based split is chosen.
if split_option == 1:
    train_programs = np.loadtxt("./data_split/train_programs.txt", dtype=str)
    test_programs = np.loadtxt("./data_split/test_programs.txt", dtype=str)
elif split_option == 0:
    train_programs = None
    test_programs = None

# Split the dataset into training and testing sets based on the chosen split option.
X_train, X_test, y_train, y_test = split_data(df, split_option, choose_y, train_programs, test_programs)

# Standardize the features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform oversampling on the training set if enabled to address class imbalance.
X_train_oversampled, y_train_oversampled = perform_oversampling(X_train_scaled, y_train, oversampling, specific_sample_count)

# Convert the numpy arrays back to pandas DataFrame to maintain column names and ease of use.
X_train_df = pd.DataFrame(X_train_oversampled, columns=X_train.columns)
y_train_df = pd.DataFrame(y_train_oversampled, columns=[f'y{choose_y+3}'])  # Reconstruct y column name based on choice
X_test_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)
y_test_df = pd.DataFrame(y_test, columns=[f'y{choose_y+3}'])

# Construct the folder name and save the processed datasets to CSV files.
folder_name = f"./data_split/training_data_seed{seed}_sp{split_option}_os{oversampling}_sc{specific_sample_count}_y{choose_y+3}"
save_datasets(X_train_df, y_train_df, X_test_df, y_test_df, folder_name)

# Count the instances of each class in the oversampled dataset and print
class_distribution = Counter(y_train_oversampled)
print(f'Number of Classes: {len(class_distribution)}\nCorresponding Distribution: {dict(class_distribution)}')

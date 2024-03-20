"""
This script is dedicated to the initial preparation and preprocessing of datasets for subsequent machine learning tasks. It focuses on addressing the common challenge of imbalanced datasets in classification problems by employing the ADASYN (Adaptive Synthetic Sampling) method for oversampling. The script offers flexibility in handling different datasets and target variables, enabling tailored data preparation steps that include splitting, scaling, and balancing the dataset.

Main functionalities include:
- Loading multiple datasets with the option to select the target variable dynamically. This accommodates varying classification objectives within the same framework.
- Splitting the dataset into training and testing sets, with support for both traditional random splits and program-based splits. The latter is particularly useful for scenarios where data from specific programs should be exclusively used for training or testing, ensuring that the model's performance is evaluated on unseen programs.
- Standardizing the feature set using `StandardScaler`, normalizing data to have a mean of 0 and a standard deviation of 1. This step is crucial for models that are sensitive to the scale of input features.
- Addressing class imbalance through oversampling with ADASYN. This approach generates synthetic samples for minority classes, aiming to create a more balanced distribution of classes in the training set. The script allows for specifying the desired sample count for each class, offering control over the extent of balancing.
- Saving the processed datasets to specified directories. This ensures that the data is readily available for the next steps in the modeling process, such as feature selection or model training.

Usage guidelines:
1. Specify the parameters at the beginning of the script according to the requirements of your project. These parameters include the choice of target variable, the type of data split, and the decision on whether to apply oversampling.
2. Ensure that the input data files are correctly named and placed in the expected directory.
3. Run the script. Processed datasets will be saved in the designated folders, ready for further analysis or direct use in machine learning models.

By automating the data preparation process, this script streamlines the initial stages of machine learning projects, allowing data scientists and researchers to focus on model selection, tuning, and evaluation.
"""
# Import necessary libraries for data manipulation and visualization.
import numpy as np  # For numerical operations and array manipulation.
import matplotlib.pyplot as plt  # For plotting and visualizing data.
import pandas as pd  # For handling and manipulating datasets in tabular form.
import seaborn as sns  # For advanced data visualization.
import os  # For creating and managing folders
import copy  # For creating shallow and deep copies of objects.
from copy import deepcopy as dc  # Importing 'deepcopy' directly for deep copying complex objects like dataframes.

# Load datasets from CSV files. Specify the separator (';') and decimal point representation ('.').
compiler_data1 = pd.read_csv('./original_data/results_2023_11_09.csv', sep=';', decimal='.')
compiler_data2 = pd.read_csv('./original_data/results_new.csv', sep=';', decimal='.')

# Merge the two datasets into one, removing any duplicate entries, and reset the index for clean access.
compiler_data_all = pd.concat([compiler_data1,compiler_data2]).drop_duplicates().reset_index(drop='index')

# Filter out rows from the dataset where the 'samplename' column contains a period ('.').
# This step may aim to exclude certain types of samples based on naming conventions.
compiler_data = dc(compiler_data_all[~compiler_data_all['samplename'].str.contains('\.')])

# Exclude samples compiled with specific compilers deemed non-important for the analysis.
# Here, 'prog_tinycc-latest-default' and 'prog_tendra-latest-default' compilers are excluded.
compiler_data = compiler_data[compiler_data.compiler != 'prog_tinycc-latest-default']
compiler_data = compiler_data[compiler_data.compiler != 'prog_tendra-latest-default']

# Perform and display a preliminary analysis on the 'samplename' column to understand its composition.
# This includes counting unique samples, displaying the frequency of each sample, and listing unique sample names.
print('Sample-Analysis')
print(f"Sample uniques: {compiler_data['samplename'].nunique()}")  # Number of unique sample names.
print(compiler_data['samplename'].value_counts())  # Frequency count of each sample name.
print(compiler_data['samplename'].unique())  # Array of unique sample names.

## Set y and X for the models
y = compiler_data.iloc[:, 2]  # set y to be the compiler column

# using all features
#X = compiler_data.drop(['compiler', 'Information_Flow', 'sampletype', 'function'], axis=1) # drop unnecessary columns
X = compiler_data.drop(['compiler', 'Information_Flow', 'sampletype', 'function', 'Halstead_Time'], axis=1) # drop unnecessary columns


X4 = compiler_data.drop(['compiler', 'Information_Flow', 'sampletype', 'function',
                         'ABC', 'B', 'Halstead_Difficulty', 'Halstead_Effort', 'Halstead_Time', 'MIwoc', 'Myers_Interval'],
                        axis=1)

# combine o levels for compilers corresponding to no obfuscation
y2 = copy.deepcopy(y)
y2[(y2 == 'prog_clang-oslatest-O0') | (y2 == 'prog_clang-oslatest-O1') | (y2 == 'prog_clang-oslatest-O2') | (y2 == 'prog_clang-oslatest-O3')] = 'prog_clang-oslatest'
y2[(y2 == 'prog_gcc-musl_oslatest-O0') | (y2 == 'prog_gcc-musl_oslatest-O1') | (y2 == 'prog_gcc-musl_oslatest-O2') | (y2 == 'prog_gcc-musl_oslatest-O3')] = 'prog_gcc-musl_oslatest'

# combine all compilers that correspond to no-obfuscation
y3 = copy.deepcopy(y2)
y3[(y3 == 'prog_clang-oslatest') | (y3 == 'prog_gcc-musl_oslatest')] = 'no obfuscation'

# combine all o levels for all compilers
y4 = copy.deepcopy(y)
y4[(y4 == 'prog_clang-oslatest-O0') | (y4 == 'prog_clang-oslatest-O1') | (y4 == 'prog_clang-oslatest-O2') |
   (y4 == 'prog_clang-oslatest-O3') | (y4 == 'prog_gcc-musl_oslatest-O0') | (y4 == 'prog_gcc-musl_oslatest-O1') |
   (y4 == 'prog_gcc-musl_oslatest-O2') | (y4 == 'prog_gcc-musl_oslatest-O3')] = 'No-Obfuscation'

y4[(y4 == 'prog_tigress-3_1-flatten_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-flatten_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-flatten_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-flatten_gcc_musl_oslatest_O3')] = 'Tigress_Flatten'
y4[(y4 == 'prog_tigress-3_1-virtualize_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-virtualize_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-virtualize_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-virtualize_gcc_musl_oslatest_O3')] = 'Tigress_Virtualize'
y4[(y4 == 'prog_tigress-3_1-opabaea_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-opabaea_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-opabaea_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-opabaea_gcc_musl_oslatest_O3')] = 'Tigress_Opabaea'
y4[(y4 == 'prog_tigress-3_1-arithmetic_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-arithmetic_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-arithmetic_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-arithmetic_gcc_musl_oslatest_O3')] = 'Tigress_Arithmetic'
y4[(y4 == 'prog_tigress-3_1-encsplit_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-encsplit_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-encsplit_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-encsplit_gcc_musl_oslatest_O3')] = 'Tigress_EncodeSplit'
y4[(y4 == 'prog_tigress-3_1-opa_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-opa_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-opa_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-opa_gcc_musl_oslatest_O3')] = 'Tigress_Opa'
y4[(y4 == 'prog_tigress-3_1-splitFlatten_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-splitFlatten_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-splitFlatten_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-splitFlatten_gcc_musl_oslatest_O3')] = 'Tigress_SplitFlatten'
y4[(y4 == 'prog_tigress-3_1-splitVirtualize_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-splitVirtualize_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-splitVirtualize_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-splitVirtualize_gcc_musl_oslatest_O3')] = 'Tigress_SplitVirtualize'
y4[(y4 == 'prog_tigress-3_1-split_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-split_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-split_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-split_gcc_musl_oslatest_O3')] = 'Tigress_Split'
y4[(y4 == 'prog_tigress-3_1-virtualizeSplit_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-virtualizeSplit_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-virtualizeSplit_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-virtualizeSplit_gcc_musl_oslatest_O3')] = 'Tigress_VirtualizeSplit'
# Continuing from the existing y4 patterns
y4[(y4 == 'prog_tigress-3_1-flattenOpa_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-flattenOpa_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-flattenOpa_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-flattenOpa_gcc_musl_oslatest_O3')] = 'Tigress_FlattenOpa'
y4[(y4 == 'prog_tigress-3_1-flattenSplitEncode_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-flattenSplitEncode_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-flattenSplitEncode_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-flattenSplitEncode_gcc_musl_oslatest_O3')] = 'Tigress_FlattenSplitEncode'
y4[(y4 == 'prog_tigress-3_1-flattenSplit_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-flattenSplit_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-flattenSplit_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-flattenSplit_gcc_musl_oslatest_O3')] = 'Tigress_FlattenSplit'
y4[(y4 == 'prog_tigress-3_1-opaFlatten_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-opaFlatten_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-opaFlatten_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-opaFlatten_gcc_musl_oslatest_O3')] = 'Tigress_OpaFlatten'
y4[(y4 == 'prog_tigress-3_1-opaSplit_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-opaSplit_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-opaSplit_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-opaSplit_gcc_musl_oslatest_O3')] = 'Tigress_OpaSplit'
y4[(y4 == 'prog_tigress-3_1-splitOpa_gcc_musl_oslatest_O0') |
   (y4 == 'prog_tigress-3_1-splitOpa_gcc_musl_oslatest_O1') |
   (y4 == 'prog_tigress-3_1-splitOpa_gcc_musl_oslatest_O2') |
   (y4 == 'prog_tigress-3_1-splitOpa_gcc_musl_oslatest_O3')] = 'Tigress_SplitOpa'


# Note: The '^' in the regex pattern ensures that the match is at the beginning of the string
print('###################################################################################################')
print('###################################################################################################')
print('###################################################################################################')
print('###################################################################################################\n\n\n')

y5 = copy.deepcopy(y4)
# Classify based on the last obfuscation technique
y5[(y5.str.contains('^Tigress_Split'))] = 'Tigress_SplitFirst'
y5[(y5.str.contains('^Tigress_Virtualize'))] = 'Tigress_VirtualizeFirst'
y5[(y5.str.contains('^Tigress_Opabaea'))] = 'Tigress_OpabaeaFirst'
y5[(y5.str.contains('^Tigress_Arithmetic'))] = 'Tigress_ArithmeticFirst'
y5[(y5.str.contains('^Tigress_Encode'))] = 'Tigress_EncodeFirst'
y5[(y5.str.contains('^Tigress_Opa'))] = 'Tigress_OpaFirst'
y5[(y5.str.contains('^Tigress_Flatten'))] = 'Tigress_FlattenFirst'

print('###################################################################################################')
print('###################################################################################################')
print('###################################################################################################')
print('###################################################################################################\n\n\n')

print('y-Analysis')
print(f'y2 uniques: {y2.nunique() }')
print(y2.value_counts())
print(y2.unique())
print('\n\n')
print(f'y3 uniques: {y3.nunique() }')
print(y3.value_counts())
print(y3.unique())
print('\n\n')
print(f'y4 uniques: {y4.nunique() }')
print(y4.value_counts())
print(y4.unique())
print('\n\n')
print(f'y5 uniques: {y5.nunique() }')
print(y5.value_counts())
print(y5.unique())
print('\n\n')

print('###################################################################################################')
print('###################################################################################################')
print('###################################################################################################')
print('###################################################################################################\n\n\n')
print('X Analysis')
print('X:')
print(X)

print('Correlations present in X')
# Set the size of the figure for the correlation heatmap
plt.figure(figsize=(10,10))

# Calculate the correlation matrix for the features in 'X'
cor = X.corr()

# Generate a mask for the upper triangle to avoid redundancy in the heatmap
mask = np.triu(np.ones_like(cor, dtype=bool))

# Create a directory named 'compiler_data' if it doesn't exist
os.makedirs('compiler_data', exist_ok=True)

# Plot the heatmap using seaborn with the upper triangle masked
sns.heatmap(cor, annot=True, cmap=plt.cm.coolwarm,
            mask=mask,
            cbar_kws = {'shrink': 0.7, 'ticks' : [-1, -.5, 0, 0.5, 1]},  # Customization for the color bar
            vmin = -1,  # Minimum value for colormap scale
            vmax = 1,   # Maximum value for colormap scale
            square = True,  # Ensure the cells of the heatmap are square
            linewidths = .5,)  # Line widths between cells
plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding

# Save the heatmap visualization as image files in PNG and EPS formats
plt.savefig('./compiler_data/corr.png')
plt.savefig('./compiler_data/corr.eps', format='eps')

# Combine 'X' with each 'y' and save as separate CSV files
for i, y in enumerate([y2, y3, y4, y5], start=2):
    # Create a new DataFrame by combining 'X' with the current 'y'
    data_combined = X.copy()  # Copy 'X' to keep the original DataFrame intact
    data_combined[f'y{i}'] = y  # Add 'y' to the DataFrame

    # Print the combined DataFrame for inspection
    print(data_combined)

    # Define the filename for saving
    file_name = f'./compiler_data/compiler_data_y{i}.csv'
    # Save the combined DataFrame as a CSV file without the index column
    data_combined.to_csv(file_name, index=False)
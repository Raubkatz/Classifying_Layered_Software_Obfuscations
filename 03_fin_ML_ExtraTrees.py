"""
This script leverages the ExtraTreesClassifier for predictive modeling, coupled with comprehensive data preparation, model evaluation, and interpretability frameworks. It is designed to automate the end-to-end process of fitting, evaluating, and interpreting ensemble tree-based machine learning models on complex datasets.

Features:
- **Data Loading and Preprocessing**: Automates the ingestion and preliminary preprocessing of datasets, ensuring they are fit for analysis and modeling.
- **Model Training with Hyperparameter Optimization**: Utilizes ExtraTreesClassifier, enhanced with Bayesian optimization (via BayesSearchCV) for hyperparameter tuning, aiming to optimize model performance.
- **Model Evaluation**: Employs a suite of metrics (accuracy, precision, recall, F1 score, confusion matrix) for a thorough evaluation of model performance, complemented by detailed classification reports.
- **Feature Importance Analysis**: Analyzes and visualizes the importance of each feature in the prediction process, providing insights into the dataset's underlying structure and influence on model decisions.
- **SHAP Value Interpretation**: Integrates SHAP (SHapley Additive exPlanations) for model interpretation, offering detailed insights into the contribution of each feature towards individual predictions.
- **Result Visualization and Storage**: Visualizes key metrics and SHAP values for interpretability, stores model artifacts, evaluation metrics, and plots for further analysis.

Workflow:
1. The script initializes by defining a parameter grid for the ExtraTreesClassifier, setting up model training and evaluation configurations.
2. It proceeds to load training and testing datasets, preparing them for the modeling process.
3. The ExtraTreesClassifier model is then trained with Bayesian optimization for hyperparameter tuning, ensuring optimal model configuration.
4. Post-training, the model is evaluated using various metrics, and a classification report is generated for a comprehensive performance overview.
5. Feature importance is calculated and visualized to understand feature contributions to the model's predictions.
6. SHAP values are computed and visualized for in-depth interpretability of model predictions on individual instances.
7. Finally, all results, including performance metrics, feature importance, and SHAP summaries, are saved for documentation and review.

The script encapsulates a robust methodology for applying ensemble tree-based models to predictive tasks, emphasizing not only on performance but also on model understanding and interpretability.

Usage:
- Ensure all dependencies are installed and datasets are located at the specified paths.
- Configure the PARAMETERS section according to the specific requirements of the dataset and predictive task at hand.
- Execute the script in an environment that supports the utilized libraries for seamless operation.
"""

from copy import deepcopy as dc
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from skopt import BayesSearchCV
import time
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

parameter_grid_extratrees = {
    'n_estimators': [80, 100, 120, 140, 160, 180, 200],
    'criterion': ['gini', 'entropy'],
    'max_features': ['log2', 'sqrt', None],
    'max_depth': list(range(10, 51, 10)),  # Example: from 10 to 50, stepping by 10
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

########################################################################################################################
# PARAMETERS ###########################################################################################################
########################################################################################################################
split_option = 1  # Defines the type of data split. `0` for a regular train-test split, `1` for a program-based split.
oversampling = True # Determines whether to apply oversampling to balance the class distribution.
specific_sample_count = 2000 # The target number of instances per class after applying oversampling.
seed = 137 # Seed for random number generators to ensure reproducibility.
choose_y = 1  # Selector for the target variable: `0` for `y3`, `1` for `y4`, `2` for `y5`.
opt_n_iter = 100 #Numbers of Iterations for Hyperparameter Optimization
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Folder for saving results and model based on parametrization
folder_name = f"./data_split/training_data_seed{seed}_sp{split_option}_os{oversampling}_sc{specific_sample_count}_y{choose_y+3}"
trained_models_dir = f"{folder_name}/trained_models_extratrees"
results_output_dir = f"{folder_name}/results_extratrees"

print(folder_name)

# Check if trained_models_dir exists, if not create it
if not os.path.exists(trained_models_dir):
    os.makedirs(trained_models_dir, exist_ok=True)

# Check if results_output_dir exists, if not create it
if not os.path.exists(results_output_dir):
    os.makedirs(results_output_dir, exist_ok=True)

output_file = f"{folder_name}/results_extratrees/classification_metrics_extratrees.txt"

X_train_path = os.path.join(folder_name, 'X_train.csv')
y_train_path = os.path.join(folder_name, 'y_train.csv')
X_test_path = os.path.join(folder_name, 'X_test.csv')
y_test_path = os.path.join(folder_name, 'y_test.csv')

# Load datasets
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()  # Use squeeze to convert DataFrame to Series (if single column)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

print("Data loading and preparation complete.")


## Generate header for csv document
def generate_header():
    header = list("seed", "Accuracy_train", "F1Score_train", "Precision_train", "Recall_train", "Accuracy_test", "F1Score_test", "Precision_test", "Recall_test")
    return dc(header)


# Initialize LightGBM model
model = ExtraTreesClassifier()
start_time_hp = time.time()

bocv = BayesSearchCV(model, parameter_grid_extratrees, n_iter=opt_n_iter, cv=5, verbose=1)

bocv.fit(X_train, y_train)

end_time_hp = time.time()
hp_search_time = end_time_hp - start_time_hp

# Use the best estimator
model = bocv.best_estimator_
best_params = bocv.best_params_
best_cv_score = bocv.best_score_

# Start timing for training best model
start_time_train = time.time()
model.fit(X_train, y_train)
end_time_train = time.time()
training_time = end_time_train - start_time_train

# Start timing for predictions
start_time_pred = time.time()
y_pred = model.predict(X_test)
end_time_pred = time.time()
prediction_time = end_time_pred - start_time_pred

# Classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate and print the classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Feature Importances
feature_importances = model.feature_importances_
feature_names = X_train.columns
# Normalize feature importances so they sum to 1
feature_importances_normalized = feature_importances / feature_importances.sum()

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances_normalized})

# Sort features based on normalized importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Generate a string representation of normalized feature importances
feature_importances_str = "\nFeature Importances (Normalized):\n"
for index, row in feature_importance_df.iterrows():
    feature_importances_str += f"{row['Feature']}: {row['Importance']:.4f}\n"

# Store metrics and model details
results = {
    "best_params": best_params,
    "best_cv_score": best_cv_score,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": conf_matrix.tolist(),
    "hp_search_time": hp_search_time,
    "training_time": training_time,
    "prediction_time": prediction_time,
}

# Update 'results' dictionary to include feature importance
results.update({
    "feature_importances": feature_importance_df.to_dict('records')  # Convert dataframe to list of dicts
})

# Print and save the metrics
print_and_save_metrics = f"""
#######################################################################################################
Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}

Timing Metrics:
Hyperparameter Search Time: {hp_search_time:.2f} seconds
Training Time: {training_time:.2f} seconds
Prediction Time: {prediction_time:.2f} seconds
Total Time: {hp_search_time + training_time + prediction_time:.2f} seconds

Best CV Score: {best_cv_score:.4f}

"""

# Append the classification report to the print and save metrics string
print_and_save_metrics += report

# Append the feature importances to the print and save metrics string
print_and_save_metrics += feature_importances_str

print(print_and_save_metrics)
with open(output_file, "a") as file:
    file.write(print_and_save_metrics)

# Save the results and model
results_pickle_file = os.path.join(results_output_dir, "classification_results_extratrees.pickle")
with open(results_pickle_file, 'wb') as handle:
    pickle.dump(results, handle)

model_filename = os.path.join(trained_models_dir, "best_extratrees_model.pkl")
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

class_names = np.unique(y_train)  # Assuming y_train contains all possible classes

# Normalized Confusion Matrix
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Assuming 'conf_matrix' and 'y_test' are your confusion matrix and test labels
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Determine the categories from 'y_test' assuming it's available
# Adjust this line as needed to match your class naming convention
categories = [x.replace('prog_', '') for x in sorted(np.unique(y_test))]

# Heatmap for Normalized Confusion Matrix with customized colors for the axis labels
plt.figure(figsize=(20, 16), dpi=100)
sns.set(font_scale=1.2)  # Adjust font scale as needed
g = sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, xticklabels=categories, yticklabels=categories)

# Custom color scheme for axis labels
# Adjust 'colors' list according to the number of categories or specific logic
colors = ['black'] * len(categories)  # This example sets all labels to black

# Apply the colors to the tick labels
for xtick, color in zip(g.axes.get_xticklabels(), colors):
    xtick.set_color(color)
for ytick, color in zip(g.axes.get_yticklabels(), colors):
    ytick.set_color(color)

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title("Normalized Confusion Matrix ExtraTrees")

# Saving the plot in both PNG and EPS formats
plt.tight_layout()
plt.savefig(os.path.join(results_output_dir, "normalized_confusion_matrix.png"), dpi=100)
plt.savefig(os.path.join(results_output_dir, "normalized_confusion_matrix.eps"), format='eps')
plt.clf()
plt.close()

# Visualizing Feature Importances
plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title('ExtraTrees Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(os.path.join(results_output_dir, "feature_importances_extratrees.png"))
plt.savefig(os.path.join(results_output_dir, "feature_importances_extratrees.eps"))
plt.clf()
plt.close()

# Initialize the explainer with the model and training data
explainer = shap.TreeExplainer(model, X_train)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test, check_additivity=False)

# Path where the SHAP plots will be saved
shap_plots_dir = os.path.join(results_output_dir, "shap_plots")
if not os.path.exists(shap_plots_dir):
    os.makedirs(shap_plots_dir, exist_ok=True)

# Assuming 'class_names' is defined and corresponds to the labels of your classes
for class_idx in range(len(class_names)):  # Adjust this range according to your number of classes
    plt.figure(figsize=(15, 12))
    shap.summary_plot(shap_values[class_idx], X_test, feature_names=X_train.columns, show=False, plot_type="dot")
    plt.title(f"SHAP Summary for {class_names[class_idx]}", fontsize=14)
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(shap_plots_dir, f"shap_summary_extratrees_{class_names[class_idx]}.png"))
    plt.savefig(os.path.join(shap_plots_dir, f"shap_summary_extratrees_{class_names[class_idx]}.eps"), format='eps')
    plt.clf()

"""
This script focuses on applying advanced ensemble and boosting machine learning techniques for classification tasks, specifically utilizing XGBoost for predictive modeling. Aimed at addressing complex datasets with potential class imbalances, it incorporates various over-sampling methods to ensure robust model training and evaluation. The workflow spans from preprocessing data, optimizing model hyperparameters using Bayesian optimization, to in-depth model evaluation and interpretability through SHAP values.

Key Features:
- Data Preparation: Streamlines data loading, preprocessing, and splitting while ensuring data is ready for modeling.
- Model Training and Hyperparameter Optimization: Leverages XGBoost, optimized with BayesSearchCV, to find the best model configuration.
- Comprehensive Model Evaluation: Utilizes accuracy, precision, recall, F1 score, and confusion matrices for detailed performance analysis.
- Feature Importance Analysis: Offers insight into which features most significantly impact model predictions.
- Model Interpretability: Incorporates SHAP values to explain the influence of each feature on individual predictions.
- Results Visualization: Provides visualizations of feature importances and SHAP values for enhanced interpretability.

Workflow:
1. Data is preprocessed and split according to specified parameters.
2. The XGBoost model undergoes hyperparameter optimization.
3. The optimized model is evaluated using a suite of metrics.
4. Feature importance and SHAP values are computed and visualized.
5. All results, including models and metrics, are saved for future reference.

Usage:
- Ensure all dependencies are installed.
- Configure the PARAMETERS section as per your dataset and requirements.
- Run the script in a suitable Python environment.

This script is designed for practitioners and researchers who require an end-to-end solution for applying ensemble methods to classification problems, with a focus on understanding model behavior and improving prediction accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error
from xgboost.sklearn import XGBClassifier
from skopt import BayesSearchCV
import shap
warnings.filterwarnings("ignore")


parameter_grid_xgb = {
    'min_child_weight': [1, 5, 10, 20, 30, 40, 50],  # Minimum sum of instance weight(hessian) needed in a child
    'gamma': [0.5, 1, 1.5, 2, 5, 7, 10],  # Minimum loss reduction required to make a further partition on a leaf node
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Subsample ratio of the training instances
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # Subsample ratio of columns when constructing each tree
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],  # Maximum depth of a tree
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25],  # Step size shrinkage used in update to prevents overfitting
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800],  # Number of gradient boosted trees
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1, 2, 5],  # L1 regularization term on weights
    'reg_lambda': [0, 0.01, 0.1, 0.5, 1, 2, 5]  # L2 regularization term on weights
}

########################################################################################################################
# PARAMETERS ###########################################################################################################
########################################################################################################################
split_option = 0  # Defines the type of data split. `0` for a regular train-test split, `1` for a program-based split.
oversampling = True # Determines whether to apply oversampling to balance the class distribution.
specific_sample_count = 2000 # The target number of instances per class after applying oversampling.
seed = 137 # Seed for random number generators to ensure reproducibility.
choose_y = 1  # Selector for the target variable: `0` for `y3`, `1` for `y4`, `2` for `y5`.
opt_n_iter = 100 #100 #Numbers of Iterations for Hyperparameter Optimization
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Folder for saving results and model based on parametrization
folder_name = f"./data_split/training_data_seed{seed}_sp{split_option}_os{oversampling}_sc{specific_sample_count}_y{choose_y+3}"
trained_models_dir = f"{folder_name}/trained_models_xgboost"
results_output_dir = f"{folder_name}/results_xgboost"

print(F'Conf: {folder_name}')

# Check if trained_models_dir exists, if not create it
if not os.path.exists(trained_models_dir):
    os.makedirs(trained_models_dir, exist_ok=True)

# Check if results_output_dir exists, if not create it
if not os.path.exists(results_output_dir):
    os.makedirs(results_output_dir, exist_ok=True)

output_file = f"{folder_name}/results_xgboost/classification_metrics_xgboost.txt"

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
model = XGBClassifier()
start_time_hp = time.time()

bocv = BayesSearchCV(model, parameter_grid_xgb, n_iter=opt_n_iter, cv=5, verbose=1)

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
results_pickle_file = os.path.join(results_output_dir, "classification_results_xgboost.pickle")
with open(results_pickle_file, 'wb') as handle:
    pickle.dump(results, handle)

model_filename = os.path.join(trained_models_dir, "best_xgboost_model.pkl")
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
plt.title("Normalized Confusion Matrix XGBoost")

# Saving the plot in both PNG and EPS formats
plt.tight_layout()
plt.savefig(os.path.join(results_output_dir, "normalized_confusion_matrix.png"), dpi=100)
plt.savefig(os.path.join(results_output_dir, "normalized_confusion_matrix.eps"), format='eps')
plt.clf()
plt.close()
#plt.show()
#plt.close()



# Visualizing Feature Importances
plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title('XGBoost Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(os.path.join(results_output_dir, "feature_importances_xgboost.png"))
plt.savefig(os.path.join(results_output_dir, "feature_importances_xgboost.eps"))
plt.clf()
plt.close()
#plt.show()
#plt.close()

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
    plt.savefig(os.path.join(shap_plots_dir, f"shap_summary_xgboost_{class_names[class_idx]}.png"))
    plt.savefig(os.path.join(shap_plots_dir, f"shap_summary_xgboost_{class_names[class_idx]}.eps"), format='eps')
    plt.clf()

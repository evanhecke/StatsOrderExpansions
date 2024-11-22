import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Load the CSV
model_metadata_df = pd.read_csv("model_metadata.csv")

print(model_metadata_df)

# Check data types
print(model_metadata_df.dtypes)
print(model_metadata_df['feature_importance'].head())  # Display the first few entries
print(model_metadata_df['feature_names'].head())  # Display the first few entries

def clean_malformed_data(data):
    if not isinstance(data, str):
        return data  # If it's already a valid object, return as-is
    try:
        # Replace multiple spaces with commas and clean brackets
        data = data.replace(" ", ",").replace(",,", ",").strip("[]")
        # Split into items and handle missing values by replacing them with 0
        cleaned_data = [float(item) if item else 0 for item in data.split(",")]
        return np.array(cleaned_data)  # Return as a numpy array
    except Exception as e:
        print(f"Error cleaning data: {data}, Error: {e}")
        return np.array([])  # Return empty array on error

# Clean 'confusionM' column
model_metadata_df['confusionM_cleaned'] = model_metadata_df['confusionM'].apply(clean_malformed_data)

# Clean 'feature_importance' column
model_metadata_df['feature_importance_cleaned'] = model_metadata_df['feature_importance'].apply(clean_malformed_data)

# Clean 'feature_names' column (special handling as it contains strings)
def clean_feature_names(data):
    if not isinstance(data, str):
        return data  # If it's already valid, return as-is
    try:
        data = data.replace(" ", ",").replace(",,", ",").strip("[]")
        return data.split(",")  # Return as a list of strings
    except Exception as e:
        print(f"Error cleaning feature names: {data}, Error: {e}")
        return []

model_metadata_df['feature_names_cleaned'] = model_metadata_df['feature_names'].apply(clean_feature_names)

import seaborn as sns

for index, row in model_metadata_df.iterrows():
    feature_importance = row['feature_importance_cleaned']
    feature_names = row['feature_names_cleaned']
    if len(feature_importance) == len(feature_names):  # Ensure matching lengths
        sns.barplot(x=feature_importance, y=feature_names, orient="h")
        plt.title(f"Feature Importance - {row['model_name']}")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.show()


# Extract and plot loss curves for each model
for index, row in model_metadata_df.iterrows():
    loss_values = eval(row['loss_values'])  # Convert string back to list
    plt.plot(loss_values, label=row['model_name'])

plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

"""
# Extract and normalize feature importance for each model
feature_importances = {}
for index, row in model_metadata_df.iterrows():
    feature_names = eval(row['feature_names'])  # Convert string back to list
    importance_values = eval(row['feature_importance'])  # Convert string back to array
    feature_importances[row['model_name']] = dict(zip(feature_names, importance_values))

# Convert to a DataFrame for plotting
importance_df = pd.DataFrame(feature_importances).fillna(0)  # Handle missing features across models
importance_df.plot(kind='bar', figsize=(12, 6))
plt.title('Feature Importance Comparison')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.legend(title="Models")
plt.show()
"""


# Extract metrics for comparison
metrics = ['accuracy', 'precision', 'recall', 'f1']
metrics_df = model_metadata_df[['model_name'] + metrics]

# Melt the DataFrame for easier plotting
melted_df = metrics_df.melt(id_vars=['model_name'], var_name='Metric', value_name='Score')

# Plot using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(data=melted_df, x='Metric', y='Score', hue='model_name')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Function to clean and parse confusion matrix strings
def clean_confusion_matrix_string(confusion_matrix_str):
    if not isinstance(confusion_matrix_str, str):
        return confusion_matrix_str  # If already a valid object, return as-is

    # Remove extra spaces and commas
    confusion_matrix_str = confusion_matrix_str.replace(" ", ",").replace(",,", ",")
    # Remove any trailing commas in rows
    confusion_matrix_str = confusion_matrix_str.strip("[]").split("],")
    cleaned_rows = []
    for row in confusion_matrix_str:
        # Convert each row to a list of integers, replacing empty fields with 0
        cleaned_row = [int(num) if num else 0 for num in row.replace("[", "").replace("]", "").split(",")]
        cleaned_rows.append(cleaned_row)
    return np.array(cleaned_rows)

# Plot confusion matrices
for index, row in model_metadata_df.iterrows():
    confusion_matrix = row['confusionM']

    # Clean and convert confusionM string to a valid numpy array
    confusion_matrix = clean_confusion_matrix_string(confusion_matrix)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {row['model_name']}")
    plt.show()


# Select a single feature and compare it against accuracy
feature_of_interest = "Feature 1"
correlation_data = {
    "Model": model_metadata_df['model_name'],
    "Feature Importance": [eval(row['feature_importance'])[0] for _, row in model_metadata_df.iterrows()],
    "Accuracy": model_metadata_df['accuracy']
}

correlation_df = pd.DataFrame(correlation_data)

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=correlation_df, x='Feature Importance', y='Accuracy', hue='Model', style='Model')
plt.title(f'Feature Importance vs. Accuracy ({feature_of_interest})')
plt.xlabel('Feature Importance')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

# Scatter plot for learning rate vs. accuracy
plt.figure(figsize=(8, 6))
sns.scatterplot(data=model_metadata_df, x='learning_rate', y='accuracy', hue='model_name', style='model_name')
plt.title('Learning Rate vs. Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()


# Select performance metrics
heatmap_data = model_metadata_df.set_index('model_name')[['accuracy', 'precision', 'recall', 'f1']]

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Model Performance Metrics Heatmap')
plt.ylabel('Models')
plt.xlabel('Metrics')
plt.show()


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


# Plot confusion matrices
for index, row in model_metadata_df.iterrows():
    # Check if confusionM is already a valid matrix format
    confusion_matrix = row['confusionM']

    # If confusionM is stored as a string, safely convert it to a numpy array
    if isinstance(confusion_matrix, str):
        confusion_matrix = np.array(eval(confusion_matrix))  # Safely parse it to a numpy array

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


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import ast
from sklearn.metrics import ConfusionMatrixDisplay


# Load the CSV
model_metadata_df = pd.read_csv("model_metadata.csv")
model_metadata_df['feature_names'] = model_metadata_df['feature_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

print(model_metadata_df)
print(model_metadata_df['feature_names'])

# Check data types
print(model_metadata_df.dtypes)
print(model_metadata_df['feature_importance'].head())  # Display the first few entries
print(model_metadata_df['feature_names'].head())  # Display the first few entries

## Extract and plot loss curves for each model
for index, row in model_metadata_df.iterrows():
    loss_values = row['loss_values']  # Directly access the loss values
    if isinstance(loss_values, str):  # If it's a string, convert it back to a list
        loss_values = eval(loss_values)
    if loss_values:  # Ensure loss_values is not None or empty
        plt.plot(loss_values, label=row['model_name'])

plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Extract feature importances
feature_importances = {}
for index, row in model_metadata_df.iterrows():
    feature_names = row['feature_names']  # List of feature names
    importance_values = eval(row['feature_importance'])  # Feature importances
    # Store importance values with feature names as columns
    feature_importances[row['model_name']] = dict(zip(feature_names, importance_values))

# Convert the dictionary to a DataFrame where each column is a model
importance_df = pd.DataFrame(feature_importances)

# Display the dataframe to ensure features are correctly saved
print(importance_df)

# Aggregate feature importances across models by summing them
total_importance = importance_df.sum(axis=1)

# Sort by total importance and get the top 10 features
top_10_features = total_importance.sort_values(ascending=False).head(10)

# Filter the importance_df to include only the top 10 features
top_10_importance_df = importance_df.loc[top_10_features.index]

# Plotting each model's feature importance for the top 10 features
top_10_importance_df.plot(kind='bar', figsize=(12, 6))

# Print the top 10 feature names to check
print("Top 10 Features:")
print(top_10_features.index)

# Set plot labels
plt.title('Feature Importance for Top 10 Features (Per Model)')
plt.xlabel('Features')
plt.ylabel('Total Importance')

# Use the correct feature names for the x-axis labels
plt.xticks(ticks=range(len(top_10_features)), labels=top_10_features.index, rotation=45, ha='right')

# Display the legend and plot
plt.legend(title="Models")
plt.tight_layout()
plt.show()

# Aggregate feature importances across models by averaging them
average_importance = importance_df.mean(axis=1)

# Sort by average importance and get the top 10 features with the highest average
top_10_average_features = average_importance.sort_values(ascending=False).head(10)

# Filter the importance_df to include only the top 10 features based on average importance
top_10_average_importance_df = importance_df.loc[top_10_average_features.index]

# Plotting each model's feature importance for the top 10 features with the highest average importance
top_10_average_importance_df.plot(kind='bar', figsize=(12, 6))

plt.title('Feature Importance for Top 10 Features with Highest Average Importance (Per Model)')
plt.xlabel('Features')
plt.ylabel('Average Importance')
plt.xticks(ticks=range(len(top_10_average_features)), labels=top_10_average_features.index, rotation=45, ha='right')
plt.legend(title="Models")
plt.tight_layout()
plt.show()



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
    # Convert list back to NumPy array
    confusion_matrix = np.array(eval(row['confusionM']))

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {row['model_name']}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
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

# Select performance metrics
heatmap_data = model_metadata_df.set_index('model_name')[['accuracy', 'precision', 'recall', 'f1']]

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Model Performance Metrics Heatmap')
plt.ylabel('Models')
plt.xlabel('Metrics')
plt.show()


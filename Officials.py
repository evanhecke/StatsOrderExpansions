import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Import CSV file into a DataFrame
file_path = "beMTPL97.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Data description:
# id: Numeric - Policy number.
# expo: Numeric - Exposure.
# claim: Factor - Indicates if a claim occurred.
# nclaims: Numeric - Number of claims.
# amount: Numeric - Aggregate claim amount.
# average: Numeric - Average claim amount.
# coverage: Factor - Insurance coverage level:
#   "TPL" = Third Party Liability only,
#   "TPL+" = TPL + Limited Material Damage,
#   "TPL++" = TPL + Comprehensive Material Damage.
# ageph: Numeric - Policyholder age.
# sex: Factor - Policyholder gender ("female" or "male").
# bm: Integer - Level occupied in the Belgian bonus-malus scale (0 to 22).
#       Higher levels indicate worse claim history (see Lemaire, 1995).
# power: Numeric - Horsepower of the vehicle in kilowatts.
# agec: Numeric - Age of the vehicle in years.
# fuel: Factor - Type of fuel of the vehicle ("gasoline" or "diesel").
# use: Factor - Use of the vehicle ("private" or "work").
# fleet: Integer - Indicates if the vehicle is part of a fleet (1 or 0).
# postcode: Postal code of the policyholder.
# long: Numeric - Longitude coordinate of the municipality where the policyholder resides.
# lat: Numeric - Latitude coordinate of the municipality where the policyholder resides.

print("DataFrame loaded:")
print(df.head())  # Display the first few rows of the DataFrame
df['average'] = pd.to_numeric(df['average'], errors='coerce').fillna(0)
print(df.head())

#Let's check duplicates
duplicate_ids = df['id'][df['id'].duplicated()].unique()
print("Duplicate IDs:", duplicate_ids)

#No duplicates, really nice.

# dropping longitude and latitude since their value is represented by postcode
df.drop(columns=['long', 'lat'], inplace=True)
print(df.head())
df = df[df['expo'] >= 0.1]
df['nclaims'] = np.round(np.divide(df['nclaims'], df['expo']))
print(max(df['nclaims']))
df = df[df['nclaims'] < 6]
df = df.drop(columns=['expo'])




df_dummies = pd.get_dummies(df, columns=['coverage', 'sex', 'fuel', 'use', 'fleet'], drop_first=True)

print(df_dummies)


# Converting some continuous features into categorical ones. (see book 3.4 for explanation why)
# Step 1: Define bins and labels for each feature

# Vehicle Age
vehicle_age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Bin edges for vehicle_age
vehicle_age_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '45+']  # Labels

# Policy Holder Age
policy_holder_age_bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 150]  # Bin edges for policy_holder_age
policy_holder_age_labels = ['18-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66+']  # Labels

# Vehicle Power
vehicle_power_bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250]  # Bin edges for vehicle_power
vehicle_power_labels = ['0-25', '25-50', '50-75', '75-100', '100-125', '125-150', '150-175', '175-200', '200+']  # Labels

# Step 2: Apply pd.cut() to categorize each feature

df_dummies['vehicle_age_cat'] = pd.cut(df_dummies['agec'], bins=vehicle_age_bins, labels=vehicle_age_labels, right=False)
df_dummies['policy_holder_age_cat'] = pd.cut(df_dummies['ageph'], bins=policy_holder_age_bins, labels=policy_holder_age_labels, right=False)
df_dummies['vehicle_power_cat'] = pd.cut(df_dummies['power'], bins=vehicle_power_bins, labels=vehicle_power_labels, right=False)

# Drop the original continuous columns
df_dummies = df_dummies.drop(columns=['agec', 'ageph', 'power'])


# Step 3: Apply one-hot encoding to the newly created categorical columns

df_dummies = pd.get_dummies(df_dummies, columns=['vehicle_age_cat', 'policy_holder_age_cat', 'vehicle_power_cat'], prefix=['vehicle_age_cat', 'policy_holder_age_cat','vehicle_power_cat'], drop_first=False)

# Step 4: Check the new DataFrame
print(df_dummies.head())
print(df_dummies.columns)

# Step 2: Define features and target variable
X = df_dummies.drop(['claim', 'nclaims', 'amount', 'average', 'id'], axis=1)  # Features
y = df_dummies['claim']               # Target variable



# Fit a Poisson distribution on y (by using its mean of course)
lambda_hat = np.mean(y)
simulated_poisson_data = stats.poisson.rvs(lambda_hat, size=len(y))

# Plot the histogram of the empirical data (discrete)
sns.histplot(y, kde=False, color='blue', alpha=0.5, discrete=True, binwidth=1, label='Empirical Data')

# Plot the fitted Poisson histogram (overlay the Poisson PMF)
sns.histplot(simulated_poisson_data, kde=False, color='red', alpha=0.5, discrete=True, binwidth=1, label='Simulated Poisson')
plt.title('Empirical vs Fitted Poisson Distribution')
plt.xlabel('y')
plt.ylabel('Frequency')
plt.legend()
plt.show()
# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the regression tree model
model = DecisionTreeRegressor(random_state=42, max_depth=20)
model.fit(X_train, y_train)

# Step 5: Predict the target values using the trained model
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)  # If predicted value > 0.2, set to 1, else 0



# Step 7: Plot predicted vs expected (true) values
plt.scatter(y_test, y_pred_binary, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()

# Step 8: Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred_binary)
print(f"Mean Squared Error: {mse}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)

# Create a heatmap for the confusion matrix using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Actual: 0', 'Actual: 1'],
            yticklabels=['Pred: 0', 'Pred: 1'],  # Classic order for Predicted
            cbar=False, square=True, linewidths=0.5)

# Add labels and title
plt.ylabel('Predicted')  # Predicted on the left
plt.xlabel('Actual')    # Actual on the top
plt.title('Confusion Matrix (Predicted vs Actual)')

# Show the plot
plt.show()

# Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_pred)  # Using predicted probabilities for ROC curve

# Calculate AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve (FPR vs Sensitivity / TPR)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (chance level)

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR) or Sensitivity')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def bagging_tree_classification(X_train, X_test, y_train, y_test, n_estimators=10, random_state=42):
    """
    Implements a bagging classifier using decision trees and displays the confusion matrix.

    Parameters:
    - X: Features (numpy array or pandas DataFrame).
    - y: Target variable (numpy array or pandas Series).
    - test_size: Fraction of data to use for testing.
    - n_estimators: Number of base models (trees).
    - random_state: Random state for reproducibility.

    Returns:
    - model: Trained bagging model.
    - accuracy: Classification accuracy.
    """

    # Create a Bagging Classifier
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.2f}")

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return model, accuracy


def random_forest_classification(X_train, X_test, y_train, y_test, n_estimators=5, random_state=42):
    """
    Implements a Random Forest classifier and displays the confusion matrix.

    Parameters:
    - X_train: Training features (numpy array or pandas DataFrame).
    - X_test: Test features (numpy array or pandas DataFrame).
    - y_train: Training target variable (numpy array or pandas Series).
    - y_test: Test target variable (numpy array or pandas Series).
    - n_estimators: Number of trees in the forest.
    - random_state: Random state for reproducibility.

    Returns:
    - model: Trained Random Forest model.
    - accuracy: Classification accuracy.
    """

    # Create a Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.2f}")

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return model, accuracy

from sklearn.datasets import make_classification


# Call the function
model, accuracy = bagging_tree_classification(X_train, X_test, y_train, y_test, n_estimators=15, random_state=42)

model, accuracy = random_forest_classification(X_train, X_test, y_train, y_test, n_estimators=15, random_state=42)



####################################################
# so far this is for the classification tree
# what if we want to know if it's 0,1,2,3,4 or 5

y = df_dummies['nclaims']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the decision tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5], yticklabels=[0, 1, 2, 3, 4, 5])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

model, accuracy = bagging_tree_classification(X_train, X_test, y_train, y_test, n_estimators=400, random_state=42)
model, accuracy = random_forest_classification(X_train, X_test, y_train, y_test, n_estimators=400, random_state=42)
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
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from graphviz import Digraph
from collections import Counter
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam
import scipy.stats as stats
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense



"""
# Import CSV file into a DataFrame
file_path = "beMTPL97.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Save to pickle for faster data loading
df.to_pickle("dataset.pkl")
"""

# Load from pickle
df = pd.read_pickle("dataset.pkl")

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

#Let's check duplicates
duplicate_ids = df['id'][df['id'].duplicated()].unique()
print("Duplicate IDs:", duplicate_ids)

#No duplicates, really nice.

# dropping longitude and latitude since their value is represented by postcode
df.drop(columns=['long', 'lat'], inplace=True)
# print(df.head())
df = df[df['expo'] >= 0.1]
df['nclaims'] = np.round(np.divide(df['nclaims'], df['expo']))
print(max(df['nclaims']))
df = df[df['nclaims'] < 6]
df = df.drop(columns=['expo'])
df['province'] = df['postcode'].astype(str).str[0].astype(int)
df = df.drop(columns=['postcode'])

df_dummies = pd.get_dummies(df, columns=['coverage', 'sex', 'fuel', 'use', 'fleet','province','bm'], drop_first=False)

print(df_dummies.head())
print(df_dummies.columns)

# Step 2: Define features and target variable
X = df_dummies.drop(['claim', 'nclaims', 'amount', 'average', 'id'], axis=1)  # Features
y = df_dummies['nclaims']               # Target variable

print(len(y))

# Preprocess numeric variables
# Identify the numeric columns for scaling (excluding the ones that are now categorical after pd.get_dummies())
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Apply MinMaxScaler to the numeric columns
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

print(type(X_train), type(y_train))
print(X_train.dtypes, y_train.dtypes)

# Initialize an empty DataFrame for storing model metadata
columns = ["model_name", "history", "mse", "mae", "accuracy", "precision", "recall", "f1", "confusionM",
           "loss_values", "predictions", "features", "feature_importance", "learning_rate", "epochs", "batch_size"]
model_metadata_df = pd.DataFrame(columns=columns)

# Function to automatically collect the metadata
"""def collect_metadata(model_name, history, X_train, y_test, y_pred, learning_rate, training_duration):
    # Ensure X_train is a DataFrame or supply feature names
    if isinstance(X_train, np.ndarray):
        feature_names = [f"Feature {i + 1}" for i in range(X_train.shape[1])]
    else:
        feature_names = X_train.columns

    if 'Bagging' in model_name or 'Random' in model_name:
        feature_importance = np.zeros(len(X_train.columns))
    else:
        # Get the weights of the first Dense layer
        weights, biases = model.layers[0].get_weights()

        # Compute feature importance as the sum of absolute weights for each input feature
        feature_importance = np.sum(np.abs(weights), axis=1)

        # Normalize for easier interpretation
        feature_importance = feature_importance / np.sum(feature_importance)

    metadata = {
        "model_name": model_name,
        "history": history,
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=1),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=1),
        "confusionM": confusion_matrix(y_test, y_pred).tolist(),
        "loss_values": history.history['loss'],
        "predictions": y_pred.tolist(),
        "feature_names": feature_names.tolist(),
        "feature_importance": feature_importance.tolist(),
        "learning_rate": learning_rate,
        "epochs": len(history.epoch),
        "train_loss": history.history.get('loss', []),
        "val_loss": history.history.get('val_loss', []),
        "train_accuracy": history.history.get('accuracy', []),
        "val_accuracy": history.history.get('val_accuracy', []),
        "model_architecture": [(layer.name, layer.get_config()) for layer in model.layers],
        "training_time": training_duration,
        "batch_size": 32
    }
    return metadata
"""

def collect_metadata(model_name, history, X_train, y_test, y_pred, learning_rate=None, training_duration=None):
    # Determine feature names
    if isinstance(X_train, np.ndarray):
        feature_names = [f"Feature {i + 1}" for i in range(X_train.shape[1])]
    else:
        feature_names = X_train.columns.tolist()

    # Initialize feature importance
    if 'Bagging' in model_name or 'Random' in model_name:
        # Bagging and Random Forest: Use estimator's feature importance if available
        if hasattr(model, "estimators_"):
            # Average feature importances across all estimators
            feature_importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
        else:
            feature_importance = np.zeros(len(feature_names))
    elif 'NN' in model_name:
        # Neural Network: Compute feature importance from weights of the first Dense layer
        weights, biases = model.layers[0].get_weights()
        feature_importance = np.sum(np.abs(weights), axis=1)
        feature_importance = feature_importance / np.sum(feature_importance)  # Normalize
    else:
        # Default: Zero feature importance
        feature_importance = np.zeros(len(feature_names))

    # Initialize metadata dictionary
    metadata = {
        "model_name": model_name,
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred) if len(set(y_test)) > 1 else None,
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=1) if len(set(y_test)) > 1 else None,
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=1) if len(set(y_test)) > 1 else None,
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=1) if len(set(y_test)) > 1 else None,
        "confusionM": confusion_matrix(y_test, y_pred).tolist() if len(set(y_test)) > 1 else None,
        "loss_values": history.history['loss'] if history and hasattr(history, 'history') else None,
        "predictions": y_pred.tolist(),
        "feature_names": feature_names,
        "feature_importance": feature_importance.tolist(),
        "learning_rate": learning_rate,
        "epochs": len(history.epoch) if history and hasattr(history, 'epoch') else None,
        "train_loss": history.history.get('loss', []) if history and hasattr(history, 'history') else None,
        "val_loss": history.history.get('val_loss', []) if history and hasattr(history, 'history') else None,
        "train_accuracy": history.history.get('accuracy', []) if history and hasattr(history, 'history') else None,
        "val_accuracy": history.history.get('val_accuracy', []) if history and hasattr(history, 'history') else None,
        "model_architecture": [(layer.name, layer.get_config()) for layer in model.layers] if hasattr(model, 'layers') else None,
        "training_time": training_duration,
        "batch_size": 32 if history and hasattr(history, 'epoch') else None,
    }
    return metadata


# Our target variable y is Poisson distributed, so we hand build an adequate loss functions
# for the statistical learning algorithms: Poisson NLL
def poisson_nll(y_true, y_pred):
    # The Poisson Negative Log-Likelihood
    # y_true is the ground truth (observed claims)
    # y_pred is the predicted rate (lambda), which should be positive

    # To avoid log(0), use a small epsilon value (e.g., 1e-10) if y_pred is very small
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.maximum(y_pred, epsilon)  # Prevent negative or zero predictions

    # Poisson NLL formula: NLL = lambda - y * log(lambda) + log(y!)
    # Note: log(y!) term is constant, so it's typically omitted during optimization.
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred))

# Define the number of epochs (iterations) to run
Nepochs = 5

#############################################################################################
# MODEL1: NN(15,25,10), with continuous variables
#############################################################################################

# Define the model
model = Sequential([
    Dense(15, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation="softplus")
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=tf.keras.losses.Poisson())

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train,
          epochs=Nepochs,
          batch_size=32,
          verbose=1)

training_end_time = time.time()
training_duration = training_end_time - training_start_time

# Retrieve the learning rate
learning_rate = model.optimizer.learning_rate.numpy()

# Evaluate the model
# Predict on the test set
y_pred = np.round(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="NN(15,20,10)Cont", history=history, X_train=X_train, y_test=y_test,
                                y_pred=y_pred, learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL2: NN(15,25,10), with categorical variables
#############################################################################################
# Load from pickle
df = pd.read_pickle("dataset.pkl")

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
df['province'] = df['postcode'].astype(str).str[0].astype(int)
df = df.drop(columns=['postcode'])

df_dummies = pd.get_dummies(df, columns=['coverage', 'sex', 'fuel', 'use', 'fleet','province', 'bm'], drop_first=False)

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
y = df_dummies['nclaims']               # Target variable

# Preprocess numeric variables
# Identify the numeric columns for scaling (excluding the ones that are now categorical after pd.get_dummies())
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

print(X.columns)

"""# Apply MinMaxScaler to the numeric columns
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])"""

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Print the shapes of the resulting datasets to verify
print("Training set (features):", X_train.shape)
print("Test set (features):", X_test.shape)
print("Training set (target):", y_train.shape)
print("Test set (target):", y_test.shape)

# Construct a Neural Network
model = Sequential([
    Dense(15, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dense(20, activation='relu'),  # Second hidden layer
    Dense(10, activation='relu'),  # Third hidden layer
    Dense(1, activation="softplus")  # Output layer for regression; softplus to ensure only pos output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Adam optimizer
              loss=tf.keras.losses.Poisson())  # Use Poisson NLL loss

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training
training_end_time = time.time()
training_duration = training_end_time - training_start_time
# Evaluate the model
# Predict on the test set
y_pred = np.round(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="NN(15,20,10)Cat", history=history, X_train=X_train, y_test=y_test,
                                y_pred=y_pred, learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)


#############################################################################################
# MODEL3: NN(100,200,75), with categorical variables
#############################################################################################
# Construct a Neural Network
model = Sequential([
    Dense(100, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dense(200, activation='relu'),  # Second hidden layer
    Dense(75, activation='relu'),  # Third hidden layer
    Dense(1, activation="softplus")  # Output layer for regression; softplus to ensure only pos output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Adam optimizer
              loss=tf.keras.losses.Poisson())  # Use Poisson NLL loss

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training
training_end_time = time.time()
training_duration = training_end_time - training_start_time
# Evaluate the model
# Predict on the test set
y_pred = np.round(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="NN(100,200,75)Cat", history=history, X_train=X_train, y_test=y_test,
                                y_pred=y_pred, learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL4: NN(500,1000,200), with categorical variables
#############################################################################################
# Construct a Neural Network
model = Sequential([
    Dense(500, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dense(1000, activation='relu'),  # Second hidden layer
    Dense(20, activation='relu'),  # Third hidden layer
    Dense(1, activation="softplus")  # Output layer for regression; softplus to ensure only pos output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Adam optimizer
              loss=tf.keras.losses.Poisson())  # Use Poisson NLL loss

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training
training_end_time = time.time()
training_duration = training_end_time - training_start_time
# Evaluate the model
# Predict on the test set
y_pred = np.round(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="NN(500,1000,200)Cat", history=history, X_train=X_train, y_test=y_test,
                                y_pred=y_pred, learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)


#############################################################################################
# MODEL5: GLM, with categorical variables
#############################################################################################
# Let's build a GLM on the same training set and compare them
# Define the GLM-like neural network
model = Sequential([
    Dense(1, activation='exponential', input_dim=X_train.shape[1])  # Exponential link function
])

# Compile the model with Poisson loss
model.compile(optimizer='adam', loss='poisson')

# Train the model
training_start_time = time.time()
history = model.fit(X_train, y_train, epochs=Nepochs, batch_size=32, verbose=1, validation_split=0.2)
training_end_time = time.time()
training_duration = training_end_time - training_start_time
# Predict using the trained Keras model
y_pred = np.round(model.predict(X_test).flatten())  # Flatten to convert predictions to 1D array

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="GLMCat", history=history, X_train=X_train, y_test=y_test, y_pred=y_pred,
                                learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL6: BaggingTree, with categorical variables
#############################################################################################

# Create a Bagging Classifier
model = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=15,
        random_state=30,
        n_jobs=-1)

# Train the model
training_start_time = time.time()
# Train the model
history = model.fit(X_train, y_train)
training_end_time = time.time()
training_duration = training_end_time - training_start_time
# Predict using the trained Keras model
y_pred = np.round(model.predict(X_test).flatten())  # Flatten to convert predictions to 1D array

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="BaggingTreeCat", history=history, X_train=X_train, y_test=y_test, y_pred=y_pred,
                                learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL7: Random Forest, with categorical variables
#############################################################################################


# Create a Random Forest Classifier
model = RandomForestClassifier(
        n_estimators=15,
        random_state=30,
        n_jobs=-1)

# Train the model
training_start_time = time.time()
# Train the model
history = model.fit(X_train, y_train)
training_end_time = time.time()
training_duration = training_end_time - training_start_time
# Predict using the trained Keras model
y_pred = np.round(model.predict(X_test).flatten())  # Flatten to convert predictions to 1D array

# Save the metadata into the dataframe
new_metadata = collect_metadata(model_name="RandomForestCat", history=history, X_train=X_train, y_test=y_test, y_pred=y_pred,
                                learning_rate=learning_rate, training_duration=training_duration)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)


######################################################################
# Save the DataFrame to a CSV file
model_metadata_df.to_csv("model_metadata.csv", index=False)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from graphviz import Digraph
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import Adam
import scipy.stats as stats
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Initialize an empty DataFrame for storing model metadata
columns = ["model_name", "history", "mse", "mae", "accuracy", "precision", "recall", "f1", "confusionM",
           "loss_values", "predictions", "features", "feature_importance", "learning_rate", "epochs", "batch_size"]
model_metadata_df = pd.DataFrame(columns=columns)

# Function to automatically collect the metadata
def collect_metadata(model_name, history, X_train, y_test, y_pred, learning_rate):
    # Ensure X_train is a DataFrame or supply feature names
    if isinstance(X_train, np.ndarray):
        feature_names = [f"Feature {i + 1}" for i in range(X_train.shape[1])]
    else:
        feature_names = X_train.columns

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
        "confusionM": confusion_matrix(y_test, y_pred),
        "loss_values": history.history['loss'],
        "predictions": y_pred,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "learning_rate": learning_rate,
        "epochs": len(history.epoch),
        "batch_size": 32
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

"""
# Load the CSV file
df = pd.read_csv("C:/Users/Evert/OneDrive/Documents/School/UNIL/2nd Year/IA/ADS/beMTPL16.csv")

# Save to pickle (for faster computations when rerunning the code)
df.to_pickle("dataset.pkl")
"""

# Load from pickle
df = pd.read_pickle("dataset.pkl")

# Define the number of epochs (iterations) to run
Nepochs = 30

#############################################################################################
# MODEL1: NN(15,25,10), with continuous variables
#############################################################################################
# Pre-processing the data
# We use minmax scaler for now. And also dummy coding (one-hot coding) for categorical variables
df = df.drop_duplicates(subset=['insurance_contract'], keep='first').reset_index(drop=True)

# Specify the features (X) and target variable (y)
X = df.drop(columns=['insurance_contract','number_of_liability_claims', 'number_of_bodily_injury_liability_claims',
                     'claim_time', 'claim_responsibility_rate', 'signal', 'driving_training_label', 'insured_birth_year',
                     'claim_value', 'policy_year', 'vehicle_model'])  # Drop all the Target Variables: NClaims, NBodilyInjuredClaims,...
y = df['number_of_liability_claims']                 # Target Variable: NClaims

# divide y by exposure
y = np.round(np.divide(y, X['exposure']))
X = X.drop(columns=['exposure'])

"""# Display the first few rows
print(df.head())
print(X.columns)"""

# Preprocess categorical variables
categorical_cols = ['vehicle_brand']
X = pd.get_dummies(X, columns=categorical_cols, prefix="Br", drop_first=False)

# Preprocess numeric variables
# Identify the numeric columns for scaling (excluding the ones that are now categorical after pd.get_dummies())
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Apply MinMaxScaler to the numeric columns
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

"""
# Print the shapes of the resulting datasets to verify
print("Training set (features):", X_train.shape)
print("Test set (features):", X_test.shape)
print("Training set (target):", y_train.shape)
print("Test set (target):", y_test.shape)"""

# Define the model
model = Sequential([
    Dense(15, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation="softplus")
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=poisson_nll)

# Train the model
history = model.fit(X_train, y_train,
          epochs=Nepochs,
          batch_size=32,
          verbose=1)

# Retrieve the learning rate
learning_rate = model.optimizer.learning_rate.numpy()

# Evaluate the model
# Predict on the test set
y_pred = np.round(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(
    model_name="NN(15,20,10)Cont",
    history=history,
    X_train=X_train,
    y_test=y_test,
    y_pred=y_pred,
    learning_rate=learning_rate
)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

#############################################################################################
# MODEL2: NN(15,25,10), with categorical variables
#############################################################################################
# Load from pickle
df = pd.read_pickle("dataset.pkl")

# Pre-processing the data
# We use minmax scaler for now. And also dummy coding (one-hot coding) for categorical variables
df = df.drop_duplicates(subset=['insurance_contract'], keep='first').reset_index(drop=True)

# Specify the features (X) and target variable (y)
X = df.drop(columns=['insurance_contract','number_of_liability_claims', 'number_of_bodily_injury_liability_claims',
                     'claim_time', 'claim_responsibility_rate', 'signal', 'driving_training_label', 'insured_birth_year',
                     'claim_value', 'policy_year', 'vehicle_model'])  # Drop all the Target Variables: NClaims, NBodilyInjuredClaims,...
y = df['number_of_liability_claims']                 # Target Variable: NClaims

# divide y by exposure
y = np.round(np.divide(y, X['exposure']))
X = X.drop(columns=['exposure'])

# Preprocess categorical variables
categorical_cols = ['vehicle_brand']
X = pd.get_dummies(X, columns=categorical_cols, prefix="Br", drop_first=False)

# Converting some continuous features into categorical ones. (see book 3.4 for explanation why)
# Step 1: Define bins and labels for each feature
# Vehicle Age
vehicle_age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]  # Bin edges for vehicle_age
vehicle_age_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51+']  # Labels

# Policy Holder Age
policy_holder_age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]  # Bin edges for policy_holder_age
policy_holder_age_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51+']  # Labels

# Driver License Age
driver_license_age_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]  # Bin edges for driver_license_age
driver_license_age_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51+']  # Labels

# Mileage
mileage_bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 100000]  # Bin edges for mileage
mileage_labels = ['0-5k', '5k-10k', '10k-15k', '15k-20k', '20k-25k', '25k-30k', '30k+']  # Labels

# Vehicle Power
vehicle_power_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 1000]  # Bin edges for vehicle_power
vehicle_power_labels = ['0-50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400+']  # Labels

# Step 2: Apply pd.cut() to categorize each feature
X['vehicle_age_cat'] = pd.cut(X['vehicle_age'], bins=vehicle_age_bins, labels=vehicle_age_labels, right=False)
X['policy_holder_age_cat'] = pd.cut(X['policy_holder_age'], bins=policy_holder_age_bins, labels=policy_holder_age_labels, right=False)
X['driver_license_age_cat'] = pd.cut(X['driver_license_age'], bins=driver_license_age_bins, labels=driver_license_age_labels, right=False)
X['mileage_cat'] = pd.cut(X['mileage'], bins=mileage_bins, labels=mileage_labels, right=False)
X['vehicle_power_cat'] = pd.cut(X['vehicle_power'], bins=vehicle_power_bins, labels=vehicle_power_labels, right=False)

# Drop the original continuous columns
X = X.drop(columns=['vehicle_age', 'policy_holder_age', 'driver_license_age', 'mileage', 'vehicle_power'])

# Step 3: Apply one-hot encoding to the newly created categorical columns
X = pd.get_dummies(X, columns=['vehicle_age_cat', 'policy_holder_age_cat', 'driver_license_age_cat', 'mileage_cat', 'vehicle_power_cat'], prefix=['vehicle_age_cat', 'policy_holder_age_cat', 'driver_license_age_cat', 'mileage_cat', 'vehicle_power_cat'], drop_first=False)

# Preprocess numeric variables
# Identify the numeric columns for scaling (excluding the ones that are now categorical after pd.get_dummies())
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Apply MinMaxScaler to the numeric columns
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
              loss=poisson_nll)  # Use Poisson NLL loss

# Train the model
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training

# Evaluate the model
# Predict on the test set
y_pred = np.round(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(
    model_name="NN(15,20,10)Cat",
    history=history,
    X_train=X_train,
    y_test=y_test,
    y_pred=y_pred,
    learning_rate=learning_rate
)

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
              loss=poisson_nll)  # Use Poisson NLL loss

# Train the model
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training

# Evaluate the model
# Predict on the test set
y_pred = np.round(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(
    model_name="NN(100,200,75)Cat",
    history=history,
    X_train=X_train,
    y_test=y_test,
    y_pred=y_pred,
    learning_rate=learning_rate
)

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
              loss=poisson_nll)  # Use Poisson NLL loss

# Train the model
history = model.fit(X_train, y_train,
          epochs=Nepochs,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training

# Evaluate the model
# Predict on the test set
y_pred = np.round(model.predict(X_test))

# Save the metadata into the dataframe
new_metadata = collect_metadata(
    model_name="NN(500,1000,200)Cat",
    history=history,
    X_train=X_train,
    y_test=y_test,
    y_pred=y_pred,
    learning_rate=learning_rate
)

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
history = model.fit(X_train, y_train, epochs=Nepochs, batch_size=32, verbose=1, validation_split=0.2)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Predict using the trained Keras model
y_pred = np.round(model.predict(X_test).flatten())  # Flatten to convert predictions to 1D array

# Save the metadata into the dataframe
new_metadata = collect_metadata(
    model_name="GLMCat",
    history=history,
    X_train=X_train,
    y_test=y_test,
    y_pred=y_pred,
    learning_rate=learning_rate
)

# Convert the dictionary to a DataFrame and concatenate
new_row_df = pd.DataFrame([new_metadata])
model_metadata_df = pd.concat([model_metadata_df, new_row_df], ignore_index=True)

print(model_metadata_df)

######################################################################
# Save the DataFrame to a CSV file
model_metadata_df.to_csv("model_metadata.csv", index=False)
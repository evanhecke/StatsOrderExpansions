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
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.stats as stats

"""
# Load the CSV file
df = pd.read_csv("C:/Users/Evert/OneDrive/Documents/School/UNIL/2nd Year/IA/ADS/beMTPL16.csv")

# Save to pickle (for faster computations when rerunning the code)
df.to_pickle("dataset.pkl")
"""

# Load from pickle
df = pd.read_pickle("dataset.pkl")

# Step 1: Pre-processing the data
# We use minmax scaler for now. And also dummy coding (one-hot coding) for categorical variables

"""# Group by 'insurance_contract' and aggregate
df = df.groupby('insurance_contract').agg({
    'policy_year': 'max',  # Take the latest policy year
    'exposure': 'max',  # Take max of exposure duration
    'insured_birth_year': 'first',  # Keep the first year of birth
    'vehicle_age': 'mean',  # Average vehicle age
    'policy_holder_age': 'mean',  # Average policy holder age
    'driver_license_age': 'mean',  # Average driver license age
    'vehicle_brand': 'first',  # Use the first vehicle brand
    'vehicle_model': 'first',  # Use the first vehicle model
    'mileage': 'mean',  # Average mileage
    'vehicle_power': 'mean',  # Average power
    'catalog_value': 'mean',  # Average catalog value
    'claim_value': 'sum',  # Sum of all claim values
    'number_of_liability_claims': 'sum',  # Total liability claims
    'number_of_bodily_injury_liability_claims': 'sum',  # Total bodily injury claims
    'claim_time': 'first',  # Use the first claim time
    'claim_responsibility_rate': 'mean',  # Average responsibility rate
    'driving_training_label': 'first',  # First driving training label
    'signal': 'sum'  # Total warnings
}).reset_index()"""

df = df.drop_duplicates(subset=['insurance_contract'], keep='first').reset_index(drop=True)

# Specify the features (X) and target variable (y)
X = df.drop(columns=['insurance_contract','number_of_liability_claims', 'number_of_bodily_injury_liability_claims',
                     'claim_time', 'claim_responsibility_rate', 'signal', 'driving_training_label', 'insured_birth_year',
                     'claim_value', 'policy_year', 'vehicle_model'])  # Drop all the Target Variables: NClaims, NBodilyInjuredClaims,...
y = df['number_of_liability_claims']                 # Target Variable: NClaims

# divide y by exposure
y = np.round(np.divide(y, X['exposure']))
X = X.drop(columns=['exposure'])

# Display the first few rows
print(df.head())
print(X.columns)

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

# Step 4: Check the new DataFrame
print(X.head())
print(X.columns)

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

# STEP2: Construct a Neural Network
# Define the MLPRegressor model
"""
mlp = MLPRegressor(
    hidden_layer_sizes=(15, 20, 10),  # Three hidden layers
    activation='relu',           # Activation function for hidden layers
    solver='adam',               # Optimization algorithm
    max_iter=30,                # Maximum number of iterations
    random_state=42,
    alpha=0.001,                 # L2 regularization term (similar to weight decay)
    verbose=True                 # Print progress during training
)

# Train the model
mlp.fit(X_train, y_train)

# STEP3: Evaluate the model
# Predict on the test set
y_pred = np.round(mlp.predict(X_test))
"""

# Define the MLP model
model = Sequential([
    Dense(15, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer + first hidden layer
    Dense(20, activation='relu'),  # Second hidden layer
    Dense(10, activation='relu'),  # Third hidden layer
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # Adam optimizer
              loss='mean_squared_error')  # MSE loss for regression

# Train the model
history = model.fit(X_train, y_train,
          epochs=30,  # Number of iterations
          batch_size=32,  # Default batch size
          verbose=1)  # Print progress during training

# Evaluate the model
# Predict on the test set
y_pred = np.round(model.predict(X_test))

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# STEP4: Plotting
# Flatten y_test and y_pred to 1D arrays
y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test
y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

# Count the frequency of each (observed, predicted) pair
point_counts = Counter(zip(y_test_flat, y_pred_flat))

# Define a scaling factor for sizes, start very small
min_size = 10  # Minimum size
max_size = 1000  # Maximum size
size_scaling = 1  # Factor to scale size gradually

# Create the size list based on frequency, starting with a small size
sizes = [min_size + point_counts[(x, y)] * size_scaling for x, y in zip(y_test_flat, y_pred_flat)]

# Create the plot with varying dot sizes based on the frequency
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_flat, y=y_pred_flat, alpha=0.7, color="blue", edgecolor=None, size=sizes, sizes=(min_size, max_size))

# Plot the diagonal line
sns.lineplot(x=[y_test_flat.min(), y_test_flat.max()], y=[y_test_flat.min(), y_test_flat.max()], color="red", linestyle="--", lw=2)

# Add titles and labels
plt.title("Observed vs Predicted Values", fontsize=14)
plt.xlabel("Observed", fontsize=12)
plt.ylabel("Predicted", fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.show()


# STEP5: Feature importance
"""# Extract the absolute weights of the first layer
input_weights = np.abs(mlp.coefs_[0])

print(input_weights)

# Calculate average weight for each feature
feature_importance = input_weights.mean(axis=1)"""

# Ensure X_train is a DataFrame or supply feature names
if isinstance(X_train, np.ndarray):
    feature_names = [f"Feature {i+1}" for i in range(X_train.shape[1])]
else:
    feature_names = X_train.columns

# Get the weights of the first Dense layer
weights, biases = model.layers[0].get_weights()

# Compute feature importance as the sum of absolute weights for each input feature
feature_importance = np.sum(np.abs(weights), axis=1)

# Normalize for easier interpretation
feature_importance = feature_importance / np.sum(feature_importance)

# Map feature importance to feature names
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# STEP 6: Select the 6 most important features
important_features = feature_importance_df['Feature'].head(6).values
print("Most important features:", important_features)

# Flatten y_test and y_pred if needed
y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test
y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

# Set up the figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()  # Flatten axes for easier indexing

for i, feature in enumerate(important_features):
    # Determine feature index
    if isinstance(feature_names, pd.Index):  # If feature_names is a pandas Index
        feature_index = feature_names.get_loc(feature)
    else:  # If feature_names is a list
        feature_index = feature_names.index(feature)

    # Extract the feature column for plotting
    feature_values = X_test.iloc[:, feature_index] if isinstance(X_test, pd.DataFrame) else X_test[:, feature_index]

    # Check if the feature is binary (0 or 1)
    is_binary = np.array_equal(np.unique(feature_values), [0, 1])

    # Plot observed values
    sns.scatterplot(
        x=feature_values,
        y=y_test_flat,
        alpha=0.3,
        color="blue",
        label="Observed",
        ax=axes[i]
    )

    # Plot predicted values
    sns.scatterplot(
        x=feature_values,
        y=y_pred_flat,
        alpha=0.3,
        color="red",
        label="Predicted",
        ax=axes[i]
    )

    # Adjust x-axis for binary features
    if is_binary:
        axes[i].set_xticks([0, 1])
        axes[i].set_xlim(-0.5, 1.5)

    axes[i].set_title(f"Observed vs Predicted: {feature}", fontsize=12)
    axes[i].set_xlabel(feature, fontsize=10)
    axes[i].set_ylabel("Response Variable", fontsize=10)
    axes[i].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()





"""# Get the weights of the first Dense layer
weights, biases = model.layers[0].get_weights()

# Compute feature importance as the sum of absolute weights for each input feature
feature_importance = np.sum(np.abs(weights), axis=1)

# Normalize for easier interpretation
feature_importance = feature_importance / np.sum(feature_importance)

# Map feature importance to feature names
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# STEP 6: Select the 6 most important features
important_features = feature_importance_df['Feature'].head(6).values  # Get the feature names of the top 6 important features
print("Most important features:", important_features)

# Set up the figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()  # Flatten axes for easier indexing

for i, feature in enumerate(important_features):
    # Check if the feature is binary (takes only 0 and 1 values)
    is_binary = sorted(X_test[feature].unique()) == [0, 1]

    # Plot observed values with circular markers and adjust transparency (alpha)
    sns.scatterplot(
        x=X_test[feature],
        y=y_test,
        alpha=0.3,  # Adjust transparency (0 is fully transparent, 1 is fully opaque)
        color="blue",
        facecolor="none",
        edgecolor="blue",
        label="Observed",
        ax=axes[i],
        marker='o',  # Circular markers for the observed values
        s=30  # Adjust size of markers to make them smaller
    )

    # Plot predicted values with square markers and adjust transparency (alpha)
    sns.scatterplot(
        x=X_test[feature],
        y=y_pred,
        alpha=0.3,  # Adjust transparency (0 is fully transparent, 1 is fully opaque)
        color="red",
        label="Predicted",
        ax=axes[i],
        marker='x',  # Square markers for the predicted values
        s=30
    )

    # Adjust x-axis for binary features
    if is_binary:
        axes[i].set_xticks([0, 1])  # Set x-axis ticks to only 0 and 1
        axes[i].set_xlim(-0.5, 1.5)  # Optional: limit range to avoid overlapping points

    axes[i].set_title(f"Observed vs Predicted: {feature}", fontsize=12)
    axes[i].set_xlabel(feature, fontsize=10)
    axes[i].set_ylabel("Response Variable", fontsize=10)
    axes[i].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()"""


# 1. Create confusion matrix directly using the discrete values
cm = confusion_matrix(y_test, y_pred)

# 2. Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    cm, annot=False, fmt="d", cmap="Blues", cbar=False,
    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test)
)

# Add annotations to the heatmap
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "red" if i == j else "black"  # Diagonal values in red
        ax.text(j + 0.5, i + 0.5, cm[i, j],
                ha="center", va="center", color=color, fontsize=12)

# Flip the vertical axis
ax.invert_yaxis()

# Add labels and title
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 3. Calculate classification metrics (accuracy, precision, recall, f1-score)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Plot Loss curve over epochs
"""# Plot the training loss over iterations
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_)
plt.title("Training Loss Curve", fontsize=14)
plt.xlabel("Epochs (iterations)", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True)
plt.show()"""

# Get the training loss values from the History object
loss_values = history.history['loss']

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label="Training Loss", color='blue')
plt.title("Training Loss Curve", fontsize=14)
plt.xlabel("Epochs (iterations)", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

"""
# Plot NN Architecture
def plot_nn_architecture_with_customizations(input_size, hidden_layers, output_size, coefs, activations):
    from graphviz import Digraph
    import numpy as np

    # Initialize Digraph
    dot = Digraph(comment="Enhanced Neural Network Architecture")
    dot.attr(rankdir='TB', size='12,10', nodesep='0.8', ranksep='1')  # Top to Bottom, adjust spacing

    # Define layers
    layers = [input_size] + hidden_layers + [output_size]
    layer_names = ['Input'] + [f"Hidden {i + 1}\n({activations[i]})" for i in range(len(hidden_layers))] + ['Output']

    # Node color palette (adjust as needed)
    layer_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    # Add nodes with vertical layout and color
    for layer_idx, (layer_size, layer_name) in enumerate(zip(layers, layer_names)):
        with dot.subgraph(name=f"cluster_{layer_idx}") as sub:
            sub.attr(label=layer_name, style='dashed', color='blue')
            sub.attr(nodesep='0.3', ranksep='1.5', rank='same')  # Forces nodes to be stacked within the same rank
            # Choose a color for the nodes in each layer
            node_color = layer_colors[layer_idx % len(layer_colors)]
            for neuron_idx in range(layer_size):
                node_label = f"L{layer_idx}_N{neuron_idx + 1}"  # Unique node name
                # Set the node color, size, and label font
                sub.node(node_label, shape='circle', style='filled', fillcolor=node_color, fixedsize='true',
                         width='0.7', height='0.7', fontsize='10', label=str(neuron_idx + 1))

    # Add edges between layers with lighter, semi-transparent color
    for layer_idx in range(len(layers) - 1):
        src_layer_size = layers[layer_idx]
        dst_layer_size = layers[layer_idx + 1]
        weights = np.abs(coefs[layer_idx])  # Absolute values of weights

        for src_neuron in range(src_layer_size):
            src_node = f"L{layer_idx}_N{src_neuron + 1}"  # Source node
            for dst_neuron in range(dst_layer_size):
                dst_node = f"L{layer_idx + 1}_N{dst_neuron + 1}"  # Destination node

                weight = weights[src_neuron, dst_neuron]
                penwidth = max(0.5, weight * 5)  # Scale edge thickness
                edge_color = 'lightgray'  # Light color for edges
                dot.edge(src_node, dst_node, penwidth=str(penwidth), color=edge_color, opacity='0.4')

    # Render the graph
    dot.render('output', format='png', engine='fdp')  # Try 'neato' or 'fdp' if 'dot' doesn't work


# Define your network architecture and details
input_size = X_train.shape[1]  # Number of input features
hidden_layers = [15, 20, 10]  # Hidden layer sizes
output_size = 1  # Single output neuron (number_of_liability_claims)
coefs = mlp.coefs_  # Extract learned weights from MLPRegressor
activations = ['ReLU', 'ReLU', 'ReLU']  # Activations used for each hidden layer

# Plot the enhanced neural network
plot_nn_architecture_with_customizations(input_size, hidden_layers, output_size, coefs, activations)
"""

# Let's build a GLM on the same training set and compare them
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define the GLM-like neural network
model = Sequential([
    Dense(1, activation='exponential', input_dim=X_train.shape[1])  # Exponential link function
])

# Compile the model with Poisson loss
model.compile(optimizer='adam', loss='poisson')

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Predict using the trained Keras model
y_pred = model.predict(X_test).flatten()  # Flatten to convert predictions to 1D array

# Create a Seaborn scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')

# Plot the 45-degree ideal fit line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')

# Add titles and labels
plt.title('Predicted vs Actual (Poisson GLM)', fontsize=14)
plt.xlabel('Actual y values', fontsize=12)
plt.ylabel('Predicted y values', fontsize=12)

# Show the legend
plt.legend()

# Show the plot
plt.show()

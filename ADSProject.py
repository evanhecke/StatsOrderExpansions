import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from sklearn.model_selection import train_test_split

"""
# Load the CSV file
df = pd.read_csv("C:/Users/Evert/OneDrive/Documents/School/UNIL/2nd Year/IA/ADS/beMTPL16.csv")

# Save to pickle (for faster computations when rerunning the code)
df.to_pickle("dataset.pkl")
"""

# Load from pickle
df = pd.read_pickle("dataset.pkl")

# Group by 'insurance_contract' and aggregate
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
}).reset_index()

# Display the first few rows
print(df.head())

# One-hot encode the 'vehicle_brand' column
df = pd.get_dummies(df, columns=['vehicle_brand'], prefix=df['vehicle_brand'], drop_first=True)  # Drop first to avoid multicollinearity


# Specify the features (X) and target variable (y)
X = df.drop(columns=['insurance_contract','number_of_liability_claims', 'number_of_bodily_injury_liability_claims',
                     'claim_time', 'claim_responsibility_rate', 'signal', 'driving_training_label',
                     'claim_value', 'policy_year', 'vehicle_model'])  # Drop all the Target Variables: NClaims, NBodilyInjuredClaims,...
y = df['number_of_liability_claims']                 # Target Variable: NClaims

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets to verify
print("Training set (features):", X_train.shape)
print("Test set (features):", X_test.shape)
print("Training set (target):", y_train.shape)
print("Test set (target):", y_test.shape)

# Plot the density function of y
sns.histplot(y, bins=range(int(y.min()), int(y.max()) + 2), kde=False, color='blue', alpha=0.5,
             discrete=True, binwidth=1)
plt.title('Empirical PMF of y')
plt.xlabel('y')
plt.ylabel('Density')
plt.xticks(range(int(y.min()), int(y.max()) + 1))
plt.show()

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

# Now let's fit a glm on the training data, using the poisson family
# Handle log transformation and polynomial terms safely:
# Avoid log(0) error: add a small constant to avoid log(0)
X_train["log_vehicle_age"] = np.log(X_train["vehicle_age"] + 1e-5)  # Adding small value to avoid log(0)
X_train["vehicle_age_squared"] = X_train["vehicle_age"] ** 2
X_train["vehicle_age_cubed"] = X_train["vehicle_age"] ** 3

X_test["log_vehicle_age"] = np.log(X_test["vehicle_age"] + 1e-5)  # Adding small value to avoid log(0)
X_test["vehicle_age_squared"] = X_test["vehicle_age"] ** 2
X_test["vehicle_age_cubed"] = X_test["vehicle_age"] ** 3

# Fit Poisson regression model (no need to specify link='log' as it is the default)
poisson_regressor = PoissonRegressor(alpha=1.0, max_iter=500)  # log link is the default
poisson_regressor.fit(X_train, y_train)

# Predictions on training data
train_predictions = poisson_regressor.predict(X_train)

# Predictions on test data
test_predictions = poisson_regressor.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")

"""
# Output model coefficients
print(f"Model Coefficients: {poisson_regressor.coef_}")
print(f"Model Intercept: {poisson_regressor.intercept_}")
"""

# Optionally, you can visualize the comparison between actual vs predicted values (for example, for the test set)
# Plotting actual vs predicted for test data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=test_predictions)
plt.title('Actual vs Predicted Number of Liability Claims (Test Set)')
plt.xlabel('Actual Number of Liability Claims')
plt.ylabel('Predicted Number of Liability Claims')
plt.show()

# GLM using some of the explanatory variables
# Add the log of exposure as an offset (accounting for exposure in the GLM)
X_train['log_exposure'] = np.log(X_train['exposure'] + 1e-5)  # Add log of exposure to the training data

# Define the GLM formula. Example includes polynomial terms for vehicle_age:
glm_formula = "number_of_liability_claims ~ vehicle_age + I(vehicle_age**2) + I(vehicle_age**3) + policy_holder_age + vehicle_power + driver_license_age + mileage + catalog_value + log_exposure"

# Fit the GLM using the Poisson family
glm_model = smf.glm(
    formula=glm_formula,
    data=pd.concat([X_train, y_train.rename('number_of_liability_claims')], axis=1),  # Combine X_train and y_train
    family=sm.families.Poisson()
).fit()

# Summary of the GLM
print(glm_model.summary())

# Predictions on training data
X_train['glm_predictions'] = glm_model.predict(X_train)

# Predictions on test data
X_test['log_exposure'] = np.log(X_test['exposure'] + 1e-5)  # Ensure log_exposure is added to test data
X_test['glm_predictions'] = glm_model.predict(X_test)

# Evaluate GLM performance
train_mse = mean_squared_error(y_train, X_train['glm_predictions'])
print(f"Train MSE GLM: {train_mse}")

test_mse = mean_squared_error(y_test, X_test['glm_predictions'])
print(f"Test MSE GLM: {test_mse}")

# Plot actual vs predicted for test data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=X_test['glm_predictions'])
plt.title('Actual vs Predicted Number of Liability Claims (Test Set)')
plt.xlabel('Actual Number of Liability Claims')
plt.ylabel('Predicted Number of Liability Claims')
plt.show()

# Create a figure with 4 subplots (2 rows, 2 columns)
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

# List of variables to compare
variables = ['insured_birth_year', 'vehicle_power', 'vehicle_age', 'policy_holder_age']

# Loop through each variable and plot
for i, var in enumerate(variables):
    sns.scatterplot(
        x=X_test[var],
        y=y_test,
        ax=axes[i],
        label='Observed',
        color='blue',
        alpha=0.5
    )
    sns.scatterplot(
        x=X_test[var],
        y=X_test['glm_predictions'],
        ax=axes[i],
        label='Predicted',
        color='orange',
        alpha=0.5
    )

    # Set titles and labels
    axes[i].set_title(f"{var.capitalize()}: Observed vs Predicted")
    axes[i].set_xlabel(var.capitalize())
    if i == 0:  # Add y-label to the first plot only
        axes[i].set_ylabel("Number of Claims")
    axes[i].legend()

# Adjust layout
plt.tight_layout()
plt.show()

#########
# I think we should transform the explanatory variables (group them by breaks of 5 for example in car age,...


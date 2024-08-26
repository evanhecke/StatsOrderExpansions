import numpy as np

# Parameters
j = 2  # Example value for j, adjust as needed
N_samples = 10000  # Number of samples for Monte Carlo simulation

# Example distributions
def sample_N():
    return np.random.poisson(5)  # N follows Poisson distribution

def sample_X():
    return np.random.normal(2, 1)  # X_i follows Normal distribution

# Monte Carlo simulation
expected_value_samples = []
for _ in range(N_samples):
    N = sample_N()
    if N > 1:
        X_values = [sample_X() for _ in range(N-1)]
        sum_X = sum(X_values)
        expected_value_samples.append(N * (sum_X ** (j+1)))

# Calculate the expected value
expected_value = np.mean(expected_value_samples)

print(f"Estimated Expected Value: {expected_value}")

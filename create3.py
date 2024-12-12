import pandas as pd
import numpy as np

# Constants
k_B = 1.38e-23  # Boltzmann constant (J/K)
q = 1.6e-19  # Elementary charge (C)
z = 1  # Charge number for monovalent ions

# Generate synthetic data
np.random.seed(0)
n_samples = 5000  # Increase sample size
T_actual = np.random.uniform(290, 350, n_samples)  # Randomized temperature
a_actual = np.exp(np.random.uniform(1, 4, n_samples))  # Randomized activity
E_0_actual = 0.2 + 0.001 * T_actual + np.random.uniform(-0.005, 0.005, n_samples)  # Add variability

# Add systematic patterns and noise
spread_factor = 1.2  # Scale factor for variability
random_variability = np.random.normal(0, 0.02, n_samples)  # Reduce noise
sensor_drift = np.random.uniform(-0.01, 0.01, n_samples)  # Simulate drift
E_actual = spread_factor * ((k_B * T_actual / (z * q)) * np.log(a_actual) + E_0_actual) + random_variability + sensor_drift

# Introduce outliers
num_outliers = int(0.05 * n_samples)
outlier_indices = np.random.choice(n_samples, num_outliers, replace=False)
E_actual[outlier_indices] += np.random.uniform(-0.1, 0.1, num_outliers)

# Simulate random missing data (introduce NaNs)
def simulate_missing_data(data, missing_rate=0.1):
    """
    Randomly introduce missing data into a dataset.
    
    Parameters:
        data: numpy array or pandas DataFrame
        missing_rate: proportion of data to be set as missing (default 10%)
    
    Returns:
        data_with_missing: data with simulated missing values
    """
    # Randomly select indices to be set as NaN
    missing_indices = np.random.rand(*data.shape) < missing_rate
    data_with_missing = data.copy()
    data_with_missing[missing_indices] = np.nan
    return data_with_missing

# Apply random missingness to each feature
a_actual_with_missing = simulate_missing_data(a_actual, missing_rate=0.1)
E_0_actual_with_missing = simulate_missing_data(E_0_actual, missing_rate=0.1)
T_actual_with_missing = simulate_missing_data(T_actual, missing_rate=0.1)
E_actual_with_missing = simulate_missing_data(E_actual, missing_rate=0.1)

# Save to CSV with missing data
data_with_missing = pd.DataFrame({
    "a_actual": a_actual_with_missing,
    "E_0_actual": E_0_actual_with_missing,
    "T_actual": T_actual_with_missing,
    "E_actual": E_actual_with_missing
})
data_with_missing.to_csv("dataset_with_missing.csv", index=False)

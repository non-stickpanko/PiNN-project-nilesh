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

# Save to CSV
data = pd.DataFrame({
    "a_actual": a_actual,
    "E_0_actual": E_0_actual,
    "T_actual": T_actual,
    "E_actual": E_actual
})
data.to_csv("dataset2.csv", index=False)




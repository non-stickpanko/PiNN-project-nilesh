import pandas as pd
import numpy as np

# Constants
k_B = 1.38e-23  # Boltzmann constant (J/K)
q = 1.6e-19  # Elementary charge (C)
z = 1  # Charge number for monovalent ions

# Generate synthetic data
np.random.seed(0)
n_samples = 500
T_actual = np.linspace(290, 350, n_samples)  # Temperature in K
a_actual = np.maximum(np.exp(np.linspace(1, 4, n_samples)), 1e-8)  # Clamp activity to avoid log(0)
E_0_actual = 0.2 + 0.001 * T_actual  # Standard potential
E_actual = (k_B * T_actual / (z * q)) * np.log(a_actual) + E_0_actual  # Voltage (V)

# Save to CSV
data = pd.DataFrame({
    "a_actual": a_actual,
    "E_0_actual": E_0_actual,
    "T_actual": T_actual,
    "E_actual": E_actual
})
data.to_csv("dataset.csv", index=False)

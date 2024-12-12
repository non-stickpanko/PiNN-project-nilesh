import matplotlib.pyplot as plt

# Show plot
plt.show()
import os
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable oneDNN optimizations (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress specific warnings from Keras
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Constants
k_B = 1.38e-23  # Boltzmann constant (J/K)
q = 1.6e-19  # Elementary charge (C)
z = 1  # Charge number for monovalent ions

# Load dataset from file
data = pd.read_csv("dataset2.csv")
a_actual = data["a_actual"].values
E_0_actual = data["E_0_actual"].values
T_actual = data["T_actual"].values
E_actual = data["E_actual"].values

# Normalize inputs and targets
inputs = np.stack([a_actual, E_0_actual], axis=1)
targets = np.stack([T_actual, E_actual], axis=1)

inputs_mean, inputs_std = inputs.mean(axis=0), inputs.std(axis=0)
targets_mean, targets_std = targets.mean(axis=0), targets.std(axis=0)

inputs_normalized = (inputs - inputs_mean) / inputs_std
targets_normalized = (targets - targets_mean) / targets_std

# Safe log to avoid NaNs
def safe_log(x, epsilon=1e-8):
    return tf.math.log(tf.clip_by_value(x, epsilon, 1e8))

# Define the PINN model
class PINN(tf.keras.Model):
    def __init__(self, hidden_units=100, hidden_layers=4):
        super(PINN, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(hidden_units, activation=tf.keras.layers.LeakyReLU(alpha=0.1))
            for _ in range(hidden_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(2)  # Outputs [Temperature, Voltage]

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

# Define physics-informed loss
def nernst_loss(inputs, predictions, targets):
    a, E_0 = inputs[:, 0], inputs[:, 1]  # Activity and standard potential
    T_pred, E_pred = predictions[:, 0], predictions[:, 1]  # Predicted temperature and voltage
    T_actual, E_actual = targets[:, 0], targets[:, 1]  # Actual temperature and voltage

    # Physics-based residual for Nernst equation with safeguards
    nernst_residual = E_pred - (k_B * tf.clip_by_value(T_pred, 1e-8, 1e8) / (z * q)) * safe_log(a) - E_0
    physics_loss = tf.reduce_mean(tf.square(nernst_residual))

    # Data loss
    data_loss = tf.reduce_mean(tf.square(predictions - targets))

    return data_loss, physics_loss


inputs_tensor = tf.convert_to_tensor(inputs_normalized, dtype=tf.float32)
targets_tensor = tf.convert_to_tensor(targets_normalized, dtype=tf.float32)



# Create a DataFrame for the normalized ground truth data
normalized_data = pd.DataFrame({
    'Normalized Temperature': targets_normalized[:, 0],
    'Normalized Voltage': targets_normalized[:, 1]
})

# Save the DataFrame to a CSV file
normalized_data.to_csv('normalized_ground_truth.csv', index=False)




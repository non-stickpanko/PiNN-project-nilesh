import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.interpolate import make_interp_spline

# Constants
k_B = 1.38e-23  # Boltzmann constant (J/K)
q = 1.6e-19  # Elementary charge (C)
z = 1  # Charge number for monovalent ions

# Load dataset from file
data = pd.read_csv("dataset.csv")
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
    def __init__(self, hidden_units=50, hidden_layers=3):
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

# Instantiate the model and optimizer
model = PINN(hidden_units=50, hidden_layers=3)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
inputs_tensor = tf.convert_to_tensor(inputs_normalized, dtype=tf.float32)
targets_tensor = tf.convert_to_tensor(targets_normalized, dtype=tf.float32)

epochs = 5000
clip_value = 1.0  # Gradient clipping threshold

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = model(inputs_tensor)
        data_loss, physics_loss = nernst_loss(inputs_tensor, predictions, targets_tensor)
        total_loss = data_loss + physics_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_value)  # Clip gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs}")
        print(f"    Data Loss: {data_loss.numpy():.6f}")
        print(f"    Physics Loss: {physics_loss.numpy():.6f}")
        print(f"    Total Loss: {total_loss.numpy():.6f}")

# Final evaluation
predictions = model(inputs_tensor).numpy()
predictions = predictions * targets_std + targets_mean  # Denormalize predictions
T_predicted, E_predicted = predictions[:, 0], predictions[:, 1]

# Calculate R² scores
r2_temp = r2_score(T_actual, T_predicted)
r2_voltage = r2_score(E_actual, E_predicted)
print(f"R² Score for Temperature: {r2_temp:.4f}")
print(f"R² Score for Voltage: {r2_voltage:.4f}")

# Plot Voltage vs Temperature with smooth interpolation
plt.figure(figsize=(10, 6))
# Ground truth
sorted_indices = np.argsort(T_actual)
T_smooth = T_actual[sorted_indices]
E_smooth = E_actual[sorted_indices]
E_smooth_interp = make_interp_spline(T_smooth, E_smooth)(T_smooth)

# Predicted
T_pred_sorted = T_predicted[sorted_indices]
E_pred_sorted = E_predicted[sorted_indices]
E_pred_interp = make_interp_spline(T_pred_sorted, E_pred_sorted)(T_pred_sorted)

plt.plot(T_smooth, E_smooth_interp, label="Ground Truth", color="blue", linewidth=2)
plt.plot(T_pred_sorted, E_pred_interp, label="Predicted", color="red", linestyle="--", linewidth=2)
plt.xlabel("Temperature (K)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Temperature (Ground Truth vs Predicted)")
plt.legend()
plt.grid(True)
plt.show()

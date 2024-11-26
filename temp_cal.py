import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

# Add noise and scaling to inputs
noise_scale = 0.01  # Noise level for random variations
scaling_factor_a = np.random.uniform(0.95, 1.05, size=a_actual.shape)  # Scaling for activity
scaling_factor_E_0 = np.random.uniform(0.95, 1.05, size=E_0_actual.shape)  # Scaling for potential

# Apply noise and scaling
a_noisy_scaled = a_actual * scaling_factor_a * (1 + np.random.normal(0, noise_scale, size=a_actual.shape))
E_0_noisy_scaled = E_0_actual * scaling_factor_E_0 + np.random.normal(0, noise_scale, size=E_0_actual.shape)

# Update the dataset with noisy and scaled values
E_actual = (k_B * T_actual / (z * q)) * np.log(a_noisy_scaled) + E_0_noisy_scaled

# Stack the data into a dataset
data = np.stack([a_noisy_scaled, E_0_noisy_scaled, T_actual, E_actual], axis=1)
inputs = data[:, :2]  # [Activity, E_0]
targets = data[:, 2:]  # [Temperature, Voltage]

# Normalize inputs and targets
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
        print(f"Epoch {epoch}/{epochs}, Total Loss: {total_loss.numpy():.6f}")

# Final evaluation
predictions = model(inputs_tensor).numpy()
predictions = predictions * targets_std + targets_mean  # Denormalize predictions
T_predicted, E_predicted = predictions[:, 0], predictions[:, 1]

# Plot Voltage vs Temperature
plt.figure(figsize=(10, 6))
plt.scatter(T_actual, E_actual, label="Ground Truth", color="blue", alpha=0.5, s=10)
plt.scatter(T_predicted, E_predicted, label="Predicted", color="red", alpha=0.5, s=10)
plt.xlabel("Temperature (K)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Temperature (Ground Truth vs Predicted)")
plt.legend()
plt.grid(True)
plt.show()

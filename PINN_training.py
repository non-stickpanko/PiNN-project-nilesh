import os
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
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

# Normalize both temperature and voltage
inputs = np.stack([a_actual, E_0_actual], axis=1)
targets = np.stack([T_actual, E_actual], axis=1)

inputs_mean, inputs_std = inputs.mean(axis=0), inputs.std(axis=0)
T_mean, T_std = T_actual.mean(), T_actual.std()  # Mean and std for temperature only
E_mean, E_std = E_actual.mean(), E_actual.std()  # Mean and std for voltage

inputs_normalized = (inputs - inputs_mean) / inputs_std
T_normalized = (T_actual - T_mean) / T_std  # Normalize temperature
E_normalized = (E_actual - E_mean) / E_std  # Normalize voltage

# Combine the normalized targets
targets_normalized = np.stack([T_normalized, E_normalized], axis=1)

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

# Instantiate the model and optimizer
model = PINN(hidden_units=100, hidden_layers=4)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,
    decay_steps=2000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Training loop
inputs_tensor = tf.convert_to_tensor(inputs_normalized, dtype=tf.float32)
targets_tensor = tf.convert_to_tensor(targets_normalized, dtype=tf.float32)

epochs = 10000
clip_value = 5.0  # Gradient clipping threshold
epoch_data = []  # To store loss values for plotting later

# Training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = model(inputs_tensor)
        data_loss, physics_loss = nernst_loss(inputs_tensor, predictions, targets_tensor)
        total_loss = 0.4 * data_loss + 0.6 * physics_loss  # Weighting physics loss more heavily

    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_value)  # Clip gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 1000 == 0:
        epoch_data.append([epoch, data_loss.numpy(), physics_loss.numpy(), total_loss.numpy()])
        print(f"Epoch {epoch}/{epochs}")
        print(f"    Data Loss: {data_loss.numpy():.6f}")
        print(f"    Physics Loss: {physics_loss.numpy():.6f}")
        print(f"    Total Loss: {total_loss.numpy():.6f}")

# Compute predictions based on total loss (no need to calculate separately)
predictions_total_loss = model(inputs_tensor)

# Denormalize predictions (both temperature and voltage)
predictions_total_loss_denormalized = predictions_total_loss.numpy()
predictions_total_loss_denormalized[:, 0] = predictions_total_loss[:, 0] * T_std + T_mean  # Denormalize temperature
predictions_total_loss_denormalized[:, 1] = predictions_total_loss[:, 1] * E_std + E_mean  # Denormalize voltage

# Clip the predicted voltage values to be within the range [0.4, 1.0]
predictions_total_loss_denormalized[:, 1] = np.clip(predictions_total_loss_denormalized[:, 1], 0.4, 1.0)

# Check range of predicted values
print("Predicted Temperature Range: ", np.min(predictions_total_loss_denormalized[:, 0]), np.max(predictions_total_loss_denormalized[:, 0]))
print("Predicted Voltage Range: ", np.min(predictions_total_loss_denormalized[:, 1]), np.max(predictions_total_loss_denormalized[:, 1]))

# Save the predictions and actual values in a CSV file
results_df = pd.DataFrame({
    'Actual Temperature': targets[:, 0],
    'Actual Voltage': targets[:, 1],
    'Predicted Temperature': predictions_total_loss_denormalized[:, 0],
    'Predicted Voltage': predictions_total_loss_denormalized[:, 1]
})

results_df.to_csv('predictions_and_actual_values.csv', index=False)
print("Data saved to 'predictions_and_actual_values.csv'.")

# Save the normalized and unnormalized values in another CSV file
normalized_and_unnormalized_df = pd.DataFrame({
    'Normalized Temperature': T_normalized,
    'Normalized Voltage': E_normalized,
    'Predicted Voltage (Before Denormalization)': predictions_total_loss[:, 1],  # Predicted voltage before denormalization
    'Predicted Temperature (Before Denormalization)': predictions_total_loss[:, 0]  # Predicted temperature before denormalization
})

normalized_and_unnormalized_df.to_csv('normalised_values.csv', index=False)
print("Normalized and unnormalized data saved to 'normalised_values.csv'.")

import pandas as pd
import matplotlib.pyplot as plt

# Load the predictions and actual values from the CSV file
data = pd.read_csv('predictions_and_actual_values.csv')

# Extract the values for plotting
actual_temp = data['Actual Temperature']
actual_voltage = data['Actual Voltage']
predicted_temp = data['Predicted Temperature']
predicted_voltage = data['Predicted Voltage']

# Load the normalized values from the second CSV file
normalized_data = pd.read_csv('normalised_values.csv')

# Extract the normalized data
normalized_temp = normalized_data['Normalized Temperature']
normalized_voltage = normalized_data['Normalized Voltage']
predicted_voltage_before_denormalized = normalized_data['Predicted Voltage (Before Denormalization)']
predicted_temp_before_denormalized = normalized_data['Predicted Temperature (Before Denormalization)']

# Create a figure with two subplots (side by side)
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# First graph: Ground Truth vs Predicted Values
axs[0].scatter(actual_voltage, actual_temp, label='Ground Truth', alpha=0.6, color='blue', s=5)
axs[0].scatter(predicted_voltage, predicted_temp, label='Predicted Values', alpha=0.6, color='red', s=5)

# Set labels and title for the first plot
axs[0].set_xlabel("Voltage (V)", fontsize=12)
axs[0].set_ylabel("Temperature (T)", fontsize=12)
axs[0].set_title("Temperature vs Voltage: Ground Truth vs Predicted Values", fontsize=14)
axs[0].legend(loc='best')
axs[0].grid(alpha=0.3)

# Second graph: Normalized Temperature vs Normalized Voltage (Predicted before denormalization)
axs[1].scatter(normalized_voltage, normalized_temp, label='Normalized Ground Truth', alpha=0.6, color='blue', s=5)
axs[1].scatter(predicted_voltage_before_denormalized, predicted_temp_before_denormalized, label='Predicted Values (Before Denormalization)', alpha=0.6, color='red', s=5)

# Set labels and title for the second plot
axs[1].set_xlabel("Normalized Voltage", fontsize=12)
axs[1].set_ylabel("Normalized Temperature", fontsize=12)
axs[1].set_title("Normalized Temperature vs Normalized Voltage", fontsize=14)
axs[1].legend(loc='best')
axs[1].grid(alpha=0.3)

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()

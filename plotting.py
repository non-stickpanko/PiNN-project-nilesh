import pandas as pd
import matplotlib.pyplot as plt

# File paths for CSVs (modify as needed)
predictions_file = 'predictions_and_actual_values.csv'
normalized_data_file = 'normalised_values.csv'
loss_data_file = 'loss_values_by_epoch.csv'

# Load the predictions and actual values from the CSV file
data = pd.read_csv(predictions_file)

# Extract the values for plotting
actual_temp = data['Actual Temperature']
actual_voltage = data['Actual Voltage']
predicted_temp = data['Predicted Temperature']
predicted_voltage = data['Predicted Voltage']

# Load the normalized values from the second CSV file
normalized_data = pd.read_csv(normalized_data_file)

# Extract the normalized data
normalized_temp = normalized_data['Normalized Temperature']
normalized_voltage = normalized_data['Normalized Voltage']
predicted_voltage_before_denormalized = normalized_data['Predicted Voltage (Before Denormalization)']
predicted_temp_before_denormalized = normalized_data['Predicted Temperature (Before Denormalization)']

# Load the loss values by epoch
loss_data = pd.read_csv(loss_data_file)

# Extract the loss data for plotting
epochs = loss_data['Epoch']
data_loss = loss_data['Data Loss']
total_loss = loss_data['Total Loss']  # Use Total Loss instead of Physics Loss

# Create a figure with four subplots (2 rows, 2 columns) with reduced size
fig, axs = plt.subplots(2, 2, figsize=(14, 12))  # Reduced figure size for less overlap

# **First Graph: Ground Truth vs Predicted Values**
axs[0, 0].scatter(actual_voltage, actual_temp, label='Ground Truth', alpha=0.6, color='blue', s=5)
axs[0, 0].scatter(predicted_voltage, predicted_temp, label='Predicted Values', alpha=0.6, color='red', s=5)

# Set labels and title for the first plot
axs[0, 0].set_xlabel("Voltage (V)", fontsize=12)
axs[0, 0].set_ylabel("Temperature (T)", fontsize=12)
axs[0, 0].set_title("Temperature vs Voltage: Ground Truth vs Predicted Values", fontsize=14)
axs[0, 0].legend(loc='best')
axs[0, 0].grid(alpha=0.3)

# **Second Graph: Normalized Temperature vs Normalized Voltage (Predicted before denormalization)**
axs[0, 1].scatter(normalized_voltage, normalized_temp, label='Normalized Ground Truth', alpha=0.6, color='blue', s=5)
axs[0, 1].scatter(predicted_voltage_before_denormalized, predicted_temp_before_denormalized, 
                  label='Predicted Values (Before Denormalization)', alpha=0.6, color='red', s=5)

# Set labels and title for the second plot
axs[0, 1].set_xlabel("Normalized Voltage", fontsize=12)
axs[0, 1].set_ylabel("Normalized Temperature", fontsize=12)
axs[0, 1].set_title("Normalized Temperature vs Normalized Voltage", fontsize=14)
axs[0, 1].legend(loc='best')
axs[0, 1].grid(alpha=0.3)

# **Third Graph: Data Loss vs Total Loss (Line plot)**
axs[1, 0].plot(epochs, data_loss, label='Data Loss', color='blue', linestyle='-', linewidth=2)
axs[1, 0].plot(epochs, total_loss, label='Total Loss', color='red', linestyle='-', linewidth=2)

# Set labels and title for the third plot
axs[1, 0].set_xlabel("Epochs", fontsize=12)
axs[1, 0].set_ylabel("Loss", fontsize=12)
axs[1, 0].set_title("Data Loss vs Total Loss", fontsize=14)
axs[1, 0].legend(loc='best')
axs[1, 0].grid(alpha=0.3)

# **Fourth Graph: Empty plot (just space)**
axs[1, 1].axis('off')  # Hide the axis

# Adjust layout to add more space between the subplots
plt.subplots_adjust(hspace=0.35, wspace=0.3)  # Increased vertical and horizontal space between plots

# Show the plot
plt.show()

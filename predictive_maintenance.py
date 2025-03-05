

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# STEP 1: Create a simple dataset to represent semiconductor equipment sensors
print("Step 1: Creating sample semiconductor sensor data...")

# Let's simulate having 1000 readings from 10 different sensors
num_samples = 1000
num_sensors = 10

# Create random sensor data (this would be your actual sensor readings in real life)
np.random.seed(42)  # For reproducibility
normal_data = np.random.randn(num_samples, num_sensors)

# Let's add some patterns that will represent normal operation
# For example, some sensors might be correlated with each other
for i in range(5):
    normal_data[:, i] = normal_data[:, i] + normal_data[:, 0] * 0.5

# Now, let's create some abnormal data (equipment starting to fail)
# We'll create 100 abnormal samples
num_abnormal = 100
abnormal_data = np.random.randn(num_abnormal, num_sensors)

# Add failure patterns - when equipment fails, some sensors show unusual readings
# For example, temperature might spike while pressure drops
abnormal_data[:, 0] = abnormal_data[:, 0] + 3  # Sensor 1 shows higher values
abnormal_data[:, 1] = abnormal_data[:, 1] - 2  # Sensor 2 shows lower values
abnormal_data[:, 2] = abnormal_data[:, 2] * 2  # Sensor 3 shows more variation

# Combine normal and abnormal data
all_data = np.vstack([normal_data, abnormal_data])

# Create labels: 0 for normal operation, 1 for failure/abnormal
labels = np.zeros(num_samples + num_abnormal)
labels[num_samples:] = 1  # Mark the abnormal samples

# Convert to a pandas DataFrame for easier handling
sensor_columns = [f'sensor_{i+1}' for i in range(num_sensors)]
df = pd.DataFrame(all_data, columns=sensor_columns)
df['failure'] = labels

print(f"Dataset created with {num_samples} normal samples and {num_abnormal} abnormal samples")
print(f"Total shape: {df.shape}")

# Show the first few rows of our dataset
print("\nFirst 5 rows of our dataset (normal samples):")
print(df.head())

print("\nLast 5 rows of our dataset (abnormal samples):")
print(df.tail())
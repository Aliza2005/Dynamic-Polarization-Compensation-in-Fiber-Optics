"""
ML Training Script for Thorlabs EPC
===================================
CORRECTIONS APPLIED:
1. Safely packs both the trained model and the MinMaxScaler into a joblib dictionary.
2. Validates output mathematically on the physical Poincaré sphere (radius 1).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

print("========================================")
print(" EPC AI BRAIN TRAINING PROTOCOL")
print("========================================\n")

# --- 1. Load the harvested data ---
csv_file = "ml_training_data.csv"
if not os.path.exists(csv_file):
    print(f"ERROR: '{csv_file}' not found.")
    print("Please run the Dashboard and click 'START ML DATA HARVEST' first.")
    exit()

print(f"Loading dataset '{csv_file}'...")
df = pd.read_csv(csv_file)

# --- 2. Auto-Detect Columns ---
if all(col in df.columns for col in ['V1', 'V2', 'V3']): v_cols = ['V1', 'V2', 'V3']
elif all(col in df.columns for col in ['V0', 'V1', 'V2']): v_cols = ['V0', 'V1', 'V2']
elif all(col in df.columns for col in ['CH1', 'CH2', 'CH3']): v_cols = ['CH1', 'CH2', 'CH3']
else:
    print("ERROR: Could not detect Voltage columns. Please ensure columns are named V0, V1, V2.")
    exit()

s_cols = ['S1', 'S2', 'S3']
if not all(col in df.columns for col in s_cols):
    print("ERROR: Could not find 'S1', 'S2', 'S3' in your CSV.")
    exit()

X = df[v_cols].values
y = df[s_cols].values

print(f"Loaded {len(X)} samples successfully.")

# --- 3. Scale the Voltages ---
# Neural Networks REQUIRE scaled inputs (-1 to 1) to avoid activation saturation
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 5. Train the AI Forward Model (Neural Network) ---
print("\nTraining Deep Neural Network (128x128x128)...")
print("This may take 30-60 seconds...")

# 3 hidden layers with 128 neurons each, using tanh (smooth curves like optical physics)
model = MLPRegressor(
    hidden_layer_sizes=(128, 128, 128), 
    activation='tanh', 
    solver='adam', 
    max_iter=2000, 
    learning_rate_init=0.001, 
    random_state=42
)
model.fit(X_train, y_train)

# --- 6. Validation and Generalization Check ---
predictions = model.predict(X_test)

# Re-normalize the neural network outputs to exactly 1.0 on the sphere
norm_preds = predictions / np.linalg.norm(predictions, axis=1, keepdims=True)
norm_actuals = y_test / np.linalg.norm(y_test, axis=1, keepdims=True)

# Calculate physical angular error on the Poincaré sphere
cos_thetas = np.clip(np.sum(norm_preds * norm_actuals, axis=1), -1.0, 1.0)
angular_errors = np.degrees(np.arccos(cos_thetas))

print("\n--- NEURAL NETWORK VALIDATION METRICS ---")
print(f"Average Angular Error:  {np.mean(angular_errors):.2f}°")
print(f"Max Angular Error:      {np.max(angular_errors):.2f}°")

if np.mean(angular_errors) > 5.0:
    print("\nWARNING: Error is still high.")
    print("The training data is likely physically corrupted by hardware latency.")
    print("If this happens repeatedly, ensure your Dashboard harvest timer is set to 350ms or higher.")
else:
    print("\nSUCCESS: Neural Network generalization is excellent.")

# --- 7. Save the model & scaler for deployment (FIXED DICT PACKING) ---
# We bundle the model and the scaler together so the dashboard can scale live inputs!
full_brain = {
    'model': model, 
    'scaler': scaler
}

output_filename = "forward_brain.joblib"
joblib.dump(full_brain, output_filename)
print(f"\nModel and Scaler packed and saved successfully to '{output_filename}'!")
print("You can now restart the main dashboard to use the AI Phase-1 jump.")
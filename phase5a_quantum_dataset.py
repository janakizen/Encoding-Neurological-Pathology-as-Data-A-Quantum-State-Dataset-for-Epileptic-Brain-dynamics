import numpy as np
import os

print("✅ Phase V-A started: Quantum-Native Dataset Construction")

OUTPUT_DIR = "outputs"
DATASET_PATH = os.path.join(OUTPUT_DIR, "quantum_epilepsy_dataset.npz")

# ----------------------------
# Load Phase IV Outputs
# ----------------------------
state_path = os.path.join(OUTPUT_DIR, "state_trajectories.npy")
transport_path = os.path.join(OUTPUT_DIR, "transport_metrics.npy")
label_path = os.path.join(OUTPUT_DIR, "labels.npy")

assert os.path.exists(state_path), "❌ Missing state trajectories from Phase IV"
assert os.path.exists(transport_path), "❌ Missing transport metrics from Phase IV"
assert os.path.exists(label_path), "❌ Missing labels from Phase IV"

state_trajectories = np.load(state_path)   # (N, T, D, D)
transport_metrics = np.load(transport_path)  # (N,)
labels = np.load(label_path)                 # (N,)

# ----------------------------
# Sanity Checks
# ----------------------------
N = state_trajectories.shape[0]

assert transport_metrics.shape[0] == N
assert labels.shape[0] == N

print(f"✔ Loaded {N} quantum samples")

# ----------------------------
# Dataset Assembly
# ----------------------------
dataset = {
    "quantum_states": state_trajectories,
    "transport_metrics": transport_metrics,
    "labels": labels
}

np.savez_compressed(DATASET_PATH, **dataset)

print(f"📦 Quantum-native dataset saved to: {DATASET_PATH}")
print("✅ Phase V-A complete")

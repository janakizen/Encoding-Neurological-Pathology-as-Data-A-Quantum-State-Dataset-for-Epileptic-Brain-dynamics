import numpy as np
import os

print("✅ Phase IV-C started: Lightweight Quantum Transport Metrics")

# Paths
STATE_PATH = "outputs/state_trajectories.npy"
OUT_PATH = "outputs/transport_metrics.npy"

assert os.path.exists(STATE_PATH), "❌ Missing state trajectories from Phase IV-B"

states = np.load(STATE_PATH)
# Expected shape: (samples, timesteps, state_dim)

print(f"Loaded state trajectories with shape: {states.shape}")

# --- VECTOR-BASED TRACE DISTANCE PROXY ---
def trace_distance_vector(v1, v2):
    return 0.5 * np.linalg.norm(v1 - v2)

transport_metrics = []

for sample in states:
    distances = []
    for t in range(len(sample) - 1):
        d = trace_distance_vector(sample[t], sample[t + 1])
        distances.append(d)
    transport_metrics.append(distances)

transport_metrics = np.array(transport_metrics)

np.save(OUT_PATH, transport_metrics)

print(f"📦 Transport metrics saved to: {OUT_PATH}")
print("✅ Phase IV-C complete")

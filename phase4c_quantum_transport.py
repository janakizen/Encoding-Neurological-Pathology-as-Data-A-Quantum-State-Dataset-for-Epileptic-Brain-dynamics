import numpy as np
import os

print("✅ Phase IV-C started: Lightweight Quantum Transport Metrics")

# =========================
# PROJECT PATHS
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE_TRAJ_PATH = os.path.join(OUTPUT_DIR, "state_trajectories.npy")
STATES_2D_PATH = os.path.join(OUTPUT_DIR, "states_2d.npy")
LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.npy")
OUT_PATH = os.path.join(OUTPUT_DIR, "transport_metrics.npy")

# =========================
# LOAD / BUILD TRAJECTORIES
# =========================
if os.path.exists(STATE_TRAJ_PATH):
    states = np.load(STATE_TRAJ_PATH)
    print(f"Loaded existing state trajectories: {states.shape}")
elif os.path.exists(STATES_2D_PATH):
    states_2d = np.load(STATES_2D_PATH)

    # Convert Phase II output into one trajectory sample
    # Shape becomes (1, timesteps, state_dim)
    states = states_2d[np.newaxis, :, :]
    np.save(STATE_TRAJ_PATH, states)

    print(f"Built state trajectories from states_2d.npy: {states.shape}")
else:
    raise FileNotFoundError(
        f"Neither {STATE_TRAJ_PATH} nor {STATES_2D_PATH} exists. "
        "Run Phase II first."
    )

# Optional labels
labels = None
if os.path.exists(LABELS_PATH):
    labels = np.load(LABELS_PATH)

# =========================
# VECTOR-BASED TRACE DISTANCE PROXY
# =========================
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

print(f"\nLoaded state trajectories with shape: {states.shape}")
print(f"Transport metrics shape: {transport_metrics.shape}")
print(f"Mean transport distance: {np.mean(transport_metrics):.6f}")
print(f"Max transport distance:  {np.max(transport_metrics):.6f}")
print(f"Min transport distance:  {np.min(transport_metrics):.6f}")

# Optional ictal/interictal summary if labels exist
if labels is not None and len(labels) == states.shape[1]:
    sample_dist = transport_metrics[0]
    inter_d = []
    ictal_d = []

    for t in range(len(sample_dist)):
        if labels[t] == 0:
            inter_d.append(sample_dist[t])
        else:
            ictal_d.append(sample_dist[t])

    if len(inter_d) > 0:
        print(f"\nMean Interictal Transport: {np.mean(inter_d):.6f}")
    if len(ictal_d) > 0:
        print(f"Mean Ictal Transport:      {np.mean(ictal_d):.6f}")
    if len(inter_d) > 0 and len(ictal_d) > 0 and np.mean(inter_d) != 0:
        print(f"Transport Ratio (Ictal / Interictal): {np.mean(ictal_d)/np.mean(inter_d):.6f}")

print(f"\n📦 Transport metrics saved to: {OUT_PATH}")
print(f"📦 State trajectories saved to: {STATE_TRAJ_PATH}")
print("✅ Phase IV-C complete")
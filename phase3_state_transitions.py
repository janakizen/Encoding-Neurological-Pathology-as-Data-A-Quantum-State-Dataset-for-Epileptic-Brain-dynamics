import numpy as np
import matplotlib.pyplot as plt
import os

print("✅ Phase III started")

# -----------------------------
# LOAD PHASE II OUTPUT
# -----------------------------
DATA_DIR = "/Users/janakinagesh/Downloads/phase1_epilepsy/outputs"
OUT_DIR = DATA_DIR

# We recompute from saved PCA states (simpler + reliable)
states_2d = np.load(f"{DATA_DIR}/states_2d.npy")
labels = np.load(f"{DATA_DIR}/labels.npy")
times = np.load(f"{DATA_DIR}/times.npy")

# -----------------------------
# TRAJECTORY VELOCITY
# -----------------------------
velocity = np.linalg.norm(
    np.diff(states_2d, axis=0),
    axis=1
)

velocity = np.concatenate([[0], velocity])

# -----------------------------
# TRANSITION ENTROPY
# -----------------------------
def transition_entropy(states, bins=20):
    dx = np.diff(states, axis=0)
    mag = np.linalg.norm(dx, axis=1)

    hist, _ = np.histogram(mag, bins=bins, density=True)
    hist = hist[hist > 0]

    return -np.sum(hist * np.log(hist))

ent_inter = transition_entropy(states_2d[labels == 0])
ent_ictal = transition_entropy(states_2d[labels == 1])

# -----------------------------
# PLOT 1: STATE TRAJECTORY
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(times, states_2d[:, 0], label="PC1")
plt.fill_between(
    times,
    states_2d[:, 0].min(),
    states_2d[:, 0].max(),
    where=labels.astype(bool),
    color="red",
    alpha=0.2,
    label="Seizure"
)
plt.xlabel("Time (s)")
plt.ylabel("PC1")
plt.title("Phase III: Neural State Trajectory")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase3_state_trajectory.png", dpi=300)
plt.close()

# -----------------------------
# PLOT 2: VELOCITY
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(times, velocity)
plt.fill_between(
    times,
    0,
    velocity.max(),
    where=labels.astype(bool),
    color="red",
    alpha=0.2
)
plt.xlabel("Time (s)")
plt.ylabel("State Velocity")
plt.title("Phase III: State Transition Velocity")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase3_state_velocity.png", dpi=300)
plt.close()

# -----------------------------
# PLOT 3: TRANSITION ENTROPY
# -----------------------------
plt.figure(figsize=(6, 4))
plt.bar(["Interictal", "Ictal"], [ent_inter, ent_ictal])
plt.ylabel("Transition Entropy")
plt.title("Phase III: Transition Entropy")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase3_transition_entropy.png", dpi=300)
plt.close()

# -----------------------------
# REPORT
# -----------------------------
print("\nTransition Entropy")
print(f"Interictal: {ent_inter:.4f}")
print(f"Ictal: {ent_ictal:.4f}")
print(f"Entropy Ratio (Ictal / Interictal): {ent_ictal / ent_inter:.4f}")

print("\n✅ Phase III complete")

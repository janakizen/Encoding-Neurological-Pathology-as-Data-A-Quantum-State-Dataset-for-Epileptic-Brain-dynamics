import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

print("✅ Phase IV-A started: Quantum-Native Encoding")

# -----------------------------
# LOAD OSCILLATORY STATES
# -----------------------------
DATA_DIR = "/Users/janakinagesh/Downloads/phase1_epilepsy/outputs"

states = np.load(f"{DATA_DIR}/states_2d.npy")
labels = np.load(f"{DATA_DIR}/labels.npy")

# Normalize to [0, π]
states_norm = (states - states.min()) / (states.max() - states.min())
states_norm = states_norm * np.pi

# -----------------------------
# DEFINE QUANTUM OPERATORS
# -----------------------------
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def rotation(theta):
    return expm(-1j * theta * sigma_z)

# -----------------------------
# ENCODE STATES
# -----------------------------
quantum_states = []

for s in states_norm:
    # Two-dimensional → single qubit superposition
    U = rotation(s[0])
    psi0 = np.array([1, 0])
    psi = U @ psi0
    quantum_states.append(psi)

quantum_states = np.array(quantum_states)

# -----------------------------
# ANALYZE HILBERT SPACE GEOMETRY
# -----------------------------
def fidelity(psi, phi):
    return np.abs(np.vdot(psi, phi)) ** 2

fid_inter = []
fid_ictal = []

for i in range(len(quantum_states) - 1):
    f = fidelity(quantum_states[i], quantum_states[i + 1])
    if labels[i] == 0:
        fid_inter.append(f)
    else:
        fid_ictal.append(f)

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(6, 4))
plt.hist(fid_inter, bins=40, alpha=0.6, label="Interictal")
plt.hist(fid_ictal, bins=40, alpha=0.6, label="Ictal")
plt.xlabel("State Fidelity")
plt.ylabel("Count")
plt.title("Quantum-State Transition Fidelity")
plt.legend()
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/phase4_quantum_fidelity.png", dpi=300)
plt.close()

print("Mean Fidelity")
print("Interictal:", np.mean(fid_inter))
print("Ictal:", np.mean(fid_ictal))

print("✅ Phase IV-A complete")

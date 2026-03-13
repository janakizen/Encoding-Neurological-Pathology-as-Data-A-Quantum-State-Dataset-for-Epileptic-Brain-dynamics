import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.linalg import logm, sqrtm
import os

# =========================
# Paths
# =========================
DATA_PATH = "data/chb01_03.edf"   # same file used earlier
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Quantum Functions
# =========================
def density_matrix(X):
    """Construct normalized density matrix from feature matrix"""
    rho = np.cov(X)
    rho = rho / np.trace(rho)
    return rho

def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-10]
    return -np.sum(eigvals * np.log2(eigvals))

def trace_distance(rho1, rho2):
    diff = rho1 - rho2
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigvals))

# =========================
# Phase IV-B
# =========================
print("✅ Phase IV-B started: Density Matrix Quantum Encoding")

raw = mne.io.read_raw_edf(DATA_PATH, preload=True, verbose=False)
raw.pick_types(eeg=True)

sfreq = raw.info["sfreq"]
data = raw.get_data()

# Simple segmentation (same logic as earlier phases)
interictal = data[:, :int(10 * sfreq)]
ictal = data[:, int(20 * sfreq):int(30 * sfreq)]

# Feature abstraction (mean-centered)
X_inter = interictal - np.mean(interictal, axis=1, keepdims=True)
X_ictal = ictal - np.mean(ictal, axis=1, keepdims=True)

# Density matrices
rho_inter = density_matrix(X_inter)
rho_ictal = density_matrix(X_ictal)

# Quantum metrics
S_inter = von_neumann_entropy(rho_inter)
S_ictal = von_neumann_entropy(rho_ictal)
T_dist = trace_distance(rho_inter, rho_ictal)

print("\nQuantum Metrics")
print(f"von Neumann Entropy (Interictal): {S_inter:.4f}")
print(f"von Neumann Entropy (Ictal):      {S_ictal:.4f}")
print(f"Trace Distance (Inter vs Ictal):  {T_dist:.4f}")

# =========================
# Visualization 1: Density Matrices
# =========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(rho_inter, cmap="viridis")
plt.title("Interictal Density Matrix")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(rho_ictal, cmap="inferno")
plt.title("Ictal Density Matrix")
plt.colorbar()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/phase4b_density_matrices.png", dpi=300)
plt.close()

# =========================
# Visualization 2: Entropy Comparison
# =========================
plt.figure()
plt.bar(["Interictal", "Ictal"], [S_inter, S_ictal])
plt.ylabel("von Neumann Entropy")
plt.title("Quantum Entropy Comparison")
plt.savefig(f"{OUTPUT_DIR}/phase4b_entropy_comparison.png", dpi=300)
plt.close()

print("\n✅ Diagrams saved:")
print(" - phase4b_density_matrices.png")
print(" - phase4b_entropy_comparison.png")
print("✅ Phase IV-B complete")

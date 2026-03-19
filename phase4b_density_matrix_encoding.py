import numpy as np
import matplotlib.pyplot as plt
import mne
import os

# =========================
# PROJECT PATHS
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, "data", "chb01_03.edf")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# TIME WINDOWS
# =========================
# Keep these consistent with your earlier phases
INTERICTAL_START = 200
INTERICTAL_END = 260

ICTAL_START = 2996
ICTAL_END = 3036

# =========================
# QUANTUM FUNCTIONS
# =========================
def density_matrix(X):
    """Construct normalized density matrix from feature matrix."""
    rho = np.cov(X)
    tr = np.trace(rho)
    if np.isclose(tr, 0):
        raise ValueError("Trace of covariance matrix is zero; cannot normalize density matrix.")
    rho = rho / tr
    return rho

def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.sum(eigvals * np.log2(eigvals))

def trace_distance(rho1, rho2):
    diff = rho1 - rho2
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigvals))

def extract_window(data, start_sec, end_sec, sfreq):
    start_idx = int(start_sec * sfreq)
    end_idx = int(end_sec * sfreq)
    return data[:, start_idx:end_idx]

# =========================
# PHASE IV-B
# =========================
print("✅ Phase IV-B started: Density Matrix Quantum Encoding")
print(f"Using EDF file: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"EDF file not found: {DATA_PATH}")

raw = mne.io.read_raw_edf(DATA_PATH, preload=True, verbose=False)
raw.pick_types(eeg=True)

sfreq = raw.info["sfreq"]
data = raw.get_data()

# =========================
# SEGMENTATION
# =========================
interictal = extract_window(data, INTERICTAL_START, INTERICTAL_END, sfreq)
ictal = extract_window(data, ICTAL_START, ICTAL_END, sfreq)

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

# Save numeric outputs too
np.save(os.path.join(OUTPUT_DIR, "rho_inter.npy"), rho_inter)
np.save(os.path.join(OUTPUT_DIR, "rho_ictal.npy"), rho_ictal)
np.save(
    os.path.join(OUTPUT_DIR, "phase4b_metrics.npy"),
    {
        "S_inter": S_inter,
        "S_ictal": S_ictal,
        "T_dist": T_dist,
    },
    allow_pickle=True
)

# =========================
# VISUALIZATION 1: DENSITY MATRICES
# =========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(rho_inter, cmap="viridis", aspect="auto")
plt.title("Interictal Density Matrix")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(rho_ictal, cmap="inferno", aspect="auto")
plt.title("Ictal Density Matrix")
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase4b_density_matrices.png"), dpi=300)
plt.close()

# =========================
# VISUALIZATION 2: ENTROPY COMPARISON
# =========================
plt.figure(figsize=(6, 4))
plt.bar(["Interictal", "Ictal"], [S_inter, S_ictal])
plt.ylabel("von Neumann Entropy")
plt.title("Quantum Entropy Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "phase4b_entropy_comparison.png"), dpi=300)
plt.close()

print("\n✅ Diagrams saved:")
print(" - phase4b_density_matrices.png")
print(" - phase4b_entropy_comparison.png")
print(" - rho_inter.npy")
print(" - rho_ictal.npy")
print(" - phase4b_metrics.npy")
print("✅ Phase IV-B complete")
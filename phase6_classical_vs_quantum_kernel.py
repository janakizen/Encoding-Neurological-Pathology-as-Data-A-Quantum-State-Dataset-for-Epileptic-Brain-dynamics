import os
import numpy as np
import matplotlib.pyplot as plt

print("🔷 Phase VI Visualization started")

# -------------------------------------------------
# PROJECT PATHS (correct handling)
# -------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
DATASET_PATH = os.path.join(OUTPUT_DIR, "quantum_epilepsy_dataset.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Project directory:", PROJECT_DIR)
print("Dataset path:", DATASET_PATH)

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "❌ Dataset not found.\n"
        "Run Phase V-B first to generate quantum_epilepsy_dataset.npz"
    )

data = np.load(DATASET_PATH, allow_pickle=True)

quantum_states = data["quantum_states"]       # (N, d, d)
transport = data["transport_metrics"]         # (N, T-1)
labels = data["labels"]

print(f"✔ Loaded {len(quantum_states)} quantum samples")

# -------------------------------------------------
# 1. Eigenvalue spectra
# -------------------------------------------------
plt.figure()

for i in range(min(5, len(quantum_states))):
    eigvals = np.linalg.eigvalsh(quantum_states[i])
    plt.plot(eigvals, marker='o', alpha=0.7)

plt.title("Eigenvalue Spectrum of Quantum Neural States")
plt.xlabel("Eigenvalue Index")
plt.ylabel("Eigenvalue")
plt.grid(True)

eig_path = os.path.join(OUTPUT_DIR, "eigenvalue_spectra.png")

plt.savefig(eig_path, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------------------------
# 2. Quantum transport dynamics
# -------------------------------------------------
plt.figure()

plt.plot(transport.mean(axis=0), marker='o')

plt.title("Average Quantum Transport Distance Over Time")
plt.xlabel("Time Step")
plt.ylabel("Trace Distance")
plt.grid(True)

transport_path = os.path.join(OUTPUT_DIR, "average_quantum_transport.png")

plt.savefig(transport_path, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------------------------
# 3. Von Neumann entropy distribution
# -------------------------------------------------
def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return -np.sum(eigvals * np.log2(eigvals))

entropies = [von_neumann_entropy(r) for r in quantum_states]

plt.figure()

plt.hist(entropies, bins=10)

plt.title("Entropy Distribution of Quantum Neural States")
plt.xlabel("Entropy")
plt.ylabel("Count")
plt.grid(True)

entropy_path = os.path.join(OUTPUT_DIR, "entropy_distribution.png")

plt.savefig(entropy_path, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------------------------------
# FINISHED
# -------------------------------------------------
print("✅ Visualizations saved:")
print(" -", eig_path)
print(" -", transport_path)
print(" -", entropy_path)

print("✅ Phase VI Visualization complete")
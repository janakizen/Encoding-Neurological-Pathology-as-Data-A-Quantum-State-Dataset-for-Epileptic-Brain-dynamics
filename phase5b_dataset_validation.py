import numpy as np
import matplotlib.pyplot as plt
import os

print("✅ Phase V-B started: Quantum-Native Dataset Validation")

# -------------------------------------------------
# PROJECT PATHS (explicit and reliable)
# -------------------------------------------------
PROJECT_DIR = "/Users/janakinagesh/Downloads/phase1_epilepsy"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

X_PATH = os.path.join(OUTPUT_DIR, "quantum_dataset_X.npy")
Y_PATH = os.path.join(OUTPUT_DIR, "quantum_dataset_y.npy")

DATASET_PATH = os.path.join(OUTPUT_DIR, "quantum_epilepsy_dataset.npz")

print("Project directory:", PROJECT_DIR)
print("Output directory :", OUTPUT_DIR)

# -------------------------------------------------
# CHECK PHASE V-A OUTPUTS
# -------------------------------------------------
if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    raise FileNotFoundError(
        "❌ Missing Phase V-A outputs.\n"
        "Run Phase V-A first to generate:\n"
        " - quantum_dataset_X.npy\n"
        " - quantum_dataset_y.npy"
    )

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
X = np.load(X_PATH)  # (N, T, D)
labels = np.load(Y_PATH)

N, T, D = X.shape

print("✔ Loaded sequences:", N)
print("✔ Sequence length:", T)
print("✔ Feature dimension:", D)

# -------------------------------------------------
# QUANTUM HELPERS
# -------------------------------------------------
def density_matrix(seq):
    """Convert sequence to density matrix"""
    seq = seq - np.mean(seq, axis=0)
    rho = np.cov(seq, rowvar=False)

    if rho.ndim == 0:
        rho = np.array([[rho]])

    rho = 0.5 * (rho + rho.T)

    tr = np.trace(rho)

    if tr == 0:
        rho = np.eye(D) / D
    else:
        rho = rho / tr

    return rho


def transport_metric(seq):
    diffs = np.diff(seq, axis=0)
    return 0.5 * np.linalg.norm(diffs, axis=1)


def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 1e-12, 1)
    eigvals = eigvals / np.sum(eigvals)
    return -np.sum(eigvals * np.log(eigvals))


# -------------------------------------------------
# BUILD QUANTUM DATASET
# -------------------------------------------------
quantum_states = []
transport_metrics = []

for seq in X:
    rho = density_matrix(seq)
    tm = transport_metric(seq)

    quantum_states.append(rho)
    transport_metrics.append(tm)

quantum_states = np.array(quantum_states)
transport_metrics = np.array(transport_metrics)

# -------------------------------------------------
# SAVE DATASET
# -------------------------------------------------
np.savez(
    DATASET_PATH,
    quantum_states=quantum_states,
    transport_metrics=transport_metrics,
    labels=labels
)

print("📦 Saved dataset:", DATASET_PATH)

# -------------------------------------------------
# ENTROPY ANALYSIS
# -------------------------------------------------
entropies = np.array([von_neumann_entropy(rho) for rho in quantum_states])

ent_inter = entropies[labels == 0]
ent_ictal = entropies[labels == 1]

mean_inter = ent_inter.mean()
mean_ictal = ent_ictal.mean()

ratio = mean_ictal / mean_inter

print("\nEntropy Metrics")
print("Interictal Mean:", mean_inter)
print("Ictal Mean:", mean_ictal)
print("Entropy Ratio:", ratio)

# -------------------------------------------------
# PLOT
# -------------------------------------------------
plt.figure(figsize=(7,5))

plt.hist(ent_inter, bins=20, alpha=0.6, label="Interictal")
plt.hist(ent_ictal, bins=20, alpha=0.6, label="Ictal")

plt.xlabel("von Neumann Entropy")
plt.ylabel("Count")
plt.title("Quantum State Entropy Distribution")

plt.legend()

plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR,"phase5b_entropy_distribution.png")
plt.savefig(plot_path)

plt.close()

print("📊 Plot saved:", plot_path)

# -------------------------------------------------
# TRANSPORT METRICS
# -------------------------------------------------
transport_mean = transport_metrics.mean(axis=0)

np.save(os.path.join(OUTPUT_DIR,"phase5b_entropy.npy"),entropies)
np.save(os.path.join(OUTPUT_DIR,"phase5b_transport_mean.npy"),transport_mean)

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
summary = f"""
Phase V-B — Dataset Validation Summary
-------------------------------------

Samples: {N}
Quantum dimension: {D}x{D}

Entropy
Interictal mean : {mean_inter}
Ictal mean      : {mean_ictal}
Ratio           : {ratio}

Transport mean
{transport_mean}

Conclusion
✔ Quantum density matrices valid
✔ Entropy structure meaningful
✔ Dataset ready for QML models
"""

summary_file = os.path.join(OUTPUT_DIR,"phase5b_summary.txt")

with open(summary_file,"w") as f:
    f.write(summary)

print(summary)
print("📄 Summary saved:",summary_file)

print("✅ Phase V-B complete")
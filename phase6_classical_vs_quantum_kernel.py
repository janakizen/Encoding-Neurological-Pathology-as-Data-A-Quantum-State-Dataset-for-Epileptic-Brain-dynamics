import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("🔷 Phase VI started: Classical vs Quantum Feature Comparison (RECTIFIED)")

# ----------------------------
# Load dataset
# ----------------------------
data = np.load("outputs/quantum_epilepsy_dataset.npz", allow_pickle=True)

states = data["quantum_states"]   # (N, d, d) complex
labels = data["labels"]

N, d, _ = states.shape
print(f"✔ Loaded {N} samples of {d}x{d} density matrices")

# ----------------------------
# Classical feature extraction
# ----------------------------
def classical_features(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals.real, 1e-12, 1.0)

    entropy = -np.sum(eigvals * np.log2(eigvals))
    purity = np.trace(rho @ rho).real

    return np.concatenate([eigvals, [entropy, purity]])

X_classical = np.array([classical_features(r) for r in states])

# ----------------------------
# Classical SVM
# ----------------------------
Xc_train, Xc_test, y_train, y_test = train_test_split(
    X_classical, labels, test_size=0.3, random_state=42
)

clf_classical = SVC(kernel="linear")
clf_classical.fit(Xc_train, y_train)

y_pred_classical = clf_classical.predict(Xc_test)
acc_classical = accuracy_score(y_test, y_pred_classical)

# ----------------------------
# Quantum kernel SVM
# ----------------------------
def quantum_kernel(A, B):
    return np.trace(A @ B).real

K = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i, j] = quantum_kernel(states[i], states[j])

idx = np.arange(N)
idx_train, idx_test, yq_train, yq_test = train_test_split(
    idx, labels, test_size=0.3, random_state=42
)

K_train = K[np.ix_(idx_train, idx_train)]
K_test = K[np.ix_(idx_test, idx_train)]

clf_quantum = SVC(kernel="precomputed")
clf_quantum.fit(K_train, yq_train)

y_pred_quantum = clf_quantum.predict(K_test)
acc_quantum = accuracy_score(yq_test, y_pred_quantum)

# ----------------------------
# Visualization (classical PCA)
# ----------------------------
pca = PCA(n_components=2)
Xc_2d = pca.fit_transform(X_classical)

plt.figure()
for label in np.unique(labels):
    plt.scatter(
        Xc_2d[labels == label, 0],
        Xc_2d[labels == label, 1],
        label=f"Class {label}",
        alpha=0.7
    )

plt.title("Classical Feature Space (Spectral + Entropy)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/classical_feature_space_pca.png", dpi=300)
plt.close()

# ----------------------------
# Results
# ----------------------------
print("\n📊 RESULTS")
print(f"Classical Feature SVM Accuracy: {acc_classical:.3f}")
print(f"Quantum Kernel SVM Accuracy:    {acc_quantum:.3f}")

print("✅ Phase VI complete")

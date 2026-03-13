import mne
import numpy as np
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

print("✅ Phase II started")

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "/Users/janakinagesh/Downloads/phase1_epilepsy/data/chb01_03.edf"
OUT_DIR = "/Users/janakinagesh/Downloads/phase1_epilepsy/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ICTAL_START = 2996
ICTAL_END = 3036

WINDOW_SEC = 2.0
STEP_SEC = 0.5

# -----------------------------
# HELPERS
# -----------------------------
def compute_plv_matrix(data):
    phases = np.angle(hilbert(data))
    n_ch = data.shape[0]
    plv = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            val = np.abs(np.mean(np.exp(1j * (phases[i] - phases[j]))))
            plv[i, j] = plv[j, i] = val
    return plv

def state_space_metrics(states):
    cov = np.cov(states.T)
    det = np.linalg.det(cov)
    entropy = 0.5 * np.log((2 * np.pi * np.e) ** states.shape[1] * det)
    return det, entropy

# -----------------------------
# LOAD EEG
# -----------------------------
raw = mne.io.read_raw_edf(DATA_PATH, preload=True, verbose=False)
raw.pick_types(eeg=True)

sfreq = raw.info["sfreq"]
data = raw.get_data()

theta = mne.filter.filter_data(data, sfreq, 4, 8, verbose=False)

# -----------------------------
# SLIDING WINDOWS
# -----------------------------
states = []
labels = []

win = int(WINDOW_SEC * sfreq)
step = int(STEP_SEC * sfreq)

for start in range(0, theta.shape[1] - win, step):
    end = start + win
    t_sec = start / sfreq

    plv = compute_plv_matrix(theta[:, start:end])
    eigvals = np.linalg.eigvalsh(plv)

    states.append(eigvals)
    labels.append(1 if ICTAL_START <= t_sec <= ICTAL_END else 0)

states = np.array(states)
labels = np.array(labels)

# -----------------------------
# PCA
# -----------------------------
pca = PCA(n_components=2)
states_2d = pca.fit_transform(states)

inter = states_2d[labels == 0]
ictal = states_2d[labels == 1]

# -----------------------------
# METRICS
# -----------------------------
vol_i, ent_i = state_space_metrics(inter)
vol_s, ent_s = state_space_metrics(ictal)

# -----------------------------
# PLOT 1: STATE SPACE
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(inter[:, 0], inter[:, 1], s=10, alpha=0.4, label="Interictal")
plt.scatter(ictal[:, 0], ictal[:, 1], s=15, color="red", label="Ictal")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Phase II: Oscillatory State Space (Theta PLV)")
plt.legend()
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/phase2_state_space_theta.png", dpi=300)
plt.close()

# -----------------------------
# PLOT 2: ENTROPY & VOLUME
# -----------------------------
plt.figure(figsize=(6, 4))
plt.bar(["Interictal", "Ictal"], [ent_i, ent_s])
plt.ylabel("Differential Entropy")
plt.title("State-Space Entropy Comparison")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase2_entropy_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
plt.bar(["Interictal", "Ictal"], [vol_i, vol_s])
plt.ylabel("Covariance Determinant (Volume)")
plt.title("State-Space Volume Comparison")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/phase2_volume_comparison.png", dpi=300)
plt.close()

# -----------------------------
# REPORT
# -----------------------------
print("\nInterictal Metrics")
print(f"Volume: {vol_i:.4e}")
print(f"Entropy: {ent_i:.4f}")

print("\nIctal Metrics")
print(f"Volume: {vol_s:.4e}")
print(f"Entropy: {ent_s:.4f}")

print("\nRatios (Ictal / Interictal)")
print(f"Volume Ratio:  {vol_s / vol_i:.4f}")
print(f"Entropy Ratio: {ent_s / ent_i:.4f}")

print("\n✅ Phase II visualizations saved to outputs/")
np.save(f"{OUT_DIR}/states_2d.npy", states_2d)
np.save(f"{OUT_DIR}/labels.npy", labels)
np.save(f"{OUT_DIR}/times.npy", np.arange(len(labels)) * STEP_SEC)

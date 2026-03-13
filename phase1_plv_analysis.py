import mne
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# -----------------------------
# FILE CONFIGURATION
# -----------------------------
FILE_PATH = "/Users/janakinagesh/Downloads/phase1_epilepsy/data/chb01_03.edf"

ICTAL_START = 2996    # seconds
ICTAL_END   = 3036

INTERICTAL_START = 200
INTERICTAL_END   = 260

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def extract_window(data, start, end, sfreq):
    return data[:, int(start * sfreq):int(end * sfreq)]

def compute_plv(sig1, sig2):
    """Phase Locking Value"""
    phase1 = np.angle(hilbert(sig1))
    phase2 = np.angle(hilbert(sig2))
    return np.abs(np.mean(np.exp(1j * (phase1 - phase2))))

def compute_pli(sig1, sig2):
    """Phase Lag Index"""
    phase1 = np.angle(hilbert(sig1))
    phase2 = np.angle(hilbert(sig2))
    return np.abs(np.mean(np.sign(np.sin(phase1 - phase2))))

def compute_plv_distribution(data):
    n_ch = data.shape[0]
    plvs = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            plvs.append(compute_plv(data[i], data[j]))
    return np.array(plvs)

def compute_pli_distribution(data):
    n_ch = data.shape[0]
    plis = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            plis.append(compute_pli(data[i], data[j]))
    return np.array(plis)

def compute_cfc(phase_data, amp_data):
    """Phase-Amplitude Coupling using Modulation Index (simplified)"""
    n_ch = phase_data.shape[0]
    cfc_list = []
    for i in range(n_ch):
        for j in range(n_ch):
            phase = np.angle(hilbert(phase_data[i]))
            amp = np.abs(hilbert(amp_data[j]))
            amp_norm = (amp - np.mean(amp)) / np.std(amp)
            mi = np.abs(np.mean(np.exp(1j * phase) * amp_norm))
            cfc_list.append(mi)
    return np.array(cfc_list)

# -----------------------------
# MAIN PHASE I
# -----------------------------
def run_phase1():
    print("✅ Phase I started")

    # -----------------------------
    # LOAD EEG
    # -----------------------------
    raw = mne.io.read_raw_edf(FILE_PATH, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    sfreq = raw.info["sfreq"]
    data = raw.get_data()

    print("Sampling frequency:", sfreq)
    print("EEG shape:", data.shape)

    # -----------------------------
    # FILTERS
    # -----------------------------
    theta = mne.filter.filter_data(data, sfreq, l_freq=4, h_freq=8, method="fir", phase="zero", verbose=False)
    gamma = mne.filter.filter_data(data, sfreq, l_freq=30, h_freq=80, method="fir", phase="zero", verbose=False)

    # -----------------------------
    # EXTRACT WINDOWS
    # -----------------------------
    theta_interictal = extract_window(theta, INTERICTAL_START, INTERICTAL_END, sfreq)
    theta_ictal = extract_window(theta, ICTAL_START, ICTAL_END, sfreq)

    gamma_interictal = extract_window(gamma, INTERICTAL_START, INTERICTAL_END, sfreq)
    gamma_ictal = extract_window(gamma, ICTAL_START, ICTAL_END, sfreq)

    # -----------------------------
    # COMPUTE METRICS
    # -----------------------------
    plv_interictal = compute_plv_distribution(theta_interictal)
    plv_ictal = compute_plv_distribution(theta_ictal)

    pli_interictal = compute_pli_distribution(theta_interictal)
    pli_ictal = compute_pli_distribution(theta_ictal)

    cfc_interictal = compute_cfc(theta_interictal, gamma_interictal)
    cfc_ictal = compute_cfc(theta_ictal, gamma_ictal)

    # -----------------------------
    # PRINT SUMMARY
    # -----------------------------
    print(f"Mean PLV: Interictal={np.mean(plv_interictal):.3f}, Ictal={np.mean(plv_ictal):.3f}")
    print(f"Mean PLI: Interictal={np.mean(pli_interictal):.3f}, Ictal={np.mean(pli_ictal):.3f}")
    print(f"Mean CFC: Interictal={np.mean(cfc_interictal):.3f}, Ictal={np.mean(cfc_ictal):.3f}")

    # -----------------------------
    # VISUALIZE
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(plv_interictal, bins=50, alpha=0.5, label="PLV Interictal")
    plt.hist(plv_ictal, bins=50, alpha=0.5, label="PLV Ictal")
    plt.hist(pli_interictal, bins=50, alpha=0.5, label="PLI Interictal")
    plt.hist(pli_ictal, bins=50, alpha=0.5, label="PLI Ictal")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Phase Synchrony Metrics (Theta)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("phase1_metrics_theta.png")
    print("✅ Plot saved as phase1_metrics_theta.png")

    # CFC plot separately
    plt.figure(figsize=(8, 5))
    plt.hist(cfc_interictal, bins=50, alpha=0.6, label="CFC Interictal")
    plt.hist(cfc_ictal, bins=50, alpha=0.6, label="CFC Ictal")
    plt.xlabel("Modulation Index (Theta-Gamma)")
    plt.ylabel("Count")
    plt.title("Cross-Frequency Coupling (Theta-Gamma)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("phase1_cfc_theta_gamma.png")
    print("✅ Plot saved as phase1_cfc_theta_gamma.png")


if __name__ == "__main__":
    run_phase1()

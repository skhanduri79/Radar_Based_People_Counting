import numpy as np
from scipy.signal import butter, filtfilt

# Load data
X = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\dataset\X_scenario1.npy")
y = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\dataset\y_scenario1.npy")

# -----------------------------
# Preprocessing functions
# -----------------------------

def dc_removal(sample):
    """
    sample: (50, 1280)
    """
    return sample - np.mean(sample, axis=1, keepdims=True)

def bandpass_filter(signal, fs=39e9, lowcut=5.65e9, highcut=7.95e9, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def bandpass_sample(sample):
    """
    sample: (50, 1280)
    """
    filtered = np.zeros_like(sample)
    for i in range(sample.shape[0]):
        filtered[i] = bandpass_filter(sample[i])
    return filtered

def clutter_removal(sample):
    """
    sample: (50, 1280)
    """
    clutter = np.mean(sample, axis=0, keepdims=True)
    return sample - clutter

def preprocess_sample(sample):
    """
    Returns:
        sample_curvelet: band-pass only (for curvelet features)
        sample_dbf: band-pass + clutter removed (for DBF features)
    """
    sample_dc = dc_removal(sample)
    sample_bp = bandpass_sample(sample_dc)

    sample_curvelet = sample_bp
    sample_dbf = clutter_removal(sample_bp)

    return sample_curvelet, sample_dbf

# -----------------------------
# Apply preprocessing
# -----------------------------

X_curvelet = []
X_dbf = []

for sample in X:
    curvelet_sig, dbf_sig = preprocess_sample(sample)
    X_curvelet.append(curvelet_sig)
    X_dbf.append(dbf_sig)

X_curvelet = np.array(X_curvelet)
X_dbf = np.array(X_dbf)

# -----------------------------
# Save outputs
# -----------------------------

np.save("X_curvelet.npy", X_curvelet)
np.save("X_dbf.npy", X_dbf)
np.save("y.npy", y)

print("Preprocessing complete.")
print("X_curvelet shape:", X_curvelet.shape)
print("X_dbf shape:", X_dbf.shape)



# def refine_signal(signal, start_idx=50, end_idx=1280):
#     """
#     Remove near-range static reflections (antenna coupling / ring-down)
#     Args:
#         signal: 1D array of shape (1280,)
#         start_idx: index to start keeping (skip near-range)
#         end_idx: index to stop (optional)
#     Returns:
#         refined_signal: 1D array of length (end_idx - start_idx)
#     """
#     refined = signal[start_idx:end_idx]
#     return refined
#
# def preprocess_X(X, start_idx=50, end_idx=1280, fs=20e9, lowcut=1e9, highcut=10e9):
#     """
#     Preprocess all radar samples
#     Args:
#         X: (num_samples, 50, 1280) raw radar data
#     Returns:
#         X_preprocessed: (num_samples, 50, L) preprocessed radar matrices
#     """
#     num_samples, num_frames, num_points = X.shape
#     L = end_idx - start_idx
#     X_preprocessed = np.zeros((num_samples, num_frames, L))
#
#     for i in range(num_samples):
#         for j in range(num_frames):
#             signal = X[i, j, :]
#             filtered = bandpass_filter(signal, fs=fs, lowcut=lowcut, highcut=highcut)
#             refined = refine_signal(filtered, start_idx=start_idx, end_idx=end_idx)
#             X_preprocessed[i, j, :] = refined
#
#     return X_preprocessed

# # Usage example
# X_preprocessed = preprocess_X(X, start_idx=50, end_idx=1280)
# print("Preprocessed X shape:", X_preprocessed.shape)
#
# np.save("X_preprocessed.npy", X_preprocessed)
# np.save("y.npy", y)
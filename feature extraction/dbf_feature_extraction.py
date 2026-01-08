import numpy as np

# -----------------------------
# Load DBF branch
# -----------------------------
X_dbf = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\preprocessing\X_dbf.npy")   # (1760, 50, 1280)

print("Loaded X_dbf:", X_dbf.shape)

# -----------------------------
# Distance Bin Feature function
# -----------------------------
def extract_dbf_features(sample, bin_size):
    """
    sample: (50, 1280)
    bin_size: number of distance samples per bin

    returns: feature vector
             [max_amp_bin1, energy_bin1, ..., max_amp_binN, energy_binN]
    """
    num_frames, num_points = sample.shape
    num_bins = num_points // bin_size

    features = []

    for b in range(num_bins):
        bin_slice = sample[:, b * bin_size:(b + 1) * bin_size]

        # Max amplitude (paper-inspired)
        max_amp = np.max(np.abs(bin_slice))

        # Energy (paper-inspired)
        energy = np.sum(bin_slice ** 2)

        features.append(max_amp)
        features.append(energy)

    return np.array(features)

# -----------------------------
# Multi-resolution DBF extraction
# -----------------------------
bin_sizes = [32, 64, 128]  # paper-aligned resolutions

X_dbf_features = []

for i, sample in enumerate(X_dbf):
    if i % 200 == 0:
        print(f"Processing sample {i}/{len(X_dbf)}")

    sample_features = []

    for bin_size in bin_sizes:
        dbf_feats = extract_dbf_features(sample, bin_size)
        sample_features.append(dbf_feats)

    # Concatenate all bin resolutions
    sample_features = np.concatenate(sample_features)
    X_dbf_features.append(sample_features)

X_dbf_features = np.array(X_dbf_features)

print("DBF feature matrix shape:", X_dbf_features.shape)

# -----------------------------
# Save DBF features
# -----------------------------
np.save("X_dbf_features.npy", X_dbf_features)
print("Saved X_dbf_features.npy")

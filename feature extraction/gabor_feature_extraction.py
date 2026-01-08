import numpy as np
from skimage.filters import gabor
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler


N_ORIENTATIONS = 6
N_SCALES = 6
FREQUENCIES = np.linspace(0.08, 0.35, N_SCALES)

EPS = 1e-8

# ================================
# FAST Gabor features for ONE sample
# ================================
def extract_fast_gabor_features(sample):
    """
    sample: (50, 1280)
    returns: ~220 features
    """

    features = []

    # Light spatial downsampling (huge speedup)
    sample_ds = sample[:, ::4]   # (50, 320)

    for theta in np.linspace(0, np.pi, N_ORIENTATIONS, endpoint=False):
        for freq in FREQUENCIES:
            real, imag = gabor(sample_ds, frequency=freq, theta=theta)
            mag = np.sqrt(real**2 + imag**2)

            # Directional energy descriptors
            features.append(np.mean(mag))
            features.append(np.std(mag))
            features.append(np.sum(mag**2))
            features.append(skew(mag.flatten()))
            features.append(kurtosis(mag.flatten()))

    # ---- Global descriptors (paper-style) ----
    flat = sample_ds.flatten()
    features.extend([
        np.mean(flat),
        np.std(flat),
        np.var(flat),
        np.min(flat),
        np.max(flat),
        skew(flat),
        kurtosis(flat),
        np.mean(np.abs(flat)),
        np.sum(flat**2)
    ])

    return np.array(features)


# ================================
# Batch extraction
# ================================
def extract_fast_gabor_matrix(X):
    feats = []

    for i in range(X.shape[0]):
        feats.append(extract_fast_gabor_features(X[i]))

        if i % 100 == 0:
            print(f"Processed {i}/{X.shape[0]} samples")

    X_feat = np.array(feats)
    X_feat = StandardScaler().fit_transform(X_feat)
    return X_feat


# ================================
# Main
# ================================
if __name__ == "__main__":
    X = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\preprocessing\X_curvelet.npy")   # (1760, 50, 1280)
    y = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\preprocessing\y.npy")

    X_gabor = extract_fast_gabor_matrix(X)

    print(" Gabor feature shape:", X_gabor.shape)

    np.save("X_gabor_features.npy", X_gabor)


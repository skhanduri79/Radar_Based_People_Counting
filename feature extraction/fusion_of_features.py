import numpy as np
from sklearn.preprocessing import StandardScaler

X_gabor = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\X_gabor_features.npy")        # (1760, 189)
X_dbf = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\X_dbf_features.npy")         # (~1760, ~140)
y = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\y.npy")

X_hybrid = np.concatenate([X_gabor, X_dbf], axis=1)

# Optional re-standardization (recommended)
X_hybrid = StandardScaler().fit_transform(X_hybrid)

print("Hybrid feature shape:", X_hybrid.shape)

np.save("X_hybrid_features.npy", X_hybrid)

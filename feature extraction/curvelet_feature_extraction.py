

import numpy as np
import pyct

# -----------------------------
# Load preprocessed data
# -----------------------------
X_curvelet = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\preprocessing\X_curvelet.npy")

print("Loaded X_curvelet with shape:", X_curvelet.shape)

# -----------------------------
# Step 2: Test curvelet transform on ONE sample
# -----------------------------
sample = X_curvelet[0]
print("Testing curvelet transform on sample shape:", sample.shape)

coeffs = pyct.fdct2(sample, is_real=True)

print("Number of scales:", len(coeffs))
for s, scale in enumerate(coeffs):
    print(f"Scale {s}: number of directions = {len(scale)}")

# -----------------------------
# Step 3: Curvelet feature extraction
# -----------------------------
def curvelet_features(sample, top_k=5):
    """
    Extract curvelet-based features from a (50 x 1280) radar sample
    """
    coeffs = pyct.fdct2(sample, is_real=True)
    features = []

    # Coarse scale
    coarse = coeffs[0][0]
    features.append(np.mean(coarse))
    features.append(np.sum(coarse ** 2))

    # Fine scale
    fine = coeffs[-1]
    fine_all = np.concatenate([c.flatten() for c in fine])
    features.append(np.sum(fine_all ** 2))
    top_vals = np.sort(np.abs(fine_all))[-top_k:]
    features.extend(top_vals.tolist())

    # Detail scales
    detail_energy = []
    for scale in coeffs[1:-1]:
        for direction in scale:
            detail_energy.append(np.sum(direction ** 2))

    detail_energy = np.array(detail_energy)
    third = len(detail_energy) // 3

    features.append(np.sum(detail_energy[:third]))           # ~45°
    features.append(np.sum(detail_energy[third:2*third]))   # ~90°
    features.append(np.sum(detail_energy[2*third:]))        # ~135°

    return np.array(features)

# -----------------------------
# Step 4: Extract features for ALL samples
# -----------------------------
X_ctf = []

for sample in X_curvelet:
    feats = curvelet_features(sample)
    X_ctf.append(feats)

X_ctf = np.array(X_ctf)

print("Final curvelet feature matrix shape:", X_ctf.shape)

# -----------------------------
# Save features
# -----------------------------
np.save("X_ctf.npy", X_ctf)
print("Saved X_ctf.npy")

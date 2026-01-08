import numpy as np
from scipy.io import savemat

# Load your existing numpy array
X_curvelet = np.load(
    r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\preprocessing\X_curvelet.npy"
)

print(X_curvelet.shape)  # should be (1760, 50, 1280)

# Save as MATLAB file
savemat(
    r"C:\Users\SAKSHI\Downloads\CurveLab-2.1.3\CurveLab-2.1.3\X_curvelet.mat",
    {"X_curvelet": X_curvelet}
)

print("Saved X_curvelet.mat successfully")

import os
import numpy as np

def load_scenario1(base_path, sample_size=50, signal_length=1280):
    X, y = [], []

    for part_folder in sorted(os.listdir(base_path)):
        part_path = os.path.join(base_path, part_folder)
        if not os.path.isdir(part_path):
            continue

        for people_folder in sorted(os.listdir(part_path), key=lambda x: int(x)):
            people_path = os.path.join(part_path, people_folder)
            if not os.path.isdir(people_path):
                continue

            try:
                label = int(people_folder)
            except ValueError:
                continue

            files = sorted(os.listdir(people_path))
            if not files:
                continue

            frames = []
            for f in files:
                file_path = os.path.join(people_path, f)
                try:
                    data = np.loadtxt(file_path)
                    # If data is 1D, make it 2D
                    if data.ndim == 1:
                        data = data[np.newaxis, :]
                    # Pad/truncate to signal_length
                    if data.shape[1] < signal_length:
                        padded = np.zeros((data.shape[0], signal_length))
                        padded[:, :data.shape[1]] = data
                        data = padded
                    elif data.shape[1] > signal_length:
                        data = data[:, :signal_length]
                    frames.append(data)
                except Exception as e:
                    print(f"Skipping {file_path}, error: {e}")

            if not frames:
                continue

            measurement = np.vstack(frames)  # shape: (num_signals, 1280)

            # Split into 50-frame samples
            for i in range(0, measurement.shape[0], sample_size):
                sample = measurement[i:i+sample_size]
                # Pad sample if less than sample_size
                if sample.shape[0] < sample_size:
                    padded_sample = np.zeros((sample_size, signal_length))
                    padded_sample[:sample.shape[0], :] = sample
                    sample = padded_sample
                X.append(sample)
                y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


# Usage
BASE = r"C:\Users\SAKSHI\PycharmProjects\CapstoneProject\dataset\scenario 1"
X, y = load_scenario1(BASE)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Example labels:", np.unique(y))


import matplotlib.pyplot as plt

# Show first sample as an image
plt.imshow(X[0], aspect='auto', cmap='jet')
plt.colorbar()
plt.title(f"Sample 0 - People count: {y[0]}")
plt.xlabel("Signal points (1280)")
plt.ylabel("Frame (50)")
plt.show()

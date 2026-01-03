# "0-10 people randomly walking in a constrained area, for a total of 88,000 radar signals (440 measurements)
This project focuses on people counting using radar sensor data.
Unlike camera-based systems, radar offers advantages such as privacy preservation, robustness to lighting conditions, and reliable performance in indoor environments.

The objective of this work is to process radar signal data and develop machine learning–based methods to estimate the number of people present in different scenarios.
Scenario 1: 0–20 people randomly walking in a constrained area at a density of Three Persons Per Square Meter​

Measurements:​

8,000 radar signals per people count​

Each measurement contains 200 received signals​

Each signal has 1,280 sampling points (3 m detection range)​

 Total dataset: 376,000 IR-UWB radar signals across all scenarios.​

​

Data Loading:​

Loaded using a custom Python script:​

Organized by folder: Scenario 1 → Number of people → measurement → frames​

Each measurement split into samples of 50 consecutive frames​

Resulting data shape:​

X: (1760, 50, 1280) → 1760 samples, 50 frames per sample, 1280 points per frame​

y: (1760,) → corresponding labels (number of people)

Signal Pre-processing​

Bandpass filter: Remove out-of-band noise and static clutter.​

Refinement step: Remove direct coupling, antenna ring-down, near-range static reflections.​

Time-Distance Window Selection: Focus on region containing human reflections.​

Resulting shape after pre-processing: X_preprocessed: (1760, 50, 1230)​
​

 Feature Extraction​

Hybrid CTF-DBF features:​

Curvelet Transform Features (CTF): Captures motion and spatial structure from radar matrices.​

Distance Bin Features (DBF): Divide distance axis into bins → compute statistics per bin​

Final hybrid feature matrix: (1760, 74)​

Each sample → 74-dimensional feature vector​

Paper used 300 features → fewer features may reduce classifier performance.​



 Accuracy Random Forest​: 0.935​

​

​

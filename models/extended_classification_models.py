import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore",category=UndefinedMetricWarning)

# --------------------------------------------------
# Config
# --------------------------------------------------
NUM_RUNS = 20
TEST_SIZE = 0.2
RANDOM_SEEDS = list(range(100, 100 + NUM_RUNS))


# --------------------------------------------------
# Load Data
# --------------------------------------------------
X = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\X_hybrid_features.npy")
y = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\y.npy")

num_classes = len(np.unique(y))
classes = [str(i) for i in range(num_classes)]


# --------------------------------------------------
# Model Definitions
# --------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, algorithm="SAMME.R"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation="relu", solver="adam", max_iter=300)
}


# --------------------------------------------------
# Metrics Storage
# --------------------------------------------------
overall_metrics = {name: {"acc": [], "prec": [], "rec": [], "f1": []} for name in models}
per_class_metrics = {name: {"prec": [], "rec": [], "f1": []} for name in models}


# --------------------------------------------------
# Multi-Run Evaluation
# --------------------------------------------------
for run_idx, seed in enumerate(RANDOM_SEEDS):
    print(f"\n>>> RUN {run_idx + 1}/{NUM_RUNS} (seed={seed})")

    for name, model in models.items():
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed
        )

        # Train
        model.set_params(random_state=seed) if hasattr(model, "random_state") else None
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Overall metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        overall_metrics[name]["acc"].append(acc)
        overall_metrics[name]["prec"].append(prec)
        overall_metrics[name]["rec"].append(rec)
        overall_metrics[name]["f1"].append(f1)

        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        prec_cls = [report[c]["precision"] if c in report else 0 for c in classes]
        rec_cls = [report[c]["recall"] if c in report else 0 for c in classes]
        f1_cls = [report[c]["f1-score"] if c in report else 0 for c in classes]

        per_class_metrics[name]["prec"].append(prec_cls)
        per_class_metrics[name]["rec"].append(rec_cls)
        per_class_metrics[name]["f1"].append(f1_cls)


# --------------------------------------------------
# Averaging Results
# --------------------------------------------------
print("\n 20-RUN AVERAGED CLASSIFICATION RESULTS \n")

for name in models:
    acc_mean  = np.mean(overall_metrics[name]["acc"])
    acc_std   = np.std(overall_metrics[name]["acc"])

    prec_mean = np.mean(overall_metrics[name]["prec"])
    prec_std  = np.std(overall_metrics[name]["prec"])

    rec_mean  = np.mean(overall_metrics[name]["rec"])
    rec_std   = np.std(overall_metrics[name]["rec"])

    f1_mean   = np.mean(overall_metrics[name]["f1"])
    f1_std    = np.std(overall_metrics[name]["f1"])

    print(f"--- {name} ---")
    print(f"Accuracy : {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"Precision: {prec_mean:.3f} ± {prec_std:.3f}")
    print(f"Recall   : {rec_mean:.3f} ± {rec_std:.3f}")
    print(f"F1-score : {f1_mean:.3f} ± {f1_std:.3f}\n")


# --------------------------------------------------
# Per-Class Metric Plotting
# --------------------------------------------------
for name in models:
    mean_prec = np.mean(per_class_metrics[name]["prec"], axis=0)
    mean_rec = np.mean(per_class_metrics[name]["rec"], axis=0)
    mean_f1 = np.mean(per_class_metrics[name]["f1"], axis=0)

    x = np.arange(num_classes)
    plt.figure(figsize=(6, 4))
    plt.plot(x, mean_prec, marker='o', label='Precision')
    plt.plot(x, mean_rec, marker='o', label='Recall')
    plt.plot(x, mean_f1, marker='o', label='F1-Score')

    plt.xticks(x, classes)
    plt.xlabel('True Count (People)')
    plt.ylabel('Score')
    plt.title(f'10-Run Averaged Per-Class Performance ({name})')
    plt.legend()
    plt.tight_layout()
    plt.show()

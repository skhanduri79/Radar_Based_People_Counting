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


# --------------------------------------------------
# Load data
# --------------------------------------------------
X = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\X_hybrid_features.npy")
y = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\y.npy")

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=50,
        algorithm="SAMME.R",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100, 200, 100),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )
}

# --------------------------------------------------
# Train / Test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
print("\nHybrid Feature Classification Results\n")

for name, model in models.items():
    print(f"--- {name} ---")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- PRINT CLASSIFICATION REPORT ---
    print(f"\n=== Classification Report ({name}) ===")
    print(classification_report(y_test, y_pred, digits=3))

    # --- PRINT SUMMARY METRICS ---
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}\n")

    # --- CLASS-WISE METRIC PLOT ---
    report = classification_report(y_test, y_pred, output_dict=True)
    num_classes = len(np.unique(y_test))
    classes = [str(i) for i in range(num_classes)]

    precision_cls = [report[c]['precision'] for c in classes]
    recall_cls = [report[c]['recall'] for c in classes]
    f1_cls = [report[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))

    plt.figure(figsize=(6, 4))
    plt.plot(x, precision_cls, marker='o', label='Precision')
    plt.plot(x, recall_cls, marker='o', label='Recall')
    plt.plot(x, f1_cls, marker='o', label='F1-Score')

    plt.xticks(x, classes)
    plt.xlabel('True Count (People)')
    plt.ylabel('Score')
    plt.title(f'Per-Class Performance ({name})')
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}\n")
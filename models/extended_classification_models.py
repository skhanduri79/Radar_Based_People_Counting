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

# --------------------------------------------------
# Classifier factory functions (ALL consistent now)
# --------------------------------------------------
def decision_tree(rs):
    return DecisionTreeClassifier(random_state=rs)

def adaboost(rs):
    return AdaBoostClassifier(
        n_estimators=50,
        algorithm="SAMME.R",
        random_state=rs
    )

def random_forest(rs):
    return RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        random_state=rs,
        n_jobs=-1
    )

def neural_network(rs):
    return MLPClassifier(
        hidden_layer_sizes=(100, 200, 100),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=rs
    )

models = {
    "Decision Tree": decision_tree,
    "AdaBoost": adaboost,
    "Random Forest": random_forest,
    "Neural Network": neural_network
}

# --------------------------------------------------
# Repeated evaluation
# --------------------------------------------------
N_RUNS = 5

print("\n5-run averaged classification results\n")

for name, model_fn in models.items():
    accs, precs, recs, f1s = [], [], [], []

    for seed in range(N_RUNS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        model = model_fn(seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        p, r, f, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        precs.append(p)
        recs.append(r)
        f1s.append(f)

    print(f"--- {name} ---")
    print(f"Accuracy : {np.mean(accs):.3f}")
    print(f"Precision: {np.mean(precs):.3f}")
    print(f"Recall   : {np.mean(recs):.3f}")
    print(f"F1-score : {np.mean(f1s):.3f}\n")

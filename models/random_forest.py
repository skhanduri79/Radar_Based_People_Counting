import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)

# ======================================================
# Metric utilities
# ======================================================

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def round_predictions(y_pred, y_min, y_max):
    y_round = np.rint(y_pred).astype(int)
    return np.clip(y_round, y_min, y_max)


def tolerance_accuracy(y_true, y_pred, tol=1):
    """
    Accuracy within ±tol people (paper-style)
    """
    y_round = np.rint(y_pred)
    return np.mean(np.abs(y_round - y_true) <= tol)


# ======================================================
# Plot: Precision / Recall / F1 vs Person Count
# ======================================================

def plot_classification_metrics(y_true, y_pred, title):
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    classes = sorted([int(k) for k in report.keys() if k.isdigit()])

    precision = [report[str(c)]['precision'] for c in classes]
    recall = [report[str(c)]['recall'] for c in classes]
    f1 = [report[str(c)]['f1-score'] for c in classes]

    plt.figure(figsize=(9, 5))
    plt.plot(classes, precision, marker='o', label='Precision')
    plt.plot(classes, recall, marker='s', label='Recall')
    plt.plot(classes, f1, marker='^', label='F1-score')

    plt.xlabel("Number of People")
    plt.ylabel("Score")
    plt.title(f"Per-Class Metrics ({title})")
    plt.xticks(classes)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ======================================================
# Train & Evaluate one feature set
# ======================================================

def train_and_evaluate(X, y, name):
    print(f"\n================ {name} =================")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- FINAL RF PARAMETERS (LOCK THESE) ----
    model = RandomForestRegressor(
        n_estimators=180,          # balanced & fast
        max_features="sqrt",       # critical for speed
        max_depth=20,              # prevents runaway trees
        min_samples_leaf=2,        # regularization
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    # ---- Predictions ----
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # ---- Regression metrics ----
    train_mae, train_rmse, train_r2 = regression_metrics(y_train, y_train_pred)
    test_mae, test_rmse, test_r2 = regression_metrics(y_test, y_test_pred)

    print("\nRegression Metrics:")
    print(f"Train MAE  : {train_mae:.3f} | Test MAE  : {test_mae:.3f}")
    print(f"Train RMSE : {train_rmse:.3f} | Test RMSE : {test_rmse:.3f}")
    print(f"Train R²   : {train_r2:.3f} | Test R²   : {test_r2:.3f}")

    # ---- Exact-match accuracy (strict) ----
    y_train_round = round_predictions(y_train_pred, y.min(), y.max())
    y_test_round = round_predictions(y_test_pred, y.min(), y.max())

    train_acc = accuracy_score(y_train, y_train_round)
    test_acc = accuracy_score(y_test, y_test_round)

    print("\nExact-Match Accuracy (after rounding):")
    print(f"Train Accuracy : {train_acc:.3f}")
    print(f"Test Accuracy  : {test_acc:.3f}")

    # ---- Tolerance-based accuracy (paper-style) ----
    tol_acc = tolerance_accuracy(y_test, y_test_pred, tol=1)
    print(f"Tolerance Accuracy (±1 person): {tol_acc:.3f}")

    # ---- Classification report ----
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_round, digits=3))

    # ---- Plot per-class metrics ----
    plot_classification_metrics(y_test, y_test_round, name)



# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    # Load data
    y = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\preprocessing\y.npy")

    X_dbf = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\X_dbf_features.npy")
    X_gabor = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\X_gabor_features.npy")
    X_hybrid = np.load(r"C:\Users\SAKSHI\Desktop\Radar_based_people_counting\feature extraction\X_hybrid_features.npy")

    print("Feature shapes:")
    # print("DBF    :", X_dbf.shape)
    # print("Gabor  :", X_gabor.shape)
    print("Hybrid :", X_hybrid.shape)

    # Run experiments
    # train_and_evaluate(X_dbf, y, "DBF only")
    # train_and_evaluate(X_gabor, y, "Gabor only")
    train_and_evaluate(X_hybrid, y, "Hybrid (DBF + Gabor)")

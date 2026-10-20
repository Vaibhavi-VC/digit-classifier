"""Train, tune and compare several classifiers on the handwritten digits dataset.

Usage:
    python train.py
    python train.py --cv 3 --seed 7
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def candidates():
    """Return {name: (pipeline, param_grid)}. Grid keys use the pipeline step prefix."""
    return {
        "Logistic Regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            {"logisticregression__C": [0.01, 0.1, 1, 10]},
        ),
        "k-Nearest Neighbors": (
            make_pipeline(StandardScaler(), KNeighborsClassifier()),
            {"kneighborsclassifier__n_neighbors": [1, 3, 5, 9]},
        ),
        "SVM (RBF)": (
            make_pipeline(StandardScaler(), SVC()),
            {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.001, 0.01]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(random_state=0)),
            {"randomforestclassifier__n_estimators": [100, 300],
             "randomforestclassifier__max_depth": [None, 15]},
        ),
    }


def plot_comparison(results, path):
    names = list(results)
    cv_scores = [results[n]["cv_accuracy"] for n in names]
    test_scores = [results[n]["test_accuracy"] for n in names]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.2 for i in x], cv_scores, 0.4, label="CV accuracy")
    ax.bar([i + 0.2 for i in x], test_scores, 0.4, label="Test accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model comparison (tuned)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_samples(model, X_test, y_test, images_test, path, n=12):
    preds = model.predict(X_test[:n])
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.5))
    for ax, img, p, t in zip(axes.ravel(), images_test[:n], preds, y_test[:n]):
        ax.imshow(img, cmap="gray_r")
        ax.set_title(f"pred {p}", color="green" if p == t else "red", fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample test predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5, help="cross-validation folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    digits = load_digits()
    X_train, X_test, y_train, y_test, _, img_test = train_test_split(
        digits.data, digits.target, digits.images,
        test_size=args.test_size, stratify=digits.target, random_state=args.seed,
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

    results, fitted = {}, {}
    for name, (pipe, grid) in candidates().items():
        search = GridSearchCV(pipe, grid, cv=args.cv, n_jobs=-1)
        search.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, search.predict(X_test))
        results[name] = {
            "cv_accuracy": round(search.best_score_, 4),
            "test_accuracy": round(test_acc, 4),
            "best_params": search.best_params_,
        }
        fitted[name] = search.best_estimator_
        print(f"{name:22s} CV={search.best_score_:.4f}  test={test_acc:.4f}  {search.best_params_}")

    # Select by cross-validation score (not test score) to avoid test-set leakage.
    best_name = max(results, key=lambda n: results[n]["cv_accuracy"])
    best = fitted[best_name]
    print(f"\nBest model (by CV): {best_name}\n")
    print(classification_report(y_test, best.predict(X_test)))

    joblib.dump(best, MODEL_DIR / "model.joblib")
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps({"best_model": best_name, "results": results}, indent=2)
    )
    plot_comparison(results, REPORT_DIR / "model_comparison.png")
    plot_samples(best, X_test, y_test, img_test, REPORT_DIR / "sample_predictions.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved model to {MODEL_DIR/'model.joblib'} and plots to {REPORT_DIR}/")


if __name__ == "__main__":
    main()

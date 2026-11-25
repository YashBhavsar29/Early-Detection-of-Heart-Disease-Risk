"""Train and save the final Heart Disease prediction pipeline.

Run from the repository root:
    python src/models/train_final_model.py

The script fetches the documented UCI dataset, exports reproducible CSV files,
fits the selected Logistic Regression pipeline on training data only, evaluates
it once on the held-out test set, and writes model/report artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42


def make_pipeline() -> Pipeline:
    """Build the selected Logistic Regression pipeline without fitting it."""
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, numeric_features),
            ("categorical", categorical_preprocessor, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def save_figures(
    X: pd.DataFrame,
    y_binary: pd.Series,
    y_test: pd.Series,
    test_predictions: pd.Series,
    test_probabilities: pd.Series,
) -> None:
    """Create report figures from the original data and final test predictions."""
    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(6, 4))
    labels = y_binary.map({0: "No disease", 1: "Disease present"})
    sns.countplot(x=labels, hue=labels, legend=False, palette="Set2")
    plt.title("Heart disease target distribution")
    plt.xlabel("")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "target_distribution.png", dpi=200)
    plt.close()

    missing = X.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=missing.index, y=missing.values, color="salmon")
    plt.title("Missing values by feature")
    plt.xlabel("Feature")
    plt.ylabel("Missing values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "missing_values.png", dpi=200)
    plt.close()

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        display_labels=["No disease", "Disease present"],
        cmap="Blues",
    )
    display.ax_.set_title("Final Logistic Regression test confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(REPORT_DIR / "final_test_confusion_matrix.png", dpi=200)
    plt.close(display.figure_)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, test_probabilities)
    test_auc = roc_auc_score(y_test, test_probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, label=f"Logistic Regression (AUC = {test_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-skill reference")
    plt.title("Final test ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "final_test_roc_curve.png", dpi=200)
    plt.close()


def main() -> None:
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    original_target = heart_disease.data.targets["num"].copy()
    raw_dataset = X.assign(num=original_target)
    raw_dataset.to_csv(RAW_DATA_DIR / "heart_disease_original.csv", index=False)

    y_binary = (original_target > 0).astype(int).rename("disease_present")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_binary,
        test_size=0.20,
        stratify=y_binary,
        random_state=RANDOM_STATE,
    )
    X_train.assign(disease_present=y_train).to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    X_test.assign(disease_present=y_test).to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    final_pipeline = make_pipeline()
    final_pipeline.fit(X_train, y_train)

    test_predictions = pd.Series(final_pipeline.predict(X_test), index=X_test.index, name="prediction")
    test_probabilities = pd.Series(
        final_pipeline.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name="disease_probability",
    )
    test_output = pd.DataFrame(
        {
            "actual_disease_present": y_test,
            "predicted_disease_present": test_predictions,
            "predicted_probability": test_probabilities,
        }
    )
    test_output.to_csv(REPORT_DIR / "final_test_predictions.csv", index=True, index_label="original_row_index")

    metrics = {
        "accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(y_test, test_predictions),
        "recall": recall_score(y_test, test_predictions),
        "f1_score": f1_score(y_test, test_predictions),
        "roc_auc": roc_auc_score(y_test, test_probabilities),
    }
    pd.DataFrame([metrics]).round(4).to_csv(REPORT_DIR / "final_test_metrics.csv", index=False)
    (REPORT_DIR / "final_test_classification_report.txt").write_text(
        classification_report(y_test, test_predictions, target_names=["No disease", "Disease present"]),
        encoding="utf-8",
    )

    joblib.dump(final_pipeline, MODEL_DIR / "heart_disease_logistic_pipeline.joblib")
    model_metadata = {
        "model_name": "Logistic Regression",
        "dataset": "UCI Heart Disease (ID 45)",
        "target_definition": "0 = num equals 0; 1 = num is between 1 and 4",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
        "test_metrics": {name: round(value, 4) for name, value in metrics.items()},
    }
    (MODEL_DIR / "heart_disease_logistic_metadata.json").write_text(
        json.dumps(model_metadata, indent=2), encoding="utf-8"
    )

    save_figures(X, y_binary, y_test, test_predictions, test_probabilities)
    print("Saved final pipeline, CSV exports, metrics, predictions, and report figures.")
    print(pd.Series(metrics).round(4).to_string())


if __name__ == "__main__":
    main()

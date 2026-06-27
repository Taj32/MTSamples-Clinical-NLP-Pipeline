import pandas as pd
import numpy as np
import mlflow
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.stage1_doctype.label_consolidation import get_class_weights


def load_splits(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(f"{prefix}_train.csv", index_col=0)
    val   = pd.read_csv(f"{prefix}_val.csv",   index_col=0)
    test  = pd.read_csv(f"{prefix}_test.csv",  index_col=0)
    return train, val, test


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def run_tfidf_baseline(
    stage: int = 2,
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
) -> dict:
    """
    Train and evaluate a TF-IDF + Logistic Regression baseline.
    Logs everything to MLflow.
    Returns dict of val metrics.
    """
    assert stage in (1, 2), "stage must be 1 or 2"
    prefix = f"data/processed/stage{stage}"
    label_col = "stage1_label" if stage == 1 else "stage2_label"

    train, val, test = load_splits(prefix)

    # Drop nulls in label or text
    train = train.dropna(subset=[label_col, "transcription"])
    val   = val.dropna(subset=[label_col, "transcription"])
    test  = test.dropna(subset=[label_col, "transcription"])

    X_train = train["transcription"]
    y_train = train[label_col]
    X_val   = val["transcription"]
    y_val   = val[label_col]
    X_test  = test["transcription"]
    y_test  = test[label_col]

    # Class weights
    weights = get_class_weights(y_train)
    labels  = sorted(y_train.unique())

    print(f"\nTraining Stage {stage} TF-IDF baseline...")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Classes: {labels}")

    mlflow.set_experiment(f"stage{stage}_tfidf_baseline")

    with mlflow.start_run(run_name=f"tfidf_lr_stage{stage}"):

        # Log hyperparameters
        mlflow.log_params({
            "max_features": max_features,
            "ngram_range": str(ngram_range),
            "C": C,
            "stage": stage,
        })

        # Build pipeline
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,       # log-scale TF dampens common terms
                min_df=2,                # ignore terms appearing in <2 docs
                strip_accents="unicode",
                analyzer="word",
            )),
            ("clf", LogisticRegression(
                C=C,
                class_weight=weights,
                max_iter=1000,
                random_state=42,
                #n_jobs=-1,
            )),
        ])

        pipeline.fit(X_train, y_train)

        # ── Validation metrics ──────────────────────────────────────────
        val_preds = pipeline.predict(X_val)
        val_macro_f1 = f1_score(y_val, val_preds, average="macro")
        val_accuracy = accuracy_score(y_val, val_preds)

        print(f"\n=== VALIDATION RESULTS ===")
        print(f"  Macro F1:  {val_macro_f1:.4f}")
        print(f"  Accuracy:  {val_accuracy:.4f}")
        print(f"\n{classification_report(y_val, val_preds, target_names=labels)}")

        mlflow.log_metrics({
            "val_macro_f1": val_macro_f1,
            "val_accuracy": val_accuracy,
        })

        # ── Confusion matrix ────────────────────────────────────────────
        Path("outputs").mkdir(exist_ok=True)
        cm = confusion_matrix(y_val, val_preds, labels=labels)
        cm_path = f"outputs/stage{stage}_tfidf_confusion_matrix.png"
        plot_confusion_matrix(cm, labels, cm_path,
                              title=f"Stage {stage} TF-IDF Baseline — Validation")
        mlflow.log_artifact(cm_path)

        # ── Test metrics (only look at once) ───────────────────────────
        test_preds = pipeline.predict(X_test)
        test_macro_f1 = f1_score(y_test, test_preds, average="macro")
        test_accuracy = accuracy_score(y_test, test_preds)

        print(f"\n=== TEST RESULTS ===")
        print(f"  Macro F1:  {test_macro_f1:.4f}")
        print(f"  Accuracy:  {test_accuracy:.4f}")

        mlflow.log_metrics({
            "test_macro_f1": test_macro_f1,
            "test_accuracy": test_accuracy,
        })

        # Log per-class metrics
        report = classification_report(
            y_test, test_preds,
            target_names=labels,
            output_dict=True
        )
        for label in labels:
            if label in report:
                mlflow.log_metrics({
                    f"test_f1_{label}": report[label]["f1-score"],
                    f"test_precision_{label}": report[label]["precision"],
                    f"test_recall_{label}": report[label]["recall"],
                })

        # Save report as artifact
        report_path = f"outputs/stage{stage}_tfidf_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        print(f"\nMLflow run logged.")

    return {
        "val_macro_f1": val_macro_f1,
        "val_accuracy": val_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_accuracy": test_accuracy,
    }
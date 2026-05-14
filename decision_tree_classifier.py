from __future__ import annotations

from pathlib import Path
import csv
import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

from segment_features import FEATURE_COLUMNS


def load_feature_table(features_csv: Path, label_column: str = "label_id"):
    rows = []
    with features_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get(label_column, "")
            if label == "":
                continue

            values = []
            missing = False
            for column in FEATURE_COLUMNS:
                value = row.get(column, "")
                if value == "":
                    missing = True
                    break
                values.append(float(value))
            if missing:
                continue

            rows.append(
                {
                    "split": row.get("split", "train"),
                    "stem": row.get("stem", ""),
                    "label": label,
                    "features": values,
                }
            )

    if not rows:
        raise ValueError(f"No usable feature rows found in {features_csv}")

    x = np.asarray([row["features"] for row in rows], dtype=np.float32)
    y = np.asarray([row["label"] for row in rows])
    splits = np.asarray([row["split"] for row in rows])
    stems = [row["stem"] for row in rows]
    return x, y, splits, stems, FEATURE_COLUMNS


def split_indices(splits: np.ndarray):
    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    if len(train_idx) == 0:
        raise ValueError("No train split rows found. Build features from the full dataset or provide a CSV with split=train rows.")
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def evaluate_split(model, x: np.ndarray, y: np.ndarray, indices: np.ndarray, labels: list[str]):
    if len(indices) == 0:
        return None

    y_true = y[indices]
    y_pred = model.predict(x[indices])
    return {
        "samples": int(len(indices)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def write_feature_importances(path: Path, feature_names: list[str], importances: np.ndarray):
    rows = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "importance"])
        for feature, importance in rows:
            writer.writerow([feature, f"{float(importance):.10f}"])


def train_decision_tree(
    features_csv: Path,
    output_dir: Path,
    label_column: str = "label_id",
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
):
    x, y, splits, _stems, feature_names = load_feature_table(features_csv, label_column)
    indices = split_indices(splits)
    labels = sorted(set(y.tolist()))

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(x[indices["train"]], y[indices["train"]])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "decision_tree.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "label_column": label_column,
            "features_csv": str(features_csv.resolve()).replace("\\", "/"),
        },
        model_path,
    )

    metrics = {
        "features_csv": str(features_csv.resolve()).replace("\\", "/"),
        "label_column": label_column,
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
        "splits": {},
    }
    for split, split_idx in indices.items():
        result = evaluate_split(model, x, y, split_idx, labels)
        if result is not None:
            metrics["splits"][split] = result

    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "tree_rules.txt").write_text(export_text(model, feature_names=feature_names), encoding="utf-8")
    write_feature_importances(output_dir / "feature_importances.csv", feature_names, model.feature_importances_)

    return {
        "model_path": model_path,
        "metrics_path": output_dir / "metrics.json",
        "rules_path": output_dir / "tree_rules.txt",
        "feature_importances_path": output_dir / "feature_importances.csv",
        "metrics": metrics,
    }

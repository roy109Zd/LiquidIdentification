from __future__ import annotations

from pathlib import Path
import csv
import json

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from segment_features import FEATURE_COLUMNS


ALGORITHM_CHOICES = (
    "decision-tree",
    "random-forest",
    "extra-trees",
    "gradient-boosting",
    "ada-boost",
    "logistic-regression",
    "linear-svm",
    "rbf-svm",
    "knn",
    "gaussian-nb",
    "mlp",
)


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


def write_linear_coefficients(path: Path, feature_names: list[str], classes: np.ndarray, coefficients: np.ndarray):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "feature", "coefficient"])
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        if len(classes) == 2 and coefficients.shape[0] == 1:
            class_names = [classes[1]]
        else:
            class_names = classes
        for class_name, class_coefficients in zip(class_names, coefficients):
            rows = sorted(zip(feature_names, class_coefficients), key=lambda item: abs(item[1]), reverse=True)
            for feature, coefficient in rows:
                writer.writerow([class_name, feature, f"{float(coefficient):.10f}"])


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not layers:
        raise ValueError("hidden_layer_sizes must contain at least one positive integer")
    if any(layer <= 0 for layer in layers):
        raise ValueError("hidden_layer_sizes values must be positive integers")
    return layers


def parse_svm_gamma(value: str | float) -> str | float:
    if isinstance(value, float):
        return value
    if value in {"scale", "auto"}:
        return value
    return float(value)


def build_classifier(
    algorithm: str,
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    n_estimators: int = 200,
    neighbors: int = 5,
    svm_c: float = 1.0,
    svm_gamma: str | float = "scale",
    max_iter: int = 2000,
    hidden_layer_sizes: str = "64,32",
):
    if algorithm == "decision-tree":
        return DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    if algorithm == "random-forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    if algorithm == "extra-trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    if algorithm == "gradient-boosting":
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth is not None else 3,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    if algorithm == "ada-boost":
        return AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )
    if algorithm == "logistic-regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=svm_c,
                        max_iter=max_iter,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if algorithm == "linear-svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearSVC(C=svm_c, class_weight="balanced", max_iter=max_iter, random_state=random_state)),
            ]
        )
    if algorithm == "rbf-svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(C=svm_c, gamma=parse_svm_gamma(svm_gamma), class_weight="balanced")),
            ]
        )
    if algorithm == "knn":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=neighbors)),
            ]
        )
    if algorithm == "gaussian-nb":
        return GaussianNB()
    if algorithm == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=parse_hidden_layers(hidden_layer_sizes),
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def unwrap_estimator(model):
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def train_classifier(
    features_csv: Path,
    output_dir: Path,
    label_column: str = "label_id",
    algorithm: str = "decision-tree",
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    n_estimators: int = 200,
    neighbors: int = 5,
    svm_c: float = 1.0,
    svm_gamma: str | float = "scale",
    max_iter: int = 2000,
    hidden_layer_sizes: str = "64,32",
):
    x, y, splits, _stems, feature_names = load_feature_table(features_csv, label_column)
    indices = split_indices(splits)
    labels = sorted(set(y.tolist()))

    model = build_classifier(
        algorithm=algorithm,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_estimators=n_estimators,
        neighbors=neighbors,
        svm_c=svm_c,
        svm_gamma=svm_gamma,
        max_iter=max_iter,
        hidden_layer_sizes=hidden_layer_sizes,
    )
    model.fit(x[indices["train"]], y[indices["train"]])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{algorithm.replace('-', '_')}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "label_column": label_column,
            "features_csv": str(features_csv.resolve()).replace("\\", "/"),
            "algorithm": algorithm,
        },
        model_path,
    )

    metrics = {
        "features_csv": str(features_csv.resolve()).replace("\\", "/"),
        "label_column": label_column,
        "algorithm": algorithm,
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "neighbors": neighbors,
        "svm_c": svm_c,
        "svm_gamma": svm_gamma,
        "max_iter": max_iter,
        "hidden_layer_sizes": hidden_layer_sizes,
        "splits": {},
    }
    for split, split_idx in indices.items():
        result = evaluate_split(model, x, y, split_idx, labels)
        if result is not None:
            metrics["splits"][split] = result

    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    estimator = unwrap_estimator(model)

    rules_path = None
    if algorithm == "decision-tree":
        rules_path = output_dir / "tree_rules.txt"
        rules_path.write_text(export_text(estimator, feature_names=feature_names), encoding="utf-8")

    feature_importances_path = None
    if hasattr(estimator, "feature_importances_"):
        feature_importances_path = output_dir / "feature_importances.csv"
        write_feature_importances(feature_importances_path, feature_names, estimator.feature_importances_)

    coefficients_path = None
    if hasattr(estimator, "coef_"):
        coefficients_path = output_dir / "linear_coefficients.csv"
        write_linear_coefficients(coefficients_path, feature_names, estimator.classes_, estimator.coef_)

    return {
        "model_path": model_path,
        "metrics_path": output_dir / "metrics.json",
        "rules_path": rules_path,
        "feature_importances_path": feature_importances_path,
        "coefficients_path": coefficients_path,
        "metrics": metrics,
    }


def train_decision_tree(
    features_csv: Path,
    output_dir: Path,
    label_column: str = "label_id",
    criterion: str = "gini",
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
):
    return train_classifier(
        features_csv=features_csv,
        output_dir=output_dir,
        label_column=label_column,
        algorithm="decision-tree",
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

from argparse import ArgumentParser
import csv
from pathlib import Path
from types import SimpleNamespace

from train_obb import DEFAULT_DATASET, LABEL_SET_ALIASES


ROOT = Path(__file__).resolve().parent
DEFAULT_SEGMENT_OUTPUT = ROOT / "runs" / "tree_segments"
DEFAULT_TREE_OUTPUT = ROOT / "runs" / "tree_classifier"
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


def parse_args():
    parser = ArgumentParser(
        description=(
            "Build segmented bottle features and train a classical machine-learning classifier. "
            "Pass --features to train from an existing features.csv. "
            "Pass --label-model to build masks from this project's trained YOLO OBB model."
        )
    )

    parser.add_argument("--features", default=None, help="Existing features.csv. If set, segmentation is skipped.")
    parser.add_argument(
        "--label-model",
        "--seg-model",
        dest="seg_model",
        default=None,
        help="Optional project-trained YOLO weights under runs/obb. Omit to build masks from label polygons.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset root containing images and labels.")
    parser.add_argument("--source", default=None, help="Optional image directory for segmentation preprocessing.")
    parser.add_argument(
        "--label-set",
        choices=sorted(LABEL_SET_ALIASES),
        default="0123",
        help="Label set used as the classifier target.",
    )
    parser.add_argument("--segments-output", default=str(DEFAULT_SEGMENT_OUTPUT), help="Output root for segmented images.")
    parser.add_argument("--tree-output", default=str(DEFAULT_TREE_OUTPUT), help="Output root for the classifier model.")
    parser.add_argument("--reuse-features", action="store_true", help="Reuse segments-output/features.csv if it exists.")

    parser.add_argument("--imgsz", type=int, default=640, help="YOLO segmentation inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO segmentation confidence threshold.")
    parser.add_argument("--device", default=None, help="Use 0 for GPU, cpu for CPU, or omit for auto.")
    parser.add_argument("--class-id", type=int, default=None, help="Optional segmentation class id to keep.")
    parser.add_argument(
        "--select",
        choices=("highest-conf", "largest-mask"),
        default="highest-conf",
        help="How to choose one mask when an image has multiple segmentation results.",
    )
    parser.add_argument(
        "--background",
        choices=("black", "white", "transparent"),
        default="black",
        help="Background used outside the selected segmentation mask.",
    )
    parser.add_argument("--crop", action="store_true", help="Crop segmented output images to the mask bounding box.")
    parser.add_argument("--overwrite-segments", action="store_true", help="Overwrite segmented images and features.csv.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars during feature generation.")

    parser.add_argument(
        "--label-column",
        choices=("label_id", "label_name"),
        default="label_id",
        help="Target column in features.csv.",
    )
    parser.add_argument(
        "--algorithm",
        choices=(*ALGORITHM_CHOICES, "all"),
        default="decision-tree",
        help="Classical ML classifier to train, or all to run every supported classifier.",
    )
    parser.add_argument(
        "--criterion",
        choices=("gini", "entropy", "log_loss"),
        default="gini",
        help="Tree split criterion for decision-tree, random-forest, and extra-trees.",
    )
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth. Omit for unlimited depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples required at a tree leaf node.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of estimators for ensemble algorithms.")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors for KNN.")
    parser.add_argument("--svm-c", type=float, default=1.0, help="Regularization strength C for SVM/logistic regression.")
    parser.add_argument("--svm-gamma", default="scale", help="Gamma for rbf-svm; use scale, auto, or a numeric value.")
    parser.add_argument("--max-iter", type=int, default=2000, help="Maximum iterations for linear models and MLP.")
    parser.add_argument("--hidden-layer-sizes", default="64,32", help="Comma-separated hidden layer sizes for MLP.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for supported classifiers.")
    return parser.parse_args()


def build_features_if_needed(args) -> Path:
    if args.features:
        features_csv = Path(args.features).resolve()
        if not features_csv.exists():
            raise FileNotFoundError(f"Feature CSV not found: {features_csv}")
        return features_csv

    segment_output = Path(args.segments_output).resolve()
    features_csv = segment_output / "features.csv"
    if args.reuse_features and features_csv.exists():
        print(f"Reusing feature CSV: {features_csv}")
        return features_csv

    segment_args = SimpleNamespace(
        model=args.seg_model,
        dataset=args.dataset,
        source=args.source,
        label_set=args.label_set,
        output=str(segment_output),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        class_id=args.class_id,
        select=args.select,
        background=args.background,
        crop=args.crop,
        overwrite=args.overwrite_segments,
        no_progress=args.no_progress,
    )
    from prepare_tree_segments import build_segment_dataset

    print("Building segmented feature dataset...")
    return build_segment_dataset(segment_args)


def metric_for_best_split(metrics: dict):
    splits = metrics.get("splits", {})
    for split in ("val", "test", "train"):
        result = splits.get(split)
        if result is not None:
            return split, float(result["accuracy"])
    return None, None


def print_result_files(result: dict):
    print(f"Classifier model: {result['model_path']}")
    print(f"Metrics: {result['metrics_path']}")
    if result["rules_path"]:
        print(f"Rules: {result['rules_path']}")
    if result["feature_importances_path"]:
        print(f"Feature importances: {result['feature_importances_path']}")
    if result["coefficients_path"]:
        print(f"Linear coefficients: {result['coefficients_path']}")


def print_accuracy_summary(rows: list[dict], output_dir: Path):
    summary_path = output_dir / "algorithm_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["algorithm", "split", "accuracy", "accuracy_percent", "status", "output_dir"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nAccuracy summary:")
    print(f"{'algorithm':<20} {'split':<6} {'accuracy':>10}  status")
    for row in rows:
        accuracy = row["accuracy"]
        accuracy_text = "" if accuracy == "" else f"{float(accuracy):.4f}"
        print(f"{row['algorithm']:<20} {row['split']:<6} {accuracy_text:>10}  {row['status']}")

    successful = [row for row in rows if row["status"] == "ok" and row["accuracy"] != ""]
    if not successful:
        raise RuntimeError("No classifier finished successfully; see failed rows above.")

    best = max(successful, key=lambda row: float(row["accuracy"]))
    print(
        "\nBest result: "
        f"{best['algorithm']} on {best['split']} accuracy={float(best['accuracy']):.4f} "
        f"({best['accuracy_percent']})"
    )
    print(f"Summary CSV: {summary_path}")


def train_one_algorithm(args, features_csv: Path, algorithm: str, output_dir: Path):
    from decision_tree_classifier import train_classifier

    print(f"Training {algorithm} from {features_csv}...")
    result = train_classifier(
        features_csv=features_csv,
        output_dir=output_dir,
        label_column=args.label_column,
        algorithm=algorithm,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        neighbors=args.neighbors,
        svm_c=args.svm_c,
        svm_gamma=args.svm_gamma,
        max_iter=args.max_iter,
        hidden_layer_sizes=args.hidden_layer_sizes,
    )
    print_result_files(result)
    return result


def main():
    args = parse_args()
    features_csv = build_features_if_needed(args)
    tree_output = Path(args.tree_output).resolve()

    if args.algorithm == "all":
        rows = []
        for algorithm in ALGORITHM_CHOICES:
            algorithm_output = tree_output / algorithm
            try:
                result = train_one_algorithm(args, features_csv, algorithm, algorithm_output)
                split, accuracy = metric_for_best_split(result["metrics"])
                rows.append(
                    {
                        "algorithm": algorithm,
                        "split": split or "",
                        "accuracy": "" if accuracy is None else f"{accuracy:.10f}",
                        "accuracy_percent": "" if accuracy is None else f"{accuracy * 100:.2f}%",
                        "status": "ok",
                        "output_dir": str(algorithm_output).replace("\\", "/"),
                    }
                )
            except Exception as exc:
                print(f"FAILED {algorithm}: {exc}")
                rows.append(
                    {
                        "algorithm": algorithm,
                        "split": "",
                        "accuracy": "",
                        "accuracy_percent": "",
                        "status": f"failed: {exc}",
                        "output_dir": str(algorithm_output).replace("\\", "/"),
                    }
                )

        print_accuracy_summary(rows, tree_output)
        return

    result = train_one_algorithm(args, features_csv, args.algorithm, tree_output)
    split, accuracy = metric_for_best_split(result["metrics"])
    if accuracy is not None:
        print(f"\nBest result: {args.algorithm} on {split} accuracy={accuracy:.4f} ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()

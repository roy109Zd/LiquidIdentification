from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

from train_obb import DEFAULT_DATASET, LABEL_SET_ALIASES


ROOT = Path(__file__).resolve().parent
DEFAULT_SEGMENT_OUTPUT = ROOT / "runs" / "tree_segments"
DEFAULT_TREE_OUTPUT = ROOT / "runs" / "tree_classifier"


def parse_args():
    parser = ArgumentParser(
        description=(
            "Build segmented bottle features and train a decision-tree classifier. "
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
        help="Label set used as the decision-tree target.",
    )
    parser.add_argument("--segments-output", default=str(DEFAULT_SEGMENT_OUTPUT), help="Output root for segmented images.")
    parser.add_argument("--tree-output", default=str(DEFAULT_TREE_OUTPUT), help="Output root for the decision-tree model.")
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
        "--criterion",
        choices=("gini", "entropy", "log_loss"),
        default="gini",
        help="Decision-tree split criterion.",
    )
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth. Omit for unlimited depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples required at a leaf node.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the decision tree.")
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


def main():
    args = parse_args()
    features_csv = build_features_if_needed(args)
    from decision_tree_classifier import train_decision_tree

    print(f"Training decision tree from {features_csv}...")
    result = train_decision_tree(
        features_csv=features_csv,
        output_dir=Path(args.tree_output).resolve(),
        label_column=args.label_column,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    print(f"Decision tree model: {result['model_path']}")
    print(f"Metrics: {result['metrics_path']}")
    print(f"Rules: {result['rules_path']}")
    print(f"Feature importances: {result['feature_importances_path']}")


if __name__ == "__main__":
    main()

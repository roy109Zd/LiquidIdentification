from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import os
import shutil
import subprocess

from convert_lcdtc_to_yolo_obb import convert_lcdtc_to_yolo_obb


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolo11m-obb.pt"
DEFAULT_DATASET = ROOT / "bottleDataset"
DEFAULT_VIEW_ROOT = ROOT / ".dataset_views"
DEFAULT_LCDTC = ROOT / "LCDTC"
DEFAULT_LCDTC_OUTPUT = DEFAULT_VIEW_ROOT / "lcdtc_obb"
LABEL_SETS = {
    "0123": {
        "label_dir": "labels_0123",
        "nc": 4,
        "names": ["none", "little", "mid", "much"],
    },
    "01": {
        "label_dir": "labels_01",
        "nc": 2,
        "names": ["none", "exist"],
    },
    "bottle": {
        "label_dir": "label_bottle",
        "nc": 1,
        "names": ["bottle"],
    },
}
LABEL_SET_ALIASES = {
    "0123": "0123",
    "labels_0123": "0123",
    "01": "01",
    "labels_01": "01",
    "bottle": "bottle",
    "label_bottle": "bottle",
    "labels_bottle": "bottle",
}


def parse_args():
    parser = ArgumentParser(description="Train YOLO11m-OBB on the bottle dataset.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL.name,
        help="Local weights path or supported Ultralytics model name. Local files are used first.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional custom dataset yaml. If set, --label-set/--dataset/--view-root are bypassed.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="Use 0 for GPU, cpu for CPU, or omit for auto.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default=None, help="Run name. Defaults to a timestamped name.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "obb"))
    parser.add_argument(
        "--label-set",
        choices=sorted(LABEL_SET_ALIASES),
        default="0123",
        help="Use 4-class labels_0123, binary labels_01, or single-class label_bottle.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset root containing images and label sets.")
    parser.add_argument("--view-root", default=str(DEFAULT_VIEW_ROOT), help="Where to build temporary YOLO dataset views.")
    parser.add_argument("--prepare-data-only", action="store_true", help="Build the selected dataset view and print its yaml path.")
    parser.add_argument(
        "--include-lcdtc",
        action="store_true",
        help="Convert LCDTC COCO labels to YOLO OBB and include them with the selected dataset.",
    )
    parser.add_argument("--lcdtc-root", default=str(DEFAULT_LCDTC), help="LCDTC root containing annotations and images.")
    parser.add_argument("--lcdtc-output", default=str(DEFAULT_LCDTC_OUTPUT), help="Where to write converted LCDTC YOLO OBB labels.")
    parser.add_argument("--rebuild-lcdtc", action="store_true", help="Rebuild the converted LCDTC labels before training.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest interrupted training run.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument(
        "--augment-geom",
        action="store_true",
        help="Enable geometry augmentation by passing rotation and translation settings to Ultralytics.",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=None,
        help="Maximum image rotation degrees for augmentation. Used with --augment-geom or when set explicitly.",
    )
    parser.add_argument(
        "--translate",
        type=float,
        default=None,
        help="Maximum image translation as a fraction of image size. Used with --augment-geom or when set explicitly.",
    )
    return parser.parse_args()


def remove_link_or_dir(path: Path):
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    if os.name == "nt" and path.is_dir() and path.stat().st_file_attributes & getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        path.rmdir()
        return
    shutil.rmtree(path)


def link_dir(source: Path, target: Path):
    source = source.resolve()
    remove_link_or_dir(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source, target_is_directory=True)
        return
    except OSError:
        if os.name != "nt":
            raise

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(target), str(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or result.stdout.strip())


def yaml_path(path: Path) -> str:
    return str(path.absolute()).replace("\\", "/")


def yaml_paths(paths) -> str:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        raise FileNotFoundError(f"No existing dataset paths found from: {paths}")
    return "\n".join(f"  - {yaml_path(path)}" for path in existing)


def resolve_model(model: str) -> str:
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)

    if not model_path.is_absolute() and model_path.parent == Path("."):
        root_model = ROOT / model
        if root_model.exists():
            return str(root_model)
        return model

    raise FileNotFoundError(
        f"Model weights not found: {model}. "
        "Use an existing local path or a supported Ultralytics model name, for example yolo11m-obb.pt."
    )


def write_dataset_yaml(path: Path, view_dir: Path, label_set: str):
    config = LABEL_SETS[label_set]
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(config["names"]))
    content = (
        f"path: {yaml_path(view_dir)}\n\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {config['nc']}\n"
        "names:\n"
        f"{names}\n"
    )
    path.write_text(content, encoding="utf-8")


def write_combined_dataset_yaml(path: Path, dataset: Path, lcdtc_dataset: Path, label_set: str):
    config = LABEL_SETS[label_set]
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(config["names"]))
    train_paths = [dataset / "images" / "train", lcdtc_dataset / "images" / "train"]
    val_paths = [dataset / "images" / "val", lcdtc_dataset / "images" / "val"]
    test_paths = [dataset / "images" / "test"]
    content = (
        f"path: {yaml_path(ROOT)}\n\n"
        "train:\n"
        f"{yaml_paths(train_paths)}\n"
        "val:\n"
        f"{yaml_paths(val_paths)}\n"
        "test:\n"
        f"{yaml_paths(test_paths)}\n\n"
        f"nc: {config['nc']}\n"
        "names:\n"
        f"{names}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def activate_label_set(dataset: Path, label_set: str):
    config = LABEL_SETS[label_set]
    selected_labels = dataset / config["label_dir"]
    active_labels = dataset / "labels"
    if not selected_labels.exists():
        raise FileNotFoundError(f"Label directory not found: {selected_labels}. Run prepare_dataset.py first.")

    link_dir(selected_labels, active_labels)


def prepare_dataset_view(dataset: Path, view_root: Path, label_set: str) -> Path:
    config = LABEL_SETS[label_set]
    images = dataset / "images"
    if not images.exists():
        raise FileNotFoundError(f"Images directory not found: {images}")
    activate_label_set(dataset, label_set)

    view_dir = view_root / config["label_dir"]
    view_dir.mkdir(parents=True, exist_ok=True)
    remove_link_or_dir(view_dir / "images")
    remove_link_or_dir(view_dir / "labels")

    data_yaml = view_dir / "bottle_obb.yaml"
    write_dataset_yaml(data_yaml, dataset, label_set)
    return data_yaml


def prepare_lcdtc_dataset_view(dataset: Path, view_root: Path, label_set: str, lcdtc_root: Path, lcdtc_output: Path, rebuild: bool) -> Path:
    config = LABEL_SETS[label_set]
    images = dataset / "images"
    if not images.exists():
        raise FileNotFoundError(f"Images directory not found: {images}")
    activate_label_set(dataset, label_set)

    lcdtc_dataset, summary = convert_lcdtc_to_yolo_obb(lcdtc_root, lcdtc_output, overwrite=rebuild)
    selected_lcdtc_labels = lcdtc_dataset / config["label_dir"]
    if not selected_lcdtc_labels.exists():
        raise FileNotFoundError(f"Converted LCDTC label directory not found: {selected_lcdtc_labels}")
    link_dir(selected_lcdtc_labels, lcdtc_dataset / "labels")

    view_dir = view_root / f"{config['label_dir']}_with_lcdtc"
    data_yaml = view_dir / "bottle_obb.yaml"
    write_combined_dataset_yaml(data_yaml, dataset, lcdtc_dataset, label_set)

    train_stats = summary.get("train", {})
    val_stats = summary.get("val", {})
    print(
        "LCDTC enabled: "
        f"{train_stats.get('images', 0)} train images, "
        f"{val_stats.get('images', 0)} val images"
    )
    return data_yaml


def main():
    args = parse_args()

    model_arg = resolve_model(args.model)
    label_set = LABEL_SET_ALIASES[args.label_set]
    if args.data and args.include_lcdtc:
        raise ValueError("--include-lcdtc cannot be combined with --data, because --data bypasses dataset view generation.")

    if args.data:
        data_path = Path(args.data)
    elif args.include_lcdtc:
        data_path = prepare_lcdtc_dataset_view(
            Path(args.dataset),
            Path(args.view_root),
            label_set,
            Path(args.lcdtc_root),
            Path(args.lcdtc_output),
            args.rebuild_lcdtc,
        )
    else:
        data_path = prepare_dataset_view(Path(args.dataset), Path(args.view_root), label_set)
    if args.prepare_data_only:
        print(data_path)
        return

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    run_name = args.name or f"bottle_yolo11m_obb_{datetime.now():%Y%m%d_%H%M%S}"

    from ultralytics import YOLO

    model = YOLO(model_arg)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": run_name,
        "task": "obb",
        "resume": args.resume,
        "exist_ok": args.exist_ok,
    }
    if args.augment_geom or args.degrees is not None:
        train_kwargs["degrees"] = 10.0 if args.degrees is None else args.degrees
    if args.augment_geom or args.translate is not None:
        train_kwargs["translate"] = 0.1 if args.translate is None else args.translate

    model.train(
        **train_kwargs,
    )


if __name__ == "__main__":
    main()

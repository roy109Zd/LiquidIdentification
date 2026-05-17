from argparse import ArgumentParser
from pathlib import Path
import shutil


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = ArgumentParser(description="Collect images whose predicted classes differ from YOLO labels.")
    parser.add_argument("--images", required=True, help="Image directory, for example bottleDataset/images/val.")
    parser.add_argument("--labels", required=True, help="Ground-truth label directory, for example bottleDataset/labels_01/val.")
    parser.add_argument("--pred-labels", required=True, help="YOLO prediction label directory, for example runs/predict/val_bottle_01/labels.")
    parser.add_argument("--pred-images", default=None, help="YOLO prediction image directory. Defaults to the parent of --pred-labels.")
    parser.add_argument("--output", required=True, help="Output directory for error cases.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    return parser.parse_args()


def read_classes(path: Path):
    if not path.exists():
        return set()

    classes = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            classes.add(line.split()[0])
    return classes


def copy_if_exists(source: Path, target: Path):
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main():
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    pred_label_dir = Path(args.pred_labels)
    pred_image_dir = Path(args.pred_images) if args.pred_images else pred_label_dir.parent
    output = Path(args.output)

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)

    errors = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = image_path.stem
        gt_classes = read_classes(label_dir / f"{stem}.txt")
        pred_classes = read_classes(pred_label_dir / f"{stem}.txt")

        if gt_classes == pred_classes:
            continue

        errors.append((stem, gt_classes, pred_classes))
        copy_if_exists(image_path, output / "images" / image_path.name)
        copy_if_exists(label_dir / f"{stem}.txt", output / "gt_labels" / f"{stem}.txt")
        copy_if_exists(pred_label_dir / f"{stem}.txt", output / "pred_labels" / f"{stem}.txt")
        copy_if_exists(pred_image_dir / image_path.name, output / "pred_images" / image_path.name)

    report_lines = ["stem,gt_classes,pred_classes"]
    for stem, gt_classes, pred_classes in errors:
        gt_text = "|".join(sorted(gt_classes)) if gt_classes else "none"
        pred_text = "|".join(sorted(pred_classes)) if pred_classes else "missing"
        report_lines.append(f"{stem},{gt_text},{pred_text}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "errors.csv").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Checked {image_dir}")
    print(f"Errors: {len(errors)}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()

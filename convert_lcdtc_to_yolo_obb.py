from argparse import ArgumentParser
from pathlib import Path
import json
import os
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "LCDTC"
DEFAULT_OUTPUT = ROOT / ".dataset_views" / "lcdtc_obb"
SPLIT_ANNOTATIONS = {
    "train": "instances_train2017.json",
    "val": "instances_val2017.json",
}
SPLIT_IMAGE_DIRS = {
    "train": "train2017",
    "val": "val2017",
}
LABEL_DIRS = ("labels_0123", "labels_01", "label_bottle")
CLASSIFY = {
    "labels_0123": ("none", "little", "mid", "much"),
    "labels_01": ("none", "exist"),
    "label_bottle": ("bottle",),
}
CATEGORY_TO_0123 = {
    "bottleempty": 0,
    "empty": 0,
    "bottlelittle": 1,
    "little": 1,
    "bottlehalf": 2,
    "half": 2,
    "bottlemuch": 3,
    "much": 3,
    "bottlefill": 3,
    "fill": 3,
}


def parse_args():
    parser = ArgumentParser(description="Convert LCDTC COCO annotations to YOLO OBB labels.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="LCDTC root containing annotations and images.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Converted dataset view root.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing converted labels and image links.")
    return parser.parse_args()


def remove_link_or_dir(path: Path):
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    if os.name == "nt" and path.is_dir() and path.stat().st_file_attributes & getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        path.rmdir()
        return
    shutil.rmtree(path)


def link_dir(source: Path, target: Path):
    source = source.resolve()
    if target.exists() or target.is_symlink():
        try:
            if target.resolve() == source:
                return
        except FileNotFoundError:
            pass
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


def normalize_category(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "")


def label_classes(class_0123: int) -> dict[str, int]:
    return {
        "labels_0123": class_0123,
        "labels_01": 0 if class_0123 == 0 else 1,
        "label_bottle": 0,
    }


def bbox_to_obb_line(class_id: int, bbox, width: int, height: int) -> str | None:
    x, y, w, h = (float(value) for value in bbox)
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    if x2 <= x1 or y2 <= y1:
        return None

    points = (
        x1 / width,
        y1 / height,
        x2 / width,
        y1 / height,
        x2 / width,
        y2 / height,
        x1 / width,
        y2 / height,
    )
    coords = " ".join(f"{value:.6f}" for value in points)
    return f"{class_id} {coords}"


def load_coco(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def convert_split(source: Path, output: Path, split: str) -> dict[str, int]:
    data = load_coco(source / "annotations" / SPLIT_ANNOTATIONS[split])
    image_dir = source / "images" / SPLIT_IMAGE_DIRS[split]
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    link_dir(image_dir, output / "images" / split)

    images = {image["id"]: image for image in data.get("images", [])}
    categories = {
        category["id"]: CATEGORY_TO_0123[normalize_category(category["name"])]
        for category in data.get("categories", [])
        if normalize_category(category["name"]) in CATEGORY_TO_0123
    }
    unknown = [
        category.get("name")
        for category in data.get("categories", [])
        if normalize_category(category.get("name", "")) not in CATEGORY_TO_0123
    ]
    if unknown:
        raise ValueError(f"Unsupported LCDTC categories in {SPLIT_ANNOTATIONS[split]}: {unknown}")

    labels = {label_dir: {image_id: [] for image_id in images} for label_dir in LABEL_DIRS}
    skipped = 0
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        image = images.get(ann.get("image_id"))
        if image is None:
            skipped += 1
            continue

        class_0123 = categories.get(ann.get("category_id"))
        if class_0123 is None:
            skipped += 1
            continue

        for label_dir, class_id in label_classes(class_0123).items():
            line = bbox_to_obb_line(class_id, ann.get("bbox", []), image["width"], image["height"])
            if line is not None:
                labels[label_dir][image["id"]].append(line)

    for label_dir, per_image in labels.items():
        split_dir = output / label_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for image_id, image in images.items():
            label_path = split_dir / f"{Path(image['file_name']).stem}.txt"
            text = "\n".join(per_image[image_id])
            if text:
                text += "\n"
            label_path.write_text(text, encoding="utf-8")

    return {
        "images": len(images),
        "annotations": len(data.get("annotations", [])),
        "skipped": skipped,
    }


def write_classify_files(output: Path):
    for label_dir, names in CLASSIFY.items():
        (output / label_dir / "classify.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def converted_ready(output: Path) -> bool:
    for split in SPLIT_ANNOTATIONS:
        image_dir = output / "images" / split
        if not image_dir.exists():
            return False
        for label_dir in LABEL_DIRS:
            labels = output / label_dir / split
            if not labels.exists() or not any(labels.glob("*.txt")):
                return False
    return True


def cached_summary(output: Path) -> dict[str, dict[str, int]]:
    summary = {}
    for split in SPLIT_ANNOTATIONS:
        image_dir = output / "images" / split
        image_count = sum(1 for path in image_dir.iterdir() if path.is_file())
        summary[split] = {
            "images": image_count,
            "annotations": 0,
            "skipped": 0,
        }
    return summary


def convert_lcdtc_to_yolo_obb(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT, overwrite: bool = False):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not (source / "annotations").exists():
        raise FileNotFoundError(f"LCDTC annotations directory not found: {source / 'annotations'}")

    if output.exists() and overwrite:
        remove_link_or_dir(output)
    output.mkdir(parents=True, exist_ok=True)

    if not overwrite and converted_ready(output):
        return output, cached_summary(output)

    for label_dir in LABEL_DIRS:
        if overwrite:
            remove_link_or_dir(output / label_dir)

    summary = {}
    for split in SPLIT_ANNOTATIONS:
        summary[split] = convert_split(source, output, split)
    write_classify_files(output)
    return output, summary


def main():
    args = parse_args()
    output, summary = convert_lcdtc_to_yolo_obb(Path(args.source), Path(args.output), args.overwrite)
    print(f"Converted LCDTC to {output}")
    for split, stats in summary.items():
        print(
            f"{split}: {stats['images']} images, "
            f"{stats['annotations']} annotations, {stats['skipped']} skipped"
        )


if __name__ == "__main__":
    main()

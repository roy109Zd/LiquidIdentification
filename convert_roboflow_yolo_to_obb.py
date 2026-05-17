from argparse import ArgumentParser
from pathlib import Path
import ast
import shutil


ROOT = Path(__file__).resolve().parent
SPLIT_ALIASES = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_MAP = {
    "empty": "none",
    "full_water_level": "much",
    "half_water_level": "mid",
    "three_quarters_level": "much",
}
TARGET_CLASSES = {
    "none": 0,
    "little": 1,
    "mid": 2,
    "much": 3,
}


def parse_args():
    parser = ArgumentParser(description="Convert a Roboflow YOLO detection dataset into this project's YOLO OBB format.")
    parser.add_argument("--source", required=True, help="Downloaded Roboflow YOLO dataset root.")
    parser.add_argument("--output", default=str(ROOT / "importedDataset"), help="Output dataset root.")
    parser.add_argument(
        "--class-map",
        nargs="*",
        default=[],
        help="Optional mappings like empty=none half_water_level=mid. Defaults support Bottle fill level.",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of reusing hard links when possible.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    return parser.parse_args()


def read_class_names(source: Path):
    candidates = [
        source / "data.yaml",
        source / "classes.txt",
        source / "obj.names",
    ]

    for path in candidates:
        if not path.exists():
            continue
        if path.name == "data.yaml":
            names = read_names_from_yaml(path)
        else:
            names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names

    raise FileNotFoundError("Could not find class names in data.yaml, classes.txt, or obj.names")


def read_names_from_yaml(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    names = []
    in_names_block = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("names:"):
            value = line.split(":", 1)[1].strip()
            if value:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict):
                    return [parsed[idx] for idx in sorted(parsed)]
                return list(parsed)
            in_names_block = True
            continue

        if in_names_block:
            if not raw_line.startswith((" ", "\t")):
                break
            key, value = line.split(":", 1)
            names.append((int(key), value.strip().strip("'\"")))

    if names:
        return [name for _, name in sorted(names)]
    return []


def build_class_map(source_names, overrides):
    name_map = DEFAULT_CLASS_MAP.copy()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --class-map item {item!r}, expected source=target")
        source, target = item.split("=", 1)
        name_map[source.strip()] = target.strip()

    id_map = {}
    for source_id, source_name in enumerate(source_names):
        target_name = name_map.get(source_name, source_name)
        if target_name not in TARGET_CLASSES:
            print(f"Skip class {source_id}={source_name!r}; no target mapping for {target_name!r}")
            continue
        id_map[str(source_id)] = TARGET_CLASSES[target_name]
    return id_map


def yolo_box_to_obb(parts):
    x, y, w, h = [float(value) for value in parts]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y - h / 2
    x3 = x + w / 2
    y3 = y + h / 2
    x4 = x - w / 2
    y4 = y + h / 2
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def convert_label_file(source: Path, target: Path, id_map):
    lines = []
    if source.exists():
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{source} is not YOLO detection format: {line}")
            source_cls = parts[0]
            if source_cls not in id_map:
                continue
            coords = yolo_box_to_obb(parts[1:])
            lines.append(" ".join([str(id_map[source_cls]), *[f"{value:.8f}" for value in coords]]))

    target.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text:
        text += "\n"
    target.write_text(text, encoding="utf-8")


def link_or_copy_image(source: Path, target: Path, copy_images: bool):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if copy_images:
        shutil.copy2(source, target)
        return
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def convert_split(source: Path, output: Path, source_split: str, target_split: str, id_map, copy_images: bool):
    image_dir = source / source_split / "images"
    label_dir = source / source_split / "labels"
    if not image_dir.exists():
        return 0

    count = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        link_or_copy_image(image_path, output / "images" / target_split / image_path.name, copy_images)
        convert_label_file(label_path, output / "labels_0123" / target_split / f"{image_path.stem}.txt", id_map)
        count += 1
    return count


def write_class_files(output: Path):
    (output / "labels_0123").mkdir(parents=True, exist_ok=True)
    (output / "labels_0123" / "classify.txt").write_text("none\nlittle\nmid\nmuch\n", encoding="utf-8")


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)

    source_names = read_class_names(source)
    id_map = build_class_map(source_names, args.class_map)
    if not id_map:
        raise ValueError("No classes are mapped into the target label set")

    total = 0
    for source_split, target_split in SPLIT_ALIASES.items():
        total += convert_split(source, output, source_split, target_split, id_map, args.copy_images)

    write_class_files(output)
    print(f"Converted {total} images into {output}")
    print(f"Class map: {id_map}")


if __name__ == "__main__":
    main()

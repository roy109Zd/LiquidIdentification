from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import random
import shutil


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "bottleDataset"
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_NAMES_01 = ("none", "exist")
LABEL_NAMES_BOTTLE = ("bottle",)


def parse_args():
    parser = ArgumentParser(description="Prepare bottle dataset labels and stratified train/val/test splits.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset root.")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for stratified shuffling.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned split without moving files.")
    return parser.parse_args()


def normalize_label_dirs(dataset: Path, dry_run: bool) -> Path:
    labels = dataset / "labels"
    labels_0123 = dataset / "labels_0123"

    if labels.exists() and not labels_0123.exists():
        print(f"Rename {labels.relative_to(dataset)} -> {labels_0123.relative_to(dataset)}")
        if not dry_run:
            labels.rename(labels_0123)
            return labels_0123
        return labels
    elif not labels_0123.exists():
        raise FileNotFoundError(f"Neither {labels} nor {labels_0123} exists.")

    return labels_0123


def remove_caches(label_root: Path):
    for cache_file in label_root.glob("*.cache"):
        cache_file.unlink()
    for cache_file in label_root.glob("*/*.cache"):
        cache_file.unlink()


def label_class(label_path: Path) -> str:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label: {label_path}")

    classes = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            classes.append(line.split()[0])

    if not classes:
        return "0"

    unique = sorted(set(classes), key=int)
    if len(unique) > 1:
        print(f"Warning: {label_path} has multiple classes {unique}; using {unique[0]} for split.")
    return unique[0]


def collect_records(dataset: Path, labels_0123: Path):
    records = []
    seen = set()

    for split in SPLITS:
        image_dir = dataset / "images" / split
        if not image_dir.exists():
            continue

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if image_path.stem in seen:
                raise ValueError(f"Duplicate image stem found across splits: {image_path.stem}")
            seen.add(image_path.stem)

            label_path = labels_0123 / split / f"{image_path.stem}.txt"
            records.append(
                {
                    "stem": image_path.stem,
                    "source_split": split,
                    "class": label_class(label_path),
                }
            )

    if not records:
        raise ValueError(f"No images found under {dataset / 'images'}")
    return records


def split_counts(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return dict(zip(SPLITS, counts))


def assign_splits(records, ratios, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["class"]].append(record)

    rng = random.Random(seed)
    assignments = {}
    summary = {}

    for cls, items in sorted(grouped.items(), key=lambda item: int(item[0])):
        items = items[:]
        rng.shuffle(items)
        counts = split_counts(len(items), ratios)
        summary[cls] = counts

        start = 0
        for split in SPLITS:
            end = start + counts[split]
            for record in items[start:end]:
                assignments[record["stem"]] = split
            start = end

    return assignments, summary


def move_dataset_files(dataset: Path, labels_0123: Path, records, assignments):
    for split in SPLITS:
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (labels_0123 / split).mkdir(parents=True, exist_ok=True)

    for record in records:
        stem = record["stem"]
        source_split = record["source_split"]
        target_split = assignments[stem]

        source_image_dir = dataset / "images" / source_split
        target_image_dir = dataset / "images" / target_split
        for sidecar in sorted(source_image_dir.glob(f"{stem}.*")):
            target = target_image_dir / sidecar.name
            if sidecar.resolve() == target.resolve():
                continue
            if target.exists():
                raise FileExistsError(f"Target already exists: {target}")
            shutil.move(str(sidecar), str(target))

        source_label = labels_0123 / source_split / f"{stem}.txt"
        target_label = labels_0123 / target_split / source_label.name
        if source_label.resolve() != target_label.resolve():
            if target_label.exists():
                raise FileExistsError(f"Target already exists: {target_label}")
            shutil.move(str(source_label), str(target_label))


def convert_label_file(source: Path, target: Path):
    converted = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            converted.append("")
            continue

        parts = line.split()
        cls = parts[0]
        if cls == "0":
            parts[0] = "0"
        elif cls in {"1", "2", "3"}:
            parts[0] = "1"
        else:
            raise ValueError(f"Unsupported class {cls!r} in {source}")
        converted.append(" ".join(parts))

    target.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(converted)
    if text:
        text += "\n"
    target.write_text(text, encoding="utf-8")


def rebuild_labels_01(dataset: Path, labels_0123: Path):
    labels_01 = dataset / "labels_01"
    if labels_01.exists():
        shutil.rmtree(labels_01)

    for split in SPLITS:
        for source in sorted((labels_0123 / split).glob("*.txt")):
            convert_label_file(source, labels_01 / split / source.name)

    (labels_01 / "classify.txt").write_text("\n".join(LABEL_NAMES_01) + "\n", encoding="utf-8")
    return labels_01


def convert_bottle_label_file(source: Path, target: Path):
    converted = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            converted.append("")
            continue

        parts = line.split()
        parts[0] = "0"
        converted.append(" ".join(parts))

    target.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(converted)
    if text:
        text += "\n"
    target.write_text(text, encoding="utf-8")


def rebuild_label_bottle(dataset: Path, labels_0123: Path):
    label_bottle = dataset / "label_bottle"
    if label_bottle.exists():
        shutil.rmtree(label_bottle)

    for split in SPLITS:
        for source in sorted((labels_0123 / split).glob("*.txt")):
            convert_bottle_label_file(source, label_bottle / split / source.name)

    (label_bottle / "classify.txt").write_text("\n".join(LABEL_NAMES_BOTTLE) + "\n", encoding="utf-8")
    return label_bottle


def print_summary(summary):
    print("\nPlanned stratified split:")
    print("class train val test total")
    for cls, counts in summary.items():
        total = sum(counts.values())
        print(f"{cls:>5} {counts['train']:>5} {counts['val']:>3} {counts['test']:>4} {total:>5}")


def main():
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    ratios = (args.train, args.val, args.test)
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive number.")
    ratios = tuple(ratio / ratio_sum for ratio in ratios)

    labels_0123 = normalize_label_dirs(dataset, args.dry_run)
    records = collect_records(dataset, labels_0123)
    assignments, summary = assign_splits(records, ratios, args.seed)
    print_summary(summary)

    if args.dry_run:
        return

    move_dataset_files(dataset, labels_0123, records, assignments)
    remove_caches(labels_0123)
    labels_01 = rebuild_labels_01(dataset, labels_0123)
    remove_caches(labels_01)
    label_bottle = rebuild_label_bottle(dataset, labels_0123)
    remove_caches(label_bottle)
    print(
        f"\nPrepared {labels_0123.relative_to(dataset)}, "
        f"{labels_01.relative_to(dataset)}, and {label_bottle.relative_to(dataset)}."
    )


if __name__ == "__main__":
    main()

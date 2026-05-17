from argparse import ArgumentParser
from pathlib import Path
import csv

import cv2
import numpy as np
from tqdm.auto import tqdm

from segment_features import FEATURE_COLUMNS, extract_segment_features
from train_obb import DEFAULT_DATASET, LABEL_SET_ALIASES, LABEL_SETS


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "runs" / "tree_segments"
PROJECT_MODEL_ROOT = ROOT / "runs" / "obb"
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = ArgumentParser(
        description=(
            "Build masked bottle-region images from label polygons or this project's trained YOLO model, "
            "then write feature CSV rows for future decision-tree training."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Project-trained YOLO weights path. Omit to build masks from label polygons.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset root containing images and labels.")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional image directory. Defaults to dataset/images/<split> for train/val/test.",
    )
    parser.add_argument(
        "--label-set",
        choices=sorted(LABEL_SET_ALIASES),
        default="0123",
        help="Label set used to read the recognition target, for example labels_0123 or labels_01.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output root for masked images and features.csv.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO segmentation inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO segmentation confidence threshold.")
    parser.add_argument("--device", default=None, help="Use 0 for GPU, cpu for CPU, or omit for auto.")
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Optional segmentation class id to keep. If omitted, all classes are candidates.",
    )
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
    parser.add_argument("--crop", action="store_true", help="Crop output images to the selected mask bounding box.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def iter_images(image_dir: Path):
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() in IMAGE_EXTS:
            yield image_path


def read_label(label_path: Path, names: list[str]):
    if not label_path.exists():
        return "", ""

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        label_id = line.split()[0]
        label_name = names[int(label_id)] if label_id.isdigit() and int(label_id) < len(names) else label_id
        return label_id, label_name

    return "", ""


def resolve_project_trained_model(model: str) -> str:
    model_path = Path(model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model_path = model_path.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Project-trained model weights not found: {model_path}")

    try:
        model_path.relative_to(PROJECT_MODEL_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "Decision-tree preprocessing only accepts weights trained by this project under runs/obb. "
            f"Got: {model_path}"
        ) from exc

    return str(model_path)


def read_label_polygons(label_path: Path, image_shape: tuple[int, int]):
    if not label_path.exists():
        return []

    height, width = image_shape
    polygons = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 9:
            continue

        coords = [float(value) for value in parts[1:9]]
        points = []
        for idx in range(0, 8, 2):
            x = int(round(coords[idx] * width))
            y = int(round(coords[idx + 1] * height))
            points.append([x, y])
        polygons.append({"class_id": int(parts[0]), "points": np.asarray(points, dtype=np.int32)})

    return polygons


def choose_label_polygon(polygons, class_id: int | None, select: str):
    candidates = []
    for polygon in polygons:
        if class_id is not None and polygon["class_id"] != class_id:
            continue
        area = abs(float(cv2.contourArea(polygon["points"])))
        if area <= 0:
            continue
        candidates.append({**polygon, "area": area})

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["area"])


def polygon_mask(image_shape: tuple[int, int], points: np.ndarray):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if mask.shape == (target_h, target_w):
        return mask
    return cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)


def choose_mask(result, class_id: int | None, select: str):
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(masks))
    classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(masks), dtype=int)

    candidates = []
    for idx, raw_mask in enumerate(masks):
        if class_id is not None and classes[idx] != class_id:
            continue

        mask = resize_mask(raw_mask > 0.5, result.orig_img.shape[:2])
        area = int(mask.sum())
        if area == 0:
            continue
        candidates.append(
            {
                "index": idx,
                "mask": mask,
                "area": area,
                "conf": float(confs[idx]),
                "class_id": int(classes[idx]),
            }
        )

    if not candidates:
        return None

    if select == "largest-mask":
        return max(candidates, key=lambda item: (item["area"], item["conf"]))
    return max(candidates, key=lambda item: (item["conf"], item["area"]))


def choose_obb_polygon(result, class_id: int | None, select: str):
    if getattr(result, "obb", None) is None or result.obb is None or len(result.obb) == 0:
        return None

    obb = result.obb
    points = obb.xyxyxyxy.cpu().numpy()
    confs = obb.conf.cpu().numpy() if obb.conf is not None else np.ones(len(points))
    classes = obb.cls.cpu().numpy().astype(int) if obb.cls is not None else np.zeros(len(points), dtype=int)

    candidates = []
    for idx, raw_points in enumerate(points):
        if class_id is not None and classes[idx] != class_id:
            continue
        polygon = raw_points.reshape(4, 2).astype(np.int32)
        area = abs(float(cv2.contourArea(polygon)))
        if area <= 0:
            continue
        candidates.append(
            {
                "points": polygon,
                "area": area,
                "conf": float(confs[idx]),
                "class_id": int(classes[idx]),
            }
        )

    if not candidates:
        return None

    if select == "largest-mask":
        return max(candidates, key=lambda item: (item["area"], item["conf"]))
    return max(candidates, key=lambda item: (item["conf"], item["area"]))


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray, background: str):
    if background == "transparent":
        image_bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        image_bgra[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
        return image_bgra

    fill_value = 255 if background == "white" else 0
    output = np.full_like(image_bgr, fill_value)
    output[mask] = image_bgr[mask]
    return output


def crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


CSV_COLUMNS = [
    "split",
    "stem",
    "masked_image",
    "label_id",
    "label_name",
    "seg_class_id",
    "seg_conf",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    *FEATURE_COLUMNS,
]


def output_path(output_root: Path, split: str, image_path: Path, background: str):
    suffix = ".png" if background == "transparent" else image_path.suffix.lower()
    return output_root / "images" / split / f"{image_path.stem}{suffix}"


def process_image(model, image_path: Path, args):
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )
    result = results[0]
    selected = choose_mask(result, args.class_id, args.select)
    if selected is None:
        selected_obb = choose_obb_polygon(result, args.class_id, args.select)
        if selected_obb is not None:
            mask = polygon_mask(result.orig_img.shape[:2], selected_obb["points"])
            selected = {
                "mask": mask,
                "area": int(mask.sum()),
                "conf": selected_obb["conf"],
                "class_id": selected_obb["class_id"],
            }
    if selected is None:
        return None

    image_bgr = result.orig_img
    mask = selected["mask"]
    bbox = mask_bbox(mask)
    masked = apply_mask(image_bgr, mask, args.background)
    if args.crop:
        masked = crop_to_bbox(masked, bbox)

    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    row = {
        "seg_class_id": selected["class_id"],
        "seg_conf": f"{selected['conf']:.6f}",
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
    }
    row.update({name: f"{value:.8f}" for name, value in extract_segment_features(image_bgr, mask, bbox).items()})
    return masked, row


def process_image_from_label(image_path: Path, label_path: Path, args):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    polygons = read_label_polygons(label_path, image_bgr.shape[:2])
    selected = choose_label_polygon(polygons, args.class_id, args.select)
    if selected is None:
        return None

    mask = polygon_mask(image_bgr.shape[:2], selected["points"])
    bbox = mask_bbox(mask)
    masked = apply_mask(image_bgr, mask, args.background)
    if args.crop:
        masked = crop_to_bbox(masked, bbox)

    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    row = {
        "seg_class_id": selected["class_id"],
        "seg_conf": "1.000000",
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
    }
    row.update({name: f"{value:.8f}" for name, value in extract_segment_features(image_bgr, mask, bbox).items()})
    return masked, row


def write_image(path: Path, image: np.ndarray, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output image exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def process_split(model, label_root: Path, names: list[str], split: str, image_dir: Path, args):
    output_root = Path(args.output)
    rows = []
    skipped = 0
    image_paths = list(iter_images(image_dir))
    progress = tqdm(
        image_paths,
        desc=f"{split} features",
        unit="img",
        dynamic_ncols=True,
        disable=getattr(args, "no_progress", False),
    )

    for image_path in progress:
        label_path = label_root / split / f"{image_path.stem}.txt"
        if model is None:
            processed = process_image_from_label(image_path, label_path, args)
        else:
            processed = process_image(model, image_path, args)
        if processed is None:
            skipped += 1
            continue

        masked, row = processed
        target = output_path(output_root, split, image_path, args.background)
        write_image(target, masked, args.overwrite)

        label_id, label_name = read_label(label_path, names)
        row.update(
            {
                "split": split,
                "stem": image_path.stem,
                "masked_image": str(target.resolve()).replace("\\", "/"),
                "label_id": label_id,
                "label_name": label_name,
            }
        )
        rows.append(row)
        progress.set_postfix(saved=len(rows), skipped=skipped)

    print(f"{split}: saved {len(rows)} masked images, skipped {skipped} images without usable masks")
    return rows


def build_segment_dataset(args) -> Path:
    dataset = Path(args.dataset).resolve()
    output_root = Path(args.output).resolve()
    label_key = LABEL_SET_ALIASES[args.label_set]
    label_config = LABEL_SETS[label_key]
    label_root = dataset / label_config["label_dir"]
    if not label_root.exists():
        raise FileNotFoundError(f"Label directory not found: {label_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    feature_csv = output_root / "features.csv"
    if feature_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Feature CSV exists: {feature_csv}. Use --overwrite to replace it.")

    if args.model:
        from ultralytics import YOLO

        model = YOLO(resolve_project_trained_model(args.model))
    else:
        model = None

    all_rows = []
    if args.source:
        source = Path(args.source).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source image directory not found: {source}")
        all_rows.extend(process_split(model, label_root, label_config["names"], "predict", source, args))
    else:
        for split in SPLITS:
            image_dir = dataset / "images" / split
            if image_dir.exists():
                all_rows.extend(process_split(model, label_root, label_config["names"], split, image_dir, args))

    with feature_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Feature CSV: {feature_csv}")
    return feature_csv


def main():
    build_segment_dataset(parse_args())


if __name__ == "__main__":
    main()

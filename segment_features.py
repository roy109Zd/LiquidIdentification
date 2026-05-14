from __future__ import annotations

import cv2
import numpy as np


BAND_COUNT = 5

GEOMETRY_FEATURE_COLUMNS = [
    "mask_area_ratio",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_aspect_ratio",
    "bbox_area_ratio",
    "mask_fill_ratio",
    "mask_center_x_norm",
    "mask_center_y_norm",
]

COLOR_FEATURE_COLUMNS = [
    "rgb_mean_0",
    "rgb_mean_1",
    "rgb_mean_2",
    "rgb_std_0",
    "rgb_std_1",
    "rgb_std_2",
    "hsv_mean_0",
    "hsv_mean_1",
    "hsv_mean_2",
    "hsv_std_0",
    "hsv_std_1",
    "hsv_std_2",
    "dark_ratio",
    "bright_ratio",
    "high_saturation_ratio",
    "bottom_top_v_diff",
    "bottom_top_s_diff",
    "vertical_brightness_center",
    "vertical_saturation_center",
]

EDGE_FEATURE_COLUMNS = [
    "edge_density",
    "horizontal_edge_strength",
]

BAND_FEATURE_COLUMNS = [
    column
    for band in range(BAND_COUNT)
    for column in (
        f"band_{band}_mask_fraction",
        f"band_{band}_h_mean",
        f"band_{band}_s_mean",
        f"band_{band}_v_mean",
        f"band_{band}_dark_ratio",
        f"band_{band}_bright_ratio",
    )
]

FEATURE_COLUMNS = [
    *GEOMETRY_FEATURE_COLUMNS,
    *COLOR_FEATURE_COLUMNS,
    *EDGE_FEATURE_COLUMNS,
    *BAND_FEATURE_COLUMNS,
]


def empty_features() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_COLUMNS}


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def bbox_geometry(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, float]:
    h, w = mask.shape
    x1, y1, x2, y2 = bbox
    bbox_width = max(0, x2 - x1)
    bbox_height = max(0, y2 - y1)
    mask_area = int(mask.sum())
    bbox_area = bbox_width * bbox_height

    if mask_area:
        ys, xs = np.where(mask)
        center_x = float(xs.mean())
        center_y = float(ys.mean())
    else:
        center_x = 0.0
        center_y = 0.0

    return {
        "mask_area_ratio": safe_ratio(mask_area, h * w),
        "bbox_width_norm": safe_ratio(bbox_width, w),
        "bbox_height_norm": safe_ratio(bbox_height, h),
        "bbox_aspect_ratio": safe_ratio(bbox_width, bbox_height),
        "bbox_area_ratio": safe_ratio(bbox_area, h * w),
        "mask_fill_ratio": safe_ratio(mask_area, bbox_area),
        "mask_center_x_norm": safe_ratio(center_x, w),
        "mask_center_y_norm": safe_ratio(center_y, h),
    }


def basic_color_features(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pixels_bgr = image_bgr[mask]
    if len(pixels_bgr) == 0:
        return {name: 0.0 for name in COLOR_FEATURE_COLUMNS}

    pixels_rgb = pixels_bgr[:, ::-1]
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels_hsv = hsv_image[mask]

    values = {}
    for prefix, pixels in (("rgb", pixels_rgb), ("hsv", pixels_hsv)):
        means = pixels.mean(axis=0)
        stds = pixels.std(axis=0)
        for channel, value in zip(("0", "1", "2"), means):
            values[f"{prefix}_mean_{channel}"] = float(value)
        for channel, value in zip(("0", "1", "2"), stds):
            values[f"{prefix}_std_{channel}"] = float(value)

    v = pixels_hsv[:, 2]
    s = pixels_hsv[:, 1]
    values["dark_ratio"] = float((v < 80).mean())
    values["bright_ratio"] = float((v > 180).mean())
    values["high_saturation_ratio"] = float((s > 80).mean())
    return values


def vertical_center(values: np.ndarray, y_positions: np.ndarray, top: int, height: int) -> float:
    weights = values.astype(np.float64)
    total = float(weights.sum())
    if total == 0 or height <= 0:
        return 0.0
    normalized_y = (y_positions.astype(np.float64) - top) / max(1, height)
    return float((normalized_y * weights).sum() / total)


def band_features(image_bgr: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, float]:
    values = {name: 0.0 for name in BAND_FEATURE_COLUMNS}
    x1, y1, x2, y2 = bbox
    bbox_width = max(0, x2 - x1)
    bbox_height = max(0, y2 - y1)
    if bbox_width == 0 or bbox_height == 0:
        return values

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    edges = np.linspace(y1, y2, BAND_COUNT + 1).astype(int)
    for band in range(BAND_COUNT):
        band_y1 = int(edges[band])
        band_y2 = int(edges[band + 1])
        if band_y2 <= band_y1:
            continue

        band_mask = mask[band_y1:band_y2, x1:x2]
        band_pixels = int(band_mask.sum())
        values[f"band_{band}_mask_fraction"] = safe_ratio(band_pixels, band_mask.size)
        if band_pixels == 0:
            continue

        pixels_hsv = hsv[band_y1:band_y2, x1:x2][band_mask]
        values[f"band_{band}_h_mean"] = float(pixels_hsv[:, 0].mean())
        values[f"band_{band}_s_mean"] = float(pixels_hsv[:, 1].mean())
        values[f"band_{band}_v_mean"] = float(pixels_hsv[:, 2].mean())
        values[f"band_{band}_dark_ratio"] = float((pixels_hsv[:, 2] < 80).mean())
        values[f"band_{band}_bright_ratio"] = float((pixels_hsv[:, 2] > 180).mean())

    return values


def vertical_contrast_features(image_bgr: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, float]:
    x1, y1, x2, y2 = bbox
    bbox_height = max(0, y2 - y1)
    if bbox_height == 0 or not mask.any():
        return {
            "bottom_top_v_diff": 0.0,
            "bottom_top_s_diff": 0.0,
            "vertical_brightness_center": 0.0,
            "vertical_saturation_center": 0.0,
        }

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask)
    pixels_hsv = hsv[ys, xs]
    band_height = max(1, bbox_height // 3)
    top_mask = mask[y1 : min(y2, y1 + band_height), x1:x2]
    bottom_mask = mask[max(y1, y2 - band_height) : y2, x1:x2]
    top_hsv = hsv[y1 : min(y2, y1 + band_height), x1:x2][top_mask]
    bottom_hsv = hsv[max(y1, y2 - band_height) : y2, x1:x2][bottom_mask]

    top_v = float(top_hsv[:, 2].mean()) if len(top_hsv) else 0.0
    bottom_v = float(bottom_hsv[:, 2].mean()) if len(bottom_hsv) else 0.0
    top_s = float(top_hsv[:, 1].mean()) if len(top_hsv) else 0.0
    bottom_s = float(bottom_hsv[:, 1].mean()) if len(bottom_hsv) else 0.0

    return {
        "bottom_top_v_diff": bottom_v - top_v,
        "bottom_top_s_diff": bottom_s - top_s,
        "vertical_brightness_center": vertical_center(pixels_hsv[:, 2], ys, y1, bbox_height),
        "vertical_saturation_center": vertical_center(pixels_hsv[:, 1], ys, y1, bbox_height),
    }


def edge_features(image_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if not mask.any():
        return {name: 0.0 for name in EDGE_FEATURE_COLUMNS}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mask_area = int(mask.sum())
    return {
        "edge_density": safe_ratio(int((edges[mask] > 0).sum()), mask_area),
        "horizontal_edge_strength": float(np.abs(sobel_y[mask]).mean() / 255.0),
    }


def extract_segment_features(image_bgr: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, float]:
    if image_bgr.size == 0 or mask.size == 0 or not mask.any():
        return empty_features()

    features = {}
    features.update(bbox_geometry(mask, bbox))
    features.update(basic_color_features(image_bgr, mask))
    features.update(vertical_contrast_features(image_bgr, mask, bbox))
    features.update(edge_features(image_bgr, mask))
    features.update(band_features(image_bgr, mask, bbox))
    return {name: float(features.get(name, 0.0)) for name in FEATURE_COLUMNS}

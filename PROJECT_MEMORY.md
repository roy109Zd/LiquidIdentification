# Project Memory

This file records project-specific preferences and operating notes for future Codex work in this repository

## Documentation Format

- Write project documentation in two complete versions: Chinese first, English second
- Do not mix Chinese and English explanations inside the same paragraph unless the term is a command, class name, path, or model name
- Keep commands copy-paste friendly, prefer single-line commands for shell use
- If multi-line zsh commands are shown, make sure `\` is the final character on the line with no trailing spaces
- Do not use sentence-final periods in documentation prose, including Chinese full-width periods and English periods
- Use UTF-8 for Markdown files

## Project Environment

- Work from WSL when running training or checking Ultralytics behavior
- Recommended environment:

```bash
cd /mnt/e/LiquidIdentification
source /root/envs/torchforge/bin/activate
```

- The `torchforge` environment has `ultralytics` installed
- The current machine has an RTX 4060 Laptop GPU with 8GB VRAM

## Dataset Notes

- The dataset root is `/mnt/e/LiquidIdentification/bottleDataset`
- The project supports three label sets:
  - `labels_0123`: `0=none`, `1=little`, `2=mid`, `3=much`
  - `labels_01`: `0=none`, `1=exist`
  - `label_bottle`: `0=bottle`
- `bottleDataset/labels` is an active link used by Ultralytics, `train_obb.py` should point it to the selected label set before training
- Do not pass `--data bottle_obb.yaml` when using `--label-set`, because that bypasses automatic label-set switching
- Use `convert_roboflow_yolo_to_obb.py` for Roboflow YOLO detection exports that need conversion to this project's OBB label format
- The default Roboflow mapping is `empty=none`, `half_water_level=mid`, `full_water_level=much`, `three_quarters_level=much`

## Preferred Training Commands

Four-class training:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_0123 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_0123_yolo11m_640_b4
```

Binary training:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_01 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_01_yolo11m_640_b4
```

Bottle-only training:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_yolo11m_640_b4
```

Geometry augmentation with translation and rotation:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_aug_yolo11m_640_b4 --augment-geom --degrees 10 --translate 0.1
```

If CUDA memory is insufficient, reduce `--batch 4` to `--batch 2`

## Maintenance Notes

- `train_obb.py --model` prefers local files in the current path or repo root; bare Ultralytics model names are passed through so Ultralytics can download them when no local file exists
- Keep README updates synchronized with script behavior
- Keep generated YOLO runs under `runs/obb/`; delete failed or accidental runs only after checking the exact directory
- Keep `.dataset_views/`, `runs/`, model weights, and cache files out of git
- For duplicate prediction boxes, prefer YOLO native `iou` and `agnostic_nms=True` prediction commands so visualizations keep the native Ultralytics style
- Do not keep custom redraw scripts for native YOLO visualizations unless there is a clear reason

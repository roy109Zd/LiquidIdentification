# LiquidIdentification

Bottle liquid-level identification with YOLO OBB

环境配置：见 [ENVIRONMENT.md](ENVIRONMENT.md)

Environment setup: see [ENVIRONMENT.md](ENVIRONMENT.md)

## 中文说明

本项目使用 Ultralytics YOLO OBB 模型识别瓶中液体状态，支持三套标签：

- `labels_0123`：四分类，`0=none`，`1=little`，`2=mid`，`3=much`
- `labels_01`：二分类，`0=none`，`1=exist`
- `label_bottle`：单类瓶子检测，`0=bottle`

### 数据集

数据集根目录：

```bash
/mnt/e/LiquidIdentification/bottleDataset
```

当前结构：

```text
bottleDataset/
  images/
    train/
    val/
    test/
  labels_0123/
    train/
    val/
    test/
  labels_01/
    train/
    val/
    test/
  label_bottle/
    train/
    val/
    test/
  labels -> labels_0123、labels_01 或 label_bottle
```

`bottleDataset/labels` 是由 `train_obb.py` 自动创建的当前标签入口，Ultralytics 默认会从 `images` 推导并查找同级的 `labels`，所以训练脚本会在启动前把 `labels` 指向所选标签集

每个标签文件使用 YOLO OBB 格式：

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

第一列是类别编号，后面 8 个数是旋转框 4 个角点的归一化坐标

### 数据处理

`prepare_dataset.py` 用于：

- 将原始 `labels` 重命名为 `labels_0123`
- 根据 `labels_0123` 创建 `labels_01`
- 将类别按 `0 -> 0`、`1/2/3 -> 1` 转换
- 根据 `labels_0123` 创建 `label_bottle`，将所有类别统一为 `0=bottle`
- 按类别分层划分 `train`、`val`、`test`

默认比例是 `8:1:1`：

```bash
python prepare_dataset.py
```

只预览划分计划，不移动文件：

```bash
python prepare_dataset.py --dry-run
```

自定义比例：

```bash
python prepare_dataset.py --train 0.7 --val 0.2 --test 0.1
```

从 Roboflow 下载普通 YOLO 检测格式后，可以用 `convert_roboflow_yolo_to_obb.py` 转成当前项目的 OBB 标签格式：

```bash
python convert_roboflow_yolo_to_obb.py --source path/to/roboflow_dataset --output importedDataset --overwrite
```

默认类名映射适配 Roboflow 的 `Bottle fill level` 数据集：

```text
empty -> none
half_water_level -> mid
full_water_level -> much
three_quarters_level -> much
```

如果下载的数据集类名不同，可以手动指定映射：

```bash
python convert_roboflow_yolo_to_obb.py --source path/to/roboflow_dataset --output importedDataset --class-map bottle=none level=mid --overwrite
```

当前划分结果：

```text
train: 175 images
val:    22 images
test:   21 images
```

四分类按图片统计：

```text
split  class 0  class 1  class 2  class 3
train       53       51       38       33
val          7        6        5        4
test         6        6        5        4
```

二分类按图片统计：

```text
split  none  exist
train    53    122
val       7     15
test      6     15
```

### 训练

训练四分类模型：

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_0123 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_0123_yolo11m_640_b4
```

训练二分类 `none/exist` 模型：

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_01 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_01_yolo11m_640_b4
```

训练单类 `bottle` 模型：

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_yolo11m_640_b4
```

启用位移和旋转数据增强：

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_aug_yolo11m_640_b4 --augment-geom --degrees 10 --translate 0.1
```

`--label-set` 支持长写和短写：

```text
labels_0123 或 0123
labels_01   或 01
label_bottle 或 bottle
```

常用参数：

```text
--model      预训练 OBB 权重，例如 yolo11m-obb.pt
--label-set  选择 labels_0123、labels_01 或 label_bottle
--epochs     训练轮数
--imgsz      训练图片尺寸
--batch      batch size
--device     0 表示 GPU，cpu 表示 CPU
--workers    dataloader worker 数量
--name       runs/obb/ 下的输出目录名
--resume     继续中断的训练
--exist-ok   允许写入已有输出目录
--augment-geom 启用位移和旋转数据增强
--degrees    最大旋转角度，默认随 --augment-geom 使用 10
--translate  最大位移比例，默认随 --augment-geom 使用 0.1
```

`--model` 会优先使用本地文件，例如项目目录下存在 `yolo11m-obb.pt` 时会直接加载本地权重；如果传入的是 `yolo11m-obb.pt` 这类 Ultralytics 支持的裸模型名且本地不存在，会交给 Ultralytics 自动下载

只检查会使用哪个 dataset yaml，不启动训练：

```bash
python train_obb.py --label-set labels_01 --prepare-data-only
python train_obb.py --label-set labels_0123 --prepare-data-only
python train_obb.py --label-set label_bottle --prepare-data-only
```

使用 `--label-set` 时不要再传 `--data bottle_obb.yaml`，否则脚本无法自动切换不同标签集

在 zsh 中写多行命令时，`\` 必须是该行最后一个字符，后面不能有空格

### 训练输出

训练结果默认保存在：

```text
runs/obb/<run_name>/
```

常见文件：

```text
weights/best.pt                 验证集指标最好的权重
weights/last.pt                 最后一轮权重，可用于继续训练
args.yaml                       本次训练参数
results.csv                     每轮训练指标
results.png                     指标曲线图
labels.jpg                      标签分布可视化
confusion_matrix.png            混淆矩阵
confusion_matrix_normalized.png 归一化混淆矩阵
train_batch*.jpg                训练 batch 可视化
val_batch*_labels.jpg           验证集真实标签可视化
val_batch*_pred.jpg             验证集预测结果可视化
```

通常使用最佳权重做预测：

```text
runs/obb/<run_name>/weights/best.pt
```

### 指标含义

训练时会看到类似输出：

```text
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  angle_loss  Instances  Size
Class  Images   Instances Box(P     R         mAP50       mAP50-95
```

关键指标：

- `box_loss`：框位置损失，越低越好
- `cls_loss`：分类损失，越低越好
- `dfl_loss`：边框细化损失，越低越好
- `angle_loss`：旋转框角度损失，越低越好
- `Box(P)`：精确率，越高表示误检越少
- `R`：召回率，越高表示漏检越少
- `mAP50`：IoU 0.5 下的平均精度
- `mAP50-95`：IoU 0.5 到 0.95 多个阈值下的平均精度，更严格

### 预测

用预训练权重预测单张图片：

```bash
yolo predict model=yolo11m-obb.pt source=/mnt/e/LiquidIdentification/testDataset/images/image4.jpg
```

预测文件夹：

```bash
yolo predict model=yolo11m-obb.pt source=/mnt/e/LiquidIdentification/testDataset/images
```

用训练好的权重预测：

```bash
yolo predict model=runs/obb/<run_name>/weights/best.pt source=/mnt/e/LiquidIdentification/testDataset/images
```

如果预测结果中出现重复框，优先使用 YOLO 原生 NMS 参数重新预测，不要手动改标签后再自绘可视化图：

```bash
yolo predict model=runs/obb/bottle_01_yolo11m_640_b4/weights/best.pt source=runs/predict_errors/val_bottle_01/images project=/mnt/e/LiquidIdentification/runs/predict_errors/val_bottle_01 name=pred_native_nms_iou50 save=True save_txt=True save_conf=True iou=0.5 agnostic_nms=True exist_ok=True
```

```bash
yolo predict model=runs/obb/bottle_01_yolo11m_640_b4/weights/best.pt source=runs/predict_errors/test_bottle_01/images project=/mnt/e/LiquidIdentification/runs/predict_errors/test_bottle_01 name=pred_native_nms_iou50 save=True save_txt=True save_conf=True iou=0.5 agnostic_nms=True exist_ok=True
```

`iou=0.5` 表示重叠区域与总区域的比例超过 0.5 时按重复框处理，`agnostic_nms=True` 表示不同类别之间也参与去重，最终可视化结果保留 Ultralytics 原生样式

## English

This project uses Ultralytics YOLO OBB to identify the liquid state in bottle images and supports three label sets:

- `labels_0123`: four classes, `0=none`, `1=little`, `2=mid`, `3=much`
- `labels_01`: binary classes, `0=none`, `1=exist`
- `label_bottle`: single-class bottle detection, `0=bottle`

### Dataset

Dataset root:

```bash
/mnt/e/LiquidIdentification/bottleDataset
```

Current layout:

```text
bottleDataset/
  images/
    train/
    val/
    test/
  labels_0123/
    train/
    val/
    test/
  labels_01/
    train/
    val/
    test/
  label_bottle/
    train/
    val/
    test/
  labels -> labels_0123, labels_01, or label_bottle
```

`bottleDataset/labels` is the active label entry created by `train_obb.py`, Ultralytics derives the label path from `images` and looks for a sibling `labels` directory, so the training script points `labels` to the selected label set before training starts

Each label file uses YOLO OBB format:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

The first value is the class id, and the following eight values are normalized coordinates of the four rotated-box corners

### Dataset Preparation

`prepare_dataset.py` is used to:

- rename the original `labels` directory to `labels_0123`
- create `labels_01` from `labels_0123`
- convert classes with `0 -> 0` and `1/2/3 -> 1`
- create `label_bottle` from `labels_0123` and convert every class to `0=bottle`
- split images and labels into `train`, `val`, and `test` with stratified class balance

The default split ratio is `8:1:1`:

```bash
python prepare_dataset.py
```

Preview the planned split without moving files:

```bash
python prepare_dataset.py --dry-run
```

Use a custom ratio:

```bash
python prepare_dataset.py --train 0.7 --val 0.2 --test 0.1
```

After downloading a normal YOLO detection export from Roboflow, use `convert_roboflow_yolo_to_obb.py` to convert it into this project's OBB label format:

```bash
python convert_roboflow_yolo_to_obb.py --source path/to/roboflow_dataset --output importedDataset --overwrite
```

The default class map supports Roboflow's `Bottle fill level` dataset:

```text
empty -> none
half_water_level -> mid
full_water_level -> much
three_quarters_level -> much
```

If the downloaded dataset uses different class names, pass an explicit mapping:

```bash
python convert_roboflow_yolo_to_obb.py --source path/to/roboflow_dataset --output importedDataset --class-map bottle=none level=mid --overwrite
```

Current split:

```text
train: 175 images
val:    22 images
test:   21 images
```

Four-class image distribution:

```text
split  class 0  class 1  class 2  class 3
train       53       51       38       33
val          7        6        5        4
test         6        6        5        4
```

Binary image distribution:

```text
split  none  exist
train    53    122
val       7     15
test      6     15
```

### Training

Train the four-class model:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_0123 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_0123_yolo11m_640_b4
```

Train the binary `none/exist` model:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_01 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_01_yolo11m_640_b4
```

Train the single-class `bottle` model:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_yolo11m_640_b4
```

Enable translation and rotation augmentation:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_aug_yolo11m_640_b4 --augment-geom --degrees 10 --translate 0.1
```

`--label-set` accepts both long and short names:

```text
labels_0123 or 0123
labels_01   or 01
label_bottle or bottle
```

Useful training parameters:

```text
--model      pretrained OBB weights, for example yolo11m-obb.pt
--label-set  choose labels_0123, labels_01, or label_bottle
--epochs     number of training epochs
--imgsz      training image size
--batch      batch size
--device     0 for GPU, cpu for CPU
--workers    number of dataloader workers
--name       output run directory under runs/obb/
--resume     resume an interrupted run
--exist-ok   allow writing into an existing output directory
--augment-geom enable translation and rotation augmentation
--degrees    maximum rotation degrees, defaults to 10 with --augment-geom
--translate  maximum translation fraction, defaults to 0.1 with --augment-geom
```

`--model` prefers local files, so a project-local `yolo11m-obb.pt` is loaded directly; if the value is a bare Ultralytics-supported model name such as `yolo11m-obb.pt` and no local file exists, Ultralytics will download it automatically

Check which dataset yaml will be used without starting training:

```bash
python train_obb.py --label-set labels_01 --prepare-data-only
python train_obb.py --label-set labels_0123 --prepare-data-only
python train_obb.py --label-set label_bottle --prepare-data-only
```

Do not pass `--data bottle_obb.yaml` when using `--label-set`, otherwise the script cannot switch between label sets

When writing multi-line commands in zsh, make sure `\` is the last character on the line, with no trailing spaces

### Training Outputs

Training outputs are saved under:

```text
runs/obb/<run_name>/
```

Common files:

```text
weights/best.pt                 best checkpoint on the validation set
weights/last.pt                 last epoch checkpoint, useful for resume
args.yaml                       training arguments
results.csv                     epoch-by-epoch metrics
results.png                     training metric plots
labels.jpg                      label distribution visualization
confusion_matrix.png            confusion matrix
confusion_matrix_normalized.png normalized confusion matrix
train_batch*.jpg                training batch visualization
val_batch*_labels.jpg           validation labels visualization
val_batch*_pred.jpg             validation predictions
```

Usually, use this checkpoint for prediction:

```text
runs/obb/<run_name>/weights/best.pt
```

### Metrics

During training, YOLO prints lines like:

```text
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  angle_loss  Instances  Size
Class  Images   Instances Box(P     R         mAP50       mAP50-95
```

Key meanings:

- `box_loss`: box position loss, lower is better
- `cls_loss`: classification loss, lower is better
- `dfl_loss`: box refinement loss, lower is better
- `angle_loss`: rotated-box angle loss, lower is better
- `Box(P)`: precision, higher means fewer false positives
- `R`: recall, higher means fewer missed objects
- `mAP50`: mean average precision at IoU 0.5
- `mAP50-95`: stricter mean average precision across IoU 0.5 to 0.95

### Prediction

Predict one image with pretrained weights:

```bash
yolo predict model=yolo11m-obb.pt source=/mnt/e/LiquidIdentification/testDataset/images/image4.jpg
```

Predict a folder:

```bash
yolo predict model=yolo11m-obb.pt source=/mnt/e/LiquidIdentification/testDataset/images
```

Predict with a trained checkpoint:

```bash
yolo predict model=runs/obb/<run_name>/weights/best.pt source=/mnt/e/LiquidIdentification/testDataset/images
```

If duplicate boxes appear in prediction results, prefer YOLO native NMS parameters and rerun prediction instead of editing labels and redrawing visualization images manually:

```bash
yolo predict model=runs/obb/bottle_01_yolo11m_640_b4/weights/best.pt source=runs/predict_errors/val_bottle_01/images project=/mnt/e/LiquidIdentification/runs/predict_errors/val_bottle_01 name=pred_native_nms_iou50 save=True save_txt=True save_conf=True iou=0.5 agnostic_nms=True exist_ok=True
```

```bash
yolo predict model=runs/obb/bottle_01_yolo11m_640_b4/weights/best.pt source=runs/predict_errors/test_bottle_01/images project=/mnt/e/LiquidIdentification/runs/predict_errors/test_bottle_01 name=pred_native_nms_iou50 save=True save_txt=True save_conf=True iou=0.5 agnostic_nms=True exist_ok=True
```

`iou=0.5` treats boxes as duplicates when overlap over union is above 0.5, `agnostic_nms=True` allows suppression across different classes, and the final visualization keeps the native Ultralytics style

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

### 分割后传统机器学习数据

`prepare_tree_segments.py` 用于给后续传统机器学习分类器准备数据。默认不使用任何已有模型，而是直接读取 YOLO OBB 标签的四点框生成 mask，只保留 mask 内的图像区域，再保存分割后的图片和一份 `features.csv`

```bash
python prepare_tree_segments.py --label-set labels_0123 --output runs/tree_segments --overwrite
```

`train_tree_classifier.py` 会把分割预处理和分类器训练串起来，中间会自动生成并读取同一份 `features.csv`。默认算法仍是 `decision-tree`，也可以通过 `--algorithm` 换成常见机器学习算法。

```bash
python train_tree_classifier.py --label-set labels_0123 --segments-output runs/tree_segments_0123 --tree-output runs/tree_classifier_0123 --overwrite-segments --max-depth 5 --min-samples-leaf 3
```

如果已经有 `features.csv`，可以跳过分割直接训练：

```bash
python train_tree_classifier.py --features runs/tree_segments/features.csv --tree-output runs/tree_classifier --max-depth 5 --min-samples-leaf 3
```

训练随机森林或 SVM：

```bash
python train_tree_classifier.py --features runs/tree_segments/features.csv --tree-output runs/tree_classifier_rf --algorithm random-forest --n-estimators 300 --max-depth 8 --min-samples-leaf 2
python train_tree_classifier.py --features runs/tree_segments/features.csv --tree-output runs/tree_classifier_svm --algorithm rbf-svm --svm-c 3.0 --svm-gamma scale
```

常用参数：

```text
--model       可选项目训练权重，必须位于 runs/obb/ 下；不传时使用标签四点框生成 mask
--label-set   读取识别标签的标签集，例如 labels_0123、labels_01 或 label_bottle
--algorithm   选择分类器：decision-tree、random-forest、extra-trees、gradient-boosting、ada-boost、logistic-regression、linear-svm、rbf-svm、knn、gaussian-nb、mlp
--criterion   决策树、随机森林、ExtraTrees 的分裂准则：gini、entropy、log_loss
--max-depth   树模型最大深度
--n-estimators 集成模型的估计器数量
--neighbors   KNN 的邻居数
--svm-c       SVM/逻辑回归的 C
--svm-gamma   RBF SVM 的 gamma
--max-iter    线性模型和 MLP 的最大迭代次数
--conf        使用模型自动生成 mask 时的置信度筛选阈值，默认 0.25
--background  mask 外背景，可选 black、white、transparent
--crop        将输出图片裁剪到 mask 外接框
--select      多个 mask 时选择 highest-conf 或 largest-mask
--class-id    只保留指定分割类别
--no-progress 关闭 tqdm 进度条
```

分类器会使用 `segment_features.py` 中定义的可解释特征：

```text
mask 几何：mask 面积比例、外接框宽高比例、mask 填充率、mask 中心位置
整体颜色：RGB/HSV 均值和标准差、暗像素比例、亮像素比例、高饱和比例
上下对比：底部和顶部的亮度差、饱和度差、亮度重心、饱和度重心
分层统计：从上到下 5 个水平带的 HSV 均值、暗亮比例、mask 占比
边缘特征：边缘密度、水平边缘强度
```

分类器训练输出：

```text
runs/tree_classifier/
  <algorithm>.joblib
  metrics.json
  tree_rules.txt              仅 decision-tree 输出
  feature_importances.csv     树模型/集成树模型输出
  linear_coefficients.csv     线性模型输出
```

输出结构：

```text
runs/tree_segments/
  images/
    train/
    val/
    test/
  features.csv
```

`features.csv` 会记录分割后的图片路径、原标签、mask 几何、颜色、分层和边缘特征，可直接复用于以上分类算法

如果需要自动打标，只允许使用本项目前面 `train_obb.py` 训练出的权重，例如：

```bash
python train_tree_classifier.py --label-model runs/obb/bottle_01_yolo11m_640_b4/weights/best.pt --label-set labels_0123 --segments-output runs/tree_segments_0123_model_labeled --tree-output runs/tree_classifier_0123_model_labeled --overwrite-segments --max-depth 5 --min-samples-leaf 3
```

`--label-model` 只接受 `runs/obb/` 下的项目训练权重，不接受 `yolo11n-seg.pt` 这类外部模型名

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

常用预测参数：

```text
conf=0.5           只保留置信度大于等于 0.5 的预测框
iou=0.5            NMS 去重的 IoU 阈值，越低去重越严格
imgsz=640          推理输入尺寸，通常和训练时保持一致
device=0           使用第 0 张 GPU，CPU 推理可写 device=cpu
save=True          保存带预测框的可视化图片
save_txt=True      保存预测标签文本
save_conf=True     在预测标签文本中额外保存置信度
project=runs/predict_errors/val_bottle_01  指定输出根目录
name=pred_conf50   指定本次预测输出目录名
exist_ok=True      允许写入已有输出目录
agnostic_nms=True  不同类别之间也参与 NMS 去重
classes=0          只预测指定类别，单类 bottle 模型通常不需要写
max_det=10         每张图片最多保留 10 个预测框
line_width=2       调整可视化框线宽
show_labels=True   可视化图片显示类别名
show_conf=True     可视化图片显示置信度
```

常用筛选和保存命令，例如只保留置信度大于等于 0.5 的结果：

```bash
yolo predict model=runs/obb/<run_name>/weights/best.pt source=/mnt/e/LiquidIdentification/testDataset/images conf=0.5 save=True save_txt=True save_conf=True
```

`conf=0.5` 表示丢弃置信度低于 0.5 的预测框，`save_conf=True` 会在导出的标签文本中同时保存每个预测框的置信度，后续排查错例或按置信度二次筛选会更方便

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

### Segmented Classical ML Data

`prepare_tree_segments.py` prepares data for classical machine-learning classifiers. By default it does not use any existing model; it reads YOLO OBB label polygons to build masks, keeps only the masked image region, then writes masked images and a `features.csv`

```bash
python prepare_tree_segments.py --label-set labels_0123 --output runs/tree_segments --overwrite
```

`train_tree_classifier.py` connects segmentation preprocessing and classifier training, automatically generating and reading the same `features.csv`. The default algorithm remains `decision-tree`; use `--algorithm` to switch to other common machine-learning classifiers.

```bash
python train_tree_classifier.py --label-set labels_0123 --segments-output runs/tree_segments_0123 --tree-output runs/tree_classifier_0123 --overwrite-segments --max-depth 5 --min-samples-leaf 3
```

If `features.csv` already exists, skip segmentation and train directly:

```bash
python train_tree_classifier.py --features runs/tree_segments/features.csv --tree-output runs/tree_classifier --max-depth 5 --min-samples-leaf 3
```

Train a random forest or SVM:

```bash
python train_tree_classifier.py --features runs/tree_segments/features.csv --tree-output runs/tree_classifier_rf --algorithm random-forest --n-estimators 300 --max-depth 8 --min-samples-leaf 2
python train_tree_classifier.py --features runs/tree_segments/features.csv --tree-output runs/tree_classifier_svm --algorithm rbf-svm --svm-c 3.0 --svm-gamma scale
```

Useful parameters:

```text
--model       optional project-trained weights under runs/obb/; omitted means masks are built from label polygons
--label-set   label set used as the recognition target, for example labels_0123, labels_01, or label_bottle
--algorithm   classifier: decision-tree, random-forest, extra-trees, gradient-boosting, ada-boost, logistic-regression, linear-svm, rbf-svm, knn, gaussian-nb, mlp
--criterion   split criterion for decision-tree, random-forest, and extra-trees: gini, entropy, log_loss
--max-depth   maximum tree depth
--n-estimators number of estimators for ensemble algorithms
--neighbors   number of neighbors for KNN
--svm-c       C for SVM/logistic regression
--svm-gamma   gamma for RBF SVM
--max-iter    maximum iterations for linear models and MLP
--conf        confidence threshold when a model is used to generate masks, default 0.25
--background  background outside the mask, one of black, white, transparent
--crop        crop output images to the mask bounding box
--select      choose highest-conf or largest-mask when multiple masks exist
--class-id    keep only one segmentation class id
--no-progress disable tqdm progress bars
```

The classifiers use interpretable features defined in `segment_features.py`:

```text
mask geometry: mask area ratio, bounding-box proportions, mask fill ratio, mask center
global color: RGB/HSV mean and std, dark ratio, bright ratio, high-saturation ratio
vertical contrast: bottom-top brightness difference, saturation difference, brightness center, saturation center
band statistics: HSV mean, dark/bright ratio, and mask fraction in 5 horizontal bands
edge features: edge density and horizontal edge strength
```

Classifier outputs:

```text
runs/tree_classifier/
  <algorithm>.joblib
  metrics.json
  tree_rules.txt              decision-tree only
  feature_importances.csv     tree and tree-ensemble models
  linear_coefficients.csv     linear models
```

Output layout:

```text
runs/tree_segments/
  images/
    train/
    val/
    test/
  features.csv
```

`features.csv` records the masked image path, original label, mask geometry, color, band, and edge features, and can be reused directly by the classifiers above

If automatic labeling is needed, only use weights trained earlier by this project through `train_obb.py`, for example:

```bash
python train_tree_classifier.py --label-model runs/obb/bottle_01_yolo11m_640_b4/weights/best.pt --label-set labels_0123 --segments-output runs/tree_segments_0123_model_labeled --tree-output runs/tree_classifier_0123_model_labeled --overwrite-segments --max-depth 5 --min-samples-leaf 3
```

`--label-model` only accepts project-trained weights under `runs/obb/`; external model names such as `yolo11n-seg.pt` are not accepted

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

Common prediction arguments:

```text
conf=0.5           keep only prediction boxes with confidence at least 0.5
iou=0.5            IoU threshold for NMS, lower values make suppression stricter
imgsz=640          inference image size, usually matching the training size
device=0           use GPU 0, or use device=cpu for CPU inference
save=True          save visualization images with prediction boxes
save_txt=True      save prediction label text files
save_conf=True     also save confidence values in prediction label text files
project=runs/predict_errors/val_bottle_01  set the output root directory
name=pred_conf50   set the output directory name for this prediction run
exist_ok=True      allow writing into an existing output directory
agnostic_nms=True  allow NMS suppression across different classes
classes=0          predict only the selected class, usually unnecessary for a single-class bottle model
max_det=10         keep at most 10 prediction boxes per image
line_width=2       adjust visualization box line width
show_labels=True   show class names in visualization images
show_conf=True     show confidence values in visualization images
```

Common filtering and saving command, for example keeping predictions with confidence at least 0.5:

```bash
yolo predict model=runs/obb/<run_name>/weights/best.pt source=/mnt/e/LiquidIdentification/testDataset/images conf=0.5 save=True save_txt=True save_conf=True
```

`conf=0.5` drops prediction boxes below 0.5 confidence, and `save_conf=True` stores each prediction confidence in the exported label text, which is useful for error analysis or a second confidence-based filtering pass

If duplicate boxes appear in prediction results, prefer YOLO native NMS parameters and rerun prediction instead of editing labels and redrawing visualization images manually:

```bash
yolo predict model=runs/obb/bottle_01_yolo11m_640_b4/weights/best.pt source=runs/predict_errors/val_bottle_01/images project=/mnt/e/LiquidIdentification/runs/predict_errors/val_bottle_01 name=pred_native_nms_iou50 save=True save_txt=True save_conf=True iou=0.5 agnostic_nms=True exist_ok=True
```

```bash
yolo predict model=runs/obb/bottle_01_yolo11m_640_b4/weights/best.pt source=runs/predict_errors/test_bottle_01/images project=/mnt/e/LiquidIdentification/runs/predict_errors/test_bottle_01 name=pred_native_nms_iou50 save=True save_txt=True save_conf=True iou=0.5 agnostic_nms=True exist_ok=True
```

`iou=0.5` treats boxes as duplicates when overlap over union is above 0.5, `agnostic_nms=True` allows suppression across different classes, and the final visualization keeps the native Ultralytics style

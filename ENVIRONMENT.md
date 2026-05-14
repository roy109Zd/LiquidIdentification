# Environment

## 中文说明

### 推荐环境

```text
OS: Windows + WSL2 Ubuntu
Python: 3.10 - 3.12
GPU: NVIDIA RTX 4060 Laptop GPU 8GB VRAM
CUDA: 使用与本机 NVIDIA 驱动匹配的 PyTorch CUDA 版本
```

项目脚本主要依赖 Python 标准库和 `ultralytics`，训练时还需要 PyTorch

### WSL 快速配置

进入项目目录：

```bash
cd /mnt/e/LiquidIdentification
```

创建虚拟环境：

```bash
python3 -m venv .venv
```

激活环境：

```bash
source .venv/bin/activate
```

升级基础工具：

```bash
python -m pip install -U pip setuptools wheel
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含 `ultralytics`、`numpy`、`opencv-python`、`scikit-learn`、`joblib` 和 `tqdm`，可覆盖 YOLO 训练、分割预处理和决策树训练脚本

如果需要 GPU 训练，请先按本机 CUDA 驱动安装对应的 PyTorch GPU 版本，再安装 `requirements.txt`

训练脚本会优先使用项目目录下的本地权重文件；如果传入 `yolo11m-obb.pt` 这类 Ultralytics 支持的裸模型名且本地不存在，会由 Ultralytics 自动下载

### 项目已有推荐环境

当前机器可直接使用已有的 `torchforge` 环境：

```bash
cd /mnt/e/LiquidIdentification
source /root/envs/torchforge/bin/activate
```

检查 GPU 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

检查 Ultralytics：

```bash
yolo checks
```

### 常用训练命令

四分类训练：

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_0123 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_0123_yolo11m_640_b4
```

二分类训练：

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_01 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_01_yolo11m_640_b4
```

只识别瓶子的单类训练：

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_yolo11m_640_b4
```

启用位移和旋转数据增强：

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_aug_yolo11m_640_b4 --augment-geom --degrees 10 --translate 0.1
```

显存不足时优先把 `--batch 4` 改成 `--batch 2`

## English

### Recommended Environment

```text
OS: Windows + WSL2 Ubuntu
Python: 3.10 - 3.12
GPU: NVIDIA RTX 4060 Laptop GPU 8GB VRAM
CUDA: use the PyTorch CUDA build that matches the local NVIDIA driver
```

The project scripts mostly use the Python standard library and `ultralytics`, and training also requires PyTorch

### Quick WSL Setup

Enter the project directory:

```bash
cd /mnt/e/LiquidIdentification
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade base tooling:

```bash
python -m pip install -U pip setuptools wheel
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `ultralytics`, `numpy`, `opencv-python`, `scikit-learn`, `joblib`, and `tqdm`, covering YOLO training, segmentation preprocessing, and decision-tree training

For GPU training, install the PyTorch GPU build that matches the local CUDA driver before installing `requirements.txt`

The training script prefers project-local weight files; if the value is a bare Ultralytics-supported model name such as `yolo11m-obb.pt` and no local file exists, Ultralytics will download it automatically

### Existing Recommended Environment

This machine can use the existing `torchforge` environment:

```bash
cd /mnt/e/LiquidIdentification
source /root/envs/torchforge/bin/activate
```

Check GPU availability:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Check Ultralytics:

```bash
yolo checks
```

### Common Training Commands

Four-class training:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_0123 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_0123_yolo11m_640_b4
```

Binary training:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set labels_01 --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_01_yolo11m_640_b4
```

Bottle-only single-class training:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_yolo11m_640_b4
```

Training with translation and rotation augmentation:

```bash
python train_obb.py --model yolo11m-obb.pt --label-set label_bottle --epochs 200 --imgsz 640 --batch 4 --device 0 --workers 2 --name bottle_only_aug_yolo11m_640_b4 --augment-geom --degrees 10 --translate 0.1
```

If GPU memory is insufficient, reduce `--batch 4` to `--batch 2`

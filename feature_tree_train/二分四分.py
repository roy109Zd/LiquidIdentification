#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整特征提取 + 多模型对比（支持 LCDTC 和 labels_picture 双数据集）
任务一：二分类 (有无液体)
任务二：四分类 (液位等级 0: empty, 1: little, 2: half, 3: much/fill)

支持两种特征集：
- 'light'  : 约 60 维，速度快
- 'full'   : 约 180 维，包含 Gabor、GLCM、傅里叶等，较慢但可能精度更高
"""

import json
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ========================= 配置参数 =========================
# --- 全局开关 -------------------------------------------------
EXTRACT_FEATURES = True   # 第一次运行设为 True 提取并保存 CSV，之后可改为 False 直接读 CSV
FEATURE_SET = 'full'      # 可选 'light' 或 'full'，决定特征丰富程度

# --- 特征选择（仅对 full 特征集有效，可进一步精简）------------
# 如果 feature_set == 'full' 且 USE_FEATURE_SELECTION = True，则仅使用重要性 Top-K 特征
USE_FEATURE_SELECTION = False    # 是否在训练前进行特征重要性筛选
TOP_K_FEATURES = 30              # 保留的特征数量（仅当 USE_FEATURE_SELECTION = True 时生效）

# --- 模型选择 -------------------------------------------------
MODELS_TO_RUN = ['decision_tree', 'random_forest', 'xgboost']  # 可任意增减

# --- 路径配置 -------------------------------------------------
LCDTC_BASE = "/root/CV/饮料瓶/LCDTC"
TRAIN_JSON = os.path.join(LCDTC_BASE, "annotations/instances_train2017.json")
VAL_JSON = os.path.join(LCDTC_BASE, "annotations/instances_val2017.json")
TRAIN_IMG_DIR = os.path.join(LCDTC_BASE, "images/train2017")
VAL_IMG_DIR = os.path.join(LCDTC_BASE, "images/val2017")

LABELS_BASE = "/root/CV/饮料瓶/labels_picture"
INFO_CSV = os.path.join(LABELS_BASE, "crop_info.csv")
CROP_IMG_DIR = LABELS_BASE

OUT_DIR = "/root/CV/饮料瓶/LCDTC/result"
os.makedirs(OUT_DIR, exist_ok=True)

# 特征 CSV 保存路径 (不同特征集会保存不同的文件)
if FEATURE_SET == 'full':
    CSV_LCDTC_TRAIN = os.path.join(OUT_DIR, "lcdtc_train_features_full.csv")
    CSV_LCDTC_VAL   = os.path.join(OUT_DIR, "lcdtc_val_features_full.csv")
    CSV_LABELS_DATA = os.path.join(OUT_DIR, "labels_picture_features_full.csv")
else:
    CSV_LCDTC_TRAIN = os.path.join(OUT_DIR, "lcdtc_train_features_light.csv")
    CSV_LCDTC_VAL   = os.path.join(OUT_DIR, "lcdtc_val_features_light.csv")
    CSV_LABELS_DATA = os.path.join(OUT_DIR, "labels_picture_features_light.csv")

# ========================= 特征提取函数 =========================
def safe_divide(a, b, eps=1e-8):
    return a / (b + eps)

def extract_features_light(img, is_already_cropped=True):
    """轻量级特征（约 60 维）"""
    if img is None or img.size == 0:
        return None
    h_target, w_target = 128, 64
    img = cv2.resize(img, (w_target, h_target))
    h, w = img.shape[:2]
    if not is_already_cropped:
        crop_start = int(w * 0.20)
        crop_end = int(w * 0.80)
        img = img[:, crop_start:crop_end]
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    features = {}

    # 水线与暗部
    blurred = cv2.GaussianBlur(gray, (5, 9), 0)
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    edge_profile = np.mean(sobel_y, axis=1)
    features['waterline_y_ratio'] = np.argmax(edge_profile) / h if edge_profile.size > 0 else 0.5
    mean_val = np.mean(gray)
    y_coords, _ = np.where(gray < mean_val)
    features['darkness_center_y'] = np.mean(y_coords) / h if len(y_coords) > 0 else 0.5

    # 上下半区对比
    top_gray, bottom_gray = gray[:h//2, :], gray[h//2:, :]
    features['tb_gray_std_ratio'] = safe_divide(np.std(top_gray), np.std(bottom_gray))
    features['tb_gray_mean_diff'] = np.mean(top_gray) - np.mean(bottom_gray)

    # 颜色直方图 (8 bins) + 颜色矩(mean, std)
    for ch, name in enumerate(['H', 'S', 'V']):
        hist = cv2.calcHist([hsv.astype(np.uint8)], [ch], None, [8], [0, 256 if ch>0 else 180]).flatten() / (h*w)
        for i in range(8):
            features[f'{name.lower()}_hist_{i}'] = hist[i]
        channel = hsv[:,:,ch]
        features[f'color_moment_{name}_mean'] = np.mean(channel)
        features[f'color_moment_{name}_std'] = np.std(channel)

    # LBP
    lbp = local_binary_pattern(gray_uint8, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(11), density=True)
    for i in range(10):
        features[f'lbp_{i}'] = lbp_hist[i]

    # 梯度方向直方图 (8 bins)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    ang[ang < 0] += 180
    grad_hist = np.histogram(ang.flatten(), bins=8, range=(0,180), weights=mag.flatten())[0]
    grad_hist = safe_divide(grad_hist, np.sum(grad_hist))
    for i in range(8):
        features[f'grad_hist_{i}'] = grad_hist[i]

    # Otsu 比例
    _, thresh = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features['otsu_white_ratio'] = np.sum(thresh > 0) / (h * w)

    # 清理异常值
    return {k: (0.0 if np.isinf(v) or np.isnan(v) else v) for k, v in features.items()}


def extract_features_full(img, is_already_cropped=True):
    """完整特征集（包含 Gabor、GLCM、傅里叶等，约 180 维）"""
    if img is None or img.size == 0:
        return None
    h_target, w_target = 128, 64
    img = cv2.resize(img, (w_target, h_target))
    h, w = img.shape[:2]
    if not is_already_cropped:
        crop_start = int(w * 0.20)
        crop_end = int(w * 0.80)
        img = img[:, crop_start:crop_end]
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    features = {}

    # ---------- 基础物理特征 ----------
    blurred = cv2.GaussianBlur(gray, (5, 9), 0)
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    edge_profile = np.mean(sobel_y, axis=1)
    features['waterline_y_ratio'] = np.argmax(edge_profile) / h if edge_profile.size > 0 else 0.5
    mean_val = np.mean(gray)
    y_coords, _ = np.where(gray < mean_val)
    features['darkness_center_y'] = np.mean(y_coords) / h if len(y_coords) > 0 else 0.5

    top_gray, bottom_gray = gray[:h//2, :], gray[h//2:, :]
    features['tb_gray_std_ratio'] = safe_divide(np.std(top_gray), np.std(bottom_gray))
    features['tb_gray_mean_diff'] = np.mean(top_gray) - np.mean(bottom_gray)

    # 四分块亮度下降趋势
    bin_h = h // 4
    bin_means = [np.mean(gray[i*bin_h:(i+1)*bin_h, :]) for i in range(4)]
    features['drop_0_1'] = bin_means[0] - bin_means[1]
    features['drop_1_2'] = bin_means[1] - bin_means[2]
    features['drop_2_3'] = bin_means[2] - bin_means[3]

    top_hsv = hsv[:h//2, :, :]
    bottom_hsv = hsv[h//2:, :, :]
    features['bottom_saturation'] = np.mean(bottom_hsv[:,:,1])
    features['diff_saturation'] = np.mean(top_hsv[:,:,1]) - features['bottom_saturation']

    # ---------- 颜色特征 ----------
    for ch, name in enumerate(['H', 'S', 'V']):
        hist = cv2.calcHist([hsv.astype(np.uint8)], [ch], None, [16], [0, 256 if ch>0 else 180]).flatten() / (h*w)
        for i in range(16):
            features[f'{name.lower()}_hist_{i}'] = hist[i]
        channel = hsv[:,:,ch]
        features[f'color_moment_{name}_mean'] = np.mean(channel)
        features[f'color_moment_{name}_std'] = np.std(channel)
        features[f'color_moment_{name}_skew'] = skew(channel.flatten())
        features[f'color_moment_{name}_kurt'] = kurtosis(channel.flatten())

    # ---------- LBP 双尺度 ----------
    lbp1 = local_binary_pattern(gray_uint8, 8, 1, method='uniform')
    hist1, _ = np.histogram(lbp1.ravel(), bins=np.arange(11), density=True)
    for i in range(10):
        features[f'lbp_r1_{i}'] = hist1[i]

    lbp2 = local_binary_pattern(gray_uint8, 16, 2, method='uniform')
    hist2, _ = np.histogram(lbp2.ravel(), bins=np.arange(19), density=True)
    for i in range(18):
        features[f'lbp_r2_{i}'] = hist2[i]

    # ---------- Gabor 滤波器 ----------
    gabor_feats = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.1, 0.2]:
            real, imag = gabor(gray_uint8, frequency=freq, theta=theta)
            resp = np.sqrt(real**2 + imag**2)
            gabor_feats.extend([np.mean(resp), np.std(resp)])
    for idx, val in enumerate(gabor_feats):
        features[f'gabor_{idx}'] = val

    # ---------- GLCM ----------
    gray_8 = (gray / 32).astype(np.uint8)  # 0-7
    glcm = graycomatrix(gray_8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=8, symmetric=True, normed=True)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        vals = graycoprops(glcm, prop).flatten()
        for i, v in enumerate(vals):
            features[f'glcm_{prop}_{i}'] = v

    # ---------- 分块统计 (4x2) ----------
    rows, cols = 4, 2
    h_block = h // rows
    w_block = w // cols
    for i in range(rows):
        for j in range(cols):
            block = gray[i*h_block:(i+1)*h_block, j*w_block:(j+1)*w_block]
            if block.size > 0:
                features[f'block_mean_{i}_{j}'] = np.mean(block)
                features[f'block_std_{i}_{j}'] = np.std(block)
                block_hsv = hsv[i*h_block:(i+1)*h_block, j*w_block:(j+1)*w_block, :]
                if block_hsv.size > 0:
                    features[f'block_h_mean_{i}_{j}'] = np.mean(block_hsv[:,:,0])
                    features[f'block_s_mean_{i}_{j}'] = np.mean(block_hsv[:,:,1])

    # ---------- 梯度方向直方图 (18 bins) ----------
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    ang[ang < 0] += 180
    grad_hist = np.histogram(ang.flatten(), bins=18, range=(0,180), weights=mag.flatten())[0]
    grad_hist = safe_divide(grad_hist, np.sum(grad_hist))
    for i in range(18):
        features[f'grad_hist_{i}'] = grad_hist[i]

    # ---------- 傅里叶功率谱 ----------
    f = fft2(gray)
    fshift = fftshift(f)
    mag_spec = np.log1p(np.abs(fshift) + 1e-8)
    cy, cx = h//2, w//2
    max_r = min(cy, cx)
    r_bins = 15
    radial = np.zeros(r_bins)
    for y in range(h):
        for x in range(w):
            r = int(np.sqrt((y-cy)**2 + (x-cx)**2))
            if r < max_r:
                idx = int(r * r_bins / max_r)
                radial[idx] += mag_spec[y, x]
    radial = safe_divide(radial, np.sum(radial))
    for i in range(r_bins):
        features[f'fft_radial_{i}'] = radial[i]

    ang_bins = 12
    angular = np.zeros(ang_bins)
    for y in range(h):
        for x in range(w):
            if x == cx and y == cy:
                continue
            angle = np.arctan2(y-cy, x-cx) + np.pi
            idx = int(angle * ang_bins / (2*np.pi)) % ang_bins
            angular[idx] += mag_spec[y, x]
    angular = safe_divide(angular, np.sum(angular))
    for i in range(ang_bins):
        features[f'fft_angular_{i}'] = angular[i]

    # ---------- Canny 边缘方向 ----------
    edges = cv2.Canny(gray_uint8, 50, 150)
    edge_y, edge_x = np.where(edges > 0)
    if len(edge_x) > 0:
        edge_angles = np.arctan2(gy[edge_y, edge_x], gx[edge_y, edge_x]) * 180 / np.pi
        edge_angles[edge_angles < 0] += 180
        edge_hist = np.histogram(edge_angles, bins=18, range=(0,180))[0]
        edge_hist = safe_divide(edge_hist, np.sum(edge_hist))
    else:
        edge_hist = np.zeros(18)
    for i in range(18):
        features[f'edge_orient_{i}'] = edge_hist[i]

    # ---------- 灰度差分 ----------
    for d in [1, 2, 4]:
        if d < w:
            diff_h = np.abs(gray[:, d:] - gray[:, :-d])
            features[f'diff_h_mean_{d}'] = np.mean(diff_h)
            features[f'diff_h_std_{d}'] = np.std(diff_h)
        if d < h:
            diff_v = np.abs(gray[d:, :] - gray[:-d, :])
            features[f'diff_v_mean_{d}'] = np.mean(diff_v)
            features[f'diff_v_std_{d}'] = np.std(diff_v)

    # ---------- 灰度分位数 ----------
    for p in [10, 25, 50, 75, 90]:
        features[f'gray_percentile_{p}'] = np.percentile(gray, p)

    # ---------- Otsu 比例 ----------
    _, thresh = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features['otsu_white_ratio'] = np.sum(thresh > 0) / (h * w)
    white_y = np.where(thresh > 0)[0]
    features['white_center_y'] = np.mean(white_y) / h if len(white_y) > 0 else 0.5

    # 清理异常值
    return {k: (0.0 if np.isinf(v) or np.isnan(v) else v) for k, v in features.items()}


def extract_features(img, is_already_cropped=True):
    """统一入口，根据 FEATURE_SET 选择特征提取函数"""
    if FEATURE_SET == 'full':
        return extract_features_full(img, is_already_cropped)
    else:
        return extract_features_light(img, is_already_cropped)


# ========================= 数据加载与 CSV 构建 =========================
def build_lcdtc_csv(json_path, img_dir, out_csv):
    with open(json_path, 'r') as f:
        data = json.load(f)
    img_id_to_file = {img['id']: img['file_name'] for img in data['images']}
    rows = []
    total = len(data['annotations'])
    for idx, ann in enumerate(data['annotations']):
        if idx % 500 == 0 or idx == total-1:
            print(f"LCDTC 进度: {idx}/{total}")
        img_file = img_id_to_file.get(ann['image_id'])
        if not img_file:
            continue
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        x, y, w, h = map(int, ann['bbox'])
        x, y = max(0, x), max(0, y)
        w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
        if w <= 0 or h <= 0:
            continue
        crop = img[y:y+h, x:x+w]
        feats = extract_features(crop, is_already_cropped=True)
        if feats:
            cat_id = ann['category_id']
            has_liquid = 0 if cat_id == 0 else 1
            amount_label = 0 if cat_id == 0 else min(cat_id, 3)
            feats.update({'source': 'LCDTC', 'image_name': img_file, 'has_liquid': has_liquid, 'amount_label': amount_label})
            rows.append(feats)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"保存 {len(df)} 个样本到 {out_csv}")


def build_labels_picture_csv(csv_path, img_dir, out_csv):
    df_info = pd.read_csv(csv_path)
    rows = []
    total = len(df_info)
    for idx, (_, row) in enumerate(df_info.iterrows()):
        if idx % 100 == 0 or idx == total-1:
            print(f"labels_picture 进度: {idx}/{total}")
        img_path = os.path.join(img_dir, row['cropped_image_path'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        feats = extract_features(img, is_already_cropped=True)
        if feats:
            split_val = row.get('new_split', 'unknown')
            has_liquid = int(row['has_liquid'])
            if 'amount_label' in row:
                amount_label = int(row['amount_label'])
            else:
                amount_label = 0 if has_liquid == 0 else 3
            feats.update({'source': f'labels_{split_val}', 'image_name': row['cropped_image_path'],
                          'has_liquid': has_liquid, 'amount_label': amount_label})
            rows.append(feats)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"保存 {len(df)} 个样本到 {out_csv}")


# ========================= 训练与评估 =========================
def get_model(name):
    if name == 'decision_tree':
        return DecisionTreeClassifier(max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42)
    elif name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                      class_weight='balanced', random_state=42, n_jobs=-1)
    elif name == 'xgboost':
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                             objective='multi:softmax' if 'amount' in name else 'binary:logistic',
                             eval_metric='mlogloss' if 'amount' in name else 'logloss',
                             random_state=42, use_label_encoder=False)
    else:
        raise ValueError(f"Unknown model: {name}")


def evaluate_models(X_train, y_train, X_test1, y_test1, X_test2, y_test2, task_name):
    """对多个模型进行训练并在两个测试集上评估"""
    print(f"\n{'='*70}")
    print(f"任务: {task_name}")
    print(f"训练样本数: {len(X_train)}")
    print(f"测试集1 (LCDTC val) 样本数: {len(X_test1)}")
    print(f"测试集2 (labels_picture test) 样本数: {len(X_test2)}")
    print(f"特征维度: {X_train.shape[1]}")
    print('='*70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test1_scaled = scaler.transform(X_test1)
    X_test2_scaled = scaler.transform(X_test2)

    results = []
    for model_name in MODELS_TO_RUN:
        print(f"\n>>> 训练 {model_name} ...")
        model = get_model(model_name)
        model.fit(X_train_scaled, y_train)

        y_pred1 = model.predict(X_test1_scaled)
        y_pred2 = model.predict(X_test2_scaled)

        acc1 = accuracy_score(y_test1, y_pred1)
        acc2 = accuracy_score(y_test2, y_pred2)

        print(f"   LCDTC 验证集准确率: {acc1:.4f}")
        print(f"   labels_picture 测试集准确率: {acc2:.4f}")
        results.append({
            'model': model_name,
            'task': task_name,
            'acc_lcdtc': acc1,
            'acc_labels': acc2
        })
    return results


def main():
    # ---------- 1. 提取特征并保存 CSV ----------
    if EXTRACT_FEATURES:
        print("========== 重新提取特征 ==========")
        print(f"特征集: {FEATURE_SET}")
        build_lcdtc_csv(TRAIN_JSON, TRAIN_IMG_DIR, CSV_LCDTC_TRAIN)
        build_lcdtc_csv(VAL_JSON, VAL_IMG_DIR, CSV_LCDTC_VAL)
        build_labels_picture_csv(INFO_CSV, CROP_IMG_DIR, CSV_LABELS_DATA)
        print("特征提取完成！\n")

    # ---------- 2. 加载 CSV ----------
    print("加载特征 CSV ...")
    df_lcdtc_train = pd.read_csv(CSV_LCDTC_TRAIN)
    df_lcdtc_val   = pd.read_csv(CSV_LCDTC_VAL)
    df_labels_all  = pd.read_csv(CSV_LABELS_DATA)

    # 分离 labels_picture 的训练集和测试集
    df_labels_train = df_labels_all[df_labels_all['source'] == 'labels_train']
    df_labels_test  = df_labels_all[df_labels_all['source'] == 'labels_test']

    # 合并训练数据
    df_train_combined = pd.concat([df_lcdtc_train, df_labels_train], ignore_index=True)

    # 确定特征列（排除非特征列）
    exclude_cols = ['source', 'image_name', 'has_liquid', 'amount_label']
    feature_cols = [c for c in df_train_combined.columns if c not in exclude_cols]
    print(f"总特征数: {len(feature_cols)}")

    X_full = df_train_combined[feature_cols].values
    y_binary = df_train_combined['has_liquid'].values
    y_4class = df_train_combined['amount_label'].values

    # ---------- 3. 可选特征重要性筛选 (仅对 full 特征集) ----------
    if FEATURE_SET == 'full' and USE_FEATURE_SELECTION:
        print("\n使用随机森林评估特征重要性，保留 Top-{} 特征...".format(TOP_K_FEATURES))
        rf_eval = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_eval.fit(X_full, y_binary)  # 用二分类任务评估重要性
        importances = rf_eval.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:TOP_K_FEATURES]
        selected_features = [feature_cols[i] for i in sorted_idx]
        print("Selected top features:", selected_features[:10])
        # 更新特征列
        feature_cols = selected_features
        X_train = df_train_combined[feature_cols].values
        X_test1 = df_lcdtc_val[feature_cols].values
        X_test2 = df_labels_test[feature_cols].values
    else:
        X_train = df_train_combined[feature_cols].values
        X_test1 = df_lcdtc_val[feature_cols].values
        X_test2 = df_labels_test[feature_cols].values

    # ---------- 4. 执行二分类任务 ----------
    y_test1_binary = df_lcdtc_val['has_liquid'].values
    y_test2_binary = df_labels_test['has_liquid'].values
    results_binary = evaluate_models(X_train, y_binary,
                                     X_test1, y_test1_binary,
                                     X_test2, y_test2_binary,
                                     task_name="二分类 (有无液体)")

    # ---------- 5. 执行四分类任务 ----------
    # 注意：四分类任务需要单独评估，这里复用同样的特征（当然也可以重新做特征选择）
    y_test1_4class = df_lcdtc_val['amount_label'].values
    y_test2_4class = df_labels_test['amount_label'].values
    results_4class = evaluate_models(X_train, y_4class,
                                     X_test1, y_test1_4class,
                                     X_test2, y_test2_4class,
                                     task_name="四分类 (液位等级 0~3)")

    # ---------- 6. 汇总结果 ----------
    print("\n" + "="*70)
    print("最终结果汇总")
    print("="*70)
    all_results = results_binary + results_4class
    df_res = pd.DataFrame(all_results)
    print(df_res.to_string(index=False))

    # 可选：保存结果到 CSV
    df_res.to_csv(os.path.join(OUT_DIR, f"results_{FEATURE_SET}.csv"), index=False)
    print(f"\n结果已保存至 {OUT_DIR}/results_{FEATURE_SET}.csv")


if __name__ == "__main__":
    main()
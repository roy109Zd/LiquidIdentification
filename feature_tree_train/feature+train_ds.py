import os
import cv2
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import kurtosis, skew
from scipy.fft import fft2, fftshift
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/CV/饮料瓶/labels_picture/'
INFO_CSV_PATH = '/root/CV/饮料瓶/labels_picture/crop_info.csv'
ERROR_DIR = '/root/CV/饮料瓶/labels_picture/g/error'

def safe_divide(a, b, eps=1e-8):
    """安全除法，避免除零和无穷"""
    return a / (b + eps)

def safe_log(x, eps=1e-8):
    """安全对数，避免负数和零"""
    return np.log1p(np.maximum(x, 0) + eps)

def extract_mega_features(img_path):
    """提取稳健的超大特征集（~450维）"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # 预处理：旋转 + 固定尺寸
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h_target, w_target = 128, 64
    img = cv2.resize(img, (w_target, h_target))

    # 取中心60%区域
    crop_start = int(w_target * 0.20)
    crop_end = int(w_target * 0.80)
    center = img[:, crop_start:crop_end]
    h, w = center.shape[:2]   # 128 x 约38

    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY).astype(np.float64)
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV).astype(np.float64)
    features = {}

    # ========== 1. 原有基础特征 ==========
    blurred = cv2.GaussianBlur(gray, (5, 9), 0)
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    edge_profile = np.mean(sobel_y, axis=1)
    features['waterline_y_ratio'] = np.argmax(edge_profile) / h if edge_profile.size > 0 else 0.5

    mean_val = np.mean(gray)
    dark_mask = gray < mean_val
    y_coords, _ = np.where(dark_mask)
    features['darkness_center_y'] = np.mean(y_coords) / h if len(y_coords) > 0 else 0.5

    top_gray = gray[:h//2, :]
    bottom_gray = gray[h//2:, :]
    features['texture_ratio_tb'] = safe_divide(np.std(top_gray), np.std(bottom_gray))

    bin_h = h // 4
    bin_means = [np.mean(gray[i*bin_h:(i+1)*bin_h, :]) for i in range(4)]
    features['drop_0_to_1'] = bin_means[0] - bin_means[1]
    features['drop_1_to_2'] = bin_means[1] - bin_means[2]
    features['drop_2_to_3'] = bin_means[2] - bin_means[3]

    top_hsv = hsv[:h//2, :, :]
    bottom_hsv = hsv[h//2:, :, :]
    features['bottom_brightness'] = np.mean(bottom_gray)
    features['bottom_saturation'] = np.mean(bottom_hsv[:,:,1])
    features['diff_brightness_tb'] = np.mean(top_gray) - features['bottom_brightness']
    features['diff_saturation_tb'] = np.mean(top_hsv[:,:,1]) - features['bottom_saturation']

    # ========== 2. 颜色直方图（16 bins） ==========
    h_hist = cv2.calcHist([hsv.astype(np.uint8)], [0], None, [16], [0, 180]).flatten() / (h*w)
    s_hist = cv2.calcHist([hsv.astype(np.uint8)], [1], None, [16], [0, 256]).flatten() / (h*w)
    v_hist = cv2.calcHist([hsv.astype(np.uint8)], [2], None, [16], [0, 256]).flatten() / (h*w)
    for i in range(16):
        features[f'h_hist_{i}'] = h_hist[i]
        features[f's_hist_{i}'] = s_hist[i]
        features[f'v_hist_{i}'] = v_hist[i]

    # 颜色矩（4阶）
    for ch, name in enumerate(['H','S','V']):
        channel = hsv[:,:,ch]
        mean = np.mean(channel)
        std = np.std(channel)
        sk = skew(channel.flatten())
        kurt = kurtosis(channel.flatten())
        features[f'color_moment_{name}_mean'] = mean
        features[f'color_moment_{name}_std'] = std
        features[f'color_moment_{name}_skew'] = sk
        features[f'color_moment_{name}_kurt'] = kurt

    # ========== 3. LBP 纹理（半径1和2，均匀模式） ==========
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    lbp_r1 = local_binary_pattern(gray_uint8, 8, 1, method='uniform')
    lbp_hist1, _ = np.histogram(lbp_r1.ravel(), bins=np.arange(11), density=True)
    for i in range(10):
        features[f'lbp_r1_{i}'] = lbp_hist1[i]
    lbp_r2 = local_binary_pattern(gray_uint8, 16, 2, method='uniform')
    lbp_hist2, _ = np.histogram(lbp_r2.ravel(), bins=np.arange(19), density=True)  # 0-17 + 1 for uniform? Actually uniform LBP with 16 neighbors yields 18 patterns (0-16 uniform + 1 mixed), total 18? We'll use 18
    for i in range(min(18, len(lbp_hist2))):
        features[f'lbp_r2_{i}'] = lbp_hist2[i]

    # ========== 4. Gabor 滤波器（2频率×4方向，节省计算） ==========
    gabor_features = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in [0.1, 0.2]:
            real, imag = gabor(gray_uint8, frequency=freq, theta=theta)
            resp = np.sqrt(real**2 + imag**2)
            gabor_features.append(np.mean(resp))
            gabor_features.append(np.std(resp))
    for idx, val in enumerate(gabor_features):
        features[f'gabor_{idx}'] = val

    # ========== 5. GLCM（灰度等级8，距离1，4方向） ==========
    gray_8 = (gray / 32).astype(np.uint8)  # 0-7, 8 levels
    glcm = graycomatrix(gray_8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in props:
        vals = graycoprops(glcm, prop).flatten()
        for i, v in enumerate(vals):
            features[f'glcm_{prop}_{i}'] = v

    # ========== 6. 分块统计（4x2网格，每块均值、标准差） ==========
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

    # ========== 7. 简化 HOG（16x4 cells，每个cell 9方向，降采样为均值和方差？为避免维数爆炸，取全局梯度直方图） ==========
    # 更好的方式：直接使用全局梯度方向直方图（类似边缘方向），而不是块级HOG
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    ang[ang < 0] += 180
    # 全局方向直方图
    grad_hist = np.histogram(ang.flatten(), bins=18, range=(0,180), weights=mag.flatten())[0]
    grad_hist = safe_divide(grad_hist, np.sum(grad_hist))
    for i in range(18):
        features[f'grad_hist_{i}'] = grad_hist[i]

    # ========== 8. 傅里叶功率谱（径向和角向） ==========
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

    # ========== 9. 边缘方向直方图（Canny） ==========
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

    # ========== 10. 灰度差分统计 ==========
    for d in [1, 2, 4]:
        if d < w:
            diff_h = np.abs(gray[:, d:] - gray[:, :-d])
            features[f'diff_h_mean_{d}'] = np.mean(diff_h)
            features[f'diff_h_std_{d}'] = np.std(diff_h)
        if d < h:
            diff_v = np.abs(gray[d:, :] - gray[:-d, :])
            features[f'diff_v_mean_{d}'] = np.mean(diff_v)
            features[f'diff_v_std_{d}'] = np.std(diff_v)

    # ========== 11. 灰度分位数 ==========
    for p in [10, 25, 50, 75, 90]:
        features[f'gray_percentile_{p}'] = np.percentile(gray, p)

    # ========== 12. Otsu 二值化特征 ==========
    _, thresh = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(thresh > 0) / (h*w)
    features['otsu_white_ratio'] = white_ratio
    white_y = np.where(thresh > 0)[0]
    features['white_center_y'] = np.mean(white_y) / h if len(white_y) > 0 else 0.5

    # 最后，对所有特征值进行裁剪，去除异常值（防止 inf 和过大值）
    for key in features:
        if np.isinf(features[key]) or np.isnan(features[key]):
            features[key] = 0.0
        elif np.abs(features[key]) > 1e6:
            features[key] = np.clip(features[key], -1e6, 1e6)

    return features

def main():
    print(">>> 提取 Mega 特征集（稳健版，特征约 450 维）...")
    os.makedirs(ERROR_DIR, exist_ok=True)

    df_info = pd.read_csv(INFO_CSV_PATH)
    results = []
    for idx, row in df_info.iterrows():
        img_path = os.path.join(BASE_DIR, row['cropped_image_path'])
        if not os.path.exists(img_path):
            continue
        feats = extract_mega_features(img_path)
        if feats:
            row_data = {
                'has_liquid': int(row['has_liquid']),
                'split': row['new_split'],
                'img_path': row['cropped_image_path']
            }
            row_data.update(feats)
            results.append(row_data)

    df = pd.DataFrame(results)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    feature_cols = [c for c in df.columns if c not in ['has_liquid', 'split', 'img_path']]
    print(f"总特征数: {len(feature_cols)}")

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df['has_liquid'].values
    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = test_df['has_liquid'].values
    test_img_paths = test_df['img_path'].values

    # 再次清理无穷值（备用）
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    svm = SVC(kernel='rbf', C=1.5, gamma='scale', class_weight='balanced', random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n========== 二分类任务 (有无液体) ==========")
    for name, model in [("Random Forest", rf), ("SVM", svm)]:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"[{name.ljust(15)}] 5折CV均值: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f}) | 测试集准确率: {acc:.3f}")
        print(classification_report(y_test, y_pred, target_names=['无液体', '有液体']))

        if name == "Random Forest":
            error_indices = np.where(y_pred != y_test)[0]
            print(f"\n>>> 随机森林预测错误数量: {len(error_indices)}")
            error_info = []
            for idx in error_indices:
                src = os.path.join(BASE_DIR, test_img_paths[idx])
                dst = os.path.join(ERROR_DIR, os.path.basename(test_img_paths[idx]))
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                error_info.append({
                    'img_path': test_img_paths[idx],
                    'true_label': y_test[idx],
                    'pred_label': y_pred[idx]
                })
            error_df = pd.DataFrame(error_info)
            error_csv_path = os.path.join(ERROR_DIR, 'error_samples.csv')
            error_df.to_csv(error_csv_path, index=False)
            print(f"错误图片已保存至: {ERROR_DIR}")
            print(f"错误样本信息已保存至: {error_csv_path}")

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:30]
            print("\n--- 随机森林特征重要性 Top30 ---")
            for i in indices:
                print(f"{feature_cols[i]:35s} : {importances[i]:.5f}")

if __name__ == "__main__":
    main()
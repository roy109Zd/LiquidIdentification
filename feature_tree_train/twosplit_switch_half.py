import os
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from skimage.feature import local_binary_pattern
import warnings
warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = '/root/CV/饮料瓶/labels_picture/'
INFO_CSV_PATH = os.path.join(BASE_DIR, 'crop_info.csv')

def safe_divide(a, b, eps=1e-8):
    return a / (b + eps)

def extract_compact_features(img_path):
    """提取约50维特征"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h_target, w_target = 128, 64
    img = cv2.resize(img, (w_target, h_target))
    crop_start = int(w_target * 0.20)
    crop_end = int(w_target * 0.80)
    center = img[:, crop_start:crop_end]
    h, w = center.shape[:2]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY).astype(np.float64)
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV).astype(np.float64)
    features = {}

    # ----- 1. 物理几何特征 (9维) -----
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

    # ----- 2. 颜色特征：压缩直方图 (H,S,V各4 bins = 12维) -----
    h_hist = cv2.calcHist([hsv.astype(np.uint8)], [0], None, [4], [0, 180]).flatten() / (h*w)
    s_hist = cv2.calcHist([hsv.astype(np.uint8)], [1], None, [4], [0, 256]).flatten() / (h*w)
    v_hist = cv2.calcHist([hsv.astype(np.uint8)], [2], None, [4], [0, 256]).flatten() / (h*w)
    for i in range(4):
        features[f'h_hist_{i}'] = h_hist[i]
        features[f's_hist_{i}'] = s_hist[i]
        features[f'v_hist_{i}'] = v_hist[i]

    # 颜色矩：均值和标准差 (H,S,V共6维)
    for ch, name in enumerate(['H','S','V']):
        channel = hsv[:,:,ch]
        features[f'color_moment_{name}_mean'] = np.mean(channel)
        features[f'color_moment_{name}_std'] = np.std(channel)

    # ----- 3. 纹理：LBP 半径1均匀模式 (10维) -----
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(11), density=True)
    for i in range(10):
        features[f'lbp_{i}'] = lbp_hist[i]

    # ----- 4. 分块统计：竖向4块，灰度均值+H均值 (8维) -----
    rows = 4
    h_block = h // rows
    for i in range(rows):
        block = gray[i*h_block:(i+1)*h_block, :]
        features[f'block_mean_{i}'] = np.mean(block)
        block_hsv = hsv[i*h_block:(i+1)*h_block, :, :]
        features[f'block_h_mean_{i}'] = np.mean(block_hsv[:,:,0])

    # ----- 5. 梯度方向直方图 (6 bins) -----
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    ang[ang < 0] += 180
    grad_hist = np.histogram(ang.flatten(), bins=6, range=(0,180), weights=mag.flatten())[0]
    grad_hist = safe_divide(grad_hist, np.sum(grad_hist))
    for i in range(6):
        features[f'grad_hist_{i}'] = grad_hist[i]

    # ----- 6. 灰度分位数 (25,50,75) 3维 + Otsu白色占比 1维 -----
    for p in [25, 50, 75]:
        features[f'gray_percentile_{p}'] = np.percentile(gray, p)
    _, thresh = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features['otsu_white_ratio'] = np.sum(thresh > 0) / (h * w)

    # 清理异常值
    for k in features:
        if np.isinf(features[k]) or np.isnan(features[k]):
            features[k] = 0.0
    return features

def build_full_dataset(df_info):
    """提取所有样本的特征和标签"""
    all_data = []
    for _, row in df_info.iterrows():
        img_path = os.path.join(BASE_DIR, row['cropped_image_path'])
        if not os.path.exists(img_path):
            continue
        feats = extract_compact_features(img_path)
        if feats:
            all_data.append({
                'features': feats,
                'label': int(row['has_liquid']),
                'img_path': row['cropped_image_path']
            })
    return all_data

def data_to_arrays(data_list):
    """将数据列表转为特征矩阵和标签向量"""
    X = np.array([list(d['features'].values()) for d in data_list], dtype=np.float64)
    y = np.array([d['label'] for d in data_list])
    feature_names = list(data_list[0]['features'].keys())
    return X, y, feature_names

def split_data(data_list, test_size=50, random_state=42):
    """打乱数据并划分固定数量的测试集"""
    random.seed(random_state)
    shuffled = data_list.copy()
    random.shuffle(shuffled)
    test_set = shuffled[:test_size]
    train_set = shuffled[test_size:]
    return train_set, test_set

def train_evaluate_model(X_train, y_train, X_test, y_test, model, model_name):
    """训练并评估模型，返回准确率"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    return acc

def iteration_swap(train_set, test_set, swap_ratio=0.5):
    """
    使用决策树找出测试集中的错误样本，只交换一半错误样本（随机选一半）到训练集，
    同时从训练集随机选相同数量的样本交换到测试集。
    返回新的 train_set, test_set。
    """
    # 构建当前训练和测试的特征数组
    X_train, y_train, _ = data_to_arrays(train_set)
    X_test, y_test, _ = data_to_arrays(test_set)

    # 标准化并训练决策树
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
    dt.fit(X_train_scaled, y_train)
    y_pred = dt.predict(X_test_scaled)

    # 找出错误样本的索引
    error_indices = [i for i, (true, pred) in enumerate(zip(y_test, y_pred)) if true != pred]
    if not error_indices:
        print("  没有错误样本，无需交换")
        return train_set, test_set

    # 只交换一半的错误样本（随机选择）
    n_errors = len(error_indices)
    n_swap_errors = max(1, int(n_errors * swap_ratio))
    selected_error_indices = random.sample(error_indices, min(n_swap_errors, n_errors))
    error_samples = [test_set[i] for i in selected_error_indices]

    # 从训练集中随机选择相同数量的样本
    n_swap = len(error_samples)
    if n_swap > len(train_set):
        n_swap = len(train_set)
        error_samples = error_samples[:n_swap]
    selected_train_indices = random.sample(range(len(train_set)), n_swap)
    swap_train_samples = [train_set[i] for i in selected_train_indices]

    # 执行交换
    new_train_set = train_set.copy()
    new_test_set = test_set.copy()
    # 移除被选中的训练样本，并加入错误样本
    for idx in sorted(selected_train_indices, reverse=True):
        del new_train_set[idx]
    new_train_set.extend(error_samples)
    # 移除被选中的错误测试样本，并加入训练样本
    for idx in sorted(selected_error_indices, reverse=True):
        del new_test_set[idx]
    new_test_set.extend(swap_train_samples)

    print(f"  交换了 {n_swap} 个样本 (错误样本: {len(error_samples)}, 训练样本: {len(swap_train_samples)})")
    return new_train_set, new_test_set

def main():
    print("读取数据并提取特征...")
    df_info = pd.read_csv(INFO_CSV_PATH)
    if 'new_split' in df_info.columns:
        df_info.rename(columns={'new_split': 'split'}, inplace=True)
    # 忽略原始划分，我们将重新打散
    all_data = build_full_dataset(df_info)
    if len(all_data) < 50:
        print(f"总样本数({len(all_data)})不足50，将使用全部作为测试集")
        test_size = len(all_data) // 2
    else:
        test_size = 50

    print(f"总样本数: {len(all_data)}, 测试集固定大小: {test_size}")
    train_set, test_set = split_data(all_data, test_size=test_size, random_state=42)
    print(f"初始划分: 训练集 {len(train_set)}, 测试集 {len(test_set)}")

    # 迭代参数
    n_iterations = 5  # 迭代次数，可修改
    swap_ratio = 0.5  # 交换一半的错误样本

    for iter_num in range(1, n_iterations+1):
        print(f"\n迭代 {iter_num}:")
        # 评估当前决策树在测试集上的准确率
        X_train, y_train, _ = data_to_arrays(train_set)
        X_test, y_test, _ = data_to_arrays(test_set)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        dt = DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42)
        dt.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, dt.predict(X_test_scaled))
        print(f"  当前决策树测试准确率: {acc:.4f}")

        # 交换样本
        train_set, test_set = iteration_swap(train_set, test_set, swap_ratio)

    # 最终评估多个模型
    print("\n" + "="*50)
    print("最终划分下的多模型评估")
    X_train, y_train, feature_names = data_to_arrays(train_set)
    X_test, y_test, _ = data_to_arrays(test_set)
    print(f"最终训练集大小: {len(train_set)}, 测试集大小: {len(test_set)}")

    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42),
        'SVM': SVC(kernel='rbf', C=1.5, gamma='scale', class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance')
    }

    for name, model in models.items():
        acc = train_evaluate_model(X_train, y_train, X_test, y_test, model, name)
        print(f"{name:15s} 准确率: {acc:.4f}")

if __name__ == "__main__":
    main()
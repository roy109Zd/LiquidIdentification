# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report

# # 导入五种模型
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC

# # --- 1. 数据加载与准备 ---
# DATA_PATH = '/root/CV/饮料瓶/labels_picture/extracted_features.csv'
# df = pd.read_csv(DATA_PATH)

# # 根据 split 列划分训练集和测试集
# train_df = df[df['split'] == 'train']
# test_df = df[df['split'] == 'test']

# # 定义特征列
# feature_cols = [
#     'proj_max_val', 'proj_max_y_ratio', 'top_mean_gray', 'bottom_mean_gray', 
#     'tb_gray_ratio', 'top_edge_density', 'bottom_edge_density', 'tb_edge_ratio', 
#     'bottom_mean_s', 'bottom_mean_v', 'global_std', 'global_entropy'
# ]

# X_train = train_df[feature_cols].values
# X_test = test_df[feature_cols].values

# # --- 2. 特征归一化 (极重要: 适配 SVM 和 KNN) ---
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # --- 3. 定义模型 ---
# # 针对小样本，限制树的深度防止过拟合
# models = {
#     "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
#     "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
#     "XGBoost": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss'),
#     "KNN": KNeighborsClassifier(n_neighbors=5),
#     "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
# }

# def train_and_evaluate(task_name, y_train, y_test):
#     print(f"\n{'='*20} 任务: {task_name} {'='*20}")
#     for name, model in models.items():
#         # 训练模型
#         model.fit(X_train_scaled, y_train)
#         # 预测
#         y_pred = model.predict(X_test_scaled)
#         # 评估
#         acc = accuracy_score(y_test, y_pred)
#         print(f"[{name.ljust(15)}] 测试集准确率: {acc:.4f}")
        
#         # 可选：打印更详细的报告
#         # if name == "Random Forest":
#         #    print(classification_report(y_test, y_pred))


# # --- 4. 分别执行两个任务 ---

# # 任务一：有没有水 (二分类 0, 1)
# y_train_has = train_df['has_liquid'].values
# y_test_has = test_df['has_liquid'].values
# train_and_evaluate("是否有液体 (0/1)", y_train_has, y_test_has)

# # 任务二：有多少水 (四分类 0, 1, 2, 3)
# y_train_amount = train_df['amount_label'].values
# y_test_amount = test_df['amount_label'].values
# train_and_evaluate("液体量级 (0,1,2,3)", y_train_amount, y_test_amount)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# --- 1. 数据加载 ---
DATA_PATH = '/root/CV/饮料瓶/labels_picture/extracted_features.csv'
df = pd.read_csv(DATA_PATH)

train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

# 假设特征是从第5列开始的 (请根据实际CSV列名调整)
feature_cols = [c for c in df.columns if c not in ['image_name', 'has_liquid', 'amount_label', 'split']]

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. 新增：分析特征重要性 (基于随机森林) ---
def plot_feature_importance(X, y, title):
    # 使用随机森林评估特征
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    # 排序并转为 DataFrame 方便显示
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='skyblue')
    plt.title(f"Feature Importances - {title}")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f'feature_importance_{title}.png') # 保存图片到本地
    print(f"\n【{title}】特征重要性排序 (从高到低):")
    print(feat_imp.sort_values(by='Importance', ascending=False).to_string(index=False))

# --- 3. 稳健的训练与评估 (引入交叉验证和类别权重) ---
models = {
    # 增加 class_weight='balanced' 防止样本不均导致瞎猜
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42),
    # XGBoost 处理多分类需特别注意客观函数
    "XGBoost": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
}

def train_and_evaluate_robust(task_name, y_train, y_test):
    print(f"\n{'='*20} 任务: {task_name} {'='*20}")
    
    # 打印该任务的特征重要性
    plot_feature_importance(X_train_scaled, y_train, "Task_" + task_name.split()[0])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # 1. 交叉验证看稳健性
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        # 2. 真实测试集测试
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        
        print(f"[{name.ljust(15)}] 5折CV均值: {np.mean(cv_scores):.3f} | 测试集准度: {test_acc:.3f}")

y_train_has = train_df['has_liquid'].values
y_test_has = test_df['has_liquid'].values
train_and_evaluate_robust("Has_Liquid (0/1)", y_train_has, y_test_has)

y_train_amount = train_df['amount_label'].values
y_test_amount = test_df['amount_label'].values
train_and_evaluate_robust("Liquid_Amount (0/1/2/3)", y_train_amount, y_test_amount)
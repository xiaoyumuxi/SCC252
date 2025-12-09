import pandas as pd
import numpy as np
import joblib
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------
# 设定参数
# ----------------------------------------------------------------------
DATA_DIR = "data"
TARGET_COLUMN = 'Label'
# If True, train on only files that contain 'Friday' in the filename.
# Set to False to include all CSV files in `data/` (recommended when you need
# multiple attack types present in training data).
USE_ONLY_FRIDAY = False

# ----------------------------------------------------------------------
# 1. 数据加载与清洗
# ----------------------------------------------------------------------
print("--- 1. 数据加载与清洗 ---")

# 获取data目录下所有CSV文件
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("在data目录中未找到任何CSV文件")

# 选择要训练的文件（默认为全量文件，除非显式要求只用 Friday）
if USE_ONLY_FRIDAY:
    selected_files = [f for f in csv_files if 'Friday' in f]
    if not selected_files:
        raise FileNotFoundError("在data目录中未找到包含'Friday'的文件")
    print("正在合并以下周五文件:")
else:
    selected_files = csv_files
    print("正在合并以下所有CSV文件:")

frames = []
for file in selected_files:
    print(f"  - {file}")
    file_path = os.path.join(DATA_DIR, file)
    frame = pd.read_csv(file_path, low_memory=False)
    frames.append(frame)

df = pd.concat(frames, ignore_index=True)
df.columns = df.columns.str.strip().str.replace(' ', '_')

# 数值列筛选
numeric_cols = df.select_dtypes(include=np.number).columns

# 处理缺失值 (NaN)
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# 处理无穷值 (Infinity)
for col in numeric_cols:
    df[col].replace([np.inf], df[col][np.isfinite(df[col])].max(), inplace=True)
    df[col].replace([-np.inf], df[col][np.isfinite(df[col])].min(), inplace=True)

# ----------------------------------------------------------------------
# 快速统计目标列的分布（帮助诊断是否只有单一类别）
# ----------------------------------------------------------------------
if TARGET_COLUMN not in df.columns:
    raise KeyError(f"找不到目标列 '{TARGET_COLUMN}'，请检查数据文件的列名是否一致")

label_counts = df[TARGET_COLUMN].astype(str).str.strip().value_counts(dropna=False)
print("\n标签分布（合并后的样本）：")
print(label_counts.to_string())

# ----------------------------------------------------------------------
# 2. 标签编码 (Label Encoding)
# ----------------------------------------------------------------------
le = LabelEncoder()
df[TARGET_COLUMN + '_Encoded'] = le.fit_transform(df[TARGET_COLUMN].astype(str))

# ----------------------------------------------------------------------
# 3. 划分数据集
# ----------------------------------------------------------------------

X = df.drop(columns=[TARGET_COLUMN, TARGET_COLUMN + '_Encoded'])
y = df[TARGET_COLUMN + '_Encoded']

# 这一步保存了训练使用的特征顺序
FEATURE_COLUMNS = X.columns.tolist()

# 划分训练集和测试集
# 如果只有一个类别，不能使用 stratify（会抛错）。只有在多类别时启用 stratify。
strat = y if len(np.unique(y)) > 1 else None
if strat is None:
    print("警告：训练数据中仅包含单一标签，已禁用 stratify。模型将学习到多数类。")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strat)

# ----------------------------------------------------------------------
# 4. 特征标准化 (Fit and Transform)
# ----------------------------------------------------------------------
# 特征标准化
scaler = StandardScaler()
# 拟合 (fit) 缩放器，记住训练集的均值和标准差
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Also scale test set for evaluation
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

# ----------------------------------------------------------------------
# 5. 训练模型并保存
# ----------------------------------------------------------------------
print("--- 2. 开始训练模型 ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"模型评估结果:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")

# 显示各类别的样本数量
unique_labels = le.inverse_transform(np.unique(y))
label_counts = {}
for label in unique_labels:
    count = np.sum(df[TARGET_COLUMN] == label)
    label_counts[label] = count
    
print("\\n各攻击类型样本数量:")
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {label}: {count}")

print("--- 模型训练完成 ---")

# **保存所有必要的组件**
joblib.dump(rf_model, 'ddos_rf_model.joblib')
joblib.dump(scaler, 'ddos_scaler.joblib')
joblib.dump(le, 'ddos_label_encoder.joblib')
joblib.dump(FEATURE_COLUMNS, 'ddos_feature_columns.joblib') # 必须保存特征顺序

print("\\n✅ 所有组件已保存，可以运行预测脚本了。")
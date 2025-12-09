import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------
# 设定参数
# ----------------------------------------------------------------------
FILE_PATH = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
TARGET_COLUMN = 'Label'

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
# ----------------------------------------------------------------------
# 1. 数据加载与清洗 (与您之前确认的清洗逻辑一致)
# ----------------------------------------------------------------------
print("--- 1. 数据加载与清洗 ---")
df = pd.read_csv(FILE_PATH, low_memory=False)
df.columns = df.columns.str.strip().str.replace(' ', '_')

# 数值列筛选
numeric_cols = df.select_dtypes(include=np.number).columns

# 处理缺失值 (NaN)
for col in numeric_cols:
    median_val = df[col].median()
    df.loc[:, col] = df[col].fillna(median_val)

# 处理无穷值 (Infinity)
for col in numeric_cols:
    finite_mask = np.isfinite(df[col])
    if finite_mask.any():
        max_finite = df.loc[finite_mask, col].max()
        min_finite = df.loc[finite_mask, col].min()
    else:
        max_finite = 0
        min_finite = 0

    # 替换无穷值
    df.loc[df[col] == np.inf, col] = max_finite
    df.loc[df[col] == -np.inf, col] = min_finite

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ----------------------------------------------------------------------
# 4. 特征标准化 (Fit and Transform)
# ----------------------------------------------------------------------
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
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
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

print("--- 模型训练完成 ---")

# **保存所有必要的组件**
joblib.dump(rf_model, f'{MODEL_DIR}/ddos_rf_model.joblib')
joblib.dump(scaler, f'{MODEL_DIR}/ddos_scaler.joblib')
joblib.dump(le, f'{MODEL_DIR}/ddos_label_encoder.joblib')
joblib.dump(FEATURE_COLUMNS, f'{MODEL_DIR}/ddos_feature_columns.joblib')

print("\n✅ 所有组件已保存，可以运行预测脚本了。")
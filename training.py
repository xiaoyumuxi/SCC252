import joblib
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
# 设置文件路径
file_path = './data/Wednesday-workingHours.pcap_ISCX.csv'

print("正在读取数据...")

# 1. 读取 CSV 文件
try:
    df = pd.read_csv(file_path)
    print(f"数据读取成功，原始形状: {df.shape}")
except FileNotFoundError:
    print("错误：未找到文件，请检查文件路径。")
    exit()

# --- 数据清理预备步骤 ---
df.columns = df.columns.str.strip()

# 2. 清理数据 (删除包含 NaN 或 Infinity 的行)
print("正在清理数据...")

# 将无穷大 (inf) 和负无穷大 (-inf) 替换为 NaN (空值)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 删除包含 NaN 的行
df.dropna(inplace=True)

print(f"清理后形状: {df.shape}")

# --- 多分类标签编码 ---
print("正在进行多分类标签编码...")

# 查看一下原始的标签都有哪些
print("原始标签类别:", df['Label'].unique())

# 3. 使用 LabelEncoder 将字符串标签转换为 0, 1, 2, 3...
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# 保存映射关系
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\n标签映射关系:")
for label, num in label_mapping.items():
    print(f"  {label} -> {num}")

# 查看编码后的分布
print("\n编码后的标签分布:")
print(df['Label'].value_counts())

# 4. 分离特征 (X) 和 标签 (y)
y = df['Label']
X = df.drop('Label', axis=1)

FEATURE_COLUMNS = X.columns.tolist()

# 5. 划分训练集和测试集
print("\n正在划分训练集和测试集 (Stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 保证切分后的类别比例与原始数据一致
)

# 6. 特征缩放 (StandardScaler)
print("正在进行特征标准化...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n多分类预处理完成！")
print("特征标准化完成。")

# --- 5. 模型训练 ---
print("\n--- 步骤 5: 训练随机森林分类器 (Random Forest) ---")

# 实例化模型-随机森林 - 使用 RandomForestClassifier
# n_estimators=100 表示使用 100 棵决策树
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 在缩放后的训练数据上进行训练
rf_model.fit(X_train, y_train)

print("模型训练完成。")

# --- 6. 模型评估 ---
print("\n--- 步骤 6: 模型评估 ---")

# 使用测试集进行预测
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# 使用 weighted average 以适应多分类和可能的不平衡数据
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
# 将 0 视为负类 (Negative)，1 视为正类 (Positive)
# TNR (特异度) 是模型正确预测负类的能力，相当于负类的召回率
recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
# 2. 类别 0 的召回率即为特异度 (TNR)
# 类别 0 对应正常流量 ('BENIGN')，在这里被认定为是假类
tnr_score = recalls[0]
FPR = 1.0 - tnr_score
# 1. 获取模型对测试集的概率输出 (AUC 必需)
y_scores = rf_model.predict_proba(X_test)

# 2. 对真实标签进行二值化 (One-Hot 编码) 以适应 OvR 策略
classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)

# 3. 获取每个类别的支持度 (样本数), 用于计算加权平均
support = y_test.value_counts().sort_index().values
total_support = np.sum(support)

roc_auc = dict()
weighted_auc_sum = 0

for i in range(len(classes)):
    # OvR 策略：计算每个类别的 AUC
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr, tpr)

    # 计算加权和
    weight = support[i] / total_support
    weighted_auc_sum += roc_auc[i] * weight

auc_weighted = weighted_auc_sum


# 打印对应的数据集合
print(f"测试集整体准确率: {accuracy:.4f}")
print(f"测试集整体精确度: {precision:.4f}")
print("---------------------------------")
print(f"测试集[BENIGN]召回率: {recall:.4f}")
print(f"测试集假阳性率: {FPR:.4f}")
print(f"测试集AUC: {auc_weighted:.4f}")

# 存入性能指标显示
performance_metrics = {
    "accuracy": f"{accuracy:.4f}",
    "precision": f"{precision:.4f}",
    "recall": f"{recall:.4f}",
    "FPR": f"{FPR:.4f}",
    "auc": f"{auc_weighted:.4f}"
}

metrics_filename = './models/ddos_performance.json'
with open(metrics_filename, 'w') as f:
    json.dump(performance_metrics, f)

print(f"- 性能指标: {metrics_filename}")

# 打印详细的分类报告
print("\n--- 多分类详细评估报告 ---")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# --- 7. 模型和预处理器保存 ---
print("\n--- 步骤 7: 模型和预处理器保存 ---")

model_filename = './models/ddos_rf_model.joblib'
scaler_filename = './models/ddos_scaler.joblib'
encoder_filename = './models/ddos_label_encoder.joblib'
feature_col = './models/ddos_feature_columns.joblib'

# 使用 joblib 保存训练好的模型 (rf_model) 和预处理器
joblib.dump(rf_model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(le, encoder_filename)
joblib.dump(FEATURE_COLUMNS, feature_col)

print(f"🎉 任务完成！")
print(f"模型和预处理器已保存为:\n- 模型: {model_filename} (内容为随机森林)\n- 缩放器: {scaler_filename}\n- 编码器: {encoder_filename}\n- 特征行列: {feature_col}")
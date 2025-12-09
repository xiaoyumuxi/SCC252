import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# 设置文件路径 (请修改为你实际的文件名)
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

# --- 3. 多分类标签编码 ---
print("正在进行多分类标签编码...")

# 查看一下原始的标签都有哪些
print("原始标签类别:", df['Label'].unique())

# 使用 LabelEncoder 将字符串标签转换为 0, 1, 2, 3...
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# 【关键步骤】保存映射关系
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

# 【关键新增】捕获并保存特征顺序，以供 API 确保输入数据列顺序一致
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

#
# --- 5. 模型训练 ---
print("\n--- 步骤 5: 训练随机森林分类器 (Random Forest) ---")

# 实例化模型 - **这里是关键修改！** 使用 RandomForestClassifier
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
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# 打印整体准确率
print(f"测试集整体准确率: {accuracy:.4f}")
print(f"测试集精确度: {precision:.4f}")
print(f"测试集召回率: {recall:.4f}")
print(f"测试集 F1-Score: {f1:.4f}")

# 打印详细的分类报告
print("\n--- 多分类详细评估报告 ---")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# --- 7. 模型和预处理器保存 ---
print("\n--- 步骤 7: 模型和预处理器保存 ---")

# **保留原始决策树的文件名，以确保 API 兼容性**
# 您的 API 仍然会加载名为 ids_decision_tree_model.joblib 的文件，
# 但实际上它是一个随机森林模型。
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
print(f"模型和预处理器已保存为:\n- 模型: {model_filename} (内容为随机森林)\n- 缩放器: {scaler_filename}\n- 编码器: {encoder_filename}")
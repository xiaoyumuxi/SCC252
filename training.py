import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # 引入 LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# 设置文件路径 (请修改为你实际的文件名)
file_path = './data/Wednesday-workingHours.pcap_ISCX.csv'

print("正在读取数据...")

# 1. 读取 CSV 文件
# engine='python' 可以避免某些文件名或格式导致的解析错误
try:
    df = pd.read_csv(file_path)
    print(f"数据读取成功，原始形状: {df.shape}")
except FileNotFoundError:
    print("错误：未找到文件，请检查文件路径。")
    exit()

# --- 数据清理预备步骤 ---
# CICIDS2017 数据集的一个常见问题是列名周围有空格（例如 " Label" 而不是 "Label"）
# 这行代码会去除所有列名首尾的空格，防止后面报错
df.columns = df.columns.str.strip()

# 2. 清理数据 (删除包含 NaN 或 Infinity 的行)
print("正在清理数据...")

# 将无穷大 (inf) 和负无穷大 (-inf) 替换为 NaN (空值)
# 这一步非常重要，因为网络流量数据中常因除以零出现无穷大值
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

# 【关键步骤】保存映射关系，方便以后查看哪个数字代表哪种攻击
#这一步很重要，否则训练完你会忘了 "2" 到底代表 DDoS 还是 Bot
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

# 5. 划分训练集和测试集
# 【重要修改】增加 stratify=y
# 原因：网络流量数据通常极度不平衡（正常流量很多，某些攻击很少）。
# stratify=y 确保训练集和测试集中，各类攻击的比例保持一致，
# 避免出现测试集中某种罕见攻击完全没出现的情况。
print("\n正在划分训练集和测试集 (Stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 保证切分后的类别比例与原始数据一致
)

# 6. 特征缩放 (StandardScaler) - 与之前相同
print("正在进行特征标准化...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n多分类预处理完成！")
print("特征标准化完成。")

# --- 5. 模型训练 ---
print("\n--- 步骤 5: 训练决策树分类器 (Decision Tree) ---")

# 实例化模型
dt_model = DecisionTreeClassifier(random_state=42)

# 在缩放后的训练数据上进行训练
dt_model.fit(X_train, y_train)

print("模型训练完成。")

# --- 6. 模型评估 ---
print("\n--- 步骤 6: 模型评估 ---")

# 使用测试集进行预测
y_pred = dt_model.predict(X_test)

# 打印整体准确率
print(f"测试集整体准确率: {accuracy_score(y_test, y_pred):.4f}")

# 打印详细的分类报告 (注意 F1-Score 对于类别不平衡问题更重要)
print("\n--- 多分类详细评估报告 ---")
# target_names 传入原始标签名，让报告更易读
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# --- 7. 模型和预处理器保存 ---
print("\n--- 步骤 7: 模型和预处理器保存 ---")

# 定义保存文件名
model_filename = './models/ids_decision_tree_model.joblib'
scaler_filename = './models/ids_standard_scaler.joblib'
encoder_filename = './models/ids_label_encoder.joblib'

# 使用 joblib 保存训练好的模型和预处理器
joblib.dump(dt_model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(le, encoder_filename)

print(f"🎉 任务完成！")
print(f"模型和预处理器已保存为:\n- 模型: {model_filename}\n- 缩放器: {scaler_filename}\n- 编码器: {encoder_filename}")

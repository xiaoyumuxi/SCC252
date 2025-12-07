import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# 假设您已将清洗后的数据保存为 'cleaned_data_scaled.csv'
file_path = 'cleaned_data_scaled.csv'
df_cleaned = pd.read_csv(file_path)

# 目标标签列名（您在清洗时创建的）
LABEL_COLUMN = 'Label_Encoded'

# X 是特征 (Features)，除了标签列以外的所有列
X = df_cleaned.drop(columns=[LABEL_COLUMN])

# y 是标签 (Labels)
y = df_cleaned[LABEL_COLUMN]

print(f"特征 (X) 维度: {X.shape}")
print(f"标签 (y) 维度: {y.shape}")

# random_state 用于确保每次运行代码时，划分结果都是一致的（可复现性）
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y # 确保训练集和测试集的 DDOs/BENIGN 比例一致
)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# 初始化随机森林分类器
# n_estimators=100 表示构建 100 棵决策树
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("\n--- 开始训练随机森林模型 ---")
# 使用训练数据 (X_train, y_train) 来拟合模型
rf_model.fit(X_train, y_train)
print("--- 模型训练完成 ---")

# 使用测试数据 (X_test) 进行预测
y_pred = rf_model.predict(X_test)

# 打印整体准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n整体准确率 (Accuracy): {accuracy:.4f}")

# 打印详细的分类报告
# P: 精度 (Precision) - 模型预测为 DDoS 的样本中，真正是 DDoS 的比例
# R: 召回率 (Recall) - 所有的 DDoS 样本中，模型正确识别出来的比例
print("\n详细分类报告:\n", classification_report(y_test, y_pred))

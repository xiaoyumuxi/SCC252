import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 文件路径
file_path = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

try:
    # 使用 low_memory=False 更好地推断数据类型，防止后续类型错误
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"错误：文件未找到在路径: {file_path}")
    exit()

# 1. 清理列名
df.columns = df.columns.str.strip().str.replace(' ', '_')

# 2. 处理缺失值 (NaN)
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# 3. 处理无穷值 (Infinity)
# 仅对数值列进行 isinf 检查，避免 TypeError
numeric_df = df[numeric_cols]
inf_cols = numeric_df.columns[np.isinf(numeric_df).any()].tolist()

for col in inf_cols:
    # 使用有限值的最大值/最小值替换无穷值
    max_finite = df[col][np.isfinite(df[col])].max()
    df[col].replace([np.inf], max_finite, inplace=True)

    min_finite = df[col][np.isfinite(df[col])].min()
    df[col].replace([-np.inf], min_finite, inplace=True)

# 修复您之前遇到的错误：只检查数值列的无穷值
inf_count = np.isinf(df[numeric_cols]).sum().sum()
if inf_count > 0:
    # 理论上执行替换后这里应该是0，如果大于0表示替换逻辑有问题
    print(f"警告：处理后仍存在 {inf_count} 个无穷值。")


# 4. 标签编码
target_column = 'Label'
le = LabelEncoder()
# 确保目标列是字符串类型，防止编码错误
df[target_column + '_Encoded'] = le.fit_transform(df[target_column].astype(str))
df.drop(columns=[target_column], inplace=True)

# =========================================================================
# 5. 特征缩放 (Feature Scaling) - 训练集必需
# =========================================================================

# 识别所有数值特征，排除编码后的标签列
numerical_features = df.select_dtypes(include=np.number).columns.tolist()
label_encoded_col = target_column + '_Encoded'
if label_encoded_col in numerical_features:
    numerical_features.remove(label_encoded_col)

# 初始化并执行标准化
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\n✅ 数据清洗和标准化完成。")

# =========================================================================
# 6. 保存数据 (推荐 Parquet 格式)
# =========================================================================

output_file_parquet = 'cleaned_data_scaled.parquet'
df.to_parquet(output_file_parquet, index=False)
print(f"数据已保存到 {output_file_parquet} (Parquet 格式，推荐用于训练集)")

# 如果仍需要 CSV 格式
output_file_csv = 'cleaned_data_scaled.csv'
df.to_csv(output_file_csv, index=False)
print(f"数据也已保存到 {output_file_csv} (CSV 格式)")
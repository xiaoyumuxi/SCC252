import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------
# 1. 加载所有保存的组件
# ----------------------------------------------------------------------
try:
    # 加载模型、缩放器、标签编码器和特征顺序
    MODEL = joblib.load('ddos_rf_model.joblib')
    SCALER = joblib.load('ddos_scaler.joblib')
    LE = joblib.load('ddos_label_encoder.joblib')
    FEATURE_COLUMNS = joblib.load('ddos_feature_columns.joblib')
    print("模型组件加载成功，准备就绪。")
except FileNotFoundError:
    print("错误：无法加载模型文件。请确保已运行 'train_and_save.py' 脚本。")
    exit()


# ----------------------------------------------------------------------
# 2. 核心预测函数
# ----------------------------------------------------------------------
def get_prediction(raw_input_data: list) -> dict:
    """
    对一组新的原始数据进行预处理和预测。

    参数:
        raw_input_data: 包含 78 个特征值的列表，顺序必须与训练时一致。

    返回:
        包含预测结果和置信度的字典。
    """

    # 检查输入数据长度
    if len(raw_input_data) != len(FEATURE_COLUMNS):
        return {
            "status": "error",
            "message": f"输入特征数量错误。需要 {len(FEATURE_COLUMNS)} 个特征，但接收到 {len(raw_input_data)} 个。"
        }

    # 转换为 DataFrame (必须保持列顺序)
    new_df = pd.DataFrame([raw_input_data], columns=FEATURE_COLUMNS)

    # 1. 清洗 (处理NaN/Inf)
    # 在生产环境中，这里需要用训练集的中位数和max/min进行填充和替换。
    # 为简化，这里用 0 填充 NaN/Inf，假设输入数据大部分是数值。
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_df.fillna(0, inplace=True)

    # 2. 特征缩放 (关键步骤：使用 SCALER.transform)
    data_scaled = SCALER.transform(new_df)

    # 🌟 修正点：将缩放后的 NumPy 数组重新转换为带有特征名称的 DataFrame 🌟
    data_scaled_df = pd.DataFrame(data_scaled, columns=FEATURE_COLUMNS)  # 确保有列名

    # 3. 模型预测
    prediction_encoded = MODEL.predict(data_scaled_df)[0]  # 传入带有列名的 DataFrame
    # 4. 预测概率 (置信度)
    prediction_proba = MODEL.predict_proba(data_scaled_df)[0]

    # 5. 反向映射标签
    prediction_label = LE.inverse_transform([prediction_encoded])[0]

    # 找出最高概率和对应标签
    max_proba = np.max(prediction_proba)

    # 6. 返回结果
    return {
        "status": "success",
        "predicted_label": prediction_label,
        "confidence": float(max_proba),
        "encoded_value": int(prediction_encoded)
    }


# ----------------------------------------------------------------------
# 3. 示例调用 (模拟网站 POST 请求)
# ----------------------------------------------------------------------

# ⚠️ 注意：这是一个包含 78 个特征值的示例数据。
# 必须确保您的网站发出的数据是相同长度和顺序。
SAMPLE_RAW_DATA = [
    54865, 3, 2, 0, 12, 0, 6, 6, 6.0, 0.0, 0, 0, 0.0, 0.0, 4000000.0,
    666666.6667, 3.0, 0.0, 3, 3, 3, 3.0, 0.0, 3, 3, 0, 0.0, 0.0, 0, 0,
    0, 0, 0, 0, 40, 0, 666666.6667, 0.0, 6, 6, 6.0, 0.0, 0.0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 9.0, 6.0, 0.0, 40, 0, 0, 0, 0, 0, 0, 2, 12, 0, 0,
    33, -1, 1, 20, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0
]

# 调用预测函数
result = get_prediction(SAMPLE_RAW_DATA)

print("\n--- 模拟网站/API 接口返回结果 ---")
print(pd.Series(result).to_json(indent=4))  # 以 JSON 格式打印结果

if result['status'] == 'success':
    if result['predicted_label'].upper() == 'BENIGN':
        print("\n🟢 预测：流量正常 (BENIGN)")
    else:
        print(f"\n🔴 预测：检测到恶意攻击！类型为 {result['predicted_label']}")
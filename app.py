import os
import logging
from datetime import datetime
from threading import Lock
import json
import pandas as pd
import numpy as np
import joblib
import sqlite3
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

# --- 配置部分 ---
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 上传最大 500MB
MAX_ALERTS = 50  # 内存中保存的最大警报数量
DB_FILE = 'ddos_detection.db'

# 全局变量 (模型组件)
MODEL = None
SCALER = None
LE = None
FEATURE_COLUMNS = None

# 模型文件路径 (与 trainning.py 保持一致)
MODEL_PATH = './models/ddos_rf_model.joblib'
SCALER_PATH = './models/ddos_scaler.joblib'
ENCODER_PATH = './models/ddos_label_encoder.joblib'
FEATURE_COLS_PATH = './models/ddos_feature_columns.joblib'
PERFORMANCE_PATH = './models/ddos_performance.json'

# 内存存储 (警报)
alerts = []
alerts_lock = Lock()

# 攻击样本库（用于 /api/stream 模拟攻击）
ATTACK_SAMPLE_LIBRARY = []          # [{'label': 'DoS Hulk', 'features': [...]}, ...]
attack_samples_lock = Lock()

# 从原始数据集中抽样构建攻击样本库的路径
ATTACK_DATASET_PATH = os.getenv(
    'ATTACK_DATASET_PATH',
    './data/Wednesday-workingHours.pcap_ISCX.csv'
)

# 频率统计时间窗（秒）
TIME_WINDOW_SECONDS = 10

# 性能指标 (示例初始值，训练后会更新)
PERFORMANCE_METRICS = {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "auc": 0.0
}


# ----------------------------------------------------------------------
# 1. 数据库初始化
# ----------------------------------------------------------------------
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        # 修复：添加 features_count 列，防止 INSERT 时报错
        c.execute('''CREATE TABLE IF NOT EXISTS detection_history
                     (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL,predicted_label TEXT NOT NULL,confidence REAL NOT NULL,threat_level TEXT NOT NULL,features_count INTEGER
                     )''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


init_db()


# ----------------------------------------------------------------------
# 2. 模型加载逻辑
# ----------------------------------------------------------------------
def load_model_components():
    """
    加载所有保存的模型组件
    """
    global MODEL, SCALER, LE, FEATURE_COLUMNS, PERFORMANCE_METRICS

    try:
        # 1. 加载模型二进制文件
        if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURE_COLS_PATH]):
            logger.warning("One or more model binary files not found.")
            return False

        MODEL = joblib.load(MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        LE = joblib.load(ENCODER_PATH)
        FEATURE_COLUMNS = joblib.load(FEATURE_COLS_PATH)

        # 2. 加载性能指标 (新增逻辑)
        if os.path.exists(PERFORMANCE_PATH):
            try:
                with open(PERFORMANCE_PATH, 'r') as f:
                    PERFORMANCE_METRICS = json.load(f)
                logger.info("✅ Performance metrics loaded.")
            except Exception as e:
                logger.warning(f"⚠️ Found metrics file but failed to load: {e}")
        else:
            logger.warning("⚠️ No performance metrics file found (ddos_performance.json). Metrics will be 0.")

        logger.info("✅ Model components loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model files: {e}")
        return False


# ----------------------------------------------------------------------
# 3. 核心预测与辅助函数
# ----------------------------------------------------------------------
def get_threat_level(label, confidence):
    # todo: 这一部分需要重写逻辑
    """根据标签和置信度确定威胁等级"""
    if label.upper() == 'BENIGN':
        return 'None'
    elif confidence > 0.9:
        return 'High'
    elif confidence > 0.7:
        return 'Medium'
    else:
        return 'Low'


def predict(raw_input_data):
    """
    核心预测逻辑，与 api.py 保持一致
    """
    if not FEATURE_COLUMNS:
        return {"status": "error", "message": "Model not loaded."}

    # 1. 检查特征数量
    if len(raw_input_data) != len(FEATURE_COLUMNS):
        return {
            "status": "error",
            "message": f"Feature mismatch. Expected {len(FEATURE_COLUMNS)}, got {len(raw_input_data)}."
        }

    try:
        # 2. 转换为 DataFrame (使用保存的列名)
        new_df = pd.DataFrame([raw_input_data], columns=FEATURE_COLUMNS)

        # 3. 数据清理 (替换 Inf/NaN 为 0，确保健壮性)
        new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_df.fillna(0, inplace=True)

        # 4. 特征缩放
        data_scaled = SCALER.transform(new_df)

        # 5. 预测
        # 使用 NumPy 数组（而非带列名的 DataFrame）传入模型，避免 scikit-learn 关于
        # "X has feature names, but RandomForestClassifier was fitted without feature names" 的警告。
        prediction_encoded = MODEL.predict(data_scaled)[0]
        prediction_proba = MODEL.predict_proba(data_scaled)[0]

        # 6. 解析结果
        prediction_label = LE.inverse_transform([prediction_encoded])[0]
        max_proba = np.max(prediction_proba)
        threat_level = get_threat_level(prediction_label, max_proba)

        return {
            "status": "success",
            "predicted_label": prediction_label,
            "confidence": float(max_proba),
            "encoded_value": int(prediction_encoded),
            "threat_level": threat_level
        }
    except Exception as e:
        logger.error(f"Prediction logic error: {e}")
        return {"status": "error", "message": str(e)}


def get_prediction(raw_input_data):
    """
    兼容性包装器：确保在调用预测前模型组件已加载。
    返回与原 `predict` 相同的字典结构。
    """
    # 如果模型或组件尚未加载，尝试加载一次
    if not all([MODEL, SCALER, LE, FEATURE_COLUMNS]):
        loaded = load_model_components()
        if not loaded:
            return {"status": "error", "message": "Model components not loaded and failed to load."}

    return predict(raw_input_data)

def build_attack_sample_library():
    """
    从原始数据集中，为每种攻击类型采样最多 5 条，构建攻击样本库。
    只使用 FEATURE_COLUMNS 中定义的特征，保证与模型输入一致。
    """
    global ATTACK_SAMPLE_LIBRARY

    with attack_samples_lock:
        # 已经构建过就直接返回
        if ATTACK_SAMPLE_LIBRARY:
            return True

        # 确保模型组件已加载，从而拿到 FEATURE_COLUMNS
        if not FEATURE_COLUMNS:
            loaded = load_model_components()
            if not loaded or not FEATURE_COLUMNS:
                logger.error("Cannot build attack sample library: FEATURE_COLUMNS not available.")
                return False

        # 检查数据集路径
        if not os.path.exists(ATTACK_DATASET_PATH):
            logger.error(f"Attack dataset file not found: {ATTACK_DATASET_PATH}")
            return False

        try:
            logger.info(f"Loading attack dataset from {ATTACK_DATASET_PATH} ...")
            df = pd.read_csv(ATTACK_DATASET_PATH)

            # 列名去空格，基础清洗
            df.columns = df.columns.str.strip()
            if 'Label' not in df.columns:
                logger.error("Dataset has no 'Label' column.")
                return False

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            # 过滤出攻击行（排除 BENIGN）
            df['Label'] = df['Label'].astype(str).str.strip()
            attack_df = df[df['Label'].str.upper() != 'BENIGN']

            if attack_df.empty:
                logger.error("No attack rows found in dataset.")
                return False

            library = []

            # 按攻击类型分组，每类最多 5 条
            for label, group in attack_df.groupby('Label'):
                sample_n = min(5, len(group))
                sampled = group.sample(n=sample_n, random_state=42)

                for _, row in sampled.iterrows():
                    features = []
                    for col in FEATURE_COLUMNS:
                        if col in sampled.columns:
                            val = row[col]
                            try:
                                val = float(val)
                            except Exception:
                                val = 0.0
                        else:
                            val = 0.0
                        features.append(val)

                    library.append({
                        "label": label,
                        "features": features
                    })

            ATTACK_SAMPLE_LIBRARY = library
            logger.info(
                f"Attack sample library built: {len(ATTACK_SAMPLE_LIBRARY)} samples "
                f"from {attack_df['Label'].nunique()} attack types."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to build attack sample library: {e}")
            return False

# ----------------------------------------------------------------------
# 4. 模型重训练逻辑
# ----------------------------------------------------------------------
def train_model_with_data(df, target_column='Label'):
    """
    使用上传的数据重新训练模型。
    
    返回值:
        字典 {'success': True/False, 'message': str, 'stats': dict}
        stats 包含: total_samples, label_distribution, new_labels_count
    """
    global PERFORMANCE_METRICS

    try:
        logger.info("Starting retraining process...")

        # 1. 简单清理
        df.columns = df.columns.str.strip()  # 去除列名空格

        # 检查是否存在 Label 列
        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found in CSV. Available columns: {', '.join(df.columns.tolist())}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg, 'stats': {}}

        # 处理 Inf 和 NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)  # 这里选择直接丢弃，保证训练质量

        # 2. 标签编码
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column].astype(str))

        # 收集统计信息
        unique_labels = set(le.classes_)
        label_dist = df[target_column].value_counts().to_dict()
        old_labels = set(LE.classes_) if LE else set()
        new_labels = unique_labels - old_labels
        stats = {
            'total_samples': len(df),
            'label_distribution': {le.inverse_transform([k])[0]: v for k, v in label_dist.items()},
            'unique_labels': list(unique_labels),
            'new_labels': list(new_labels),
            'new_labels_count': len(new_labels)
        }
        logger.info(f"Data statistics: {stats}")

        # 3. 分离特征和标签
        X = df.drop(columns=[target_column])
        y = df[target_column]

        feature_columns_list = X.columns.tolist()

        # 4. 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 5. 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 转换回 DataFrame 格式以保持一致性
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns_list)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns_list)

        # 6. 训练 Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)

        # 7. 评估
        y_pred = rf_model.predict(X_test_scaled)

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

        PERFORMANCE_METRICS = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "FPR": 1.0 - recall_score(y_test, y_pred, average=None, zero_division=0)[0],
            "auc": auc_weighted
        }

        # 8. 保存所有组件 (覆盖旧文件)
        os.makedirs('./models', exist_ok=True)
        joblib.dump(rf_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(le, ENCODER_PATH)
        joblib.dump(feature_columns_list, FEATURE_COLS_PATH)

        logger.info(f"Retraining complete. Accuracy: {PERFORMANCE_METRICS['accuracy']:.4f}")
        # 保存性能指标
        with open(PERFORMANCE_PATH, 'w') as f:
            json.dump(PERFORMANCE_METRICS, f)
        logger.info(f"Performance metrics saved to {PERFORMANCE_PATH}")

        return {'success': True, 'message': f'Retraining complete. Accuracy: {PERFORMANCE_METRICS["accuracy"]:.4f}', 'stats': stats}

    except Exception as e:
        logger.error(f"Train model with data failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': str(e), 'stats': {}}


# ----------------------------------------------------------------------
# 5. API 路由接口
# ----------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """
    预测接口
    POST Body: {"features": [v1, v2, ...]}
    """
    if not all([MODEL, SCALER, LE, FEATURE_COLUMNS]):
        return jsonify({"status": "error", "message": "Model not fully loaded."}), 503

    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"status": "error", "message": "Missing 'features' in JSON."}), 400

        features = data['features']

        # 调用预测
        result = predict(features)

        if result['status'] == 'success':
            # 保存到数据库
            try:
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute(
                    "INSERT INTO detection_history (timestamp, predicted_label, confidence, threat_level, features_count) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     result['predicted_label'],
                     result['confidence'],
                     result['threat_level'],
                     len(features)))
                conn.commit()
                conn.close()
            except Exception as db_e:
                logger.error(f"Database insert error: {db_e}")

            # 处理警报 (非正常流量)
            if result['predicted_label'].upper() != 'BENIGN':
                with alerts_lock:
                    alert = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": result['predicted_label'],
                        "confidence": result['confidence'],
                        "level": result['threat_level']
                    }
                    alerts.append(alert)
                    # 修复：使用全局变量 MAX_ALERTS
                    if len(alerts) > MAX_ALERTS:
                        alerts.pop(0)

        return jsonify(result)

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """获取最新警报"""
    # todo: 这里需要增加有关警报的逻辑，需要统计10s内出现了多少个非BEGINE的攻击，1~5低5~7中7以上高
    with alerts_lock:
        return jsonify(list(reversed(alerts)))


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取历史记录"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "SELECT timestamp, predicted_label, confidence, threat_level FROM detection_history ORDER BY timestamp DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                "timestamp": row[0],
                "type": row[1],
                "confidence": row[2],
                "level": row[3],  # 直接使用 threat_level
                "threat_level": row[3]
            })
        return jsonify(history)
    except Exception as e:
        return jsonify([]), 500


@app.route('/api/performance', methods=['GET'])
def get_performance():
    """获取模型性能"""
    return jsonify(PERFORMANCE_METRICS)
@app.route('/api/stream', methods=['GET'])
def get_attack_stream_sample():
    """
    从攻击样本库中随机选取一条样本，作为模拟攻击流。
    可选查询参数: ?label=XXX  指定某一种攻击类型。

    额外返回:
    - attack_frequency: 在 TIME_WINDOW_SECONDS 内的攻击次数（随机模拟）
    - frequency_level: Low / Medium / High
    """
    try:
        # 如果还没构建过攻击样本库，先构建一次
        if not ATTACK_SAMPLE_LIBRARY:
            ok = build_attack_sample_library()
            if not ok or not ATTACK_SAMPLE_LIBRARY:
                return jsonify({
                    "status": "error",
                    "message": "Attack sample library not available. Check dataset and logs."
                }), 500

        # 可选：按 label 过滤指定攻击类型
        label_filter = request.args.get('label')
        candidates = ATTACK_SAMPLE_LIBRARY

        if label_filter:
            candidates = [s for s in ATTACK_SAMPLE_LIBRARY if s['label'] == label_filter]
            if not candidates:
                return jsonify({
                    "status": "error",
                    "message": f"No samples found for label '{label_filter}'."
                }), 404

        # 1) 从候选库中随机选取一条攻击样本
        idx = np.random.randint(0, len(candidates))
        chosen = candidates[idx]

        features = chosen["features"]
        label = chosen["label"]

        names = FEATURE_COLUMNS if FEATURE_COLUMNS else [
            f"f_{i}" for i in range(len(features))
        ]

        # 2) 随机生成攻频：TIME_WINDOW_SECONDS 内的攻击次数，这里模拟 1~10 次
        attack_frequency = int(np.random.randint(1, 11))  # [1, 10]

        # 3) 根据频率划分等级：
        #    <5 次   → Low
        #    5~6 次  → Medium
        #    ≥7 次   → High
        if attack_frequency < 5:
            frequency_level = "Low"
        elif attack_frequency < 7:
            frequency_level = "Medium"
        else:
            frequency_level = "High"

        return jsonify({
            "status": "success",
            "features": features,
            "feature_names": names,
            "label": label,                        # 真正的攻击类型名称（如 DoS Hulk）
            "attack_frequency": attack_frequency,  # TIME_WINDOW_SECONDS 内攻击次数
            "frequency_level": frequency_level,    # Low / Medium / High
            "time_window_seconds": TIME_WINDOW_SECONDS
        })

    except Exception as e:
        logger.error(f"/api/stream error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/stream', methods=['GET'])
def get_sample_data():
    # todo: 这里需要模拟一个攻击流，已经讲过了的
    # 示例数据 (长度需要与模型一致)
    sample = [0] * (len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 78)
    names = FEATURE_COLUMNS if FEATURE_COLUMNS else [f"f_{i}" for i in range(78)]
    return jsonify({"features": sample, "feature_names": names})


@app.route('/api/random', methods=['GET'])
def get_random_data():
    """生成随机数据"""
    count = len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 78
    data = np.random.uniform(0, 1000, count).tolist()
    names = FEATURE_COLUMNS if FEATURE_COLUMNS else [f"f_{i}" for i in range(count)]
    return jsonify({"features": data, "feature_names": names})


@app.route('/api/upload-and-retrain', methods=['POST'])
def upload_and_retrain():
    """
    上传 CSV 并重训练
    """
    try:
        if 'files' not in request.files:
            return jsonify({"status": "error", "message": "No files part"}), 400

        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"status": "error", "message": "No selected file"}), 400

        dfs = []
        for file in files:
            if file and file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception as e:
                    return jsonify({"status": "error", "message": f"Error reading {file.filename}: {e}"}), 400

        if not dfs:
            return jsonify({"status": "error", "message": "No valid CSV files found"}), 400

        full_df = pd.concat(dfs, ignore_index=True)

        # 启动训练
        result = train_model_with_data(full_df)

        if result['success']:
            # 重新加载模型
            if load_model_components():
                return jsonify({
                    "status": "success",
                    "message": result['message'],
                    "stats": result['stats'],
                    "performance": PERFORMANCE_METRICS
                })
            else:
                return jsonify({"status": "error", "message": "Training succeeded but reload failed."}), 500
        else:
            return jsonify({"status": "error", "message": result['message'], "details": result['stats']}), 400

    except Exception as e:
        logger.error(f"Retrain API error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ----------------------------------------------------------------------
# 程序入口
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 启动时加载模型
    if not load_model_components():
        logger.warning("⚠️ Warning: Model components could not be loaded at startup.")
        logger.warning("   Please ensure 'trainning.py' has been run and generated files in './models/'.")

    app.run(host='127.0.0.1', port=5000, debug=True)
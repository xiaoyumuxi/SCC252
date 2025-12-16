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

# --- Configuration ---
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# App config
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max upload size: 500MB
MAX_ALERTS = 50  # Max number of alerts stored in memory
DB_FILE = 'ddos_detection.db'

# Global variables (model components)
MODEL = None
SCALER = None
LE = None
FEATURE_COLUMNS = None

# Model file paths (consistent with trainning.py)
MODEL_PATH = './models/ddos_rf_model.joblib'
SCALER_PATH = './models/ddos_scaler.joblib'
ENCODER_PATH = './models/ddos_label_encoder.joblib'
FEATURE_COLS_PATH = './models/ddos_feature_columns.joblib'
PERFORMANCE_PATH = './models/ddos_performance.json'

# In-memory storage (alerts)
alerts = []
alerts_lock = Lock()

# Attack sample library (used by /api/stream to simulate attack traffic)
ATTACK_SAMPLE_LIBRARY = []          # [{'label': 'DoS Hulk', 'features': [...]}, ...]
attack_samples_lock = Lock()

# Path to sample attack data from the raw dataset
ATTACK_DATASET_PATH = os.getenv(
    'ATTACK_DATASET_PATH',
    './data/Wednesday-workingHours.pcap_ISCX.csv'
)

# Frequency counting time window (seconds)
TIME_WINDOW_SECONDS = 10

# Performance metrics (initial example values; updated after training)
PERFORMANCE_METRICS = {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "auc": 0.0
}


# ----------------------------------------------------------------------
# 1. Database initialization
# ----------------------------------------------------------------------
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        # Fix: Add features_count column to prevent INSERT errors
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
# 2. Model loading logic
# ----------------------------------------------------------------------
def load_model_components():
    """
    Load all saved model components.
    """
    global MODEL, SCALER, LE, FEATURE_COLUMNS, PERFORMANCE_METRICS

    try:
        # 1. Load model binary files
        if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURE_COLS_PATH]):
            logger.warning("One or more model binary files not found.")
            return False

        MODEL = joblib.load(MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        LE = joblib.load(ENCODER_PATH)
        FEATURE_COLUMNS = joblib.load(FEATURE_COLS_PATH)

        # 2. Load performance metrics (additional logic)
        if os.path.exists(PERFORMANCE_PATH):
            try:
                with open(PERFORMANCE_PATH, 'r') as f:
                    PERFORMANCE_METRICS = json.load(f)
                logger.info("Performance metrics loaded.")
            except Exception as e:
                logger.warning(f"Found metrics file but failed to load: {e}")
        else:
            logger.warning("No performance metrics file found (ddos_performance.json). Metrics will be 0.")

        logger.info("Model components loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading model files: {e}")
        return False


# ----------------------------------------------------------------------
# 3. Core prediction logic and helper functions
# ----------------------------------------------------------------------
def get_threat_level(label, confidence):
    # TODO: This part needs a rewritten logic
    """Determine threat level based on label and confidence."""
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
    Core prediction logic (kept consistent with api.py).
    """
    if not FEATURE_COLUMNS:
        return {"status": "error", "message": "Model not loaded."}

    # 1. Check feature length
    if len(raw_input_data) != len(FEATURE_COLUMNS):
        return {
            "status": "error",
            "message": f"Feature mismatch. Expected {len(FEATURE_COLUMNS)}, got {len(raw_input_data)}."
        }

    try:
        # 2. Convert to DataFrame (using saved column names)
        new_df = pd.DataFrame([raw_input_data], columns=FEATURE_COLUMNS)

        # 3. Data cleaning (replace Inf/NaN with 0 to ensure robustness)
        new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_df.fillna(0, inplace=True)

        # 4. Feature scaling
        data_scaled = SCALER.transform(new_df)

        # 5. Predict
        # Pass NumPy array (not a DataFrame with feature names) to avoid scikit-learn warning:
        # "X has feature names, but RandomForestClassifier was fitted without feature names"
        prediction_encoded = MODEL.predict(data_scaled)[0]
        prediction_proba = MODEL.predict_proba(data_scaled)[0]

        # 6. Parse results
        prediction_label = LE.inverse_transform([prediction_encoded])[0]
        max_proba = np.max(prediction_proba)
        threat_level = get_threat_level(prediction_label, max_proba)

        # Get probability distribution (top 5 probabilities above 1%)
        proba_dict = {}
        classes = LE.classes_
        for i, prob in enumerate(prediction_proba):
            if prob > 0.01:  # Only show probabilities > 1%
                proba_dict[classes[i]] = float(prob)

        # Sort by probability descending, take top 5
        top_probabilities = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:5])

        return {
            "status": "success",
            "predicted_label": prediction_label,
            "confidence": float(max_proba),
            "encoded_value": int(prediction_encoded),
            "threat_level": threat_level,
            "probabilities": top_probabilities,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Prediction logic error: {e}")
        return {"status": "error", "message": str(e)}


def get_prediction(raw_input_data):
    """
    Compatibility wrapper: ensures model components are loaded before predicting.
    Returns the same dict structure as `predict`.
    """
    # If model/components not loaded, try loading once
    if not all([MODEL, SCALER, LE, FEATURE_COLUMNS]):
        loaded = load_model_components()
        if not loaded:
            return {"status": "error", "message": "Model components not loaded and failed to load."}

    return predict(raw_input_data)

def build_attack_sample_library():
    """
    Build an attack sample library from the raw dataset:
    - Sample up to 5 rows per attack label
    - Use only FEATURE_COLUMNS to match model input format
    """
    global ATTACK_SAMPLE_LIBRARY

    with attack_samples_lock:
        # If already built, return directly
        if ATTACK_SAMPLE_LIBRARY:
            return True

        # Ensure model components are loaded to get FEATURE_COLUMNS
        if not FEATURE_COLUMNS:
            loaded = load_model_components()
            if not loaded or not FEATURE_COLUMNS:
                logger.error("Cannot build attack sample library: FEATURE_COLUMNS not available.")
                return False

        # Validate dataset path
        if not os.path.exists(ATTACK_DATASET_PATH):
            logger.error(f"Attack dataset file not found: {ATTACK_DATASET_PATH}")
            return False

        try:
            logger.info(f"Loading attack dataset from {ATTACK_DATASET_PATH} ...")
            df = pd.read_csv(ATTACK_DATASET_PATH)

            # Strip column names and basic cleanup
            df.columns = df.columns.str.strip()
            if 'Label' not in df.columns:
                logger.error("Dataset has no 'Label' column.")
                return False

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            # Filter attack rows (exclude BENIGN)
            df['Label'] = df['Label'].astype(str).str.strip()
            attack_df = df[df['Label'].str.upper() != 'BENIGN']

            if attack_df.empty:
                logger.error("No attack rows found in dataset.")
                return False

            library = []

            # Group by attack label, sample up to 5 per label
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
# 4. Model retraining logic
# ----------------------------------------------------------------------
def train_model_with_data(df, target_column='Label'):
    """
    Retrain the model using uploaded data.

    Returns:
        dict {'success': True/False, 'message': str, 'stats': dict}
        stats includes: total_samples, label_distribution, new_labels_count
    """
    global PERFORMANCE_METRICS

    try:
        logger.info("Starting retraining process...")

        # 1. Basic cleanup
        df.columns = df.columns.str.strip()  # Strip column name spaces

        # Check if target column exists
        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found in CSV. Available columns: {', '.join(df.columns.tolist())}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg, 'stats': {}}

        # Handle Inf and NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)  # Drop rows with NaN to ensure training quality

        # 2. Label encoding
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column].astype(str))

        # Collect statistics
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

        # 3 Split features and labels
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # (A) Drop duplicates to reduce train/test leakage
        df2 = pd.concat([X, y], axis=1).drop_duplicates()
        X = df2.drop(columns=[target_column])
        y = df2[target_column]

        # (B) Keep numeric columns only (avoid leakage from IDs/timestamps/IPs)
        X = X.select_dtypes(include=[np.number])

        # (C) Enforce the same feature columns/order as the currently loaded model (recommended)
        # This prevents "same length but wrong order" leading to always-BENIGN predictions.
        if FEATURE_COLUMNS:
            X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
            feature_columns_list = list(FEATURE_COLUMNS)
        else:
            feature_columns_list = X.columns.tolist()

        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5. Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame to keep consistent column order
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns_list)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns_list)

        # 6. Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)

        # 7. Evaluation
        y_pred = rf_model.predict(X_test_scaled)

        # ====== AUC calculation (fix binary-case indexing issue) ======
        # 1) Probability output for the test set (required for AUC)
        #    Must match training input: use X_test_scaled (not X_test)
        y_scores = rf_model.predict_proba(X_test_scaled)

        # 2) Classes
        classes = np.unique(y_test)
        n_classes = len(classes)

        # 3) Weighted AUC
        if n_classes == 2:
            # Binary: label_binarize returns (n,1), cannot loop over [:, i]
            y_bin = label_binarize(y_test, classes=classes).ravel()  # (n,)

            # For binary predict_proba is usually (n,2), use positive class column
            if y_scores.ndim == 2 and y_scores.shape[1] >= 2:
                score_pos = y_scores[:, 1]
            else:
                score_pos = np.asarray(y_scores).ravel()

            fpr, tpr, _ = roc_curve(y_bin, score_pos)
            auc_weighted = float(auc(fpr, tpr))

        else:
            # Multiclass: OvR AUC for each class, weighted by support
            y_test_binarized = label_binarize(y_test, classes=classes)  # (n, n_classes)

            support = y_test.value_counts().sort_index().values
            total_support = np.sum(support)

            roc_auc = dict()
            weighted_auc_sum = 0.0

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr, tpr)

                weight = support[i] / total_support
                weighted_auc_sum += roc_auc[i] * weight

            auc_weighted = float(weighted_auc_sum)
        # ====== End AUC calculation ======

        PERFORMANCE_METRICS = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "FPR": 1.0 - recall_score(y_test, y_pred, average=None, zero_division=0)[0],
            "auc": auc_weighted
        }

        # 8. Save all components (overwrite old files)
        os.makedirs('./models', exist_ok=True)
        joblib.dump(rf_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(le, ENCODER_PATH)
        joblib.dump(feature_columns_list, FEATURE_COLS_PATH)

        logger.info(f"Retraining complete. Accuracy: {PERFORMANCE_METRICS['accuracy']:.4f}")

        # Save performance metrics
        with open(PERFORMANCE_PATH, 'w') as f:
            json.dump(PERFORMANCE_METRICS, f)
        logger.info(f"Performance metrics saved to {PERFORMANCE_PATH}")

        return {
            'success': True,
            'message': f'Retraining complete. Accuracy: {PERFORMANCE_METRICS["accuracy"]:.4f}',
            'stats': stats
        }

    except Exception as e:
        logger.error(f"Train model with data failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': str(e), 'stats': {}}

# ----------------------------------------------------------------------
# 5. API routes
# ----------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """
    Prediction endpoint.
    POST Body: {"features": [v1, v2, ...]}
    """
    if not all([MODEL, SCALER, LE, FEATURE_COLUMNS]):
        return jsonify({"status": "error", "message": "Model not fully loaded."}), 503

    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"status": "error", "message": "Missing 'features' in JSON."}), 400

        features = data['features']

        # Run prediction
        result = predict(features)

        if result['status'] == 'success':
            # Save to database
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

            # Handle alerts (non-benign traffic)
            if result['predicted_label'].upper() != 'BENIGN':
                with alerts_lock:
                    alert = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": result['predicted_label'],
                        "confidence": result['confidence'],
                        "level": result['threat_level']
                    }
                    alerts.append(alert)
                    # Fix: use global MAX_ALERTS
                    if len(alerts) > MAX_ALERTS:
                        alerts.pop(0)

        return jsonify(result)

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get latest alerts."""
    with alerts_lock:
        return jsonify(list(reversed(alerts)))


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get detection history."""
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
                "level": row[3],
                "threat_level": row[3]
            })
        return jsonify(history)
    except Exception as e:
        return jsonify([]), 500


@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get model performance metrics."""
    return jsonify(PERFORMANCE_METRICS)

import random
import time
from flask import request, jsonify

# Fixed "attack interval" per mode
MODE_INTERVAL_MS = {
    "Low": 2000,     # once every 2 seconds
    "Medium": 1000,  # once every 1 second
    "High": 500,     # once every 0.5s (= twice per second)
}

# Thresholds used by frontend to map count within last 10 seconds
LEVEL_THRESHOLDS = {
    "low_max": 5,      # 0~5 => Low
    "medium_max": 10,  # 6~10 => Medium
    # >=11 => High
}

def count_to_level(c: int) -> str:
    if c <= LEVEL_THRESHOLDS["low_max"]:
        return "Low"
    if c <= LEVEL_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"

# ====== Place near imports at the top of app.py (if already present, no need to duplicate) ======
import os
import random
import time
import pandas as pd
from threading import Lock
from flask import request, jsonify

# ====== Anomaly traffic CSV cache (added) ======
ANOMALY_TRAFFIC_DF = None
ANOMALY_FEATURE_COLS = None
ANOMALY_LABEL_COL = None
anomaly_traffic_lock = Lock()

def _resolve_anomaly_csv_path() -> str:
    """
    Handle different working directories:
    Prefer locating stream/anomaly_traffic.csv relative to this file.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "stream", "anomaly_traffic.csv"),
        os.path.join(os.getcwd(), "stream", "anomaly_traffic.csv"),
        "stream/anomaly_traffic.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def _load_anomaly_traffic():
    """
    Load anomaly_traffic.csv once and cache:
      - ANOMALY_TRAFFIC_DF
      - ANOMALY_FEATURE_COLS
      - ANOMALY_LABEL_COL (if detected)
    """
    global ANOMALY_TRAFFIC_DF, ANOMALY_FEATURE_COLS, ANOMALY_LABEL_COL

    if ANOMALY_TRAFFIC_DF is not None and ANOMALY_FEATURE_COLS is not None:
        return

    with anomaly_traffic_lock:
        if ANOMALY_TRAFFIC_DF is not None and ANOMALY_FEATURE_COLS is not None:
            return

        csv_path = _resolve_anomaly_csv_path()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"anomaly traffic csv not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # 1) Try to detect a label column (optional)
        label_col = None
        for c in df.columns:
            if str(c).strip().lower() in ("label", "class", "target", "y"):
                label_col = c
                break

        # 2) Prefer using model FEATURE_COLUMNS if available and fully present
        feature_cols = None
        try:
            if "FEATURE_COLUMNS" in globals() and FEATURE_COLUMNS:
                if all(col in df.columns for col in FEATURE_COLUMNS):
                    feature_cols = list(FEATURE_COLUMNS)
        except Exception:
            pass

        # 3) Fallback: use all numeric columns (excluding label)
        if feature_cols is None:
            tmp = df.copy()
            if label_col and label_col in tmp.columns:
                tmp = tmp.drop(columns=[label_col])
            tmp = tmp.select_dtypes(include=["number"])
            feature_cols = list(tmp.columns)

        if not feature_cols:
            raise ValueError("No numeric feature columns found in anomaly_traffic.csv")

        # 4) Enforce feature count consistency if model expects it
        expected = None
        if "FEATURE_COLUMNS" in globals() and FEATURE_COLUMNS:
            expected = len(FEATURE_COLUMNS)

        if expected is not None and len(feature_cols) != expected:
            raise ValueError(
                f"Feature count mismatch: csv has {len(feature_cols)} cols, "
                f"model expects {expected}. Check anomaly_traffic.csv columns."
            )

        ANOMALY_TRAFFIC_DF = df
        ANOMALY_FEATURE_COLS = feature_cols
        ANOMALY_LABEL_COL = label_col


# ====== Keep the same mode intervals and threshold logic ======
MODE_INTERVAL_MS = {
    "Low": 2000,
    "Medium": 1000,
    "High": 500,
}

LEVEL_THRESHOLDS = {
    "low_max": 5,
    "medium_max": 10,
}

def count_to_level(c: int) -> str:
    if c <= LEVEL_THRESHOLDS["low_max"]:
        return "Low"
    if c <= LEVEL_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"


# ====== Replace the original /api/stream route with this one ======
@app.route('/api/stream', methods=['GET'])
def get_attack_stream_sample():

    try:
        # 1) Load anomaly traffic CSV (cached)
        _load_anomaly_traffic()

        df = ANOMALY_TRAFFIC_DF
        feature_cols = ANOMALY_FEATURE_COLS
        label_col = ANOMALY_LABEL_COL

        # 2) Window size in seconds
        default_window = int(TIME_WINDOW_SECONDS) if "TIME_WINDOW_SECONDS" in globals() else 10
        window_s = int(request.args.get("window", default_window))
        window_ms = max(1000, window_s * 1000)

        # 3) Mode: fixed HIGH (ignore any ?mode=)
        mode = "High"
        interval = MODE_INTERVAL_MS[mode]

        # 4) Label filter (only if CSV has a label column)
        label_filter = request.args.get("label")
        df_candidates = df
        if label_filter:
            if not label_col:
                return jsonify({
                    "status": "error",
                    "message": "label filter requested but anomaly_traffic.csv has no label column."
                }), 400
            df_candidates = df[df[label_col] == label_filter]
            if df_candidates.empty:
                return jsonify({
                    "status": "error",
                    "message": f"No samples found for label '{label_filter}'."
                }), 404

        # 5) Generate offsets by fixed interval (HIGH)
        offsets = list(range(0, window_ms, interval))
        attack_frequency = len(offsets)

        # 6) Randomly sample one row per event as features
        start_ms = int(time.time() * 1000)
        stream = []
        n = len(df_candidates)

        for off in offsets:
            ridx = random.randrange(n)
            row = df_candidates.iloc[ridx]

            feats = pd.to_numeric(row[feature_cols], errors="coerce").fillna(0.0).astype(float).tolist()

            lbl = None
            if label_col:
                lbl = row[label_col]
            elif label_filter:
                lbl = label_filter
            else:
                lbl = "ANOMALY"

            stream.append({
                "features": feats,
                "label": str(lbl) if lbl is not None else "ANOMALY",
                "at_ms": off,
                "ts": start_ms + off
            })

        return jsonify({
            "status": "ok",
            "mode": mode,
            "time_window_seconds": window_s,
            "attack_frequency": attack_frequency,
            "frequency_level": count_to_level(attack_frequency),
            "thresholds": LEVEL_THRESHOLDS,
            "stream": stream
        })

    except Exception as e:
        logger.error(f"/api/stream error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/random', methods=['GET'])
def get_random_data():
    """Generate random feature data."""
    count = len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 78
    data = np.random.uniform(0, 1000, count).tolist()
    names = FEATURE_COLUMNS if FEATURE_COLUMNS else [f"f_{i}" for i in range(count)]
    return jsonify({"features": data, "feature_names": names})


@app.route('/api/upload-and-retrain', methods=['POST'])
def upload_and_retrain():
    """
    Upload CSV files and retrain the model.
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

        # Start training
        result = train_model_with_data(full_df)

        if result['success']:
            # Reload model components
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
# Entry point
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Load model at startup
    if not load_model_components():
        logger.warning("Warning: Model components could not be loaded at startup.")
        logger.warning("Please ensure 'trainning.py' has been run and generated files in './models/.")

    app.run(host='127.0.0.1', port=5050, debug=True)

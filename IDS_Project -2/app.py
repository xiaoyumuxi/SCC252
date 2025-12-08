import os
import json
import logging
from datetime import datetime
from threading import Lock
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np
import joblib
import sqlite3
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure file upload
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global variables for model components
MODEL = None
SCALER = None
LE = None
FEATURE_COLUMNS = None

# Initialize database
def init_db():
    conn = sqlite3.connect('ddos_detection.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detection_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT NOT NULL,
                  predicted_label TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  threat_level TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('dashboard.html')

# In-memory storage for alerts and performance metrics
alerts = []
alerts_lock = Lock()

# Sample performance metrics (in a real app, these would be calculated from test data)
PERFORMANCE_METRICS = {
    "accuracy": 0.98,
    "precision": 0.97,
    "recall": 0.96,
    "f1_score": 0.96,
    "auc": 0.99
}

def load_model_components():
    """
    Load all saved model components
    """
    global MODEL, SCALER, LE, FEATURE_COLUMNS
    
    try:
        MODEL = joblib.load('ddos_rf_model.joblib')
        SCALER = joblib.load('ddos_scaler.joblib')
        LE = joblib.load('ddos_label_encoder.joblib')
        FEATURE_COLUMNS = joblib.load('ddos_feature_columns.joblib')
        logger.info("Model components loaded successfully.")
        return True
    except FileNotFoundError as e:
        logger.error(f"Error loading model files: {e}")
        return False

def train_model_with_data(df, target_column='Label'):
    """
    Train a new model with the provided dataframe
    """
    global PERFORMANCE_METRICS
    
    # Data cleaning (same as in training script)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Handle missing values (NaN)
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Handle infinite values (Infinity)
    for col in numeric_cols:
        df[col].replace([np.inf], df[col][np.isfinite(df[col])].max(), inplace=True)
        df[col].replace([-np.inf], df[col][np.isfinite(df[col])].min(), inplace=True)
    
    # Label encoding
    le = LabelEncoder()
    df[target_column + '_Encoded'] = le.fit_transform(df[target_column].astype(str))
    
    # Split dataset
    X = df.drop(columns=[target_column, target_column + '_Encoded'])
    y = df[target_column + '_Encoded']
    
    # Save feature columns
    feature_columns = X.columns.tolist()
    
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    
    # Also scale test set for evaluation
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Update performance metrics
    PERFORMANCE_METRICS = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": 0.99  # Placeholder
    }
    
    # Save all components
    joblib.dump(rf_model, 'ddos_rf_model.joblib')
    joblib.dump(scaler, 'ddos_scaler.joblib')
    joblib.dump(le, 'ddos_label_encoder.joblib')
    joblib.dump(feature_columns, 'ddos_feature_columns.joblib')
    
    return True

def get_threat_level(label, confidence):
    """
    Determine threat level based on label and confidence
    """
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
    Make prediction using the loaded model
    """
    # Check input data length
    if len(raw_input_data) != len(FEATURE_COLUMNS):
        return {
            "status": "error",
            "message": f"Input feature count error. Need {len(FEATURE_COLUMNS)} features, but received {len(raw_input_data)}."
        }

    # Convert to DataFrame (must maintain column order)
    new_df = pd.DataFrame([raw_input_data], columns=FEATURE_COLUMNS)

    # 1. Clean (handle NaN/Inf)
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    new_df.fillna(0, inplace=True)

    # 2. Feature scaling (key step: use SCALER.transform)
    data_scaled = SCALER.transform(new_df)

    # Convert scaled numpy array back to DataFrame with feature names
    data_scaled_df = pd.DataFrame(data_scaled, columns=FEATURE_COLUMNS)

    # 3. Model prediction
    prediction_encoded = MODEL.predict(data_scaled_df)[0]
    
    # 4. Prediction probability (confidence)
    prediction_proba = MODEL.predict_proba(data_scaled_df)[0]

    # 5. Inverse map labels
    prediction_label = LE.inverse_transform([prediction_encoded])[0]

    # Find highest probability
    max_proba = np.max(prediction_proba)

    # Determine threat level
    threat_level = get_threat_level(prediction_label, max_proba)

    # Return result
    return {
        "status": "success",
        "predicted_label": prediction_label,
        "confidence": float(max_proba),
        "encoded_value": int(prediction_encoded),
        "threat_level": threat_level
    }

# API Routes

@app.route('/api/sample', methods=['GET'])
def get_sample_data():
    """
    Return sample data for testing
    """
    # This is the same sample data from your api.py file
    sample_features = [
        54865, 3, 2, 0, 12, 0, 6, 6, 6.0, 0.0, 0, 0, 0.0, 0.0, 4000000.0,
        666666.6667, 3.0, 0.0, 3, 3, 3, 3.0, 0.0, 3, 3, 0, 0.0, 0.0, 0, 0,
        0, 0, 0, 0, 40, 0, 666666.6667, 0.0, 6, 6, 6.0, 0.0, 0.0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 9.0, 6.0, 0.0, 40, 0, 0, 0, 0, 0, 0, 2, 12, 0, 0,
        33, -1, 1, 20, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0
    ]
    
    # Generate feature names based on the number of features
    feature_names = [f"feature_{i}" for i in range(len(sample_features))]
    
    return jsonify({
        "features": sample_features,
        "feature_names": feature_names
    })

@app.route('/api/random', methods=['GET'])
def get_random_data():
    """
    Generate random data for testing
    """
    # Generate random features
    if FEATURE_COLUMNS:
        random_features = [np.random.uniform(-10000, 10000) for _ in range(len(FEATURE_COLUMNS))]
        feature_names = FEATURE_COLUMNS
    else:
        # Fallback if model not loaded
        random_features = [np.random.uniform(-10000, 10000) for _ in range(78)]
        feature_names = [f"feature_{i}" for i in range(78)]
    
    return jsonify({
        "features": [float(x) for x in random_features],
        "feature_names": feature_names
    })

@app.route('/api/simulate-attack', methods=['GET'])
def simulate_attack():
    """
    Generate data that simulates a DDoS attack
    """
    # Generate features that resemble a DDoS attack pattern
    if FEATURE_COLUMNS and len(FEATURE_COLUMNS) >= 78:
        # Create attack-like pattern
        attack_features = []
        
        # First set of features - high packet rates
        attack_features.extend([np.random.uniform(10000, 100000) for _ in range(10)])
        
        # Protocol features - TCP floods
        attack_features.extend([6, 6, 6, 0, 0])  # TCP protocol indicators
        
        # Flow features - many packets in flow
        attack_features.extend([np.random.uniform(1000, 10000) for _ in range(10)])
        
        # Duration features - long flows
        attack_features.extend([np.random.uniform(100, 1000) for _ in range(5)])
        
        # Packet size features - small packets (common in SYN floods)
        attack_features.extend([40, 0, 40, 0])  # Small packet sizes
        
        # Rate features - high rates
        attack_features.extend([np.random.uniform(100000, 1000000) for _ in range(10)])
        
        # Flags - SYN flood pattern
        attack_features.extend([1, 0, 0, 0, 0])  # SYN flag set, others not
        
        # More flow features
        attack_features.extend([np.random.uniform(100, 1000) for _ in range(10)])
        
        # Header features
        attack_features.extend([20, 0, 1, 20, 0, 0, 0, 0])
        
        # Downsample to exactly 78 features if needed
        while len(attack_features) < 78:
            attack_features.append(np.random.uniform(0, 1000))
        
        # Take only first 78 if we have more
        attack_features = attack_features[:78]
        
        feature_names = FEATURE_COLUMNS[:78] if len(FEATURE_COLUMNS) >= 78 else [f"feature_{i}" for i in range(78)]
    else:
        # Fallback if model not loaded or feature count unknown
        attack_features = [np.random.uniform(1000, 100000) if i % 3 == 0 else np.random.uniform(0, 1000) 
                          for i in range(78)]
        feature_names = [f"feature_{i}" for i in range(78)]
    
    return jsonify({
        "features": [float(x) for x in attack_features],
        "feature_names": feature_names
    })

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """
    Predict endpoint for DDoS detection
    """
    if not MODEL or not SCALER or not LE or not FEATURE_COLUMNS:
        return jsonify({
            "status": "error",
            "message": "Model components not loaded. Please check server logs."
        }), 500

    try:
        data = request.get_json()
        features = data['features']
        
        # Make prediction
        result = predict(features)
        
        # If prediction successful and it's not benign, add to alerts and database
        if result['status'] == 'success':
            # Store in database
            conn = sqlite3.connect('ddos_detection.db')
            c = conn.cursor()
            c.execute("INSERT INTO detection_history (timestamp, predicted_label, confidence, threat_level) VALUES (?, ?, ?, ?)",
                      (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result['predicted_label'], result['confidence'], result['threat_level']))
            conn.commit()
            conn.close()
            
            # Add to in-memory alerts if it's not benign
            if result['predicted_label'].upper() != 'BENIGN':
                with alerts_lock:
                    alert = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": result['predicted_label'],
                        "confidence": result['confidence'],
                        "level": result['threat_level']
                    }
                    alerts.append(alert)
                    # Keep only last 50 alerts
                    if len(alerts) > 50:
                        alerts.pop(0)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """
    Get recent alerts
    """
    with alerts_lock:
        # Return alerts in reverse order (newest first)
        return jsonify(list(reversed(alerts)))

@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Get full detection history
    """
    try:
        conn = sqlite3.connect('ddos_detection.db')
        c = conn.cursor()
        c.execute("SELECT timestamp, predicted_label, confidence, threat_level FROM detection_history ORDER BY timestamp DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "timestamp": row[0],
                "type": row[1],
                "confidence": row[2],
                "level": row[3]
            })
        
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify([]), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """
    Get model performance metrics
    """
    return jsonify(PERFORMANCE_METRICS)

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """
    Retrain the model
    """
    try:
        # Execute the training script
        os.system("python trainning.py > retrain.log 2>&1")
        
        # Reload model components
        if load_model_components():
            return jsonify({
                "status": "success",
                "message": "Model retrained and loaded successfully."
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Model retrained but failed to load. Check retrain.log for details."
            }), 500
            
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/upload-and-retrain', methods=['POST'])
def upload_and_retrain():
    """
    Upload CSV files and retrain model with them
    """
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No files uploaded"
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({
                "status": "error",
                "message": "No files selected"
            }), 400
        
        # Process uploaded CSV files
        dataframes = []
        for file in files:
            if file and (file.filename.endswith('.csv')):
                # Read CSV file
                df = pd.read_csv(file, low_memory=False)
                dataframes.append(df)
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid file format for {file.filename}. Only CSV files are allowed."
                }), 400
        
        # Combine all dataframes
        if len(dataframes) > 1:
            combined_df = pd.concat(dataframes, ignore_index=True)
        else:
            combined_df = dataframes[0]
        
        # Train model with combined data
        if train_model_with_data(combined_df):
            # Reload model components
            if load_model_components():
                return jsonify({
                    "status": "success",
                    "message": f"Model successfully retrained with {len(dataframes)} file(s) containing {len(combined_df)} rows."
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Model trained but failed to load components."
                }), 500
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to train model with provided data."
            }), 500
            
    except Exception as e:
        logger.error(f"Upload and retraining error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None
    })

if __name__ == '__main__':
    # Load model components on startup
    if not load_model_components():
        logger.warning("Failed to load model components. Server will start but prediction will not work.")
    
    # Run the Flask app
    app.run(host='127.0.0.1', port=5000, debug=True)
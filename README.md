# DDoS Detection System

A real-time DDoS attack detection system using machine learning with a React frontend and Flask backend.

## Project Structure

- `template/`: React frontend application
- `app.py`: Flask backend server (contains all prediction and retraining logic)
- `training.py`: Model training script for initial training
- `examples/`: Example scripts for prediction and retraining
- `models/`: Trained model components (joblib files)
- `data/`: Dataset files (CSV)

## Setup Instructions

1. Create a virtual environment and install Python dependencies:
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1    # Windows PowerShell
   # or source .venv/bin/activate  # Unix/Linux
   
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Train the model (if not already done):
   ```
   python training.py
   ```

3. Start the Flask backend server:
   ```
   python app.py
   ```

4. In a new terminal, navigate to the template directory and install Node dependencies:
   ```
   cd template
   npm install
   npm run dev
   ```

## API Endpoints

### Prediction Endpoints
- **POST `/api/predict`** - Make a prediction based on provided features
  - Body: `{"features": [v1, v2, ..., v78]}`
  - Response: `{"status": "success", "predicted_label": "...", "confidence": 0.99, "threat_level": "High"}`

### Data Generation Endpoints
- **GET `/api/sample`** - Get sample feature data with feature names
- **GET `/api/random`** - Generate random feature data for testing

### Monitoring Endpoints
- **GET `/api/alerts`** - Get recent security alerts (last 50)
- **GET `/api/history`** - Get full detection history (last 100 records)
- **GET `/api/performance`** - Get model performance metrics (accuracy, precision, recall, f1_score)
- **GET `/health`** - Health check endpoint

### Model Retraining Endpoints
- **POST `/api/upload-and-retrain`** - Upload CSV file(s) and retrain the model with new data
  - Required CSV format: Must contain a `Label` column with attack type labels
  - Supported labels: BENIGN, DDoS, PortScan, Bot, Infiltration, Web Attack, FTP-Patator, SSH-Patator, DoS variants, etc.
  - Response includes: new_labels_count, label_distribution, and updated performance metrics
  - Note: Retraining may take several minutes for large datasets

## Frontend Features

The React frontend provides a user-friendly interface for:

1. **Traffic Analysis**: Submit network traffic features for DDoS detection
2. **Editable Features**: Click on any feature value to edit it directly
3. **Random Data Generation**: Generate random data to test the model
4. **Attack Simulation**: Generate simulated DDoS attack data patterns
5. **Model Retraining**: Upload new CSV files with attack data to retrain the model
6. **Threat Visualization**: View detected threats in real-time with severity levels
7. **Performance Monitoring**: Monitor model performance metrics through charts
8. **Detection History**: Persistent storage of all detection results

## Model Retraining Guide

### Using the Python Script

The easiest way to retrain with new data is to use the provided retraining script:

```bash
# Activate your virtual environment first
python examples/retrain_with_new_data.py ./data/new_attack_data.csv
```

**Requirements for the CSV file:**
- Must contain a `Label` column (with or without leading/trailing spaces)
- Labels can be any attack type (e.g., DDoS, PortScan, Bot, Infiltration, etc.)
- Other columns are treated as features and must match the model's expected feature count
- Null/Inf values are automatically cleaned

**Expected output:**
```
✅ 重训练成功！
📊 数据统计:
  • 总样本数: 500000
  • 唯一标签数: 5
  • 新增标签数: 1
  • 新增标签: NewAttackType

📈 模型性能指标:
  • 准确率 (Accuracy): 0.9994
  • 精确度 (Precision): 0.9994
  • 召回率 (Recall): 0.9994
  • F1-Score: 0.9994
```

### Using the Frontend

1. Click "Choose Files" in the Model Retraining section
2. Select one or more CSV files with attack data
3. Click "Upload and Retrain Model"
4. Wait for retraining to complete (check the console for progress)
5. New attack types will be automatically detected and added to the model
6. Model performance metrics will be updated and displayed

### Using the REST API

```bash
curl -F "files=@./data/attack_data.csv" http://127.0.0.1:5000/api/upload-and-retrain
```

Response:
```json
{
  "status": "success",
  "message": "Retraining complete. Accuracy: 0.9994",
  "stats": {
    "total_samples": 500000,
    "unique_labels": ["BENIGN", "DDoS", "PortScan", "Bot", "NewAttackType"],
    "new_labels": ["NewAttackType"],
    "new_labels_count": 1,
    "label_distribution": {...}
  },
  "performance": {
    "accuracy": 0.9994,
    "precision": 0.9994,
    "recall": 0.9994,
    "f1_score": 0.9994
  }
}
```

## Dataset

The system is trained on the CICIDS2017 dataset, which includes the following attack types:

- **BENIGN**: Normal traffic
- **DDoS**: Distributed Denial of Service
- **PortScan**: Port scanning attacks
- **Bot**: Botnet activity
- **Infiltration**: Network penetration
- **Web Attack**: SQL Injection, XSS, Brute Force
- **FTP-Patator**: FTP password brute force
- **SSH-Patator**: SSH password brute force
- **DoS Variants**: Hulk, slowloris, Slowhttptest, GoldenEye
- **Heartbleed**: Heartbleed vulnerability exploit

You can retrain the model with additional or custom attack types by providing CSV files in the same format.

## Editing Data

To edit feature values in the frontend:
1. Click on any feature value in the "Raw Feature Vector" display
2. Modify the value in the input field that appears
3. Press Enter to save or click the checkmark button
4. Press Escape or click the X button to cancel

## Generating Test Data

### Random Data
To generate random data for testing:
1. Click the "🎲 Random Data" button in the Traffic Analyzer section
2. The system will generate random values for all features
3. Click "Analyze Traffic Pattern" to test the model with this data

### Simulated Attack Data
To generate simulated DDoS attack data:
1. Click the "⚔️ Simulate Attack" button in the Traffic Analyzer section
2. The system will generate data patterns typical of DDoS attacks
3. Click "Analyze Traffic Pattern" to test the model with this attack data

## Threat Intelligence Panel

The right panel shows threat intelligence information with two tabs:

1. **Recent Alerts**: Shows live alerts for malicious traffic detections (refreshes every 5 seconds)
2. **Detection History**: Shows persistent history of all detections (both benign and malicious)

Each entry displays:
- Timestamp of detection
- Type of traffic (BENIGN or specific attack type)
- Confidence score of the detection
- Threat level (None, Low, Medium, High)
- Threat level (None, Low, Medium, High)

## Uploading CSV Files for Retraining

To retrain the model with your own data:

1. Click the "Choose Files" button in the Model Retraining section
2. Select one or more CSV files (they must have the same format as the original dataset)
3. Click "Upload and Retrain Model"
4. Wait for the process to complete (you'll receive a notification)
5. The model will be automatically updated and ready for new predictions

## Model Components

The trained model consists of four serialized components:
1. `ddos_rf_model.joblib` - Random Forest classifier
2. `ddos_scaler.joblib` - Feature scaler
3. `ddos_label_encoder.joblib` - Label encoder
4. `ddos_feature_columns.joblib` - Feature column names

These components are automatically loaded when the Flask server starts.
# DDoS Detection System

A real-time DDoS attack detection system using machine learning with a React frontend and Flask backend.

## Project Structure

- `template/`: React frontend application
- `web/`: Web dependencies
- `app.py`: Flask backend server
- `api.py`: Prediction functions
- `trainning.py`: Model training script
- `*.joblib`: Serialized model components

## Setup Instructions

1. Install Python dependencies:
   ```
   pip install -r web/requirements.txt
   ```

2. Train the model (if not already done):
   ```
   python trainning.py
   ```

3. Start the Flask backend server:
   ```
   python app.py
   ```

4. In a new terminal, navigate to the template directory and install Node dependencies:
   ```
   cd template
   npm install
   ```

5. Start the React frontend:
   ```
   npm run dev
   ```

## API Endpoints

- GET `/api/sample` - Get sample feature data
- GET `/api/random` - Generate random feature data
- GET `/api/simulate-attack` - Generate DDoS attack simulation data
- POST `/api/predict` - Make a prediction based on provided features
- GET `/api/alerts` - Get recent security alerts (last 50)
- GET `/api/history` - Get full detection history (last 100 records)
- GET `/api/performance` - Get model performance metrics
- POST `/api/retrain` - Retrain the model using the original dataset
- POST `/api/upload-and-retrain` - Upload CSV files and retrain the model with new data
- GET `/health` - Health check endpoint

## Frontend Features

The React frontend provides a user-friendly interface for:

1. **Traffic Analysis**: Submit network traffic features for DDoS detection
2. **Editable Features**: Click on any feature value to edit it directly
3. **Random Data Generation**: Generate random data to test the model
4. **Attack Simulation**: Generate simulated DDoS attack data patterns
5. **Model Retraining**: Retrain the model using either the original dataset or by uploading new CSV files
6. **Threat Visualization**: View detected threats in real-time with severity levels
7. **Performance Monitoring**: Monitor model performance metrics through charts
8. **Detection History**: Persistent storage of all detection results

## Editing Data

To edit feature values:
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
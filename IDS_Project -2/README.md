# AI-Powered DDoS Detection System

## Project Overview
This project is a real-time Intrusion Detection System (IDS) designed to detect **DDoS attacks** and **Port Scans** using Machine Learning. It utilizes a **Random Forest** classifier trained on the **CICIDS2017** dataset and features a modern, interactive web dashboard for threat monitoring.

## Key Features
* **Real-time Traffic Simulation**: Simulate Normal, PortScan, and DDoS traffic patterns with a single click.
* **Live Threat Intelligence**: Visual logs and alerts categorized by severity (High/Medium/Low).
* **Interactive Dashboard**: Cyberpunk-themed UI visualizing model performance metrics (Accuracy, Precision, Recall, F1).
* **Model Adaptivity**: Supports online retraining via CSV file upload to adapt to new threat patterns.

## Tech Stack
* **Backend**: Python, Flask, SQLite
* **ML Core**: Scikit-learn (Random Forest), Pandas, NumPy, Joblib
* **Frontend**: HTML5, Bootstrap 5, Chart.js (Radar Charts)

## Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone [your-repo-link]
    cd IDS_Project
    ```

2.  **Install dependencies**:
    ```bash
    pip install flask pandas scikit-learn numpy joblib flask-cors
    ```

3.  **Run the application**:
    ```bash
    python app.py
    ```

4.  **Access the Dashboard**:
    Open your browser and navigate to: `http://127.0.0.1:5000`

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Loads the main dashboard interface. |
| `POST` | `/api/predict` | Accepts feature JSON, returns prediction & threat level. |
| `GET` | `/api/simulate-attack` | Generates synthetic DDoS feature vectors. |
| `POST` | `/api/upload-and-retrain` | Uploads CSV to retrain the Random Forest model. |

## Team Members
* **Backend & Model**: [Teammate Name]
* **Frontend & Visualization**: [Your Name]
* ...

---
*University of Lancaster - SCC.252 Secure Cyber Systems*
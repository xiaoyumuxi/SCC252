import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    classification_report
)

# Set dataset path
file_path = './data/Wednesday-workingHours.pcap_ISCX.csv'

print("Reading dataset...")

# 1) Read CSV file
try:
    df = pd.read_csv(file_path)
    print(f"Loaded successfully. Raw shape: {df.shape}")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    raise SystemExit(1)

# --- Data cleaning prep ---
df.columns = df.columns.str.strip()

# 2) Clean data (drop rows containing NaN or Infinity)
print("Cleaning data...")

# Replace +inf/-inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows containing NaN
df.dropna(inplace=True)

print(f"After cleaning, shape: {df.shape}")

# --- Multi-class label encoding ---
print("Encoding labels...")

# Show original label classes
print("Original label classes:", df['Label'].unique())

# 3) Encode string labels to 0,1,2,3...
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'].astype(str))

# Save label mapping
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nLabel mapping:")
for label, num in label_mapping.items():
    print(f"  {label} -> {num}")

# Show encoded label distribution
print("\nEncoded label distribution:")
print(df['Label'].value_counts())

# 4) Split features (X) and labels (y)
y = df['Label']
X = df.drop('Label', axis=1)

FEATURE_COLUMNS = X.columns.tolist()

# 5) Train/test split (stratified)
print("\nSplitting train/test sets (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6) Feature scaling (StandardScaler)
print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing complete.")

# --- 5) Model training ---
print("\n--- Step 5: Training Random Forest classifier ---")

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train
rf_model.fit(X_train, y_train)

print("Training complete.")

# --- 6) Model evaluation ---
print("\n--- Step 6: Evaluating model ---")

# Predict
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

# FPR using class-0 recall as TNR (kept as your original logic)
recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
tnr_score = recalls[0]
FPR = 1.0 - tnr_score

# Probabilities for AUC
y_scores = rf_model.predict_proba(X_test)

# Binarize for OvR AUC
classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=classes)

# Weighted AUC by class support
support = y_test.value_counts().sort_index().values
total_support = np.sum(support)

roc_auc = {}
weighted_auc_sum = 0.0

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr, tpr)

    weight = support[i] / total_support
    weighted_auc_sum += roc_auc[i] * weight

auc_weighted = float(weighted_auc_sum)

print(f"Test Accuracy:  {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print("---------------------------------")
print(f"Test Recall:    {recall:.4f}")
print(f"Test FPR:       {FPR:.4f}")
print(f"Test AUC:       {auc_weighted:.4f}")

# Save performance metrics
performance_metrics = {
    "accuracy": f"{accuracy:.4f}",
    "precision": f"{precision:.4f}",
    "recall": f"{recall:.4f}",
    "FPR": f"{FPR:.4f}",
    "auc": f"{auc_weighted:.4f}"
}

os.makedirs('./models', exist_ok=True)
metrics_filename = './models/ddos_performance.json'
with open(metrics_filename, 'w') as f:
    json.dump(performance_metrics, f)

print(f"Saved metrics to: {metrics_filename}")

# Detailed classification report
print("\n--- Detailed classification report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# --- 7) Save model and preprocessors ---
print("\n--- Step 7: Saving model and preprocessors ---")

model_filename = './models/ddos_rf_model.joblib'
scaler_filename = './models/ddos_scaler.joblib'
encoder_filename = './models/ddos_label_encoder.joblib'
feature_col_filename = './models/ddos_feature_columns.joblib'

joblib.dump(rf_model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(le, encoder_filename)
joblib.dump(FEATURE_COLUMNS, feature_col_filename)

print("Done.")
print(
    "Saved artifacts:\n"
    f"- Model:          {model_filename}\n"
    f"- Scaler:         {scaler_filename}\n"
    f"- Label encoder:  {encoder_filename}\n"
    f"- Feature columns:{feature_col_filename}"
)

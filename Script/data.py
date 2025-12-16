import pandas as pd
import os
import glob  # used to find files matching a pattern


# --- 1) Load and merge all CSV files in a folder ---
def load_data_from_folder(folder_path='./data'):
    """Traverse a folder, read all CSV files, and merge them into one DataFrame."""

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        print(f"Please make sure the '{folder_path}' folder exists and contains CSV files.")
        return pd.DataFrame()

    # Find all CSV files
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not all_files:
        print(f"Error: No .csv files found in folder {folder_path}")
        return pd.DataFrame()

    list_of_dfs = []
    print(f"Starting to read {len(all_files)} files...\n")

    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            list_of_dfs.append(df)
            print(f"✓ Loaded: {os.path.basename(filename)}")
            print(f"  - Rows: {len(df)}, Columns: {len(df.columns)}")
            print(f"  - Column names: {list(df.columns)}")
        except Exception as e:
            print(f"✗ Failed to read {os.path.basename(filename)}: {e}")
            continue

    # Merge all DataFrames
    if list_of_dfs:
        combined_df = pd.concat(list_of_dfs, ignore_index=True)
        print(f"\nAll files merged. Total rows: {len(combined_df)}\n")
        return combined_df
    else:
        print("\nNo data files could be loaded.")
        return pd.DataFrame()


# --- 2) Analyze label counts and proportions ---
def analyze_labels(df, label_column='Label'):
    """Count each label and compute proportions."""
    if df.empty or label_column not in df.columns:
        print(f"Error: DataFrame is empty or missing column '{label_column}'.")
        return None, None

    label_counts = df[label_column].value_counts()
    print("\n--- Label count statistics ---")
    print(label_counts)

    label_proportions = df[label_column].value_counts(normalize=True)
    print("\n--- Label proportion statistics ---")
    print(label_proportions)

    return label_counts, label_proportions


# --- 3) Simulate normal daily traffic by sampling according to proportions ---
def simulate_daily_traffic(df, proportions, total_sample_size=1000, label_column='Label'):
    """Sample traffic based on label proportions to simulate daily traffic."""
    if df.empty or proportions is None:
        print("Error: Cannot simulate daily traffic. Missing data or proportions.")
        return pd.DataFrame()

    print(f"\n--- Simulating daily traffic (total samples: {total_sample_size}) ---")

    sample_counts = (proportions * total_sample_size).round().astype(int)

    # Ensure the total matches total_sample_size
    diff = total_sample_size - sample_counts.sum()
    if diff != 0:
        most_common_label = sample_counts.idxmax()
        sample_counts[most_common_label] += diff

    print("Sampling counts by label:")
    print(sample_counts)

    simulated_normal_traffic = pd.DataFrame()
    for label, count in sample_counts.items():
        label_data = df[df[label_column] == label]
        sampled_data = label_data.sample(n=count, replace=True, random_state=42)
        simulated_normal_traffic = pd.concat([simulated_normal_traffic, sampled_data])

    simulated_normal_traffic = simulated_normal_traffic.sample(frac=1).reset_index(drop=True)
    print(f"Daily traffic simulation complete. Total rows: {len(simulated_normal_traffic)}")
    return simulated_normal_traffic


# --- 4) Simulate anomalous traffic by boosting one label (optional utility) ---
def simulate_anomalous_traffic(df, anomalous_label='Bot', anomalous_count=100):
    """Simulate abnormal traffic by sampling more of one label."""
    print(f"\n--- Simulating anomalous traffic (add {anomalous_count} rows of label '{anomalous_label}') ---")

    if anomalous_label not in df['Label'].unique():
        print(f"Warning: Label '{anomalous_label}' not found in the dataset. Cannot simulate.")
        return pd.DataFrame()

    anomalous_data = df[df['Label'] == anomalous_label]
    simulated_anomaly = anomalous_data.sample(n=anomalous_count, replace=True, random_state=42)

    print("Anomalous traffic simulation complete.")
    return simulated_anomaly


# --- Main program ---
if __name__ == "__main__":
    FOLDER_PATH = '../data'
    LABEL_COLUMN = ' Label'
    TOTAL_NORMAL_SAMPLE = 5000  # number of samples for normal traffic

    # 1) Load all data from folder
    original_df = load_data_from_folder(FOLDER_PATH)

    if not original_df.empty:
        # Normal traffic: sample from full dataset
        proportions = original_df[LABEL_COLUMN].value_counts(normalize=True)
        normal_sampled = original_df.sample(
            n=min(TOTAL_NORMAL_SAMPLE, len(original_df)),
            replace=(len(original_df) < TOTAL_NORMAL_SAMPLE),
            random_state=42
        )
        normal_sampled.to_csv('../stream/normal_traffic.csv', index=False)
        print(
            f"Saved normal traffic to normal_traffic.csv. Rows: {len(normal_sampled)}, "
            f"Label distribution: {normal_sampled[LABEL_COLUMN].value_counts().to_dict()}"
        )

        # Anomalous traffic: sample from non-BENIGN labels
        anomaly_df = original_df[original_df[LABEL_COLUMN] != 'BEGINE']
        if not anomaly_df.empty:
            anomaly_sampled = anomaly_df.sample(
                n=min(TOTAL_NORMAL_SAMPLE, len(anomaly_df)),
                replace=(len(anomaly_df) < TOTAL_NORMAL_SAMPLE),
                random_state=42
            )

            # Adjust label distribution for anomaly traffic
            label_distribution = {
                'BENIGN': 4194,
                'DoS Hulk': 450,
                'PortScan': 291,
                'DoS GoldenEye': 14,
                'FTP-Patator': 14,
                'SSH-Patator': 12,
                'DoS slowloris': 10,
                'DoS Slowhttptest': 10,
                'Web Attack � Brute Force': 2,
                'Bot': 2,
                'Web Attack � XSS': 1
            }

            label_distribution['BENIGN'] = 0
            remaining_total = sum(label_distribution.values())
            new_distribution = {
                label: (count / remaining_total) * TOTAL_NORMAL_SAMPLE
                for label, count in label_distribution.items()
                if count > 0
            }

            adjusted_anomaly_sampled = pd.DataFrame()
            for label, count in new_distribution.items():
                label_data = anomaly_df[anomaly_df[LABEL_COLUMN] == label]
                sampled_data = label_data.sample(n=int(count), replace=True, random_state=42)
                adjusted_anomaly_sampled = pd.concat([adjusted_anomaly_sampled, sampled_data])

            adjusted_anomaly_sampled = adjusted_anomaly_sampled.sample(frac=1).reset_index(drop=True)
            adjusted_anomaly_sampled.to_csv('../stream/anomaly_traffic.csv', index=False)
            print(
                f"Saved anomaly traffic to anomaly_traffic.csv. Rows: {len(adjusted_anomaly_sampled)}, "
                f"Label distribution: {adjusted_anomaly_sampled[LABEL_COLUMN].value_counts().to_dict()}"
            )
        else:
            print("No non-BEGINE label data found. Cannot generate anomaly traffic file.")

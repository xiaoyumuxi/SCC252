import pandas as pd
import os
import glob  # 用于查找匹配特定模式的文件名


# --- 1. 读取文件data里面的所有数据 (更新为读取文件夹) ---
def load_data_from_folder(folder_path='./data'):
    """遍历指定文件夹，读取并合并所有CSV文件的数据。"""

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹未找到在 {folder_path}")
        print(f"请确保 '{folder_path}' 文件夹存在并包含 CSV 数据文件。")
        return pd.DataFrame()

    # 使用 glob 查找所有 CSV 文件 (如果你的是其他格式，请修改 *.csv)
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not all_files:
        print(f"错误：在 {folder_path} 文件夹中未找到任何 .csv 文件。")
        return pd.DataFrame()

    list_of_dfs = []
    print(f"开始读取 {len(all_files)} 个文件...\n")

    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            list_of_dfs.append(df)
            print(f"✓ 成功读取: {os.path.basename(filename)}")
            print(f"  - 行数: {len(df)}, 列数: {len(df.columns)}")
            print(f"  - 列名: {list(df.columns)}")
        except Exception as e:
            print(f"✗ 读取文件 {os.path.basename(filename)} 时发生错误: {e}")
            continue

    # 将所有读取到的 DataFrame 合并
    if list_of_dfs:
        combined_df = pd.concat(list_of_dfs, ignore_index=True)
        print(f"\n所有文件合并完成，总共 {len(combined_df)} 条数据。\n")
        return combined_df
    else:
        print("\n未能读取任何数据文件。")
        return pd.DataFrame()


# --- 2. 进行数据处理, 统计不同Label的数量 ---
# --- 3. 获取对应的比例 ---
def analyze_labels(df, label_column='Label'):
    """统计不同标签的数量和比例。"""
    if df.empty or label_column not in df.columns:
        print(f"错误：DataFrame为空或缺少 '{label_column}' 列。")
        return None, None

    # 统计数量
    label_counts = df[label_column].value_counts()
    print("\n--- 标签（Label）数量统计 ---")
    print(label_counts)

    # 获取对应的比例
    label_proportions = df[label_column].value_counts(normalize=True)
    print("\n--- 标签（Label）比例统计 ---")
    print(label_proportions)

    return label_counts, label_proportions


# --- 4. 从对应的数据里面按照比例抽取数据从而模拟一个网站的日常 ---
def simulate_daily_traffic(df, proportions, total_sample_size=1000, label_column='Label'):
    """按照现有比例抽取样本来模拟日常流量。"""
    if df.empty or proportions is None:
        print("错误：无法进行日常流量模拟，缺少数据或比例信息。")
        return pd.DataFrame()

    print(f"\n--- 模拟日常流量 (总共抽取 {total_sample_size} 条) ---")

    sample_counts = (proportions * total_sample_size).round().astype(int)

    # 确保总和是 total_sample_size
    diff = total_sample_size - sample_counts.sum()
    if diff != 0:
        # 将差异加到数量最多的那个标签上
        most_common_label = sample_counts.idxmax()
        sample_counts[most_common_label] += diff

    print("日常流量各标签抽取数量:")
    print(sample_counts)

    simulated_normal_traffic = pd.DataFrame()
    for label, count in sample_counts.items():
        label_data = df[df[label_column] == label]
        sampled_data = label_data.sample(n=count, replace=True, random_state=42)
        simulated_normal_traffic = pd.concat([simulated_normal_traffic, sampled_data])

    simulated_normal_traffic = simulated_normal_traffic.sample(frac=1).reset_index(drop=True)
    print(f"日常流量模拟完成，总共 {len(simulated_normal_traffic)} 条。")
    return simulated_normal_traffic


# --- 5. 模拟不正常的流量 ---
def simulate_anomalous_traffic(df, anomalous_label='Bot', anomalous_count=100):
    """模拟异常流量，例如增加Bot或Error标签的数量。"""
    print(f"\n--- 模拟非正常流量 (增加 {anomalous_label} 标签 {anomalous_count} 条) ---")

    if anomalous_label not in df['Label'].unique():
        print(f"警告：原始数据中没有标签 '{anomalous_label}'，无法模拟。")
        return pd.DataFrame()

    anomalous_data = df[df['Label'] == anomalous_label]
    simulated_anomaly = anomalous_data.sample(n=anomalous_count, replace=True, random_state=42)

    print("非正常流量模拟完成。")
    return simulated_anomaly


# --- 主程序执行 ---
if __name__ == "__main__":
    FOLDER_PATH = '../data'
    LABEL_COLUMN = ' Label'
    TOTAL_NORMAL_SAMPLE = 5000  # 普通流量采样数量

    # 1. 读取文件夹中的所有数据
    original_df = load_data_from_folder(FOLDER_PATH)

    if not original_df.empty:
        # 普通流量：全体数据按比例采样
        proportions = original_df[LABEL_COLUMN].value_counts(normalize=True)
        normal_sampled = original_df.sample(n=min(TOTAL_NORMAL_SAMPLE, len(original_df)), replace=(len(original_df) < TOTAL_NORMAL_SAMPLE), random_state=42)
        normal_sampled.to_csv('../stream/normal_traffic.csv', index=False)
        print(f"普通流量已保存到 normal_traffic.csv，行数: {len(normal_sampled)}，标签分布: {normal_sampled[LABEL_COLUMN].value_counts().to_dict()}")

        # 异常流量：非BEGINE标签，按比例采样
        anomaly_df = original_df[original_df[LABEL_COLUMN] != 'BEGINE']
        if not anomaly_df.empty:
            anomaly_props = anomaly_df[LABEL_COLUMN].value_counts(normalize=True)
            anomaly_sampled = anomaly_df.sample(n=min(TOTAL_NORMAL_SAMPLE, len(anomaly_df)), replace=(len(anomaly_df) < TOTAL_NORMAL_SAMPLE), random_state=42)
            
            # 调整异常流量的标签分布
            label_distribution = {'BENIGN': 4194, 'DoS Hulk': 450, 'PortScan': 291, 'DoS GoldenEye': 14, 'FTP-Patator': 14, 'SSH-Patator': 12, 'DoS slowloris': 10, 'DoS Slowhttptest': 10, 'Web Attack � Brute Force': 2, 'Bot': 2, 'Web Attack � XSS': 1}
            label_distribution['BENIGN'] = 0  # 将'BENIGN'的数量设为0
            remaining_total = sum(label_distribution.values())  # 计算其余标签的总数
            new_distribution = {label: (count / remaining_total) * TOTAL_NORMAL_SAMPLE for label, count in label_distribution.items() if count > 0}  # 计算新的分布

            # 根据新的分布抽样
            adjusted_anomaly_sampled = pd.DataFrame()
            for label, count in new_distribution.items():
                label_data = anomaly_df[anomaly_df[LABEL_COLUMN] == label]
                sampled_data = label_data.sample(n=int(count), replace=True, random_state=42)
                adjusted_anomaly_sampled = pd.concat([adjusted_anomaly_sampled, sampled_data])

            adjusted_anomaly_sampled = adjusted_anomaly_sampled.sample(frac=1).reset_index(drop=True)
            adjusted_anomaly_sampled.to_csv('../stream/anomaly_traffic.csv', index=False)
            print(f"异常流量已保存到 anomaly_traffic.csv，行数: {len(adjusted_anomaly_sampled)}，标签分布: {adjusted_anomaly_sampled[LABEL_COLUMN].value_counts().to_dict()}")
        else:
            print("没有非BEGINE标签的数据，无法生成异常流量文件。")
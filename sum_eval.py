import os
import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


SKIP_FOLDERS = [
    "unlearn-N1-A1-bt-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-A1-bt-fn-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-A1-bt-rd-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-A1-bt-cp-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",

    "unlearn-N1-A1-pc-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-A1-pc-fn-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-A1-pc-rd-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-A1-pc-cp-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",

    "unlearn-N1-A1-sin-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-A1-sin-fn-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-A1-sin-rd-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-A1-sin-cp-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
]

CHOOSE_FOLDER = {
    # "unlearn-N20-A1-yrb-lr0.0002_WD0.0_loraRank256_loraDrop0.0_GradStep10_reg5.0"
    # "unlearn-N20-A1-bld-lr0.0002_WD0.0_loraRank256_loraDrop0.0_GradStep10_reg5.0"
}



def extract_metrics(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) >= 2:
                return data[-2]
    except Exception as e:
        e
        # print(f"❌ Error reading {path}: {e}")
    return None

def extract_metrics_from_base(path):
    """从base目录的JSON数据中提取所需参数，并将其缩放到[0,1]范围"""
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) >= 1:
                last_item = data[-1]
                if last_item.get("type") != "statistics":
                    return None
                
                avg_scores = last_item.get("average_scores", {})
                scaled_params = {}
                if "model_llm_score" in avg_scores:
                    scaled_params["llm_score"] = avg_scores["model_llm_score"] / 10
                if "llm_score" in avg_scores:
                    scaled_params["rel_score"] = avg_scores["llm_score"] / 10
                    
                return scaled_params
    except Exception as e:
        e
        # print(f"❌ Error reading {path}: {e}")
    return None
    

def parse_all_methods_in_one_config(config_name, unlearn_root, recovery_root, base_root):
    """
    For a given config path (like unlearn_deepseek_7b_log/unlearn-N1-...),
    extract all results from its methods (dpo/npo/...) and matching recovery results.
    """
    config_path = os.path.join(unlearn_root, config_name)
    records = []

    for method in os.listdir(config_path):
        method_path = os.path.join(config_path, method)
        if not os.path.isdir(method_path):
            continue

        # 按 epoch 聚合多个 dataset_tag
        epoch_rows = defaultdict(dict)

        # 1. 解析unlearn数据
        for fname in os.listdir(method_path):
            if not fname.startswith("epoch-") or not fname.endswith(".json"):
                continue
            m = re.match(r'epoch-(\d+)-([a-zA-Z0-9_]+)\.json', fname)
            if not m:
                continue
            epoch = int(m.group(1))
            dataset_tag = m.group(2)

            unlearn_json = os.path.join(method_path, fname)
            unlearn_metrics = extract_metrics(unlearn_json)
            if unlearn_metrics:
                for k, v in unlearn_metrics.items():
                    epoch_rows[epoch][f"unlearn_{dataset_tag}_{k}"] = v

        # 2. 解析recovery数据（独立于unlearn的epoch）
        rec_dir = os.path.join(recovery_root, config_name, method)
        if os.path.isdir(rec_dir):
            for rfname in os.listdir(rec_dir):
                m = re.match(r'^recovery-epoch-(\d+)-([a-zA-Z0-9_]+)-([^.]+)\.json$', rfname)
                if not m:
                    continue
                epoch = int(m.group(1))
                dataset_tag = m.group(2)
                recover_method = m.group(3)
                
                recovery_json = os.path.join(rec_dir, rfname)
                recovery_metrics = extract_metrics(recovery_json)
                if recovery_metrics:
                    for k, v in recovery_metrics.items():
                        epoch_rows[epoch][f"recovery_{dataset_tag}_{recover_method}_{k}"] = v
        
        # 3. 解析base数据（独立于unlearn的epoch）
        base_dir = os.path.join(base_root, config_name, method)
        if os.path.isdir(base_dir):
            for bfname in os.listdir(base_dir):
                m_base = re.match(r'^epoch-(\d+)-([a-zA-Z0-9_]+)\.json$', bfname)
                if not m_base:
                    continue
                epoch = int(m_base.group(1))
                dataset_tag = m_base.group(2)
                
                base_json = os.path.join(base_dir, bfname)
                base_metrics = extract_metrics_from_base(base_json)
                if base_metrics:
                    for k, v in base_metrics.items():
                        epoch_rows[epoch][f"base_{dataset_tag}_{k}"] = v

        # 组装每一行
        for epoch, row_data in epoch_rows.items():
            row = {
                "method": method,
                "epoch": epoch,
                # "lr": float(lr),
                # "wd": float(wd),
                # "loraRank": int(loraRank),
                # "loraDrop": float(loraDrop),
                # "reg": float(reg),
            }
            row.update(row_data)
            records.append(row)
            # print(row)

    return pd.DataFrame(records)


def scan_all_configs_and_save(unlearn_root, recovery_root, base_root):
    """
    Traverse all config folders in unlearn_root, extract evaluation results,
    and save a summary.csv in each config folder.
    """
    for config_name in os.listdir(unlearn_root):
        if CHOOSE_FOLDER:
            if config_name not in CHOOSE_FOLDER:
                continue
        if config_name in SKIP_FOLDERS:
            print(f"⚠️ Skipping folder: {config_name}")
            continue
        if not re.match(r'.*-lr[\d\.e-]+_WD[\d\.]+_loraRank\d+_loraDrop[\d\.]+_GradStep[\d\.]+_reg[\d\.]+', config_name):
            print(f"❌ Invalid config name: {config_name}")
            continue
        print(f"🔍 Processing config: {config_name}")

        config_path = os.path.join(unlearn_root, config_name)
        if not os.path.isdir(config_path):
            continue

        df = parse_all_methods_in_one_config(config_name, unlearn_root, recovery_root, base_root)
        if not df.empty:
            metrics = set()
            # Extract unique metrics from the DataFrame
            for col in df.columns:
                m = re.match(r'^(unlearn|recovery|base)_[A-Za-z0-9_]+_([A-Za-z0-9_]+)$', col)
                if m:
                    metrics.add(m.group(2))
            # Sort metrics for consistent plotting
            for metric in metrics:
                plot_attribute_progression(df, attribute=metric, output_dir=os.path.join(config_path, "plots"))

            # Save the DataFrame to a CSV file
            output_path = os.path.join(config_path, "summary.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Saved summary to {output_path}")
        else:
            print(f"⚠️ No data found in {config_path}")


def plot_attribute_progression(df: pd.DataFrame, attribute: str, output_dir: str = "plots"):
    """
    For each method, plot the progression of the selected attribute (e.g., 'EM') across epochs,
    from unlearn_* and recovery_* metrics. Saves one plot per method.

    Parameters:
        df: the DataFrame containing all results.
        attribute: the metric to plot (e.g., "EM").
        output_dir: directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = df["method"].unique()

    for method in methods:
        print(f"Method: {method}; Attr: {attribute}")
        df_method = df[df["method"] == method].sort_values("epoch")

        plt.figure(figsize=(10, 6))
        skip_keywords = ["entro", "oracle"]
        for col in df_method.columns:
            if attribute in col and col.startswith(("unlearn_", "recovery_", "base_")): # ("unlearn_", "recovery_", "base_")
                if any(skip in col for skip in skip_keywords):
                    continue
                # 筛选出该列非空的行
                valid_rows = df_method[~df_method[col].isnull()]
                # print(f"col:{col}; attr:{attribute}; row:{valid_rows}")
                if not valid_rows.empty:
                    print(f"绘制 {col}：{len(valid_rows)} 个数据点")
                    plt.plot(
                        valid_rows["epoch"],
                        valid_rows[col],
                        marker='o',
                        label=col,
                        alpha=0.8
                    )

        plt.ylim(0, 1.05)
        plt.title(f"{attribute} progression for method: {method}")
        plt.xlabel("Epoch")
        plt.ylabel(f"Average {attribute} error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{method}_{attribute}_progression.png")
        plt.savefig(plot_path)
        plt.close()

    return f"✅ All plots saved to {output_dir}"


if __name__ == "__main__":
    scan_all_configs_and_save(
        unlearn_root="unlearn_deepseek_7b_log",
        recovery_root="recovery_deepseek_7b_log",
        base_root="base_deepseek_7b_log"
    )
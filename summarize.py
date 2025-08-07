import os
import json
import re
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


SKIP_FOLDERS = [
    "unlearn-N1-lr0.001_WD0.0_loraRank32_loraDrop0.0_reg1.0",
    "unlearn-N1-lr0.0005_WD0.0_loraRank32_loraDrop0.0_reg1.0-1",
    "unlearn-N1-lr0.0001_WD0.0_loraRank32_loraDrop0.0_reg1.0",

    "unlearn-N1-lr0.0005_WD0.0_loraRank64_loraDrop0.0_reg1.0",

    "unlearn-N3-A1-lr0.0001_WD0.0_loraRank32_loraDrop0.0_reg10.0",
    "unlearn-N3-A1-lr0.0005_WD0.0_loraRank32_loraDrop0.0_reg2.0",
    "unlearn-N3-A1-lr0.0005_WD0.0_loraRank32_loraDrop0.0_reg0.0005"
]



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


def parse_all_methods_in_one_config(config_name, unlearn_root, recovery_root):
    """
    For a given config path (like unlearn_llama_7b_log/unlearn-N1-...),
    extract all results from its methods (dpo/npo/...) and matching recovery results.
    """
    config_path = os.path.join(unlearn_root, config_name)
    # match = re.search(r'-lr([\d\.]+)_WD([\d\.]+)_loraRank(\d+)_loraDrop([\d\.]+)_reg([\d\.]+)', config_name)
    # if not match:
    #     print(f"❌ Invalid config path: {config_name}")
    #     return pd.DataFrame()
    # lr, wd, loraRank, loraDrop, reg = match.groups()

    records = []

    for method in os.listdir(config_path):
        method_path = os.path.join(config_path, method)
        if not os.path.isdir(method_path):
            continue

        # 按 epoch 聚合多个 dataset_tag
        epoch_rows = defaultdict(dict)

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

            for flip_val in ["0", "1"]:
                recovery_json = os.path.join(
                    recovery_root, config_name, method,
                    f"recovery-epoch-{epoch}-{dataset_tag}-flip_logit-{flip_val}.json"
                )
                recovery_metrics = extract_metrics(recovery_json)

                if recovery_metrics:
                    for k, v in recovery_metrics.items():
                        epoch_rows[epoch][f"recovery_{dataset_tag}_{k}_flip{flip_val}"] = v

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

    return pd.DataFrame(records)


def scan_all_configs_and_save(unlearn_root, recovery_root):
    """
    Traverse all config folders in unlearn_root, extract evaluation results,
    and save a summary.csv in each config folder.
    """
    for config_name in os.listdir(unlearn_root):
        if config_name in SKIP_FOLDERS:
            print(f"⚠️ Skipping folder: {config_name}")
            continue
        if not re.match(r'.*-lr[\d\.]+_WD[\d\.]+_loraRank\d+_loraDrop[\d\.]+_reg[\d\.]+', config_name):
            print(f"❌ Invalid config name: {config_name}")
            continue
        print(f"🔍 Processing config: {config_name}")

        config_path = os.path.join(unlearn_root, config_name)
        if not os.path.isdir(config_path):
            continue

        df = parse_all_methods_in_one_config(config_name, unlearn_root, recovery_root)
        if not df.empty:
            metrics = set()
            # Extract unique metrics from the DataFrame
            for col in df.columns:
                m = re.match(r'(unlearn|recovery)_[a-zA-Z0-9_]+_([a-zA-Z0-9_]+)(_flip\d)?', col)
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
        df_method = df[df["method"] == method].sort_values("epoch")

        plt.figure(figsize=(10, 6))

        for col in df_method.columns:
            if attribute in col and col.startswith(("unlearn_", "recovery_")):
                plt.plot(df_method["epoch"], df_method[col], marker='o', label=col)

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
        unlearn_root="unlearn_llama_7b_log",
        recovery_root="recovery_llama_7b_log"
    )
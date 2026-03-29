import os
import json
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def extract_parameters_from_ft(json_data):
    """从 fine-tuned 目录的 JSON 数据中提取所需参数"""
    if len(json_data) < 2:
        return None

    # 获取倒数第二个 item
    second_last_item = json_data[-2]
    params = {}

    if "year_of_birth" in second_last_item:
        params["err_year"] = second_last_item["year_of_birth"]
    if "address_postcode" in second_last_item:
        params["err_post"] = second_last_item["address_postcode"]
    if "social_insurance_number" in second_last_item:
        params["err_sin"] = second_last_item["social_insurance_number"]
    if "blood_type" in second_last_item:
        params["err_blood"] = second_last_item["blood_type"]

    return params if params else None


def parse_folder_name(folder_name: str):
    """解析文件夹名称，提取学习率、权重衰减等参数"""
    pattern = (
        r"lr(\d+\.?\d*(?:e-?\d+)?)_WD(\d+\.?\d*)_loraRank(\d+)_loraDrop(\d+\.?\d*)"
        r"(?:_GradStep(\d+))?"
    )
    match = re.match(pattern, folder_name)
    if match:
        return {
            "learning_rate": match.group(1),
            "weight_decay": match.group(2),
            "lora_rank": match.group(3),
            "lora_drop": match.group(4),
            "grad_step": match.group(5) if match.group(5) else None,
        }
    return None


def generate_plot(data, output_path, folder_params):
    """生成折线图并保存到指定路径，只绘制 fine-tuned 指标"""
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 8))

    base_colors = [
        "#1f77b4",  # 蓝色
        "#ff7f0e",  # 橙色
        "#2ca02c",  # 绿色
        "#d62728",  # 红色
        "#9467bd",  # 紫色
        "#8c564b",  # 棕色
        "#e377c2",  # 粉色
        "#7f7f7f",  # 灰色
        "#bcbd22",  # 黄绿色
        "#17becf",  # 青色
    ]

    def create_variations(base_color, n):
        """基于基础颜色创建 n 个细微变化的颜色"""
        hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(base_color))
        variations = []

        sat_range = np.linspace(max(0, hsv[1] - 0.08), min(1, hsv[1] + 0.08), n)
        val_range = np.linspace(max(0.6, hsv[2] - 0.1), min(1, hsv[2] + 0.05), n)

        for i in range(n):
            h = hsv[0]
            s = sat_range[i]
            v = val_range[i]
            rgb = matplotlib.colors.hsv_to_rgb((h, s, v))
            variations.append(matplotlib.colors.to_hex(rgb))

        return variations

    markers = ['o', 's', '^', 'D', 'v', '<', '>']

    datasets = list(data.keys())
    dataset_color_map = {
        dataset: base_colors[i % len(base_colors)]
        for i, dataset in enumerate(datasets)
    }

    for dataset, attrs in data.items():
        num_attrs = len(attrs)
        if num_attrs == 0:
            continue

        base_color = dataset_color_map[dataset]
        color_variations = create_variations(base_color, num_attrs)

        for attr_idx, (attr, epochs_data) in enumerate(attrs.items()):
            if not epochs_data:
                continue

            sorted_epochs = sorted(epochs_data.items(), key=lambda x: int(x[0]))
            epochs, values = zip(*sorted_epochs)
            epochs_int = [int(e) for e in epochs]

            color = color_variations[attr_idx]
            marker = markers[attr_idx % len(markers)]

            ax.plot(
                epochs_int,
                values,
                label=f"{dataset}-{attr}",
                marker=marker,
                color=color,
                linewidth=2,
                markersize=6,
            )

    title = (
        f"Fine-tuned Parameter Trends "
        f"(lr={folder_params['learning_rate']}, "
        f"WD={folder_params['weight_decay']}, "
        f"loraRank={folder_params['lora_rank']}, "
        f"loraDrop={folder_params['lora_drop']}), "
        f"GradStep={folder_params['grad_step']}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)

    # 如果这些 err_* 本身不是 [0,1]，建议删掉这一行
    # ax.set_ylim(0, 1.05)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_ft_folders(ft_root):
    """只处理 fine-tuned 文件夹，提取数据并生成图表"""
    plots_dir = os.path.join(ft_root, "metric_plots")
    os.makedirs(plots_dir, exist_ok=True)

    for folder_name in os.listdir(ft_root):
        ft_folder_path = os.path.join(ft_root, folder_name)

        if not os.path.isdir(ft_folder_path) or folder_name == "metric_plots":
            continue

        folder_params = parse_folder_name(folder_name)
        if not folder_params:
            print(f"Skipping folder with invalid name: {folder_name}")
            continue

        data = defaultdict(lambda: defaultdict(dict))

        for filename in os.listdir(ft_folder_path):
            if filename.endswith('.json') and re.match(r"epoch-\d+-[\w_]+\.json", filename):
                epoch_match = re.search(r"epoch-(\d+)-", filename)
                dataset_match = re.search(r"-([\w_]+)\.json", filename)

                if epoch_match and dataset_match:
                    epoch = epoch_match.group(1)
                    dataset = dataset_match.group(1)

                    file_path = os.path.join(ft_folder_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)

                        params = extract_parameters_from_ft(json_data)
                        if params:
                            for attr, value in params.items():
                                data[dataset][attr][epoch] = value

                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")

        has_data = any(
            any(epochs_data for epochs_data in attrs.values())
            for attrs in data.values()
        )

        if has_data:
            plot_path = os.path.join(plots_dir, f"{folder_name}_ft_metrics.png")
            generate_plot(data, plot_path, folder_params)
            print(f"Generated plot for folder {folder_name}: {plot_path}")
        else:
            print(f"No valid fine-tuned data found in folder {folder_name}, skipping plot generation")


if __name__ == "__main__":
    ft_root = "pretrain_gpt2_log"

    if not os.path.isdir(ft_root):
        print(f"Error: Fine-tuned directory does not exist - {ft_root}")
    else:
        print(f"Starting processing fine-tuned directory: {ft_root}")
        process_ft_folders(ft_root)
        print("Processing completed")
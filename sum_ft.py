import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap

def extract_parameters_from_base(json_data):
    """从base目录的JSON数据中提取所需参数，并将其缩放到[0,1]范围"""
    if len(json_data) == 0:
        return None
    
    # 获取最后一个item
    last_item = json_data[-1]
    if last_item.get("type") != "statistics":
        return None
    
    avg_scores = last_item.get("average_scores", {})
    
    # 将参数除以10缩放到[0,1]范围
    scaled_params = {}
    if "model_llm_score" in avg_scores:
        scaled_params["llm_score"] = avg_scores["model_llm_score"] / 10
    if "llm_score" in avg_scores:
        scaled_params["rel_score"] = avg_scores["llm_score"] / 10
        
    return scaled_params

def extract_parameters_from_ft(json_data):
    """从fine-tuned目录的JSON数据中提取所需参数"""
    if len(json_data) < 2:
        return None
    
    # 获取倒数第二个item
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

def parse_folder_name(folder_name):
    """解析文件夹名称，提取学习率、权重衰减等参数"""
    pattern = r"lr(\d+\.?\d*e?-?\d*)_WD(\d+\.?\d*)_loraRank(\d+)_loraDrop(\d+\.?\d*)_GradStep(\d+)"
    match = re.match(pattern, folder_name)
    if match:
        return {
            "learning_rate": match.group(1),
            "weight_decay": match.group(2),
            "lora_rank": match.group(3),
            "lora_drop": match.group(4),
            "grad_step": match.group(5)
        }
    return None

def get_base_llm_score(bs_root):
    """从base.json获取base_llm_score作为0epoch的llm_score基准值"""
    base_json_path = os.path.join(bs_root, "base.json")
    try:
        with open(base_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if len(json_data) > 0:
            last_item = json_data[-1]
            if last_item.get("type") == "statistics" and last_item.get("model_type") == "base":
                avg_scores = last_item.get("average_scores", {})
                if "base_llm_score" in avg_scores:
                    return avg_scores["base_llm_score"] / 10  # 缩放到[0,1]范围
    except Exception as e:
        print(f"Error processing base.json: {str(e)}")
    return None

def get_ft_base_params(ft_root, dataset):
    """从ft_root的base-{dataset}.json获取0epoch的err_*基准值"""
    ft_base_json_path = os.path.join(ft_root, f"base-{dataset}.json")
    try:
        if os.path.exists(ft_base_json_path):
            with open(ft_base_json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if len(json_data) >= 2:
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
                return params
    except Exception as e:
        print(f"Error processing {ft_base_json_path}: {str(e)}")
    
    # 只返回实际存在的参数，不添加默认值
    return {}

def generate_plot(data, output_path, folder_params):
    """生成折线图并保存到指定路径，优化颜色区分和参数显示"""
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 为不同数据集定义明显不同的基础颜色
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
        "#17becf"   # 青色
    ]
    
    # 为同一数据集内的参数创建细微的颜色变化（亮度和饱和度）
    def create_variations(base_color, n):
        """基于基础颜色创建n个细微变化的颜色"""
        hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(base_color))
        variations = []
        
        # 变化范围（小范围调整以保持同一色系）
        sat_range = np.linspace(max(0, hsv[1]-0.08), min(1, hsv[1]+0.08), n)
        val_range = np.linspace(max(0.6, hsv[2]-0.1), min(1, hsv[2]+0.05), n)
        
        for i in range(n):
            h = hsv[0]
            s = sat_range[i]
            v = val_range[i]
            rgb = matplotlib.colors.hsv_to_rgb((h, s, v))
            variations.append(matplotlib.colors.to_hex(rgb))
        
        return variations
    
    # 不同参数使用的标记
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # 获取所有数据集并分配基础颜色
    datasets = list(data.keys())
    dataset_color_map = {
        dataset: base_colors[i % len(base_colors)] 
        for i, dataset in enumerate(datasets)
    }
    
    # 绘制每条线
    for dataset_idx, (dataset, attrs) in enumerate(data.items()):
        # 为当前数据集的参数创建颜色变化
        num_attrs = len(attrs)
        if num_attrs == 0:
            continue
            
        # 获取当前数据集的颜色变化
        base_color = dataset_color_map[dataset]
        color_variations = create_variations(base_color, num_attrs)
        
        # 为每个属性绘制折线
        for attr_idx, (attr, epochs_data) in enumerate(attrs.items()):
            # 排序数据点（按数值排序，处理非均匀epoch）
            sorted_epochs = sorted(epochs_data.items(), key=lambda x: int(x[0]))
            epochs, values = zip(*sorted_epochs)
            
            # 转换epochs为整数用于正确排序
            epochs_int = [int(e) for e in epochs]
            
            # 获取样式
            color = color_variations[attr_idx]
            marker = markers[attr_idx % len(markers)]
            
            # 绘制折线
            ax.plot(epochs_int, values, label=f"{dataset}-{attr}", 
                    marker=marker, 
                    color=color,
                    linewidth=2,
                    markersize=6)
    
    # 添加标题和标签（使用英文）
    title = f"Parameter Trends (lr={folder_params['learning_rate']}, WD={folder_params['weight_decay']}, " \
            f"loraRank={folder_params['lora_rank']}, loraDrop={folder_params['lora_drop']}), " \
            f"GradStep={folder_params['grad_step']}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)

    ax.set_ylim(0, 1.05)
    
    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_folders(bs_root, ft_root):
    """处理所有文件夹，提取数据并生成图表，只保留存在的参数"""
    # 创建统一存放图像的文件夹
    plots_dir = os.path.join(bs_root, "metric_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 获取base_llm_score作为0epoch的llm_score基准值
    base_llm_score = get_base_llm_score(bs_root)
    if base_llm_score is None:
        print("Warning: Could not get base_llm_score from base.json")
    
    # 遍历base目录下的所有子文件夹
    for folder_name in os.listdir(bs_root):
        bs_folder_path = os.path.join(bs_root, folder_name)
        
        # 只处理目录，跳过图像文件夹
        if not os.path.isdir(bs_folder_path) or folder_name == "metric_plots":
            continue
        
        # 解析文件夹名称
        folder_params = parse_folder_name(folder_name)
        if not folder_params:
            print(f"Skipping folder with invalid name: {folder_name}")
            continue
        
        # 检查对应的fine-tuned文件夹是否存在
        ft_folder_path = os.path.join(ft_root, folder_name)
        if not os.path.isdir(ft_folder_path):
            print(f"Corresponding fine-tuned folder not found: {folder_name}")
            continue
        
        # 收集数据 - 只保留实际存在的参数
        data = defaultdict(lambda: defaultdict(dict))  # 结构: {dataset: {attr: {epoch: value}}}
        datasets_found = set()
        
        # 处理base目录下的JSON文件
        for filename in os.listdir(bs_folder_path):
            if filename.endswith('.json') and re.match(r"epoch-\d+-[\w_]+\.json", filename):
                epoch_match = re.search(r"epoch-(\d+)-", filename)
                dataset_match = re.search(r"-([\w_]+)\.json", filename)
                
                if epoch_match and dataset_match:
                    epoch = epoch_match.group(1)
                    dataset = dataset_match.group(1)
                    datasets_found.add(dataset)
                    
                    file_path = os.path.join(bs_folder_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        params = extract_parameters_from_base(json_data)
                        if params:
                            for attr, value in params.items():
                                data[dataset][attr][epoch] = value
                    
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
        
        # 处理fine-tuned目录下的JSON文件
        for filename in os.listdir(ft_folder_path):
            if filename.endswith('.json') and re.match(r"epoch-\d+-[\w_]+\.json", filename):
                epoch_match = re.search(r"epoch-(\d+)-", filename)
                dataset_match = re.search(r"-([\w_]+)\.json", filename)
                
                if epoch_match and dataset_match:
                    epoch = epoch_match.group(1)
                    dataset = dataset_match.group(1)
                    datasets_found.add(dataset)
                    
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
        
        # 添加0epoch的基准值（只添加存在的参数）
        if base_llm_score is not None:
            for dataset in datasets_found:
                # 只在有该数据集的llm_score数据时添加基准值
                if "llm_score" in data[dataset]:
                    data[dataset]["llm_score"]["0"] = base_llm_score
                
                # 只在有该数据集的rel_score数据时添加基准值
                if "rel_score" in data[dataset]:
                    data[dataset]["rel_score"]["0"] = 1
                
                # 获取并添加ft的基准值（只添加存在的参数）
                ft_base_params = get_ft_base_params(ft_root, dataset)
                for attr, value in ft_base_params.items():
                    if attr in data[dataset]:  # 只添加已存在的参数的基准值
                        data[dataset][attr]["0"] = value
        
        # 检查是否有数据
        has_data = any(any(epochs_data for epochs_data in attrs.values()) for attrs in data.values())
        
        # 如果有数据，生成并保存图表到统一的图像文件夹
        if has_data:
            plot_path = os.path.join(plots_dir, f"{folder_name}_metrics.png")
            generate_plot(data, plot_path, folder_params)
            print(f"Generated plot for folder {folder_name}: {plot_path}")
        else:
            print(f"No valid data found in folder {folder_name}, skipping plot generation")

if __name__ == "__main__":
    import matplotlib  # 延迟导入，仅在主程序中需要
    
    # 可修改为实际的根目录路径
    bs_root = "base_deepseek_7b_log"
    ft_root = "fine_tuned_deepseek_7b_log"
    
    # 检查根目录是否存在
    if not os.path.isdir(bs_root):
        print(f"Error: Base directory does not exist - {bs_root}")
    elif not os.path.isdir(ft_root):
        print(f"Error: Fine-tuned directory does not exist - {ft_root}")
    else:
        print(f"Starting processing base directory: {bs_root} and fine-tuned directory: {ft_root}")
        process_folders(bs_root, ft_root)
        print("Processing completed")

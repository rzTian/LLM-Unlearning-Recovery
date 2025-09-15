import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_path):
    """
    解析单个result.log文件，提取loss、eval_loss、grad_norm和epoch数据
    :param log_path: result.log文件路径
    :return: 包含多种指标数据的字典，或None（解析失败时）
    """
    # 数据存储结构
    data = {
        'loss': [],        # 存储(epoch, loss)
        'eval_loss': [],   # 存储(epoch, eval_loss)
        'grad_norm': []    # 存储(epoch, grad_norm)
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            log_content = f.read()
            
            # 尝试提取所有包含字典的行 - 更宽松的匹配模式
            pattern = re.compile(r'(\{.*?\})', re.DOTALL)
            matches = pattern.findall(log_content)
            
            if not matches:
                print(f"  未找到任何包含字典的行")
                return None
                
            print(f"  找到 {len(matches)} 个可能包含数据的字典")
            
            for i, dict_str in enumerate(matches, 1):
                try:
                    # 处理可能的格式问题
                    dict_str = dict_str.replace("'", '"')  # 单引号转双引号
                    dict_str = dict_str.replace('None', 'null')  # None转null
                    dict_str = dict_str.replace('True', 'true')  # True转true
                    dict_str = dict_str.replace('False', 'false')  # False转false
                    
                    log_data = json.loads(dict_str)
                    epoch = log_data.get('epoch')
                    
                    if not isinstance(epoch, (int, float)):
                        continue  # 跳过没有有效epoch的记录
                    
                    # 提取loss
                    if 'loss' in log_data and isinstance(log_data['loss'], (int, float)):
                        # 避免同一epoch重复添加
                        if not any(round(e, 2) == round(epoch, 2) for e, _ in data['loss']):
                            data['loss'].append((epoch, log_data['loss']))
                    
                    # 提取eval_loss
                    if 'eval_loss' in log_data and isinstance(log_data['eval_loss'], (int, float)):
                        # 避免同一epoch重复添加
                        if not any(round(e, 2) == round(epoch, 2) for e, _ in data['eval_loss']):
                            data['eval_loss'].append((epoch, log_data['eval_loss']))
                    
                    # 提取grad_norm
                    if 'grad_norm' in log_data and isinstance(log_data['grad_norm'], (int, float)):
                        # 避免同一epoch重复添加
                        if not any(round(e, 2) == round(epoch, 2) for e, _ in data['grad_norm']):
                            data['grad_norm'].append((epoch, log_data['grad_norm']))
                            
                except json.JSONDecodeError as e:
                    if i <= 10:  # 只显示前10个解析错误
                        print(f"  字典 {i} 解析失败: {str(e)[:50]}...")
                except Exception as e:
                    if i <= 10:
                        print(f"  字典 {i} 处理错误: {str(e)[:50]}...")
    
    except Exception as e:
        print(f"  读取文件时异常: {str(e)}")
        return None
    
    # 按epoch排序
    for key in data:
        data[key].sort(key=lambda x: x[0])
    
    # 检查是否有有效数据
    print(f"  成功解析到:")
    print(f"    loss数据: {len(data['loss'])} 条")
    print(f"    eval_loss数据: {len(data['eval_loss'])} 条")
    print(f"    grad_norm数据: {len(data['grad_norm'])} 条")
    
    # 至少需要一种损失数据和梯度数据才能绘图
    has_loss_data = len(data['loss']) > 0 or len(data['eval_loss']) > 0
    if not has_loss_data or len(data['grad_norm']) == 0:
        print(f"  警告：未解析到足够的有效数据")
        return None
        
    return data

def create_plot(data, save_path):
    """
    生成并保存折线图，包含loss、eval_loss和grad_norm
    :param data: 包含多种指标数据的字典
    :param save_path: 图片保存路径
    """
    # 创建画布和双轴
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 左侧y轴：损失数据（loss和eval_loss）
    color1 = 'tab:blue'    # loss颜色
    color2 = 'tab:green'   # eval_loss颜色
    
    # 绘制loss
    if data['loss']:
        epochs_loss = [e for e, _ in data['loss']]
        losses = [l for _, l in data['loss']]
        ax1.plot(epochs_loss, losses, color=color1, marker='o', markersize=4, 
                 linewidth=2, label='Training Loss')
        ax1.set_ylabel('Loss', color='black')
        ax1.tick_params(axis='y')
    
    # 绘制eval_loss
    if data['eval_loss']:
        epochs_eval = [e for e, _ in data['eval_loss']]
        eval_losses = [l for _, l in data['eval_loss']]
        ax1.plot(epochs_eval, eval_losses, color=color2, marker='^', markersize=4, 
                 linewidth=2, linestyle='-.', label='Evaluation Loss')
    
    # 右侧y轴：梯度范数
    ax2 = ax1.twinx()
    color3 = 'tab:orange'
    if data['grad_norm']:
        epochs_grad = [e for e, _ in data['grad_norm']]
        grad_norms = [g for _, g in data['grad_norm']]
        ax2.plot(epochs_grad, grad_norms, color=color3, marker='s', markersize=4, 
                 linestyle='--', linewidth=2, label='Gradient Norm')
        ax2.set_ylabel('Gradient Norm', color=color3)
        ax2.tick_params(axis='y', labelcolor=color3)
    
    # 设置共同的x轴
    ax1.set_xlabel('Epoch')
    
    # 添加标题和图例
    plt.title('Training Dynamics: Loss and Gradient Norm vs Epoch')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best', fontsize=10)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_all_logs(target_dir):
    """
    遍历目标目录下所有result.log文件，生成并保存图表
    :param target_dir: 目标根目录
    """
    processed_count = 0
    skipped_count = 0
    
    print(f"开始处理目录：{os.path.abspath(target_dir)}")
    
    # 遍历所有子文件夹和文件
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file == "result.log":
                log_path = os.path.join(root, file)
                print(f"\n处理日志文件: {log_path}")
                
                # 解析日志
                data = parse_training_log(log_path)
                if not data:
                    skipped_count += 1
                    continue
                
                # 生成保存路径（修改部分）
                # 获取最后一个文件夹名
                last_folder = os.path.basename(root)
                # 获取倒数第二个文件夹路径
                parent_folder = os.path.dirname(root)
                # 创建plots目录
                plots_dir = os.path.join(parent_folder, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                # 生成图片文件名和路径
                save_path = os.path.join(plots_dir, f"{last_folder}_dynamic.png")
                
                # 创建并保存图表
                create_plot(data, save_path)
                print(f"  图表已保存至: {save_path}")
                
                processed_count += 1
    
    print(f"\n处理完成！共处理 {processed_count} 个日志文件，跳过 {skipped_count} 个文件")

if __name__ == "__main__":
    # 目标目录（请根据实际情况修改）
    TARGET_DIRECTORY = "unlearn_deepseek_7b_log"
    
    # 执行处理
    if os.path.isdir(TARGET_DIRECTORY):
        process_all_logs(TARGET_DIRECTORY)
    else:
        print(f"错误：目录不存在 - {TARGET_DIRECTORY}")

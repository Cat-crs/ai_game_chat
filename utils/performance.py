"""
性能优化模块，用于调整AI模型的参数和MCTS的搜索深度。
"""
import torch
import psutil
import os
import time


def get_system_info():
    """
    获取系统信息
    
    返回:
        dict: 系统信息字典
    """
    # CPU信息
    cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
    cpu_count_logical = psutil.cpu_count(logical=True)  # 逻辑核心数
    cpu_freq = psutil.cpu_freq()
    
    # 内存信息
    memory = psutil.virtual_memory()
    
    # GPU信息
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_available else []
    
    return {
        'cpu_count': cpu_count,
        'cpu_count_logical': cpu_count_logical,
        'cpu_freq': cpu_freq,
        'memory_total': memory.total,
        'memory_available': memory.available,
        'gpu_available': gpu_available,
        'gpu_count': gpu_count,
        'gpu_names': gpu_names
    }


def optimize_model_parameters(model, device):
    """
    优化模型参数
    
    参数:
        model: 神经网络模型
        device: 设备类型
        
    返回:
        model: 优化后的模型
    """
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 如果是CPU设备，尝试使用量化技术
    if device.type == 'cpu':
        try:
            # 使用PyTorch的量化功能
            model_fp32 = model
            model_int8 = torch.quantization.quantize_dynamic(
                model_fp32,  # 原始模型
                {torch.nn.Linear, torch.nn.Conv2d},  # 要量化的层类型
                dtype=torch.qint8  # 量化数据类型
            )
            return model_int8
        except Exception as e:
            print(f"量化模型失败: {e}")
            return model
    
    return model


def optimize_mcts_parameters(system_info):
    """
    优化MCTS参数
    
    参数:
        system_info (dict): 系统信息字典
        
    返回:
        dict: MCTS参数字典
    """
    # 根据系统资源调整MCTS参数
    cpu_count = system_info['cpu_count']
    memory_gb = system_info['memory_total'] / (1024 ** 3)  # 转换为GB
    
    # 基础参数
    base_simulations = 100
    
    # 根据CPU核心数调整模拟次数
    if cpu_count >= 8:
        simulations = base_simulations * 2
    elif cpu_count >= 4:
        simulations = base_simulations
    else:
        simulations = base_simulations // 2
        
    # 根据内存大小调整
    if memory_gb >= 16:
        simulations = int(simulations * 1.5)
    elif memory_gb < 4:
        simulations = int(simulations * 0.5)
        
    # 如果有GPU，增加模拟次数
    if system_info['gpu_available']:
        simulations = int(simulations * 3)
        
    # 其他MCTS参数
    c_puct = 1.0  # 探索常数
    
    return {
        'num_simulations': max(10, simulations),  # 确保至少有10次模拟
        'c_puct': c_puct
    }


def measure_inference_time(model, board_size=15, device='cpu', num_runs=10):
    """
    测量模型推理时间
    
    参数:
        model: 神经网络模型
        board_size (int): 棋盘大小
        device: 设备类型
        num_runs (int): 运行次数
        
    返回:
        float: 平均推理时间（毫秒）
    """
    # 创建随机输入
    x = torch.randn(1, 3, board_size, board_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            model(x)
            
    # 测量时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(x)
    end_time = time.time()
    
    # 计算平均时间（毫秒）
    avg_time = (end_time - start_time) * 1000 / num_runs
    
    return avg_time


def optimize_batch_size(system_info):
    """
    优化批次大小
    
    参数:
        system_info (dict): 系统信息字典
        
    返回:
        int: 优化后的批次大小
    """
    # 根据系统资源调整批次大小
    memory_gb = system_info['memory_total'] / (1024 ** 3)  # 转换为GB
    
    # 基础批次大小
    base_batch_size = 32
    
    # 根据内存大小调整
    if memory_gb >= 16:
        batch_size = base_batch_size * 2
    elif memory_gb >= 8:
        batch_size = base_batch_size
    else:
        batch_size = base_batch_size // 2
        
    # 如果有GPU，增加批次大小
    if system_info['gpu_available']:
        batch_size = batch_size * 4
        
    return max(8, batch_size)  # 确保至少为8


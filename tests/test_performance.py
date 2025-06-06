"""
性能测试脚本，用于测试项目在不同环境下的性能表现。
"""
import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.gomoku import Gomoku
from game.go import Go
from ai.model import create_model
from ai.mcts import MCTS
from utils.performance import get_system_info, measure_inference_time


def test_model_inference_time():
    """测试模型推理时间"""
    print("===== 测试模型推理时间 =====")
    
    # 获取系统信息
    system_info = get_system_info()
    print(f"系统信息: {system_info}")
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试不同棋盘大小的推理时间
    board_sizes = [9, 15, 19]
    inference_times = []
    
    for board_size in board_sizes:
        print(f"\n测试棋盘大小: {board_size}x{board_size}")
        
        # 创建模型
        model = create_model(board_size=board_size, device=device)
        
        # 测量推理时间
        avg_time = measure_inference_time(model, board_size=board_size, device=device, num_runs=20)
        
        print(f"平均推理时间: {avg_time:.2f} ms")
        inference_times.append(avg_time)
        
    # 绘制推理时间图表
    plt.figure(figsize=(10, 6))
    plt.bar(board_sizes, inference_times)
    plt.xlabel('棋盘大小')
    plt.ylabel('推理时间 (ms)')
    plt.title('不同棋盘大小的模型推理时间')
    plt.savefig('inference_time.png')
    print("推理时间图表已保存为 inference_time.png")


def test_mcts_performance():
    """测试MCTS性能"""
    print("\n===== 测试MCTS性能 =====")
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    board_size = 15
    model = create_model(board_size=board_size, device=device)
    
    # 测试不同模拟次数的MCTS性能
    simulation_counts = [10, 50, 100]
    decision_times = []
    
    for num_simulations in simulation_counts:
        print(f"\n测试模拟次数: {num_simulations}")
        
        # 创建MCTS
        mcts = MCTS(model, num_simulations=num_simulations, device=device)
        
        # 创建游戏
        gomoku = Gomoku()
        
        # 测量决策时间
        start_time = time.time()
        x, y = mcts.get_best_move(gomoku)
        end_time = time.time()
        
        decision_time = (end_time - start_time) * 1000  # 转换为毫秒
        print(f"决策时间: {decision_time:.2f} ms")
        decision_times.append(decision_time)
        
    # 绘制决策时间图表
    plt.figure(figsize=(10, 6))
    plt.bar(simulation_counts, decision_times)
    plt.xlabel('模拟次数')
    plt.ylabel('决策时间 (ms)')
    plt.title('不同模拟次数的MCTS决策时间')
    plt.savefig('decision_time.png')
    print("决策时间图表已保存为 decision_time.png")


def test_memory_usage():
    """测试内存使用情况"""
    print("\n===== 测试内存使用情况 =====")
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试不同棋盘大小的内存使用情况
    board_sizes = [9, 15, 19]
    memory_usages = []
    
    for board_size in board_sizes:
        print(f"\n测试棋盘大小: {board_size}x{board_size}")
        
        # 记录初始内存使用
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        initial_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        # 创建模型
        model = create_model(board_size=board_size, device=device)
        
        # 记录模型内存使用
        model_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        model_memory_usage = (model_memory - initial_memory) / (1024 ** 2)  # 转换为MB
        
        if device.type == 'cuda':
            print(f"模型内存使用: {model_memory_usage:.2f} MB")
        else:
            print("在CPU设备上无法准确测量内存使用")
            
        memory_usages.append(model_memory_usage if device.type == 'cuda' else 0)
        
    # 如果是GPU设备，绘制内存使用图表
    if device.type == 'cuda':
        plt.figure(figsize=(10, 6))
        plt.bar(board_sizes, memory_usages)
        plt.xlabel('棋盘大小')
        plt.ylabel('内存使用 (MB)')
        plt.title('不同棋盘大小的模型内存使用')
        plt.savefig('memory_usage.png')
        print("内存使用图表已保存为 memory_usage.png")


def test_self_play_speed():
    """测试自我对弈速度"""
    print("\n===== 测试自我对弈速度 =====")
    
    # 创建设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    board_size = 9  # 使用小棋盘加快测试
    model = create_model(board_size=board_size, device=device)
    
    # 创建MCTS
    mcts = MCTS(model, num_simulations=10, device=device)
    
    # 创建游戏
    gomoku = Gomoku(board_size=board_size)
    
    # 测量自我对弈速度
    print("执行自我对弈...")
    start_time = time.time()
    
    # 模拟一局游戏
    moves = 0
    while not gomoku.is_game_over() and moves < 30:  # 最多30步
        x, y = mcts.get_best_move(gomoku)
        gomoku.place_stone(x, y)
        moves += 1
        
    end_time = time.time()
    
    game_time = end_time - start_time
    moves_per_second = moves / game_time
    
    print(f"游戏步数: {moves}")
    print(f"游戏时间: {game_time:.2f} 秒")
    print(f"每秒步数: {moves_per_second:.2f}")


if __name__ == "__main__":
    test_model_inference_time()
    test_mcts_performance()
    test_memory_usage()
    test_self_play_speed()


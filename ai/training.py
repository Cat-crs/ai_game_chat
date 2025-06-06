"""
模型训练模块，实现神经网络模型的训练功能。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import time
from collections import deque


class TrainingExample:
    """
    训练样本类，用于存储自我对弈生成的训练数据。
    
    属性:
        state: 棋盘状态
        current_player: 当前玩家
        policy: 策略概率分布
        value: 价值（游戏结果）
    """
    
    def __init__(self, state, current_player, policy, value):
        """
        初始化训练样本
        
        参数:
            state (numpy.ndarray): 棋盘状态
            current_player (Player): 当前玩家
            policy (numpy.ndarray): 策略概率分布
            value (float): 价值（游戏结果）
        """
        self.state = state
        self.current_player = current_player
        self.policy = policy
        self.value = value


class Trainer:
    """
    模型训练器类，用于训练神经网络模型。
    
    属性:
        model: 神经网络模型
        optimizer: 优化器
        device: 设备类型
        batch_size: 批次大小
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        train_examples_history: 训练样本历史
        train_examples_max_history_size: 训练样本历史最大大小
    """
    
    def __init__(self, model, device='cpu', batch_size=64, epochs=10, lr=0.001, weight_decay=1e-4, max_history_size=20000):
        """
        初始化训练器
        
        参数:
            model: 神经网络模型
            device (str): 设备类型
            batch_size (int): 批次大小
            epochs (int): 训练轮数
            lr (float): 学习率
            weight_decay (float): 权重衰减
            max_history_size (int): 训练样本历史最大大小
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_examples_history = []
        self.train_examples_max_history_size = max_history_size
        
    def train(self, examples):
        """
        训练模型
        
        参数:
            examples (list): 训练样本列表
            
        返回:
            tuple: (策略损失, 价值损失)
        """
        # 将新样本添加到历史中
        self.train_examples_history.extend(examples)
        
        # 如果历史样本超过最大大小，随机删除一些旧样本
        if len(self.train_examples_history) > self.train_examples_max_history_size:
            print(f"样本数量超过限制，从 {len(self.train_examples_history)} 减少到 {self.train_examples_max_history_size}")
            self.train_examples_history = np.random.choice(
                self.train_examples_history,
                size=self.train_examples_max_history_size,
                replace=False
            ).tolist()
            
        # 准备训练数据
        train_data = self._prepare_training_data(self.train_examples_history)
        
        # 训练模型
        self.model.train()
        
        policy_losses = []
        value_losses = []
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # 打乱数据
            indices = np.arange(len(train_data["states"]))
            np.random.shuffle(indices)
            
            # 批次训练
            batch_count = int(np.ceil(len(indices) / self.batch_size))
            
            for batch in tqdm(range(batch_count)):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                # 准备批次数据
                states = torch.from_numpy(train_data["states"][batch_indices]).to(self.device)
                policies = torch.from_numpy(train_data["policies"][batch_indices]).to(self.device)
                values = torch.from_numpy(train_data["values"][batch_indices]).to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                policy_logits, value_preds = self.model(states)
                
                # 计算损失
                policy_loss = -torch.sum(policies * policy_logits) / policies.size(0)
                value_loss = torch.mean((values - value_preds.squeeze(1)) ** 2)
                total_loss = policy_loss + value_loss
                
                # 反向传播
                total_loss.backward()
                self.optimizer.step()
                
                # 记录损失
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                
            # 打印每个epoch的平均损失
            avg_policy_loss = np.mean(policy_losses[-batch_count:])
            avg_value_loss = np.mean(value_losses[-batch_count:])
            print(f"策略损失: {avg_policy_loss:.4f}, 价值损失: {avg_value_loss:.4f}")
            
        return np.mean(policy_losses), np.mean(value_losses)
        
    def _prepare_training_data(self, examples):
        """
        准备训练数据
        
        参数:
            examples (list): 训练样本列表
            
        返回:
            dict: 训练数据字典
        """
        from .model import prepare_input
        
        n = len(examples)
        board_size = examples[0].state.shape[0]
        
        # 初始化数据数组
        states = np.zeros((n, 3, board_size, board_size), dtype=np.float32)
        policies = np.zeros((n, board_size * board_size), dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        
        # 填充数据
        for i, example in enumerate(examples):
            # 准备输入状态
            x = prepare_input(example.state, example.current_player)
            states[i] = x.numpy()
            
            # 策略和价值
            policies[i] = example.policy
            values[i] = example.value
            
        return {
            "states": states,
            "policies": policies,
            "values": values
        }
        
    def save_model(self, filepath):
        """
        保存模型
        
        参数:
            filepath (str): 模型保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"模型已保存到 {filepath}")
        
    def load_model(self, filepath):
        """
        加载模型
        
        参数:
            filepath (str): 模型加载路径
            
        返回:
            bool: 是否成功加载
        """
        if not os.path.exists(filepath):
            print(f"模型文件 {filepath} 不存在")
            return False
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已从 {filepath} 加载")
        return True


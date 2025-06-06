"""
神经网络模型模块，实现基于PyTorch的深度学习模型。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DualNetworkModel(nn.Module):
    """
    双头神经网络模型，用于棋类游戏的策略和价值评估。
    
    属性:
        board_size (int): 棋盘大小
        num_channels (int): 卷积层通道数
        num_res_blocks (int): 残差块数量
    """
    
    def __init__(self, board_size=15, input_channels=3, num_channels=32, num_res_blocks=3):
        """
        初始化神经网络模型
        
        参数:
            board_size (int): 棋盘大小，默认为15（适合五子棋）
            input_channels (int): 输入通道数，默认为3（黑子、白子、当前玩家）
            num_channels (int): 卷积层通道数，默认为32
            num_res_blocks (int): 残差块数量，默认为3
        """
        super(DualNetworkModel, self).__init__()
        
        self.board_size = board_size
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        
        # 输入卷积层
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 策略头（输出每个位置的落子概率）
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 价值头（输出当前局面的评估值）
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, input_channels, board_size, board_size]
            
        返回:
            tuple: (policy_output, value_output)
                policy_output (torch.Tensor): 策略输出，形状为 [batch_size, board_size * board_size]
                value_output (torch.Tensor): 价值输出，形状为 [batch_size, 1]
        """
        # 输入层
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 残差块
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class ResBlock(nn.Module):
    """
    残差块，用于构建深层神经网络。
    """
    
    def __init__(self, num_channels):
        """
        初始化残差块
        
        参数:
            num_channels (int): 通道数
        """
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 输出张量
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


def prepare_input(board_state, current_player):
    """
    将棋盘状态转换为神经网络输入格式
    
    参数:
        board_state (numpy.ndarray): 棋盘状态矩阵
        current_player (Player): 当前玩家
        
    返回:
        torch.Tensor: 神经网络输入张量
    """
    # 导入Player枚举
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from game.board import Player
    
    board_size = board_state.shape[0]
    
    # 创建3个通道：黑子、白子、当前玩家
    x = np.zeros((3, board_size, board_size), dtype=np.float32)
    
    # 黑子通道
    x[0] = (board_state == Player.BLACK.value).astype(np.float32)
    
    # 白子通道
    x[1] = (board_state == Player.WHITE.value).astype(np.float32)
    
    # 当前玩家通道（全1表示当前玩家是黑方，全0表示当前玩家是白方）
    x[2] = np.ones((board_size, board_size), dtype=np.float32) if current_player == Player.BLACK else np.zeros((board_size, board_size), dtype=np.float32)
    
    # 转换为PyTorch张量
    x = torch.from_numpy(x)
    
    # 添加批次维度
    x = x.unsqueeze(0)
    
    return x


def create_model(board_size=15, device='cpu'):
    """
    创建神经网络模型
    
    参数:
        board_size (int): 棋盘大小
        device (str): 设备类型，'cpu'或'cuda'
        
    返回:
        DualNetworkModel: 神经网络模型
    """
    model = DualNetworkModel(board_size=board_size)
    model.to(device)
    return model


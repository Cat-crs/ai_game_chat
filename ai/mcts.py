"""
蒙特卡洛树搜索(MCTS)模块，实现AI决策算法。
"""
import numpy as np
import math
import torch
from copy import deepcopy
import time


class MCTSNode:
    """
    蒙特卡洛树搜索节点类
    
    属性:
        game: 游戏对象
        parent: 父节点
        move: 到达该节点的落子位置
        children: 子节点列表
        visits: 访问次数
        value_sum: 价值总和
        prior: 先验概率
    """
    
    def __init__(self, game, parent=None, move=None, prior=0.0):
        """
        初始化MCTS节点
        
        参数:
            game: 游戏对象
            parent (MCTSNode): 父节点
            move (tuple): 到达该节点的落子位置 (x, y)
            prior (float): 先验概率
        """
        self.game = deepcopy(game)  # 深拷贝游戏状态
        self.parent = parent
        self.move = move
        self.children = {}  # 子节点字典 {move: node}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        
    def is_expanded(self):
        """
        检查节点是否已展开
        
        返回:
            bool: 是否已展开
        """
        return len(self.children) > 0
        
    def select_child(self, c_puct=1.0):
        """
        根据UCB公式选择子节点
        
        参数:
            c_puct (float): 探索常数
            
        返回:
            tuple: (选中的子节点, 对应的落子位置)
        """
        # UCB公式: Q(s,a) + c_puct * P(s,a) * sqrt(sum(N(s,b))) / (1 + N(s,a))
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        for move, child in self.children.items():
            # 计算UCB分数
            if child.visits > 0:
                q_value = child.value_sum / child.visits
            else:
                q_value = 0
                
            exploration = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            ucb_score = q_value + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_move = move
                best_child = child
                
        return best_child, best_move
        
    def expand(self, policy):
        """
        展开节点
        
        参数:
            policy (numpy.ndarray): 策略概率分布
        """
        legal_moves = self.game.get_legal_moves()
        board_size = self.game.board.size
        
        for x, y in legal_moves:
            move = (x, y)
            move_idx = x * board_size + y
            
            # 创建新游戏状态
            new_game = deepcopy(self.game)
            new_game.place_stone(x, y)
            
            # 创建子节点
            self.children[move] = MCTSNode(
                game=new_game,
                parent=self,
                move=move,
                prior=policy[move_idx]
            )
            
    def update(self, value):
        """
        更新节点统计信息
        
        参数:
            value (float): 节点价值
        """
        self.visits += 1
        self.value_sum += value
        
    def get_value(self):
        """
        获取节点价值
        
        返回:
            float: 节点价值
        """
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits


class MCTS:
    """
    蒙特卡洛树搜索类
    
    属性:
        model: 神经网络模型
        num_simulations: 模拟次数
        c_puct: 探索常数
        device: 设备类型
    """
    
    def __init__(self, model, num_simulations=800, c_puct=1.0, device='cpu'):
        """
        初始化MCTS
        
        参数:
            model: 神经网络模型
            num_simulations (int): 模拟次数
            c_puct (float): 探索常数
            device (str): 设备类型
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        
    def search(self, game):
        """
        执行MCTS搜索
        
        参数:
            game: 游戏对象
            
        返回:
            numpy.ndarray: 落子概率分布
        """
        # 创建根节点
        root = MCTSNode(game)
        
        # 如果游戏已结束，直接返回均匀分布
        if game.is_game_over():
            board_size = game.board.size
            return np.ones(board_size * board_size) / (board_size * board_size)
        
        # 获取根节点的策略和价值
        state = game.get_state()
        current_player = game.get_current_player()
        
        from .model import prepare_input
        x = prepare_input(state, current_player)
        x = x.to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.model(x)
            
        policy = torch.exp(policy_logits).cpu().numpy()[0]
        
        # 只保留合法落子位置的概率
        legal_moves = game.get_legal_moves()
        board_size = game.board.size
        mask = np.zeros(board_size * board_size, dtype=np.float32)
        
        for x, y in legal_moves:
            mask[x * board_size + y] = 1
            
        policy = policy * mask
        
        # 如果没有合法落子，返回均匀分布
        if np.sum(policy) == 0:
            policy = mask
            
        # 归一化
        policy = policy / np.sum(policy)
        
        # 展开根节点
        root.expand(policy)
        
        # 执行模拟
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 选择阶段：从根节点到叶节点
            while node.is_expanded() and not node.game.is_game_over():
                node, move = node.select_child(self.c_puct)
                search_path.append(node)
                
            # 如果游戏未结束，展开叶节点
            game_over = node.game.is_game_over()
            value = 0.0
            
            if not game_over:
                state = node.game.get_state()
                current_player = node.game.get_current_player()
                
                x = prepare_input(state, current_player)
                x = x.to(self.device)
                
                with torch.no_grad():
                    policy_logits, value = self.model(x)
                    
                policy = torch.exp(policy_logits).cpu().numpy()[0]
                value = value.item()
                
                # 只保留合法落子位置的概率
                legal_moves = node.game.get_legal_moves()
                board_size = node.game.board.size
                mask = np.zeros(board_size * board_size, dtype=np.float32)
                
                for x, y in legal_moves:
                    mask[x * board_size + y] = 1
                    
                policy = policy * mask
                
                # 如果没有合法落子，返回均匀分布
                if np.sum(policy) == 0:
                    policy = mask
                    
                # 归一化
                policy = policy / np.sum(policy)
                
                # 展开节点
                node.expand(policy)
            else:
                # 游戏结束，获取胜负结果
                winner = node.game.get_winner()
                
                # 导入Player枚举
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from game.board import Player
                
                if winner is None:  # 平局
                    value = 0.0
                elif winner == Player.BLACK:
                    value = 1.0 if node.game.get_current_player() == Player.BLACK else -1.0
                else:  # winner == Player.WHITE
                    value = 1.0 if node.game.get_current_player() == Player.WHITE else -1.0
                    
            # 反向传播
            for node in reversed(search_path):
                # 从当前玩家的角度更新价值
                if node != search_path[-1]:
                    value = -value
                    
                node.update(value)
                
        # 计算落子概率分布
        visit_counts = np.zeros(board_size * board_size, dtype=np.float32)
        
        for move, child in root.children.items():
            x, y = move
            visit_counts[x * board_size + y] = child.visits
            
        # 归一化
        if np.sum(visit_counts) > 0:
            visit_counts = visit_counts / np.sum(visit_counts)
            
        return visit_counts
        
    def get_best_move(self, game, temperature=0.0):
        """
        获取最佳落子位置
        
        参数:
            game: 游戏对象
            temperature (float): 温度参数，控制探索程度
            
        返回:
            tuple: 最佳落子位置 (x, y)
        """
        board_size = game.board.size
        visit_counts = self.search(game)
        
        if temperature == 0:
            # 选择访问次数最多的落子位置
            move_idx = np.argmax(visit_counts)
        else:
            # 根据温度参数进行采样
            visit_counts = np.power(visit_counts, 1.0 / temperature)
            visit_counts = visit_counts / np.sum(visit_counts)
            move_idx = np.random.choice(board_size * board_size, p=visit_counts)
            
        x = move_idx // board_size
        y = move_idx % board_size
        
        return x, y


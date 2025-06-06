"""
自我对弈模块，实现AI自我学习功能。
"""
import numpy as np
import torch
import os
import time
from tqdm import tqdm
import random
from copy import deepcopy

# 导入自定义模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game.gomoku import Gomoku
from game.go import Go
from game.board import Player
from ai.model import create_model, prepare_input
from ai.mcts import MCTS
from ai.training import Trainer, TrainingExample


class SelfPlay:
    """
    自我对弈类，用于生成训练数据并提升AI能力。
    
    属性:
        game_type (str): 游戏类型，'gomoku'或'go'
        board_size (int): 棋盘大小
        model: 神经网络模型
        mcts: 蒙特卡洛树搜索
        device (str): 设备类型
        model_dir (str): 模型保存目录
    """
    
    def __init__(self, game_type='gomoku', board_size=15, model=None, num_simulations=800, device='cpu', model_dir='models'):
        """
        初始化自我对弈
        
        参数:
            game_type (str): 游戏类型，'gomoku'或'go'
            board_size (int): 棋盘大小
            model: 神经网络模型，如果为None则创建新模型
            num_simulations (int): MCTS模拟次数
            device (str): 设备类型
            model_dir (str): 模型保存目录
        """
        self.game_type = game_type
        self.board_size = board_size
        self.device = device
        self.model_dir = model_dir
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建或加载模型
        if model is None:
            self.model = create_model(board_size=board_size, device=device)
        else:
            self.model = model
            
        # 创建MCTS
        self.mcts = MCTS(self.model, num_simulations=num_simulations, device=device)
        
        # 创建训练器
        self.trainer = Trainer(self.model, device=device)
        
    def execute_episode(self, temperature=1.0):
        """
        执行一局自我对弈
        
        参数:
            temperature (float): 温度参数，控制探索程度
            
        返回:
            list: 训练样本列表
        """
        # 创建游戏
        if self.game_type == 'gomoku':
            game = Gomoku(board_size=self.board_size)
        else:  # go
            game = Go(board_size=self.board_size)
            
        # 训练样本列表
        train_examples = []
        
        # 游戏状态历史
        state_history = []
        
        # 执行游戏
        step = 0
        
        while not game.is_game_over():
            # 记录当前状态
            state = game.get_state()
            current_player = game.get_current_player()
            state_history.append((deepcopy(state), current_player))
            
            # 计算落子概率分布
            pi = self.mcts.search(game)
            
            # 根据温度参数调整概率分布
            if temperature == 0:
                # 选择最佳落子
                best_idx = np.argmax(pi)
                pi = np.zeros_like(pi)
                pi[best_idx] = 1.0
            else:
                # 根据温度参数调整概率分布
                pi = np.power(pi, 1.0 / temperature)
                pi = pi / np.sum(pi)
                
            # 保存训练样本（先不设置价值，后面根据游戏结果更新）
            train_examples.append(TrainingExample(
                state=deepcopy(state),
                current_player=current_player,
                policy=pi,
                value=0.0  # 临时值，后面更新
            ))
            
            # 选择落子位置
            move_idx = np.random.choice(len(pi), p=pi)
            x = move_idx // self.board_size
            y = move_idx % self.board_size
            
            # 执行落子
            game.place_stone(x, y)
            
            # 每10步打印一次棋盘状态
            step += 1
            if step % 10 == 0:
                print(f"Step {step}:")
                print(game)
                
        # 游戏结束，获取胜负结果
        winner = game.get_winner()
        
        # 更新训练样本的价值
        for i in range(len(train_examples)):
            if winner is None:  # 平局
                value = 0.0
            else:
                # 从当前玩家的角度计算价值
                player = state_history[i][1]
                value = 1.0 if player == winner else -1.0
                
            train_examples[i].value = value
            
        print("游戏结束:")
        print(game)
        if winner:
            print(f"获胜者: {'黑方' if winner == Player.BLACK else '白方'}")
        else:
            print("平局")
            
        return train_examples
        
    def learn(self, num_episodes=100, epochs=10, batch_size=64, temperature_schedule=None):
        """
        通过自我对弈学习
        
        参数:
            num_episodes (int): 自我对弈局数
            epochs (int): 每次训练的轮数
            batch_size (int): 批次大小
            temperature_schedule (list): 温度参数调度表，如果为None则使用默认值
            
        返回:
            tuple: (策略损失, 价值损失)
        """
        # 默认温度参数调度
        if temperature_schedule is None:
            temperature_schedule = [(0.0, 0.8), (0.5, 0.4), (0.75, 0.2), (1.0, 0.1)]
            
        # 训练样本列表
        all_examples = []
        
        # 执行自我对弈
        for i in range(num_episodes):
            print(f"Episode {i+1}/{num_episodes}")
            
            # 根据进度选择温度参数
            progress = i / num_episodes
            temperature = 1.0
            
            for threshold, temp in temperature_schedule:
                if progress >= threshold:
                    temperature = temp
                    
            print(f"使用温度参数: {temperature}")
            
            # 执行一局自我对弈
            examples = self.execute_episode(temperature=temperature)
            all_examples.extend(examples)
            
            # 每10局保存一次模型
            if (i + 1) % 10 == 0:
                self.save_model(f"{self.model_dir}/{self.game_type}_model_{i+1}.pth")
                
        # 训练模型
        print("训练模型...")
        self.trainer.epochs = epochs
        self.trainer.batch_size = batch_size
        policy_loss, value_loss = self.trainer.train(all_examples)
        
        # 保存最终模型
        self.save_model(f"{self.model_dir}/{self.game_type}_model_final.pth")
        
        return policy_loss, value_loss
        
    def save_model(self, filepath):
        """
        保存模型
        
        参数:
            filepath (str): 模型保存路径
        """
        self.trainer.save_model(filepath)
        
    def load_model(self, filepath):
        """
        加载模型
        
        参数:
            filepath (str): 模型加载路径
            
        返回:
            bool: 是否成功加载
        """
        return self.trainer.load_model(filepath)
        
    def play_against_human(self, human_first=True):
        """
        与人类玩家对弈
        
        参数:
            human_first (bool): 人类玩家是否先手
        """
        # 创建游戏
        if self.game_type == 'gomoku':
            game = Gomoku(board_size=self.board_size)
        else:  # go
            game = Go(board_size=self.board_size)
            
        # 设置AI玩家
        ai_player = Player.WHITE if human_first else Player.BLACK
        
        print("===== AI对弈 =====")
        print(f"游戏类型: {'五子棋' if self.game_type == 'gomoku' else '围棋'}")
        print(f"{'人类' if human_first else 'AI'}先手")
        print("输入格式：列行，如A1表示左上角，Q退出")
        
        print(game)
        
        while not game.is_game_over():
            current_player = game.get_current_player()
            
            if current_player == ai_player:
                # AI回合
                print("AI思考中...")
                x, y = self.mcts.get_best_move(game, temperature=0.0)
                print(f"AI落子在 {chr(ord('A') + y)}{x+1}")
                game.place_stone(x, y)
            else:
                # 人类回合
                while True:
                    move = input("请输入落子位置（如A1）或Q退出: ")
                    
                    if move.upper() == 'Q':
                        print("游戏已退出")
                        return
                        
                    # 解析落子位置
                    if len(move) < 2:
                        print("输入无效，请重新输入")
                        continue
                        
                    col = move[0].upper()
                    if not ('A' <= col <= chr(ord('A') + self.board_size - 1)):
                        print("输入无效，请重新输入")
                        continue
                        
                    try:
                        row = int(move[1:]) - 1
                        if row < 0 or row >= self.board_size:
                            print("输入无效，请重新输入")
                            continue
                    except ValueError:
                        print("输入无效，请重新输入")
                        continue
                        
                    y = ord(col) - ord('A')
                    x = row
                    
                    if not game.place_stone(x, y):
                        print("落子无效，请重新输入")
                        continue
                        
                    break
                    
            print(game)
            
        # 游戏结束
        winner = game.get_winner()
        if winner:
            if winner == ai_player:
                print("AI获胜！")
            else:
                print("恭喜，你获胜了！")
        else:
            print("游戏平局！")


def main():
    """主函数"""
    # 创建自我对弈
    self_play = SelfPlay(game_type='gomoku', board_size=15, num_simulations=100)
    
    # 学习
    print("开始自我学习...")
    self_play.learn(num_episodes=5, epochs=5, batch_size=32)
    
    # 与人类玩家对弈
    self_play.play_against_human()


if __name__ == "__main__":
    main()


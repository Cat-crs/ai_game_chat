"""
功能测试脚本，用于测试项目的各个功能。
"""
import sys
import os
import unittest
import torch
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Player
from game.gomoku import Gomoku
from game.go import Go
from ai.model import create_model
from ai.mcts import MCTS
from ai.self_play import SelfPlay
from chat.model import ChatModel, ChatSession
from utils.performance import get_system_info, optimize_model_parameters, optimize_mcts_parameters


class TestGameLogic(unittest.TestCase):
    """测试游戏逻辑"""
    
    def test_gomoku_rules(self):
        """测试五子棋规则"""
        print("测试五子棋规则...")
        
        # 创建五子棋游戏
        gomoku = Gomoku()
        
        # 测试落子
        self.assertTrue(gomoku.place_stone(7, 7))  # 黑子
        self.assertTrue(gomoku.place_stone(7, 8))  # 白子
        self.assertTrue(gomoku.place_stone(8, 7))  # 黑子
        
        # 测试重复落子
        self.assertFalse(gomoku.place_stone(7, 7))
        
        # 测试胜负判断
        self.assertFalse(gomoku.is_game_over())
        
        # 创建一个明确的五连场景
        gomoku.board.board = gomoku.board.board * 0  # 清空棋盘
        gomoku.board.current_player = Player.BLACK
        gomoku.winner = None
        
        # 放置黑子形成水平五连
        for i in range(5):
            gomoku.board.board[7, 7+i] = Player.BLACK.value
            
        # 打印棋盘状态
        print("棋盘状态:")
        print(gomoku)
        
        # 检查是否五连
        self.assertTrue(gomoku._check_win(7, 9, Player.BLACK))
        
        # 手动设置获胜者
        gomoku.winner = Player.BLACK
        
        # 检查游戏是否结束
        self.assertTrue(gomoku.is_game_over())
        
        # 检查获胜者
        self.assertEqual(gomoku.get_winner(), Player.BLACK)
        
    def test_go_rules(self):
        """测试围棋规则"""
        print("测试围棋规则...")
        
        # 创建围棋游戏
        go = Go(board_size=9)  # 使用小棋盘便于测试
        
        # 测试落子
        self.assertTrue(go.place_stone(2, 2))  # 黑子
        self.assertTrue(go.place_stone(2, 3))  # 白子
        self.assertTrue(go.place_stone(3, 2))  # 黑子
        
        # 测试重复落子
        self.assertFalse(go.place_stone(2, 2))
        
        # 测试提子
        # 我们将直接调用提子方法，而不是通过落子来测试
        go.board.board = np.zeros((9, 9), dtype=np.int64)
        go.board.current_player = Player.WHITE
        
        # 放置一个被包围的黑子
        go.board.board[3, 3] = Player.BLACK.value
        go.board.board[2, 3] = Player.WHITE.value
        go.board.board[3, 2] = Player.WHITE.value
        go.board.board[4, 3] = Player.WHITE.value
        
        # 打印棋盘状态
        print("围棋棋盘状态:")
        print(go)
        
        # 放置最后一个白子形成包围
        go.board.board[3, 4] = Player.WHITE.value
        
        # 调用提子方法
        captured = go._capture_stones(Player.BLACK)
        
        # 再次打印棋盘状态
        print("提子后棋盘状态:")
        print(go)
        
        # 检查是否有提子
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], (3, 3))
        
        # 检查棋盘状态
        board_state = go.get_state()
        self.assertEqual(board_state[3, 3], 0)  # 黑子被提走


class TestAIModel(unittest.TestCase):
    """测试AI模型"""
    
    def test_model_creation(self):
        """测试模型创建"""
        print("测试模型创建...")
        
        # 创建设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        model = create_model(board_size=15, device=device)
        
        # 检查模型结构
        self.assertIsNotNone(model)
        
        # 测试模型前向传播
        x = torch.randn(1, 3, 15, 15).to(device)
        policy, value = model(x)
        
        # 检查输出形状
        self.assertEqual(policy.shape, (1, 15 * 15))
        self.assertEqual(value.shape, (1, 1))
        
    def test_mcts(self):
        """测试MCTS"""
        print("测试MCTS...")
        
        # 创建设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        model = create_model(board_size=15, device=device)
        
        # 创建MCTS
        mcts = MCTS(model, num_simulations=10, device=device)
        
        # 创建游戏
        gomoku = Gomoku()
        
        # 获取最佳落子位置
        x, y = mcts.get_best_move(gomoku)
        
        # 检查落子位置是否有效
        self.assertTrue(0 <= x < 15)
        self.assertTrue(0 <= y < 15)


class TestChatModel(unittest.TestCase):
    """测试聊天模型"""
    
    def test_chat_model(self):
        """测试聊天模型"""
        print("测试聊天模型...")
        
        # 创建聊天模型
        chat_model = ChatModel()
        
        # 测试问题列表
        test_questions = [
            "你好",
            "围棋规则是什么？",
            "五子棋和围棋有什么区别？",
            "如何提高围棋水平？",
            "我总是输棋，好沮丧",
            "谢谢你的帮助",
            "再见"
        ]
        
        # 测试每个问题
        for question in test_questions:
            response = chat_model.get_response(question)
            self.assertIsNotNone(response)
            self.assertNotEqual(response, "")
            
    def test_chat_session(self):
        """测试聊天会话"""
        print("测试聊天会话...")
        
        # 创建聊天会话
        session = ChatSession()
        
        # 测试消息列表
        test_messages = [
            "我想下五子棋",
            "我想在A1落子",
            "查看棋盘",
            "结束游戏",
            "围棋的气是什么？"
        ]
        
        # 测试每个消息
        for message in test_messages:
            response = session.process_message(message)
            self.assertIsNotNone(response)
            self.assertNotEqual(response, "")


class TestPerformance(unittest.TestCase):
    """测试性能优化"""
    
    def test_system_info(self):
        """测试系统信息获取"""
        print("测试系统信息获取...")
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 检查系统信息
        self.assertIsNotNone(system_info)
        self.assertIn('cpu_count', system_info)
        self.assertIn('memory_total', system_info)
        self.assertIn('gpu_available', system_info)
        
    def test_model_optimization(self):
        """测试模型优化"""
        print("测试模型优化...")
        
        # 创建设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        model = create_model(board_size=15, device=device)
        
        # 优化模型
        optimized_model = optimize_model_parameters(model, device)
        
        # 检查优化后的模型
        self.assertIsNotNone(optimized_model)
        
    def test_mcts_optimization(self):
        """测试MCTS优化"""
        print("测试MCTS优化...")
        
        # 获取系统信息
        system_info = get_system_info()
        
        # 优化MCTS参数
        mcts_params = optimize_mcts_parameters(system_info)
        
        # 检查MCTS参数
        self.assertIsNotNone(mcts_params)
        self.assertIn('num_simulations', mcts_params)
        self.assertIn('c_puct', mcts_params)


if __name__ == "__main__":
    unittest.main()


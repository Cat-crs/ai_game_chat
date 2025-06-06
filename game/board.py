"""
棋盘类模块，提供通用的棋盘表示和操作。
"""
import numpy as np
from enum import Enum


class Player(Enum):
    """玩家枚举类，表示黑白双方"""
    BLACK = 1
    WHITE = 2
    EMPTY = 0


class Board:
    """
    通用棋盘类，用于表示棋盘状态和基本操作。
    
    属性:
        size (int): 棋盘大小，如9、13、15、19等
        board (numpy.ndarray): 棋盘状态矩阵
        current_player (Player): 当前玩家
        move_history (list): 着子历史记录
    """
    
    def __init__(self, size=15):
        """
        初始化棋盘
        
        参数:
            size (int): 棋盘大小，默认为15（适合五子棋）
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = Player.BLACK  # 黑方先行
        self.move_history = []
        
    def reset(self):
        """重置棋盘状态"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = Player.BLACK
        self.move_history = []
        
    def is_valid_move(self, x, y):
        """
        检查落子是否有效
        
        参数:
            x (int): 横坐标
            y (int): 纵坐标
            
        返回:
            bool: 落子是否有效
        """
        # 检查坐标是否在棋盘范围内
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        
        # 检查该位置是否已有棋子
        return self.board[x, y] == Player.EMPTY.value
    
    def place_stone(self, x, y):
        """
        在指定位置落子
        
        参数:
            x (int): 横坐标
            y (int): 纵坐标
            
        返回:
            bool: 落子是否成功
        """
        if not self.is_valid_move(x, y):
            return False
        
        self.board[x, y] = self.current_player.value
        self.move_history.append((x, y, self.current_player))
        
        # 切换玩家
        self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        
        return True
    
    def undo_move(self):
        """
        撤销上一步落子
        
        返回:
            bool: 撤销是否成功
        """
        if not self.move_history:
            return False
        
        x, y, player = self.move_history.pop()
        self.board[x, y] = Player.EMPTY.value
        self.current_player = player  # 恢复到上一步的玩家
        
        return True
    
    def get_state(self):
        """
        获取当前棋盘状态
        
        返回:
            numpy.ndarray: 棋盘状态矩阵的副本
        """
        return self.board.copy()
    
    def get_legal_moves(self):
        """
        获取所有合法落子位置
        
        返回:
            list: 所有合法落子位置的坐标列表 [(x1, y1), (x2, y2), ...]
        """
        legal_moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_valid_move(x, y):
                    legal_moves.append((x, y))
        return legal_moves
    
    def is_full(self):
        """
        检查棋盘是否已满
        
        返回:
            bool: 棋盘是否已满
        """
        return len(self.get_legal_moves()) == 0
    
    def __str__(self):
        """
        返回棋盘的字符串表示，用于打印显示
        
        返回:
            str: 棋盘的字符串表示
        """
        symbols = {
            Player.EMPTY.value: '.',
            Player.BLACK.value: 'X',
            Player.WHITE.value: 'O'
        }
        
        # 添加列标签
        result = '  ' + ' '.join([chr(ord('A') + i) for i in range(self.size)]) + '\n'
        
        # 添加行标签和棋盘内容
        for i in range(self.size):
            row = f"{i+1:2d} " if i < 9 else f"{i+1} "
            row += ' '.join([symbols[self.board[i, j]] for j in range(self.size)])
            result += row + '\n'
            
        return result


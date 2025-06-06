"""
五子棋游戏规则模块，实现五子棋特有的规则和胜负判断。
"""
from .board import Board, Player


class Gomoku:
    """
    五子棋游戏类，实现五子棋特有的规则和胜负判断。
    
    属性:
        board (Board): 棋盘对象
        winner (Player): 获胜者
    """
    
    def __init__(self, board_size=15):
        """
        初始化五子棋游戏
        
        参数:
            board_size (int): 棋盘大小，默认为15x15
        """
        self.board = Board(board_size)
        self.winner = None
        
    def reset(self):
        """重置游戏状态"""
        self.board.reset()
        self.winner = None
        
    def place_stone(self, x, y):
        """
        在指定位置落子，并检查是否获胜
        
        参数:
            x (int): 横坐标
            y (int): 纵坐标
            
        返回:
            bool: 落子是否成功
        """
        current_player = self.board.current_player
        if not self.board.place_stone(x, y):
            return False
        
        # 检查是否获胜
        if self._check_win(x, y, current_player):
            self.winner = current_player
            
        return True
    
    def _check_win(self, x, y, player):
        """
        检查指定玩家在指定位置落子后是否获胜（五子连珠）
        
        参数:
            x (int): 横坐标
            y (int): 纵坐标
            player (Player): 玩家
            
        返回:
            bool: 是否获胜
        """
        directions = [
            (1, 0),   # 水平
            (0, 1),   # 垂直
            (1, 1),   # 右下对角线
            (1, -1)   # 右上对角线
        ]
        
        board_state = self.board.get_state()
        player_value = player.value
        
        for dx, dy in directions:
            count = 1  # 当前位置已经有一个棋子
            
            # 正向检查
            for i in range(1, 5):  # 最多检查4步
                nx, ny = x + i * dx, y + i * dy
                if (nx < 0 or nx >= self.board.size or 
                    ny < 0 or ny >= self.board.size or 
                    board_state[nx, ny] != player_value):
                    break
                count += 1
                
            # 反向检查
            for i in range(1, 5):  # 最多检查4步
                nx, ny = x - i * dx, y - i * dy
                if (nx < 0 or nx >= self.board.size or 
                    ny < 0 or ny >= self.board.size or 
                    board_state[nx, ny] != player_value):
                    break
                count += 1
                
            if count >= 5:
                return True
                
        return False
    
    def is_game_over(self):
        """
        检查游戏是否结束
        
        返回:
            bool: 游戏是否结束
        """
        return self.winner is not None or self.board.is_full()
    
    def get_winner(self):
        """
        获取获胜者
        
        返回:
            Player: 获胜者，如果没有获胜者则返回None
        """
        return self.winner
    
    def get_current_player(self):
        """
        获取当前玩家
        
        返回:
            Player: 当前玩家
        """
        return self.board.current_player
    
    def get_legal_moves(self):
        """
        获取所有合法落子位置
        
        返回:
            list: 所有合法落子位置的坐标列表 [(x1, y1), (x2, y2), ...]
        """
        return self.board.get_legal_moves()
    
    def get_state(self):
        """
        获取当前棋盘状态
        
        返回:
            numpy.ndarray: 棋盘状态矩阵的副本
        """
        return self.board.get_state()
    
    def __str__(self):
        """
        返回游戏状态的字符串表示
        
        返回:
            str: 游戏状态的字符串表示
        """
        result = str(self.board)
        
        if self.winner:
            result += f"\n获胜者: {'黑方' if self.winner == Player.BLACK else '白方'}"
        elif self.board.is_full():
            result += "\n游戏平局"
        else:
            result += f"\n当前玩家: {'黑方' if self.board.current_player == Player.BLACK else '白方'}"
            
        return result


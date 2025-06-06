"""
围棋游戏规则模块，实现围棋特有的规则和胜负判断。
"""
import numpy as np
from .board import Board, Player
from collections import deque


class Go:
    """
    围棋游戏类，实现围棋特有的规则和胜负判断。
    
    属性:
        board (Board): 棋盘对象
        size (int): 棋盘大小
        ko_point (tuple): 劫争点，禁止下一手落子的位置
        black_captures (int): 黑方提子数
        white_captures (int): 白方提子数
        game_ended (bool): 游戏是否结束
        passes (int): 连续弃权次数
    """
    
    def __init__(self, board_size=19):
        """
        初始化围棋游戏
        
        参数:
            board_size (int): 棋盘大小，默认为19x19
        """
        self.board = Board(board_size)
        self.size = board_size
        self.ko_point = None
        self.black_captures = 0
        self.white_captures = 0
        self.game_ended = False
        self.passes = 0
        self.last_board_state = None
        
    def reset(self):
        """重置游戏状态"""
        self.board.reset()
        self.ko_point = None
        self.black_captures = 0
        self.white_captures = 0
        self.game_ended = False
        self.passes = 0
        self.last_board_state = None
        
    def place_stone(self, x, y):
        """
        在指定位置落子，并处理提子等逻辑
        
        参数:
            x (int): 横坐标
            y (int): 纵坐标
            
        返回:
            bool: 落子是否成功
        """
        # 检查游戏是否已结束
        if self.game_ended:
            return False
            
        # 检查是否为劫争点
        if self.ko_point and self.ko_point == (x, y):
            return False
            
        # 保存当前棋盘状态用于后续检查
        current_state = self.board.get_state().copy()
        current_player = self.board.current_player
        opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK
        
        # 尝试落子
        if not self.board.is_valid_move(x, y):
            return False
            
        # 实际落子
        self.board.board[x, y] = current_player.value
        
        # 检查并提取对方的死子
        captured_stones = self._capture_stones(opponent)
        
        # 如果没有提子，检查自己的这个子组是否有气
        if not captured_stones:
            # 如果自己的这个子组没有气，则是自杀，不允许落子
            if not self._has_liberty(x, y):
                # 恢复棋盘状态
                self.board.board = current_state
                return False
                
        # 更新提子数
        if current_player == Player.BLACK:
            self.black_captures += len(captured_stones)
        else:
            self.white_captures += len(captured_stones)
            
        # 处理劫争
        self.ko_point = None
        if len(captured_stones) == 1 and self._is_ko_situation(x, y, captured_stones[0]):
            self.ko_point = captured_stones[0]
            
        # 重置连续弃权次数
        self.passes = 0
        
        # 保存上一个棋盘状态
        self.last_board_state = current_state
        
        # 切换玩家
        self.board.current_player = opponent
        self.board.move_history.append((x, y, current_player))
        
        return True
        
    def pass_move(self):
        """
        玩家选择弃权
        
        返回:
            bool: 操作是否成功
        """
        if self.game_ended:
            return False
            
        # 切换玩家
        self.board.current_player = Player.WHITE if self.board.current_player == Player.BLACK else Player.BLACK
        self.board.move_history.append((-1, -1, self.board.current_player))  # 使用(-1, -1)表示弃权
        
        # 增加连续弃权次数
        self.passes += 1
        
        # 如果连续两次弃权，游戏结束
        if self.passes >= 2:
            self.game_ended = True
            
        # 重置劫争点
        self.ko_point = None
        
        return True
        
    def _capture_stones(self, player):
        """
        检查并提取指定玩家的死子
        
        参数:
            player (Player): 要检查的玩家
            
        返回:
            list: 被提取的死子坐标列表
        """
        captured = []
        board_state = self.board.board
        
        # 检查整个棋盘
        for x in range(self.size):
            for y in range(self.size):
                # 如果是指定玩家的棋子，且没有气，则提取
                if board_state[x, y] == player.value and not self._has_liberty(x, y):
                    # 获取整个连通棋子组
                    group = self._get_connected_group(x, y)
                    
                    # 提取这些棋子
                    for gx, gy in group:
                        board_state[gx, gy] = Player.EMPTY.value
                        captured.append((gx, gy))
                        
        return captured
        
    def _has_liberty(self, x, y):
        """
        检查指定位置的棋子组是否有气
        
        参数:
            x (int): 横坐标
            y (int): 纵坐标
            
        返回:
            bool: 是否有气
        """
        board_state = self.board.board
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
            
        if board_state[x, y] == Player.EMPTY.value:
            return True
            
        # 获取棋子所在的连通组
        stone_color = board_state[x, y]
        group = self._get_connected_group(x, y)
        
        # 检查组中任何棋子是否有气
        for gx, gy in group:
            # 检查四个方向
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = gx + dx, gy + dy
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    board_state[nx, ny] == Player.EMPTY.value):
                    return True
                    
        return False
        
    def _get_connected_group(self, x, y):
        """
        获取与指定位置棋子相连的所有同色棋子
        
        参数:
            x (int): 横坐标
            y (int): 纵坐标
            
        返回:
            list: 连通棋子组的坐标列表
        """
        board_state = self.board.board
        stone_color = board_state[x, y]
        
        if stone_color == Player.EMPTY.value:
            return []
            
        # 使用BFS查找连通棋子组
        group = []
        queue = deque([(x, y)])
        visited = set([(x, y)])
        
        while queue:
            cx, cy = queue.popleft()
            group.append((cx, cy))
            
            # 检查四个方向
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    board_state[nx, ny] == stone_color and 
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    
        return group
        
    def _is_ko_situation(self, x, y, captured):
        """
        检查是否为劫争情况
        
        参数:
            x (int): 当前落子的横坐标
            y (int): 当前落子的纵坐标
            captured (tuple): 被提取的棋子坐标
            
        返回:
            bool: 是否为劫争
        """
        # 如果提了一个子，且这个子的四周只有一个我方棋子（刚下的那个）
        cx, cy = captured
        
        # 检查四个方向
        adjacent_my_stones = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < self.size and 0 <= ny < self.size and 
                self.board.board[nx, ny] == self.board.current_player.value):
                adjacent_my_stones += 1
                
        # 如果只有一个相邻的我方棋子，且那个棋子是刚下的，则为劫争
        return adjacent_my_stones == 1 and (abs(cx - x) + abs(cy - y)) == 1
        
    def score_game(self):
        """
        计算游戏得分（简化版，仅计算棋子数和提子数）
        
        返回:
            tuple: (黑方得分, 白方得分)
        """
        if not self.game_ended:
            return None
            
        # 简化版得分计算，实际围棋需要计算领地
        board_state = self.board.board
        black_stones = np.sum(board_state == Player.BLACK.value)
        white_stones = np.sum(board_state == Player.WHITE.value)
        
        black_score = black_stones + self.black_captures
        white_score = white_stones + self.white_captures + 6.5  # 贴目
        
        return black_score, white_score
        
    def get_winner(self):
        """
        获取获胜者
        
        返回:
            Player: 获胜者，如果没有获胜者则返回None
        """
        if not self.game_ended:
            return None
            
        black_score, white_score = self.score_game()
        
        if black_score > white_score:
            return Player.BLACK
        else:
            return Player.WHITE
            
    def is_game_over(self):
        """
        检查游戏是否结束
        
        返回:
            bool: 游戏是否结束
        """
        return self.game_ended
        
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
        if self.game_ended:
            return []
            
        legal_moves = []
        
        # 遍历所有空点
        for x in range(self.size):
            for y in range(self.size):
                if self.board.board[x, y] == Player.EMPTY.value and (x, y) != self.ko_point:
                    # 尝试落子
                    board_copy = self.board.board.copy()
                    current_player = self.board.current_player
                    
                    # 模拟落子
                    self.board.board[x, y] = current_player.value
                    
                    # 检查是否有提子
                    opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK
                    captured = self._capture_stones(opponent)
                    
                    # 如果没有提子，检查自己的这个子组是否有气
                    if not captured and not self._has_liberty(x, y):
                        # 恢复棋盘状态
                        self.board.board = board_copy
                        continue
                        
                    # 恢复棋盘状态
                    self.board.board = board_copy
                    
                    # 这是一个合法的落子点
                    legal_moves.append((x, y))
                    
        return legal_moves
        
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
        
        if self.game_ended:
            black_score, white_score = self.score_game()
            result += f"\n游戏结束\n黑方得分: {black_score}\n白方得分: {white_score}"
            result += f"\n获胜者: {'黑方' if black_score > white_score else '白方'}"
        else:
            result += f"\n当前玩家: {'黑方' if self.board.current_player == Player.BLACK else '白方'}"
            result += f"\n黑方提子: {self.black_captures}\n白方提子: {self.white_captures}"
            
        if self.ko_point:
            result += f"\n劫争点: {chr(ord('A') + self.ko_point[1])}{self.ko_point[0] + 1}"
            
        return result


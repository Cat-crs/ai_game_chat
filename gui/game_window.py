"""
游戏窗口模块，提供图形界面进行游戏交互。
"""
import pygame
import numpy as np
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Player
from game.gomoku import Gomoku
from game.go import Go


class GameWindow:
    """
    游戏窗口类，提供图形界面进行游戏交互。
    
    属性:
        width (int): 窗口宽度
        height (int): 窗口高度
        game_type (str): 游戏类型，'gomoku'或'go'
        game: 游戏对象
        ai: AI对象
        human_player (Player): 人类玩家
        ai_player (Player): AI玩家
        board_size (int): 棋盘大小
        cell_size (int): 棋盘格子大小
        board_offset (tuple): 棋盘偏移量
        screen: Pygame屏幕对象
        font: Pygame字体对象
        running (bool): 游戏是否运行中
    """
    
    def __init__(self, width=800, height=600, game_type='gomoku', ai=None, human_first=True):
        """
        初始化游戏窗口
        
        参数:
            width (int): 窗口宽度
            height (int): 窗口高度
            game_type (str): 游戏类型，'gomoku'或'go'
            ai: AI对象
            human_first (bool): 人类玩家是否先手
        """
        self.width = width
        self.height = height
        self.game_type = game_type
        self.ai = ai
        
        # 创建游戏对象
        if game_type == 'gomoku':
            self.game = Gomoku()
            self.board_size = 15
        else:  # go
            self.game = Go()
            self.board_size = 19
            
        # 设置玩家
        self.human_player = Player.BLACK if human_first else Player.WHITE
        self.ai_player = Player.WHITE if human_first else Player.BLACK
        
        # 计算棋盘格子大小和偏移量
        self.cell_size = min(width, height) // (self.board_size + 2)
        self.board_offset = (
            (width - self.cell_size * self.board_size) // 2,
            (height - self.cell_size * self.board_size) // 2
        )
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"AI {'五子棋' if game_type == 'gomoku' else '围棋'}")
        self.font = pygame.font.SysFont('simhei', 24)
        
        # 游戏状态
        self.running = True
        self.message = ""
        self.thinking = False
        
    def run(self):
        """运行游戏"""
        # 如果AI先手，让AI落子
        if self.game.get_current_player() == self.ai_player and self.ai:
            self.ai_move()
            
        # 主循环
        while self.running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 左键点击
                    if not self.game.is_game_over() and self.game.get_current_player() == self.human_player:
                        self.handle_click(event.pos)
                        
            # 绘制界面
            self.draw()
            
            # 更新屏幕
            pygame.display.flip()
            
            # 控制帧率
            pygame.time.delay(30)
            
        # 退出Pygame
        pygame.quit()
        
    def handle_click(self, pos):
        """
        处理鼠标点击事件
        
        参数:
            pos (tuple): 鼠标点击位置
        """
        # 计算点击的棋盘坐标
        x = (pos[0] - self.board_offset[0]) // self.cell_size
        y = (pos[1] - self.board_offset[1]) // self.cell_size
        
        # 检查坐标是否在棋盘范围内
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            # 尝试落子
            if self.game.place_stone(y, x):  # 注意：游戏中的坐标是(y, x)，而不是(x, y)
                # 检查游戏是否结束
                if self.game.is_game_over():
                    winner = self.game.get_winner()
                    if winner == self.human_player:
                        self.message = "恭喜，你赢了！"
                    elif winner == self.ai_player:
                        self.message = "AI赢了！"
                    else:
                        self.message = "平局！"
                else:
                    # 如果游戏未结束，让AI落子
                    if self.ai:
                        self.ai_move()
                        
    def ai_move(self):
        """AI落子"""
        if self.game.is_game_over():
            return
            
        self.thinking = True
        self.draw()
        pygame.display.flip()
        
        # 获取AI的落子位置
        x, y = self.ai.get_best_move(self.game)
        
        self.thinking = False
        
        # 执行落子
        self.game.place_stone(x, y)
        
        # 检查游戏是否结束
        if self.game.is_game_over():
            winner = self.game.get_winner()
            if winner == self.human_player:
                self.message = "恭喜，你赢了！"
            elif winner == self.ai_player:
                self.message = "AI赢了！"
            else:
                self.message = "平局！"
                
    def draw(self):
        """绘制游戏界面"""
        # 填充背景
        self.screen.fill((240, 230, 200))
        
        # 绘制棋盘
        self.draw_board()
        
        # 绘制棋子
        self.draw_stones()
        
        # 绘制状态信息
        self.draw_status()
        
    def draw_board(self):
        """绘制棋盘"""
        # 绘制棋盘背景
        board_rect = pygame.Rect(
            self.board_offset[0] - self.cell_size // 2,
            self.board_offset[1] - self.cell_size // 2,
            self.cell_size * self.board_size + self.cell_size,
            self.cell_size * self.board_size + self.cell_size
        )
        pygame.draw.rect(self.screen, (220, 180, 100), board_rect)
        
        # 绘制棋盘线
        for i in range(self.board_size):
            # 横线
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (self.board_offset[0], self.board_offset[1] + i * self.cell_size),
                (self.board_offset[0] + (self.board_size - 1) * self.cell_size, self.board_offset[1] + i * self.cell_size)
            )
            
            # 竖线
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (self.board_offset[0] + i * self.cell_size, self.board_offset[1]),
                (self.board_offset[0] + i * self.cell_size, self.board_offset[1] + (self.board_size - 1) * self.cell_size)
            )
            
        # 绘制星位
        if self.game_type == 'gomoku':
            # 五子棋的星位
            star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        else:
            # 围棋的星位
            if self.board_size == 19:
                star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
            else:
                star_points = [(2, 2), (2, 6), (6, 2), (6, 6)]
                
        for y, x in star_points:
            pygame.draw.circle(
                self.screen,
                (0, 0, 0),
                (self.board_offset[0] + x * self.cell_size, self.board_offset[1] + y * self.cell_size),
                self.cell_size // 8
            )
            
    def draw_stones(self):
        """绘制棋子"""
        board_state = self.game.get_state()
        
        for y in range(self.board_size):
            for x in range(self.board_size):
                if board_state[y, x] == Player.BLACK.value:
                    # 黑子
                    pygame.draw.circle(
                        self.screen,
                        (0, 0, 0),
                        (self.board_offset[0] + x * self.cell_size, self.board_offset[1] + y * self.cell_size),
                        self.cell_size // 2 - 2
                    )
                elif board_state[y, x] == Player.WHITE.value:
                    # 白子
                    pygame.draw.circle(
                        self.screen,
                        (255, 255, 255),
                        (self.board_offset[0] + x * self.cell_size, self.board_offset[1] + y * self.cell_size),
                        self.cell_size // 2 - 2
                    )
                    pygame.draw.circle(
                        self.screen,
                        (0, 0, 0),
                        (self.board_offset[0] + x * self.cell_size, self.board_offset[1] + y * self.cell_size),
                        self.cell_size // 2 - 2,
                        1
                    )
                    
    def draw_status(self):
        """绘制状态信息"""
        # 绘制当前玩家
        if not self.game.is_game_over():
            current_player = self.game.get_current_player()
            player_text = "你的回合" if current_player == self.human_player else "AI思考中..."
            player_color = (0, 0, 0) if current_player == Player.BLACK else (255, 255, 255)
            
            # 绘制当前玩家的棋子
            pygame.draw.circle(
                self.screen,
                player_color,
                (self.width // 2 - 100, 30),
                15
            )
            if current_player == Player.WHITE:
                pygame.draw.circle(
                    self.screen,
                    (0, 0, 0),
                    (self.width // 2 - 100, 30),
                    15,
                    1
                )
                
            # 绘制当前玩家的文本
            player_surface = self.font.render(player_text, True, (0, 0, 0))
            self.screen.blit(player_surface, (self.width // 2 - 70, 20))
            
        # 绘制消息
        if self.message:
            message_surface = self.font.render(self.message, True, (200, 0, 0))
            self.screen.blit(message_surface, (self.width // 2 - 100, self.height - 40))
            
        # 绘制思考中提示
        if self.thinking:
            thinking_surface = self.font.render("AI思考中...", True, (0, 0, 200))
            self.screen.blit(thinking_surface, (self.width // 2 - 50, 60))


def main():
    """主函数"""
    # 创建游戏窗口
    window = GameWindow(game_type='gomoku')
    
    # 运行游戏
    window.run()


if __name__ == "__main__":
    main()


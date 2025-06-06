"""
主程序，整合游戏和聊天功能。
"""
import pygame
import sys
import os
import threading
import torch
import argparse

from gui.game_window import GameWindow
from gui.chat_window import ChatWindow
from game.board import Player
from game.gomoku import Gomoku
from game.go import Go
from ai.model import create_model
from ai.mcts import MCTS
from ai.self_play import SelfPlay
from chat.model import ChatSession


class MainWindow:
    """
    主窗口类，整合游戏和聊天功能。
    
    属性:
        width (int): 窗口宽度
        height (int): 窗口高度
        screen: Pygame屏幕对象
        font: Pygame字体对象
        running (bool): 主窗口是否运行中
        game_type (str): 游戏类型，'gomoku'或'go'
        ai_model: AI模型
        ai_mcts: MCTS对象
        chat_session: 聊天会话对象
    """
    
    def __init__(self, width=1200, height=800, game_type='gomoku', model_path=None):
        """
        初始化主窗口
        
        参数:
            width (int): 窗口宽度
            height (int): 窗口高度
            game_type (str): 游戏类型，'gomoku'或'go'
            model_path (str): 模型路径
        """
        self.width = width
        self.height = height
        self.game_type = game_type
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"AI {'五子棋' if game_type == 'gomoku' else '围棋'} + 聊天")
        self.font = pygame.font.SysFont('simhei', 24)
        
        # 主窗口状态
        self.running = True
        self.current_mode = 'menu'  # 'menu', 'game', 'chat', 'self_play'
        
        # 创建AI模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 获取系统信息
        from utils.performance import get_system_info, optimize_model_parameters, optimize_mcts_parameters
        self.system_info = get_system_info()
        print(f"系统信息: {self.system_info}")
        
        # 创建并优化模型
        board_size = 15 if game_type == 'gomoku' else 19
        self.ai_model = create_model(board_size=board_size, device=self.device)
        self.ai_model = optimize_model_parameters(self.ai_model, self.device)
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.ai_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型已从 {model_path} 加载")
            
        # 优化MCTS参数
        mcts_params = optimize_mcts_parameters(self.system_info)
        print(f"MCTS参数: {mcts_params}")
        
        # 创建MCTS
        self.ai_mcts = MCTS(
            self.ai_model, 
            num_simulations=mcts_params['num_simulations'], 
            c_puct=mcts_params['c_puct'],
            device=self.device
        )
        
        # 创建聊天会话
        self.chat_session = ChatSession()
        
        # 子窗口
        self.game_window = None
        self.chat_window = None
        self.self_play_thread = None
        
    def run(self):
        """运行主窗口"""
        # 主循环
        while self.running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # ESC键，返回菜单
                        if self.current_mode != 'menu':
                            self.current_mode = 'menu'
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 左键点击
                    if self.current_mode == 'menu':
                        self.handle_menu_click(event.pos)
                        
            # 绘制界面
            if self.current_mode == 'menu':
                self.draw_menu()
                
            # 更新屏幕
            pygame.display.flip()
            
            # 控制帧率
            pygame.time.delay(30)
            
        # 退出Pygame
        pygame.quit()
        
    def handle_menu_click(self, pos):
        """
        处理菜单点击事件
        
        参数:
            pos (tuple): 鼠标点击位置
        """
        # 五子棋按钮
        gomoku_rect = pygame.Rect(self.width // 2 - 150, 200, 300, 50)
        if gomoku_rect.collidepoint(pos):
            self.start_game('gomoku')
            
        # 围棋按钮
        go_rect = pygame.Rect(self.width // 2 - 150, 300, 300, 50)
        if go_rect.collidepoint(pos):
            self.start_game('go')
            
        # 聊天按钮
        chat_rect = pygame.Rect(self.width // 2 - 150, 400, 300, 50)
        if chat_rect.collidepoint(pos):
            self.start_chat()
            
        # 自我学习按钮
        self_play_rect = pygame.Rect(self.width // 2 - 150, 500, 300, 50)
        if self_play_rect.collidepoint(pos):
            self.start_self_play()
            
        # 退出按钮
        exit_rect = pygame.Rect(self.width // 2 - 150, 600, 300, 50)
        if exit_rect.collidepoint(pos):
            self.running = False
            
    def draw_menu(self):
        """绘制菜单界面"""
        # 填充背景
        self.screen.fill((240, 230, 200))
        
        # 绘制标题
        title_surface = self.font.render(f"AI {'五子棋' if self.game_type == 'gomoku' else '围棋'} + 聊天", True, (0, 0, 0))
        self.screen.blit(title_surface, (self.width // 2 - title_surface.get_width() // 2, 100))
        
        # 绘制按钮
        self.draw_button("五子棋", pygame.Rect(self.width // 2 - 150, 200, 300, 50))
        self.draw_button("围棋", pygame.Rect(self.width // 2 - 150, 300, 300, 50))
        self.draw_button("聊天", pygame.Rect(self.width // 2 - 150, 400, 300, 50))
        self.draw_button("自我学习", pygame.Rect(self.width // 2 - 150, 500, 300, 50))
        self.draw_button("退出", pygame.Rect(self.width // 2 - 150, 600, 300, 50))
        
    def draw_button(self, text, rect):
        """
        绘制按钮
        
        参数:
            text (str): 按钮文本
            rect (pygame.Rect): 按钮矩形
        """
        # 绘制按钮背景
        pygame.draw.rect(self.screen, (200, 200, 200), rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
        
        # 绘制按钮文本
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (rect.centerx - text_surface.get_width() // 2, rect.centery - text_surface.get_height() // 2))
        
    def start_game(self, game_type):
        """
        开始游戏
        
        参数:
            game_type (str): 游戏类型，'gomoku'或'go'
        """
        self.game_type = game_type
        self.current_mode = 'game'
        
        # 创建游戏窗口
        self.game_window = GameWindow(
            width=self.width,
            height=self.height,
            game_type=game_type,
            ai=self.ai_mcts,
            human_first=True
        )
        
        # 运行游戏窗口
        self.game_window.run()
        
        # 游戏结束后返回菜单
        self.current_mode = 'menu'
        
    def start_chat(self):
        """开始聊天"""
        self.current_mode = 'chat'
        
        # 创建聊天窗口
        self.chat_window = ChatWindow(
            width=self.width,
            height=self.height,
            chat_session=self.chat_session
        )
        
        # 运行聊天窗口
        self.chat_window.run()
        
        # 聊天结束后返回菜单
        self.current_mode = 'menu'
        
    def start_self_play(self):
        """开始自我学习"""
        self.current_mode = 'self_play'
        
        # 创建自我学习线程
        self.self_play_thread = threading.Thread(target=self._self_play_thread)
        self.self_play_thread.daemon = True
        self.self_play_thread.start()
        
        # 显示自我学习界面
        self._show_self_play_ui()
        
        # 自我学习结束后返回菜单
        self.current_mode = 'menu'
        
    def _self_play_thread(self):
        """自我学习线程"""
        # 优化批次大小
        from utils.performance import optimize_batch_size
        batch_size = optimize_batch_size(self.system_info)
        print(f"自我学习批次大小: {batch_size}")
        
        # 创建自我对弈对象
        self_play = SelfPlay(
            game_type=self.game_type,
            board_size=15 if self.game_type == 'gomoku' else 19,
            model=self.ai_model,
            num_simulations=self.ai_mcts.num_simulations,
            device=self.device,
            model_dir='models'
        )
        
        # 执行自我学习
        self_play.learn(num_episodes=5, epochs=5, batch_size=batch_size)
        
    def _show_self_play_ui(self):
        """显示自我学习界面"""
        # 填充背景
        self.screen.fill((240, 230, 200))
        
        # 绘制标题
        title_surface = self.font.render("AI自我学习中...", True, (0, 0, 0))
        self.screen.blit(title_surface, (self.width // 2 - title_surface.get_width() // 2, 100))
        
        # 绘制提示
        tip_surface = self.font.render("按ESC键返回菜单", True, (0, 0, 0))
        self.screen.blit(tip_surface, (self.width // 2 - tip_surface.get_width() // 2, 200))
        
        # 更新屏幕
        pygame.display.flip()
        
        # 等待ESC键
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False
                        
            # 控制帧率
            pygame.time.delay(30)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AI游戏+聊天')
    parser.add_argument('--game', type=str, default='gomoku', choices=['gomoku', 'go'], help='游戏类型')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    args = parser.parse_args()
    
    # 创建主窗口
    window = MainWindow(game_type=args.game, model_path=args.model)
    
    # 运行主窗口
    window.run()


if __name__ == "__main__":
    main()


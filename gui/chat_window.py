"""
聊天窗口模块，提供图形界面进行聊天交互。
"""
import pygame
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.model import ChatSession


class ChatWindow:
    """
    聊天窗口类，提供图形界面进行聊天交互。
    
    属性:
        width (int): 窗口宽度
        height (int): 窗口高度
        chat_session: 聊天会话对象
        screen: Pygame屏幕对象
        font: Pygame字体对象
        running (bool): 聊天窗口是否运行中
        messages (list): 消息列表
        input_text (str): 输入文本
        input_active (bool): 输入框是否激活
    """
    
    def __init__(self, width=400, height=600, chat_session=None):
        """
        初始化聊天窗口
        
        参数:
            width (int): 窗口宽度
            height (int): 窗口高度
            chat_session: 聊天会话对象
        """
        self.width = width
        self.height = height
        self.chat_session = chat_session if chat_session else ChatSession()
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AI聊天助手")
        self.font = pygame.font.SysFont('simhei', 18)
        
        # 聊天状态
        self.running = True
        self.messages = []
        self.input_text = ""
        self.input_active = False
        
        # 添加欢迎消息
        self.add_message("AI", "您好！我是一个围棋和五子棋AI助手，可以回答您关于围棋和五子棋的问题，也可以和您下棋。有什么我可以帮您的吗？")
        
    def run(self):
        """运行聊天窗口"""
        # 主循环
        while self.running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 左键点击
                    # 检查是否点击了输入框
                    input_rect = pygame.Rect(10, self.height - 40, self.width - 20, 30)
                    self.input_active = input_rect.collidepoint(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if self.input_active:
                        if event.key == pygame.K_RETURN:
                            # 回车键，发送消息
                            if self.input_text:
                                self.send_message(self.input_text)
                                self.input_text = ""
                        elif event.key == pygame.K_BACKSPACE:
                            # 退格键，删除一个字符
                            self.input_text = self.input_text[:-1]
                        else:
                            # 其他键，添加到输入文本
                            self.input_text += event.unicode
                            
            # 绘制界面
            self.draw()
            
            # 更新屏幕
            pygame.display.flip()
            
            # 控制帧率
            pygame.time.delay(30)
            
        # 退出Pygame
        pygame.quit()
        
    def send_message(self, text):
        """
        发送消息
        
        参数:
            text (str): 消息文本
        """
        # 添加用户消息
        self.add_message("用户", text)
        
        # 处理消息并获取回复
        response = self.chat_session.process_message(text)
        
        # 添加AI回复
        self.add_message("AI", response)
        
    def add_message(self, sender, text):
        """
        添加消息
        
        参数:
            sender (str): 发送者
            text (str): 消息文本
        """
        self.messages.append((sender, text))
        
        # 如果消息太多，删除最早的消息
        if len(self.messages) > 100:
            self.messages.pop(0)
            
    def draw(self):
        """绘制聊天界面"""
        # 填充背景
        self.screen.fill((240, 240, 240))
        
        # 绘制消息区域
        self.draw_messages()
        
        # 绘制输入框
        self.draw_input_box()
        
    def draw_messages(self):
        """绘制消息区域"""
        # 绘制消息区域背景
        message_rect = pygame.Rect(5, 5, self.width - 10, self.height - 50)
        pygame.draw.rect(self.screen, (255, 255, 255), message_rect)
        pygame.draw.rect(self.screen, (200, 200, 200), message_rect, 1)
        
        # 绘制消息
        y = 10
        for sender, text in self.messages:
            # 绘制发送者
            sender_color = (0, 0, 200) if sender == "AI" else (200, 0, 0)
            sender_surface = self.font.render(f"{sender}:", True, sender_color)
            self.screen.blit(sender_surface, (10, y))
            y += 20
            
            # 绘制消息文本（自动换行）
            words = text.split(' ')
            line = ""
            for word in words:
                test_line = line + word + " "
                # 如果当前行加上新单词超过宽度，换行
                if self.font.size(test_line)[0] > self.width - 30:
                    text_surface = self.font.render(line, True, (0, 0, 0))
                    self.screen.blit(text_surface, (20, y))
                    y += 20
                    line = word + " "
                else:
                    line = test_line
                    
            # 绘制最后一行
            if line:
                text_surface = self.font.render(line, True, (0, 0, 0))
                self.screen.blit(text_surface, (20, y))
                y += 30
                
            # 如果超出显示区域，停止绘制
            if y > self.height - 60:
                break
                
    def draw_input_box(self):
        """绘制输入框"""
        # 绘制输入框背景
        input_rect = pygame.Rect(10, self.height - 40, self.width - 20, 30)
        pygame.draw.rect(self.screen, (255, 255, 255), input_rect)
        pygame.draw.rect(self.screen, (0, 0, 0) if self.input_active else (200, 200, 200), input_rect, 2)
        
        # 绘制输入文本
        input_surface = self.font.render(self.input_text, True, (0, 0, 0))
        self.screen.blit(input_surface, (15, self.height - 35))
        
        # 如果输入框激活，绘制光标
        if self.input_active and pygame.time.get_ticks() % 1000 < 500:
            cursor_pos = self.font.size(self.input_text)[0] + 15
            pygame.draw.line(self.screen, (0, 0, 0), (cursor_pos, self.height - 35), (cursor_pos, self.height - 15), 2)


def main():
    """主函数"""
    # 创建聊天窗口
    window = ChatWindow()
    
    # 运行聊天窗口
    window.run()


if __name__ == "__main__":
    main()


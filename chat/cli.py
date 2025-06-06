"""
聊天命令行界面模块，提供简单的文本界面进行聊天测试。
"""
from .model import ChatSession


def chat_cli():
    """聊天命令行界面"""
    print("===== AI聊天助手 =====")
    print("您可以询问关于围棋和五子棋的问题，或者输入'退出'结束聊天")
    
    # 创建聊天会话
    session = ChatSession()
    
    # 显示欢迎消息
    print("AI: 您好！我是一个围棋和五子棋AI助手，可以回答您关于围棋和五子棋的问题，也可以和您下棋。有什么我可以帮您的吗？")
    
    while True:
        # 获取用户输入
        user_input = input("您: ")
        
        # 检查是否退出
        if user_input.lower() in ["退出", "再见", "拜拜", "exit", "quit", "bye"]:
            print("AI: 再见！希望我的回答对您有所帮助。如果您想再次下棋或有其他问题，随时欢迎回来。")
            break
            
        # 处理用户输入
        response = session.process_message(user_input)
        
        # 显示AI回复
        print(f"AI: {response}")


if __name__ == "__main__":
    chat_cli()


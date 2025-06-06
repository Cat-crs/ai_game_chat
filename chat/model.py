"""
聊天模型模块，实现简单的问答功能。
"""
import re
import random


class ChatModel:
    """
    简单聊天模型类，用于处理用户的问题并提供回答。
    
    属性:
        knowledge_base (dict): 知识库
        context (list): 对话上下文
        max_context_length (int): 最大上下文长度
    """
    
    def __init__(self, knowledge_base=None, max_context_length=10):
        """
        初始化聊天模型
        
        参数:
            knowledge_base (dict): 知识库，如果为None则使用默认知识库
            max_context_length (int): 最大上下文长度
        """
        from .knowledge_base import KNOWLEDGE_BASE
        self.knowledge_base = knowledge_base if knowledge_base else KNOWLEDGE_BASE
        self.context = []
        self.max_context_length = max_context_length
        
    def preprocess(self, text):
        """
        预处理文本
        
        参数:
            text (str): 输入文本
            
        返回:
            list: 处理后的词列表
        """
        # 转换为小写
        text = text.lower()
        
        # 简单分词（按空格分割）
        words = text.split()
        
        return words
        
    def get_response(self, query):
        """
        获取对用户查询的回答
        
        参数:
            query (str): 用户查询
            
        返回:
            str: 回答
        """
        # 更新上下文
        self.context.append(query)
        if len(self.context) > self.max_context_length:
            self.context.pop(0)
            
        # 检查是否是问候
        if self._is_greeting(query):
            return self.knowledge_base["问候"]
            
        # 检查是否是感谢
        if self._is_thanks(query):
            return self.knowledge_base["感谢"]
            
        # 检查是否是告别
        if self._is_goodbye(query):
            return self.knowledge_base["再见"]
            
        # 检查是否是请求建议
        if self._is_asking_for_advice(query):
            return self.knowledge_base["建议"]
            
        # 检查是否是需要鼓励
        if self._is_need_encouragement(query):
            return self.knowledge_base["鼓励"]
            
        # 直接检查关键词匹配
        for key, value in self.knowledge_base.items():
            # 跳过特殊回答模板
            if key in ["不知道", "问候", "感谢", "再见", "建议", "鼓励"]:
                continue
                
            # 检查查询中是否包含关键词
            query_lower = query.lower()
            key_lower = key.lower()
            
            # 如果查询中包含知识库键的所有关键词，则返回对应的回答
            key_words = key_lower.split()
            if len(key_words) > 0 and all(word in query_lower for word in key_words):
                return value
                
        # 如果没有直接匹配，尝试使用预处理和匹配算法
        processed_query = self.preprocess(query)
        best_match = self._find_best_match(processed_query)
        
        if best_match:
            return self.knowledge_base[best_match]
        else:
            return self.knowledge_base["不知道"]
            
    def _is_greeting(self, query):
        """
        检查是否是问候
        
        参数:
            query (str): 用户查询
            
        返回:
            bool: 是否是问候
        """
        greetings = ["你好", "早上好", "下午好", "晚上好", "嗨", "您好", "hello", "hi", "hey"]
        return any(greeting in query.lower() for greeting in greetings)
        
    def _is_thanks(self, query):
        """
        检查是否是感谢
        
        参数:
            query (str): 用户查询
            
        返回:
            bool: 是否是感谢
        """
        thanks = ["谢谢", "感谢", "多谢", "thank", "thanks"]
        return any(thank in query.lower() for thank in thanks)
        
    def _is_goodbye(self, query):
        """
        检查是否是告别
        
        参数:
            query (str): 用户查询
            
        返回:
            bool: 是否是告别
        """
        goodbyes = ["再见", "拜拜", "回头见", "下次见", "goodbye", "bye"]
        return any(goodbye in query.lower() for goodbye in goodbyes)
        
    def _is_asking_for_advice(self, query):
        """
        检查是否是请求建议
        
        参数:
            query (str): 用户查询
            
        返回:
            bool: 是否是请求建议
        """
        advice_keywords = ["建议", "怎么学", "如何学", "入门", "开始", "新手", "初学者"]
        return any(keyword in query.lower() for keyword in advice_keywords)
        
    def _is_need_encouragement(self, query):
        """
        检查是否是需要鼓励
        
        参数:
            query (str): 用户查询
            
        返回:
            bool: 是否是需要鼓励
        """
        encouragement_keywords = ["输了", "失败", "不行", "太难", "放弃", "沮丧", "气馁"]
        return any(keyword in query.lower() for keyword in encouragement_keywords)
        
    def _find_best_match(self, processed_query):
        """
        在知识库中查找最匹配的回答
        
        参数:
            processed_query (list): 处理后的查询词列表
            
        返回:
            str: 最匹配的知识库键，如果没有匹配则返回None
        """
        best_match = None
        best_score = 0
        
        # 将查询词转换为集合，便于后续计算
        query_set = set(processed_query)
        
        for key in self.knowledge_base:
            # 预处理知识库键
            processed_key = self.preprocess(key)
            key_set = set(processed_key)
            
            # 计算关键词匹配
            # 1. 检查查询中的关键词是否出现在知识库键中
            if any(word in key_set for word in query_set):
                # 2. 计算匹配分数
                # 共同词的数量
                common_words = query_set.intersection(key_set)
                # 匹配分数 = 共同词数量 / 查询词数量
                score = len(common_words) / len(query_set) if query_set else 0
                
                if score > best_score:
                    best_score = score
                    best_match = key
                    
        # 如果最佳匹配分数太低，返回None
        if best_score < 0.2:
            return None
            
        return best_match
        
    def _calculate_match_score(self, query_words, key_words):
        """
        计算查询和知识库键之间的匹配分数
        
        参数:
            query_words (list): 查询词列表
            key_words (list): 知识库键词列表
            
        返回:
            float: 匹配分数，范围为0-1
        """
        # 如果没有词，返回0
        if not query_words or not key_words:
            return 0
            
        # 计算共同词的数量
        common_words = set(query_words) & set(key_words)
        
        # 计算Jaccard相似度
        jaccard = len(common_words) / (len(set(query_words)) + len(set(key_words)) - len(common_words))
        
        return jaccard


class ChatSession:
    """
    聊天会话类，用于管理用户和AI之间的对话。
    
    属性:
        chat_model (ChatModel): 聊天模型
        game_context (dict): 游戏上下文
    """
    
    def __init__(self, chat_model=None):
        """
        初始化聊天会话
        
        参数:
            chat_model (ChatModel): 聊天模型，如果为None则创建新模型
        """
        self.chat_model = chat_model if chat_model else ChatModel()
        self.game_context = {
            "current_game": None,  # 当前游戏对象
            "game_type": None,     # 游戏类型：'gomoku'或'go'
            "ai_player": None      # AI玩家：Player.BLACK或Player.WHITE
        }
        
    def process_message(self, message):
        """
        处理用户消息
        
        参数:
            message (str): 用户消息
            
        返回:
            str: AI回复
        """
        # 检查是否是游戏相关命令
        game_command = self._check_game_command(message)
        if game_command:
            return self._handle_game_command(game_command, message)
            
        # 否则，使用聊天模型获取回答
        return self.chat_model.get_response(message)
        
    def _check_game_command(self, message):
        """
        检查是否是游戏相关命令
        
        参数:
            message (str): 用户消息
            
        返回:
            str: 命令类型，如果不是命令则返回None
        """
        # 转换为小写，便于匹配
        message_lower = message.lower()
        
        # 开始游戏命令
        if any(word in message_lower for word in ["开始", "开局", "玩", "下棋", "对弈"]):
            if "五子棋" in message_lower or "五子" in message_lower:
                return "start_gomoku"
            elif "围棋" in message_lower or "围" in message_lower:
                return "start_go"
            else:
                return "start_unknown"
                
        # 落子命令
        if re.search(r'[A-Za-z][0-9]{1,2}', message):
            return "move"
            
        # 结束游戏命令
        if any(word in message_lower for word in ["结束", "停止", "退出"]) and any(word in message_lower for word in ["游戏", "棋"]):
            return "end_game"
            
        # 查看棋盘命令
        if any(word in message_lower for word in ["查看", "显示"]) and any(word in message_lower for word in ["棋盘", "局面"]):
            return "show_board"
            
        return None
        
    def _handle_game_command(self, command, message):
        """
        处理游戏相关命令
        
        参数:
            command (str): 命令类型
            message (str): 用户消息
            
        返回:
            str: AI回复
        """
        # 这里只是返回一个提示，实际游戏逻辑需要在主程序中实现
        if command == "start_gomoku":
            return "您想开始一局五子棋游戏。请在主界面选择'五子棋'模式开始游戏。"
            
        elif command == "start_go":
            return "您想开始一局围棋游戏。请在主界面选择'围棋'模式开始游戏。"
            
        elif command == "start_unknown":
            return "您想开始一局游戏，但没有指定游戏类型。请选择'五子棋'或'围棋'。"
            
        elif command == "move":
            return "您想在某个位置落子。请在游戏界面点击棋盘进行落子。"
            
        elif command == "end_game":
            return "您想结束当前游戏。请在游戏界面点击'结束游戏'按钮。"
            
        elif command == "show_board":
            return "您想查看当前棋盘。请查看游戏界面上的棋盘。"
            
        return "我不太理解您的游戏命令。请尝试在游戏界面直接操作。"


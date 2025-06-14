"""
知识库模块，提供围棋和五子棋相关的知识和术语。
"""

# 围棋知识库
GO_KNOWLEDGE = {
    # 基本规则
    "围棋规则": "围棋是一种两人对弈的棋类游戏，使用19×19的棋盘和黑白棋子。黑方先行，双方轮流在棋盘交叉点上放置棋子。"
               "围棋的目标是通过围地和吃子获得更多的点数。当一个棋子或一组相连的棋子被对方的棋子完全包围，没有气（空点）时，这些棋子会被提走。"
               "游戏结束时，拥有更多领地和吃子的一方获胜。",
    
    "围棋棋盘": "标准围棋棋盘是19×19的方格，棋子放在交叉点上。棋盘上有9个小黑点，称为星位，用于定位和参考。",
    
    "围棋落子": "围棋中，棋子放在交叉点上，而不是方格内。黑方先行，之后双方轮流落子。",
    
    "围棋气": "围棋中的'气'是指一个棋子或一组相连棋子相邻的空点。当一个棋子或一组棋子没有气时，它们会被提走（吃掉）。",
    
    "围棋提子": "当一个棋子或一组相连的棋子被对方的棋子完全包围，没有气时，这些棋子会被提走。提走的棋子作为俘虏计入对方的得分。",
    
    "围棋禁入点": "围棋中的禁入点是指某一方不能落子的点。主要有两种情况：自杀点（落子后立即没有气）和劫争点（避免无限循环）。",
    
    "围棋劫争": "劫争是围棋中的一种特殊情况，指的是双方可以无限循环吃子的局面。为了避免无限循环，规则规定刚被吃的一个子的位置，对方不能立即落子。",
    
    "围棋计分": "围棋的计分方式有两种：区域计分法（日本规则）和数子计分法（中国规则）。区域计分法计算实际围地和提子数，数子计分法计算棋盘上的棋子和围地。",
    
    "围棋贴目": "由于黑方先行有优势，所以通常会给白方一定的贴目（让子）作为补偿，一般为6.5目或7.5目。",
    
    # 基本术语
    "围棋术语": "围棋有许多专业术语，如气、提子、劫争、目、征子、打吃、枷、官子等。",
    
    "围棋目": "围棋中的'目'是计分单位，指一个交叉点的空间。",
    
    "围棋征子": "征子是一种战术，通过连续的打吃迫使对方的棋子沿着一条路线逃跑，最终被吃掉。",
    
    "围棋打吃": "打吃是指威胁吃掉对方的棋子，迫使对方必须应对。",
    
    "围棋枷": "枷是一种战术，通过在对方棋子的关键位置放置棋子，限制对方棋子的活动。",
    
    "围棋官子": "官子是指围棋终盘时的落子，主要目的是确定边界和最大化得分。",
    
    # 基本战术和策略
    "围棋入门策略": "围棋入门策略包括：控制角落、占据边缘、避免过早进入中央、保持棋子的联系、注意棋子的气等。",
    
    "围棋布局": "围棋的布局（布石）是指游戏开始阶段的落子，目的是建立有利的局面。常见的布局有星、小目、高目等。",
    
    "围棋中盘": "围棋的中盘是指布局结束后、官子开始前的阶段，主要是双方的战斗和侵消。",
    
    "围棋收官": "围棋的收官是指游戏的最后阶段，主要是确定地盘的边界和最大化得分。",
    
    # 进阶知识
    "围棋定式": "定式是指在特定局部位置上，经过长期实践证明是最佳或较好的一系列着法。",
    
    "围棋死活": "死活问题是围棋中的基本问题，判断一组棋子是否可以存活或被吃掉。",
    
    "围棋段位": "围棋的段位从低到高分为1-9段，其中9段最高。段位之上是职业级别。",
    
    "围棋AI": "近年来，围棋AI取得了巨大进步，如AlphaGo、AlphaZero等，它们已经能够击败世界顶级职业棋手。"
}

# 五子棋知识库
GOMOKU_KNOWLEDGE = {
    # 基本规则
    "五子棋规则": "五子棋是一种两人对弈的棋类游戏，使用15×15的棋盘和黑白棋子。黑方先行，双方轮流在棋盘交叉点上放置棋子。"
                "谁先在横、竖或斜线上连成五个或更多同色棋子，谁就获胜。",
    
    "五子棋棋盘": "标准五子棋棋盘是15×15的方格，棋子放在交叉点上。",
    
    "五子棋落子": "五子棋中，棋子放在交叉点上，而不是方格内。黑方先行，之后双方轮流落子。",
    
    "五子棋胜负": "在五子棋中，谁先在横、竖或斜线上连成五个或更多同色棋子，谁就获胜。",
    
    # 基本术语
    "五子棋术语": "五子棋有一些专业术语，如连、活三、活四、冲四、禁手等。",
    
    "五子棋连": "连是指同色棋子在一条直线上相连的情况。",
    
    "五子棋活三": "活三是指在一条直线上有三个相连的同色棋子，两端都是空点，可以形成活四的情况。",
    
    "五子棋活四": "活四是指在一条直线上有四个相连的同色棋子，一端是空点，可以形成五连的情况。",
    
    "五子棋冲四": "冲四是指在一条直线上有四个相连的同色棋子，但只有一端可以落子形成五连的情况。",
    
    "五子棋禁手": "禁手是指在标准五子棋规则中，黑方不能形成的特定棋型，如三三禁手、四四禁手和长连禁手。",
    
    # 基本战术和策略
    "五子棋入门策略": "五子棋入门策略包括：控制中央、形成活三和活四、阻止对方形成活三和活四、注意禁手等。",
    
    "五子棋开局": "五子棋的开局通常是在棋盘中央或其附近落子，以便控制更多的方向。",
    
    "五子棋进攻": "五子棋的进攻主要是形成活三、活四等威胁，迫使对方防守，从而获得更多的主动权。",
    
    "五子棋防守": "五子棋的防守主要是阻止对方形成活三、活四等威胁，同时寻找反击的机会。",
    
    # 进阶知识
    "五子棋定式": "定式是指在特定局部位置上，经过长期实践证明是最佳或较好的一系列着法。",
    
    "五子棋变化": "五子棋的变化非常多，即使是同一个开局，也可能有多种不同的发展路线。",
    
    "五子棋AI": "近年来，五子棋AI也取得了很大进步，已经能够击败人类顶级选手。"
}

# 通用棋类知识库
GENERAL_KNOWLEDGE = {
    "棋类游戏": "棋类游戏是一种策略性游戏，通常由两名玩家轮流在棋盘上移动棋子。常见的棋类游戏包括围棋、五子棋、国际象棋、中国象棋等。",
    
    "棋类历史": "棋类游戏有着悠久的历史，其中围棋起源于中国，历史可以追溯到4000多年前。五子棋也起源于中国，是围棋的简化版本。",
    
    "棋类益处": "下棋有很多益处，包括提高逻辑思维能力、增强记忆力、培养耐心和专注力、锻炼决策能力等。",
    
    "棋类学习": "学习棋类游戏的方法包括：学习基本规则、研究经典对局、解决棋谱问题、参加比赛、找老师指导等。",
    
    "棋类比赛": "棋类比赛有很多级别，从业余比赛到世界锦标赛。比赛通常有时间限制，使用积分系统来评定选手的水平。"
}

# 回答模板
ANSWER_TEMPLATES = {
    "不知道": "对不起，我不太了解这个问题。您可以问我一些关于围棋或五子棋的基本规则和策略。",
    
    "问候": "您好！我是一个围棋和五子棋AI助手，可以回答您关于围棋和五子棋的问题，也可以和您下棋。有什么我可以帮您的吗？",
    
    "感谢": "不客气！如果您还有其他问题，随时可以问我。",
    
    "再见": "再见！希望我的回答对您有所帮助。如果您想再次下棋或有其他问题，随时欢迎回来。",
    
    "建议": "作为一个初学者，我建议您先了解基本规则，然后多练习基本战术。您可以从小棋盘开始，逐渐过渡到标准棋盘。多看一些入门教程和经典对局也会很有帮助。",
    
    "鼓励": "下棋是需要时间和耐心的，不要因为初期的失败而气馁。每一局棋都是一次学习的机会，重要的是从中吸取经验教训。坚持练习，您会看到自己的进步的！"
}

# 围棋和五子棋的区别
DIFFERENCES = {
    "围棋和五子棋的区别": "围棋和五子棋有以下主要区别：\n"
                    "1. 规则：围棋的目标是通过围地和吃子获得更多的点数，而五子棋的目标是先连成五个同色棋子。\n"
                    "2. 棋盘：标准围棋棋盘是19×19的，而五子棋通常是15×15的。\n"
                    "3. 复杂度：围棋的复杂度远高于五子棋，有更多的变化和策略。\n"
                    "4. 提子：围棋有提子规则，而五子棋没有。\n"
                    "5. 终局：围棋通常会下满整个棋盘，而五子棋一旦有一方连成五子，游戏就结束了。"
}

# 合并所有知识库
KNOWLEDGE_BASE = {}
KNOWLEDGE_BASE.update(GO_KNOWLEDGE)
KNOWLEDGE_BASE.update(GOMOKU_KNOWLEDGE)
KNOWLEDGE_BASE.update(GENERAL_KNOWLEDGE)
KNOWLEDGE_BASE.update(DIFFERENCES)
KNOWLEDGE_BASE.update(ANSWER_TEMPLATES)


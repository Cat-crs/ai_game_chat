"""
命令行界面模块，提供简单的文本界面进行游戏测试。
"""
from .gomoku import Gomoku
from .go import Go
from .board import Player


def parse_move(move_str, board_size):
    """
    解析用户输入的落子位置
    
    参数:
        move_str (str): 用户输入的落子位置，如"A1", "B2"等
        board_size (int): 棋盘大小
        
    返回:
        tuple: (x, y) 坐标，如果输入无效则返回None
    """
    if not move_str or len(move_str) < 2:
        return None
        
    # 处理列坐标（字母）
    col = move_str[0].upper()
    if not ('A' <= col <= chr(ord('A') + board_size - 1)):
        return None
    y = ord(col) - ord('A')
    
    # 处理行坐标（数字）
    try:
        x = int(move_str[1:]) - 1
        if x < 0 or x >= board_size:
            return None
    except ValueError:
        return None
        
    return x, y


def gomoku_cli():
    """五子棋命令行界面"""
    print("===== 五子棋游戏 =====")
    print("输入格式：列行，如A1表示左上角，Q退出")
    
    game = Gomoku()
    print(game)
    
    while not game.is_game_over():
        current_player = "黑方" if game.get_current_player() == Player.BLACK else "白方"
        move = input(f"{current_player}请输入落子位置（如A1）或Q退出: ")
        
        if move.upper() == 'Q':
            print("游戏已退出")
            return
            
        coords = parse_move(move, game.board.size)
        if coords is None:
            print("输入无效，请重新输入")
            continue
            
        x, y = coords
        if not game.place_stone(x, y):
            print("落子无效，请重新输入")
            continue
            
        print(game)
        
    if game.get_winner():
        winner = "黑方" if game.get_winner() == Player.BLACK else "白方"
        print(f"游戏结束，{winner}获胜！")
    else:
        print("游戏结束，平局！")


def go_cli():
    """围棋命令行界面"""
    print("===== 围棋游戏 =====")
    print("输入格式：列行，如A1表示左上角，P表示弃权，Q退出")
    
    game = Go()
    print(game)
    
    while not game.is_game_over():
        current_player = "黑方" if game.get_current_player() == Player.BLACK else "白方"
        move = input(f"{current_player}请输入落子位置（如A1）、P弃权或Q退出: ")
        
        if move.upper() == 'Q':
            print("游戏已退出")
            return
            
        if move.upper() == 'P':
            game.pass_move()
            print("玩家选择弃权")
            print(game)
            continue
            
        coords = parse_move(move, game.size)
        if coords is None:
            print("输入无效，请重新输入")
            continue
            
        x, y = coords
        if not game.place_stone(x, y):
            print("落子无效，请重新输入")
            continue
            
        print(game)
        
    if game.is_game_over():
        black_score, white_score = game.score_game()
        print(f"游戏结束\n黑方得分: {black_score}\n白方得分: {white_score}")
        winner = "黑方" if black_score > white_score else "白方"
        print(f"{winner}获胜！")


if __name__ == "__main__":
    game_type = input("请选择游戏类型（1:五子棋，2:围棋）: ")
    
    if game_type == "1":
        gomoku_cli()
    elif game_type == "2":
        go_cli()
    else:
        print("选择无效，程序退出")


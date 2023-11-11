"""
人机对战脚本 - 与训练好的AI对弈

使用方法:
    python -m scripts.human_play [--model MODEL_FILE]

参数:
    --model: 模型文件路径，默认使用 outputs/model_fast_final.pth
"""

import argparse
from pathlib import Path

from game import Board, MCTSPlayer
from model import PolicyValueNet

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
DEFAULT_MODEL = "model_fast_final.pth"


def resolve_model_path(path_str: str) -> Path:
    """解析模型路径，自动在 outputs/ 目录下查找"""
    path = Path(path_str)
    if path.exists():
        return path
    outputs_path = OUTPUTS_DIR / path.name
    if outputs_path.exists():
        return outputs_path
    return path


class HumanPlayer:
    """人类玩家"""

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        """获取人类玩家的落子"""
        while True:
            try:
                move_str = input("请输入落子坐标 (行,列): ")
                row, col = map(int, move_str.strip().split(","))
                move = row * board.width + col

                if move in board.availables:
                    return move
                print("该位置已有棋子，请重新输入")
            except (ValueError, IndexError):
                print("输入格式错误，请使用 '行,列' 格式，如: 7,7")


def print_board(board, p1, p2):
    """打印棋盘"""
    print("\n  ", end="")
    for x in range(board.width):
        print(f"{x:3}", end="")
    print("\n")

    for i in range(board.height):
        print(f"{i:2} ", end="")
        for j in range(board.width):
            loc = i * board.width + j
            p = board.states.get(loc, -1)
            if p == p1:
                print(" X ", end="")
            elif p == p2:
                print(" O ", end="")
            else:
                print(" · ", end="")
        print()
    print()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="五子棋人机对战")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"模型文件名或路径，默认: {DEFAULT_MODEL}（自动在 outputs/ 目录查找）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 棋盘设置
    # width, height, n_in_row = 15, 15, 5
    width, height, n_in_row = 9, 9, 5
    board = Board(width=width, height=height, n_in_row=n_in_row)
    board.init_board()

    # 加载AI
    model_path = resolve_model_path(args.model)

    try:
        policy_net = PolicyValueNet(width, height, model_file=str(model_path))
        ai_player = MCTSPlayer(policy_net.policy_value_fn, c_puct=5, n_playout=400)
        print(f"已加载模型: {model_path}")
    except FileNotFoundError:
        print(f"未找到模型文件 {model_path}，使用随机AI")
        ai_player = MCTSPlayer(c_puct=5, n_playout=400)

    human = HumanPlayer()

    # 设置玩家
    p1, p2 = board.players
    human.set_player_ind(p1)
    ai_player.set_player_ind(p2)
    players = {p1: human, p2: ai_player}

    print("\n===== 五子棋人机对战 =====")
    print("你执 X (先手)，AI 执 O")
    print("输入格式: 行,列 (例如: 7,7)")
    print("=" * 26)

    print_board(board, p1, p2)

    # 游戏主循环
    while True:
        current = board.current_player
        player = players[current]

        if isinstance(player, HumanPlayer):
            move = player.get_action(board)
        else:
            print("AI思考中...")
            move = player.get_action(board)
            row, col = move // board.width, move % board.width
            print(f"AI落子: {row},{col}")

        board.do_move(move)
        print_board(board, p1, p2)

        end, winner = board.game_end()
        if end:
            if winner == -1:
                print("平局!")
            elif winner == p1:
                print("恭喜你赢了!")
            else:
                print("AI获胜!")
            break


if __name__ == "__main__":
    main()

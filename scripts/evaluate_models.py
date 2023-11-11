"""模型对战评估工具 - 用于准确对比模型强弱"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from game.board import Board
from game.mcts import MCTSPlayer
from model.policy_value_net import PolicyValueNet

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def resolve_model_path(path_str: str) -> Path:
    """解析模型路径，自动在 outputs/ 目录下查找"""
    path = Path(path_str)
    if path.exists():
        return path
    # 尝试在 outputs/ 目录下查找
    outputs_path = OUTPUTS_DIR / path.name
    if outputs_path.exists():
        return outputs_path
    # 都不存在，返回原路径（让后续报错）
    return path


class ModelEvaluator:
    """模型评估器：让两个模型对战多局"""

    def __init__(self, board_size=8, n_in_row=5):
        self.board_size = board_size
        self.n_in_row = n_in_row

    def play_one_game(self, player1, player2, player1_first=True):
        """
        对弈一局

        Returns:
            1: player1胜, 2: player2胜, 0: 平局
        """
        board = Board(self.board_size, self.board_size, self.n_in_row)
        board.init_board(start_player=0)

        # 根据先手设置玩家顺序 (board.players = [1, 2])
        player_map = {1: player1, 2: player2} if player1_first else {1: player2, 2: player1}

        # 重置MCTS树
        player1.reset_player()
        player2.reset_player()

        while True:
            current = board.get_current_player()  # 1 或 2
            player = player_map[current]
            move = player.get_action(board)
            board.do_move(move)

            end, winner = board.game_end()
            if end:
                if winner == -1:
                    return 0  # 平局
                return 1 if player_map[winner] is player1 else 2

    def evaluate(self, model1_path, model2_path, n_games=20, n_playout=400):
        """
        评估两个模型，返回对战结果

        Args:
            n_games: 对战局数（必须是偶数，保证先手公平）
        """
        # 解析模型路径
        path1 = resolve_model_path(model1_path)
        path2 = resolve_model_path(model2_path)

        print(f"\n{'=' * 50}")
        print("模型对战评估")
        print(f"{'=' * 50}")
        print(f"模型1: {path1}")
        print(f"模型2: {path2}")
        print(f"对局: {n_games}局, MCTS: {n_playout}次模拟")
        print(f"{'=' * 50}\n")

        # 加载模型
        net1 = PolicyValueNet(self.board_size, self.board_size, model_file=str(path1))
        net2 = PolicyValueNet(self.board_size, self.board_size, model_file=str(path2))

        player1 = MCTSPlayer(net1.policy_value_fn, c_puct=5, n_playout=n_playout)
        player2 = MCTSPlayer(net2.policy_value_fn, c_puct=5, n_playout=n_playout)

        wins1, wins2, draws = 0, 0, 0

        for i in range(n_games):
            player1_first = i % 2 == 0  # 轮流先手
            first_name = "模型1" if player1_first else "模型2"
            print(f"第{i + 1:2d}局 ({first_name}先手)...", end=" ", flush=True)

            result = self.play_one_game(player1, player2, player1_first)

            if result == 1:
                wins1 += 1
                print("模型1 胜 ✓")
            elif result == 2:
                wins2 += 1
                print("模型2 胜 ✓")
            else:
                draws += 1
                print("平局 ○")

        # 输出结果
        print(f"\n{'=' * 50}")
        print("结果统计:")
        print(f"模型1: {wins1}胜 / {draws}平 / {wins2}负  (胜率: {wins1 / n_games * 100:.1f}%)")
        print(f"模型2: {wins2}胜 / {draws}平 / {wins1}负  (胜率: {wins2 / n_games * 100:.1f}%)")
        print(f"{'=' * 50}")

        if wins1 > wins2:
            print(f">>> 模型1 更强 (领先 {(wins1 - wins2) / n_games * 100:.1f}%)")
        elif wins2 > wins1:
            print(f">>> 模型2 更强 (领先 {(wins2 - wins1) / n_games * 100:.1f}%)")
        else:
            print(">>> 两模型实力相当")
        print()

        return {"wins1": wins1, "wins2": wins2, "draws": draws}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="模型对战评估")
    parser.add_argument("model1", help="模型1路径")
    parser.add_argument("model2", help="模型2路径")
    parser.add_argument("-n", "--games", type=int, default=20, help="对战局数(偶数)")
    parser.add_argument("-p", "--playout", type=int, default=400, help="MCTS模拟次数")
    parser.add_argument("-s", "--size", type=int, default=9, help="棋盘大小")
    args = parser.parse_args()

    # 确保对局数为偶数
    n_games = args.games if args.games % 2 == 0 else args.games + 1

    evaluator = ModelEvaluator(board_size=args.size)
    evaluator.evaluate(args.model1, args.model2, n_games=n_games, n_playout=args.playout)


if __name__ == "__main__":
    main()

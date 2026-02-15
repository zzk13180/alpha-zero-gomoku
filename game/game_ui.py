"""游戏控制模块"""

import numpy as np


class Game:
    """游戏控制类"""

    # 温度退火阈值：前 temp_threshold 步使用高温度探索
    TEMP_THRESHOLD = 12

    def __init__(self, board):
        self.board = board

    def self_play(self, player, temp=1.0):
        """
        执行自我对弈，返回 (winner, [(状态, MCTS概率, 胜负)])

        温度退火策略:
        - 前 TEMP_THRESHOLD 步使用 temp=1.0 鼓励探索
        - 之后使用 temp=1e-3 选择最优动作
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []
        move_count = 0

        while True:
            # 温度退火：开局探索，中后局精确
            current_temp = temp if move_count < self.TEMP_THRESHOLD else 1e-3
            move, move_probs = player.get_action(
                self.board, temp=current_temp, return_prob=True
            )
            # 存储每次落子前的棋盘状态
            states.append(self.board.current_state())
            # 存储每次 MCTS 计算的动作概率分布
            mcts_probs.append(move_probs)
            # 存储每次落子时的当前玩家
            current_players.append(self.board.current_player)
            self.board.do_move(move)
            move_count += 1

            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                return winner, list(zip(states, mcts_probs, winners_z, strict=True))

"""游戏控制模块"""

import numpy as np


class Game:
    """游戏控制类"""

    def __init__(self, board):
        self.board = board

    def self_play(self, player, temp=1.0):
        """执行自我对弈，返回 (winner, [(状态, MCTS概率, 胜负)])"""
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []

        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=True)
            # 存储每次落子前的棋盘状态
            states.append(self.board.current_state())
            # 存储每次 MCTS 计算的动作概率分布
            mcts_probs.append(move_probs)
            # 存储每次落子时的当前玩家
            current_players.append(self.board.current_player)
            self.board.do_move(move)

            end, winner = self.board.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                return winner, list(zip(states, mcts_probs, winners_z, strict=True))

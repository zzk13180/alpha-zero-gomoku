"""五子棋棋盘模块"""

import numpy as np


class Board:
    """五子棋棋盘，位置 = 行 * 宽度 + 列"""

    def __init__(self, width=15, height=15, n_in_row=5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.players = [1, 2]  # 两个玩家

    def init_board(self, start_player=0):
        """初始化棋盘"""
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))  # 所有位置可用
        self.states = {}
        self.last_move = -1

    def current_state(self):
        """获取4通道棋盘状态: [我方棋子, 对方棋子, 最后落子, 先手标记]"""
        state = np.zeros((4, self.height, self.width))

        if self.states:
            moves, players = np.array(list(zip(*self.states.items(), strict=True)))
            curr_moves = moves[players == self.current_player]
            state[0][curr_moves // self.width, curr_moves % self.width] = 1.0
            oppo_moves = moves[players != self.current_player]
            state[1][oppo_moves // self.width, oppo_moves % self.width] = 1.0
            state[2][self.last_move // self.width, self.last_move % self.width] = 1.0

        if len(self.states) % 2 == 0:
            state[3][:, :] = 1.0

        return state[:, ::-1, :]

    def do_move(self, move):
        """执行落子"""
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[1] if self.current_player == self.players[0] else self.players[0]
        )
        self.last_move = move

    def has_a_winner(self):
        """检查最后落子位置周围是否连成五子"""
        if self.last_move == -1:
            return False, -1

        n = self.n_in_row
        if len(self.states) < 2 * n - 1:
            return False, -1

        move = self.last_move
        h, w = move // self.width, move % self.width
        player = self.states[move]

        # 四个方向：横、竖、左斜、右斜
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dh, dw in directions:
            count = 1
            # 向一个方向延伸
            for i in range(1, n):
                nh, nw = h + i * dh, w + i * dw
                if 0 <= nh < self.height and 0 <= nw < self.width:
                    if self.states.get(nh * self.width + nw) == player:
                        count += 1
                    else:
                        break
                else:
                    break
            # 向相反方向延伸
            for i in range(1, n):
                nh, nw = h - i * dh, w - i * dw
                if 0 <= nh < self.height and 0 <= nw < self.width:
                    if self.states.get(nh * self.width + nw) == player:
                        count += 1
                    else:
                        break
                else:
                    break

            if count >= n:
                return True, player

        return False, -1

    def game_end(self):
        """检查游戏是否结束"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        if not self.availables:
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

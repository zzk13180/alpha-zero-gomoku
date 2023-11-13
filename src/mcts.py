import js
import numpy as np


class Board:
    def __init__(self, width=9, height=9, n_in_row=5):
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.players = [1, 2]
        self.init_board()

    def init_board(self, start_player=0):
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def current_state(self):
        state = np.zeros((4, self.height, self.width), dtype=np.float32)
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            curr_moves = moves[players == self.current_player]
            state[0][curr_moves // self.width, curr_moves % self.width] = 1.0
            oppo_moves = moves[players != self.current_player]
            state[1][oppo_moves // self.width, oppo_moves % self.width] = 1.0
            state[2][self.last_move // self.width, self.last_move % self.width] = 1.0

        if len(self.states) % 2 == 0:
            state[3][:, :] = 1.0

        return state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[1] if self.current_player == self.players[0] else self.players[0]
        )
        self.last_move = move

    def has_a_winner(self):
        if self.last_move == -1:
            return False, -1

        n = self.n_in_row
        if len(self.states) < 2 * n - 1:
            return False, -1

        move = self.last_move
        h, w = move // self.width, move % self.width
        player = self.states[move]

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dh, dw in directions:
            count = 1
            for i in range(1, n):
                nh, nw = h + i * dh, w + i * dw
                if 0 <= nh < self.height and 0 <= nw < self.width:
                    if self.states.get(nh * self.width + nw) == player:
                        count += 1
                    else:
                        break
                else:
                    break
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
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        if not self.availables:
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(), key=lambda x: x[1]._get_ucb(c_puct))

    def _get_ucb(self, c_puct):
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + self._u

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    async def _playout(self, state):
        node = self._root
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, leaf_value = await self._policy(state)

        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:  # Tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        node.update_recursive(-leaf_value)

    async def get_move_probs(self, state, temp=1e-3):
        import copy

        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            await self._playout(state_copy)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


async def policy_value_fn_proxy(board):
    legal_positions = board.availables
    current_state = board.current_state()
    input_data = current_state.astype(np.float32).flatten()

    result = await js.predict(input_data)

    log_probs_js = result[0]
    value = result[1]

    log_probs = np.array(log_probs_js.to_py())
    probs = np.exp(log_probs)

    action_probs = zip(legal_positions, probs[legal_positions])
    return action_probs, value


class AlphaZeroPlayer:
    def __init__(self, n_playout=400):
        self.mcts = MCTS(policy_value_fn_proxy, c_puct=5, n_playout=n_playout)

    def reset_player(self):
        self.mcts.update_with_move(-1)

    async def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move_acts, move_probs = await self.mcts.get_move_probs(board)
            move = np.random.choice(move_acts, p=move_probs)
            self.mcts.update_with_move(-1)
            return int(move)
        else:
            print("WARNING: no sensible moves")
            return -1

    def observe_opponent_move(self, move):
        self.mcts.update_with_move(move)


global_board = None
global_ai = None


def init_game(width=9, height=9, n_playout=400):
    global global_board, global_ai
    global_board = Board(width=width, height=height)
    global_ai = AlphaZeroPlayer(n_playout=n_playout)
    return True


def make_move(move):
    global global_board, global_ai
    if global_board and move in global_board.availables:
        global_board.do_move(move)
        if global_ai:
            global_ai.observe_opponent_move(move)
        return True
    return False


async def get_ai_move():
    global global_board, global_ai
    if global_board and global_ai:
        move = await global_ai.get_action(global_board)
        return move
    return None


def check_game_end():
    global global_board
    if global_board:
        return global_board.game_end()
    return False, -1


def set_ai_level(n_playout):
    global global_ai
    if global_ai:
        global_ai.mcts._n_playout = int(n_playout)


def restore_from_history(history_list, width=9, height=9):
    init_game(width=width, height=height)
    for item in history_list:
        r = item["row"]
        c = item["col"]
        move = int(r * width + c)
        make_move(move)

"""蒙特卡洛树搜索 (MCTS) 模块"""

import copy

import numpy as np


def softmax(x):
    """softmax函数"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class TreeNode:
    """MCTS树节点，使用UCB公式: Q + c_puct * P * sqrt(N_parent) / (1 + N)"""

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._P = prior_p

    def expand(self, action_priors):
        """扩展节点"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """选择UCB值最大的子节点"""
        return max(self._children.items(), key=lambda x: x[1]._get_ucb(c_puct))

    def _get_ucb(self, c_puct):
        """计算UCB值"""
        u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._Q + u

    def update(self, leaf_value):
        """更新节点统计值"""
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """递归更新祖先节点（正负号交替）"""
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return len(self._children) == 0

    def is_root(self):
        return self._parent is None


class MCTS:
    """蒙特卡洛树搜索"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """执行一次模拟: 选择->扩展->评估->回传"""
        node = self._root

        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()

        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """执行n次模拟，返回动作和概率"""
        for _ in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits, strict=True)
        act_probs = softmax(np.log(np.array(visits) + 1e-10) / temp)

        return acts, act_probs

    def update_with_move(self, last_move):
        """根据落子更新根节点"""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


def _default_policy(board):
    """默认均匀随机策略"""
    probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, probs, strict=True), 0


class MCTSPlayer:
    """基于MCTS的AI玩家"""

    def __init__(self, policy_value_function=None, c_puct=5, n_playout=400, is_selfplay=False):
        if policy_value_function is None:
            policy_value_function = _default_policy
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False):
        """获取下一步动作"""
        if not board.availables:
            print("棋盘已满")
            return None

        move_probs = np.zeros(board.width * board.height)
        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs

        if self._is_selfplay:
            # 添加Dirichlet噪声: (1-ε)*p + ε*noise, ε=0.25
            move = np.random.choice(
                acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
            )
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        return move

    def __str__(self):
        return f"MCTS Player {self.player}"

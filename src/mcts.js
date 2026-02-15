/**
 * 蒙特卡洛树搜索 (MCTS) — 纯 JavaScript 实现
 * 与 Python 端算法完全一致，使用 ONNX Runtime Web 做神经网络推理
 */

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

class TreeNode {
  constructor(parent, priorP) {
    this.parent = parent;
    this.children = new Map(); // action -> TreeNode
    this.nVisits = 0;
    this.Q = 0;
    this.P = priorP;
  }

  expand(actionPriors) {
    for (const [action, prob] of actionPriors) {
      if (!this.children.has(action)) {
        this.children.set(action, new TreeNode(this, prob));
      }
    }
  }

  select(cPuct) {
    let bestScore = -Infinity;
    let bestAction = -1;
    let bestChild = null;

    for (const [action, child] of this.children) {
      const ucb = child.Q + cPuct * child.P *
        Math.sqrt(this.nVisits) / (1 + child.nVisits);
      if (ucb > bestScore) {
        bestScore = ucb;
        bestAction = action;
        bestChild = child;
      }
    }
    return [bestAction, bestChild];
  }

  update(leafValue) {
    this.nVisits += 1;
    this.Q += (leafValue - this.Q) / this.nVisits;
  }

  updateRecursive(leafValue) {
    if (this.parent) {
      this.parent.updateRecursive(-leafValue);
    }
    this.update(leafValue);
  }

  isLeaf() { return this.children.size === 0; }
  isRoot() { return this.parent === null; }
}

export class MCTS {
  /**
   * @param {Function} policyValueFn - async (board) => { actionProbs: [[action, prob]], value: number }
   * @param {number} cPuct - 探索常数
   * @param {number} nPlayout - 每步模拟次数
   */
  constructor(policyValueFn, cPuct = 5, nPlayout = 400) {
    this.root = new TreeNode(null, 1.0);
    this.policyFn = policyValueFn;
    this.cPuct = cPuct;
    this.nPlayout = nPlayout;
  }

  async playout(state) {
    let node = this.root;

    // 1. 选择：沿树向下直到叶节点
    while (!node.isLeaf()) {
      const [action, child] = node.select(this.cPuct);
      state.doMove(action);
      node = child;
    }

    // 2. 评估
    const { actionProbs, value } = await this.policyFn(state);
    const [end, winner] = state.gameEnd();

    let leafValue;
    if (!end) {
      // 3. 扩展
      node.expand(actionProbs);
      leafValue = value;
    } else {
      if (winner === -1) {
        leafValue = 0.0;
      } else {
        leafValue = winner === state.currentPlayer ? 1.0 : -1.0;
      }
    }

    // 4. 回传
    node.updateRecursive(-leafValue);
  }

  async getMoveProbs(state, temp = 1e-3) {
    for (let i = 0; i < this.nPlayout; i++) {
      const stateCopy = state.clone();
      await this.playout(stateCopy);
    }

    const acts = [];
    const visits = [];
    for (const [act, child] of this.root.children) {
      acts.push(act);
      visits.push(child.nVisits);
    }

    const logVisits = visits.map(v => Math.log(v + 1e-10) / temp);
    const probs = softmax(logVisits);
    return { acts, probs };
  }

  updateWithMove(lastMove) {
    if (this.root.children.has(lastMove)) {
      this.root = this.root.children.get(lastMove);
      this.root.parent = null;
    } else {
      this.root = new TreeNode(null, 1.0);
    }
  }
}

/**
 * AI 玩家：MCTS + ONNX 神经网络
 */
export class AlphaZeroPlayer {
  /**
   * @param {ort.InferenceSession} session - ONNX 推理会话
   * @param {number} boardSize
   * @param {number} nPlayout
   */
  constructor(session, boardSize, nPlayout = 400) {
    this.session = session;
    this.boardSize = boardSize;

    // 包装策略价值函数
    const policyValueFn = async (board) => {
      const inputData = board.currentState();
      const tensor = new ort.Tensor('float32', inputData,
        [1, 4, boardSize, boardSize]);
      const results = await this.session.run({ input: tensor });

      const logProbs = results.log_probs.data;
      const value = results.value.data[0];

      // 合法动作概率归一化
      const legalMoves = [...board.availables];
      const rawProbs = legalMoves.map(m => Math.exp(logProbs[m]));
      const probSum = rawProbs.reduce((a, b) => a + b, 0);
      const actionProbs = legalMoves.map((m, i) => [m, probSum > 0 ? rawProbs[i] / probSum : 1.0 / legalMoves.length]);

      return { actionProbs, value };
    };

    this.mcts = new MCTS(policyValueFn, 5, nPlayout);
  }

  async getAction(board) {
    if (board.availables.size === 0) return -1;
    const { acts, probs } = await this.mcts.getMoveProbs(board);
    // 选择概率最高的动作
    let bestIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIdx]) bestIdx = i;
    }
    const move = acts[bestIdx];
    // 保留子树，复用搜索结果（比 reset 更高效）
    this.mcts.updateWithMove(move);
    return move;
  }

  observeOpponentMove(move) {
    this.mcts.updateWithMove(move);
  }

  reset() {
    this.mcts.updateWithMove(-1);
  }

  setNPlayout(n) {
    this.mcts.nPlayout = n;
  }
}

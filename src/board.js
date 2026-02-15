/**
 * 五子棋棋盘逻辑 (纯 JavaScript)
 * 与 Python 端 Board 类完全一致的状态表示
 */
export class Board {
  constructor(size = 9, nInRow = 5) {
    this.size = size;
    this.nInRow = nInRow;
    this.players = [1, 2];
    this.init();
  }

  init(startPlayer = 0) {
    this.currentPlayer = this.players[startPlayer];
    this.states = new Map();
    this.availables = new Set();
    for (let i = 0; i < this.size * this.size; i++) {
      this.availables.add(i);
    }
    this.lastMove = -1;
  }

  clone() {
    const b = new Board(this.size, this.nInRow);
    b.currentPlayer = this.currentPlayer;
    b.states = new Map(this.states);
    b.availables = new Set(this.availables);
    b.lastMove = this.lastMove;
    return b;
  }

  doMove(move) {
    this.states.set(move, this.currentPlayer);
    this.availables.delete(move);
    this.currentPlayer = this.currentPlayer === this.players[0]
      ? this.players[1]
      : this.players[0];
    this.lastMove = move;
  }

  /**
   * 生成 4 通道棋盘状态 Float32Array, 与 Python 端 current_state() 一致
   * 通道0: 当前玩家棋子
   * 通道1: 对手棋子
   * 通道2: 最后落子
   * 通道3: 先手标记
   */
  currentState() {
    const s = this.size;
    const state = new Float32Array(4 * s * s);

    for (const [move, player] of this.states) {
      const r = Math.floor(move / s);
      const c = move % s;
      const flippedR = s - 1 - r; // numpy [:, ::-1, :]
      const idx = flippedR * s + c;

      if (player === this.currentPlayer) {
        state[0 * s * s + idx] = 1.0;
      } else {
        state[1 * s * s + idx] = 1.0;
      }
    }

    if (this.lastMove >= 0) {
      const r = Math.floor(this.lastMove / s);
      const c = this.lastMove % s;
      const flippedR = s - 1 - r;
      state[2 * s * s + flippedR * s + c] = 1.0;
    }

    if (this.states.size % 2 === 0) {
      for (let i = 0; i < s * s; i++) {
        state[3 * s * s + i] = 1.0;
      }
    }

    return state;
  }

  hasWinner() {
    if (this.lastMove === -1) return [false, -1];
    if (this.states.size < 2 * this.nInRow - 1) return [false, -1];

    const move = this.lastMove;
    const h = Math.floor(move / this.size);
    const w = move % this.size;
    const player = this.states.get(move);
    const n = this.nInRow;

    const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];

    for (const [dh, dw] of directions) {
      let count = 1;
      for (let i = 1; i < n; i++) {
        const nh = h + i * dh, nw = w + i * dw;
        if (nh >= 0 && nh < this.size && nw >= 0 && nw < this.size &&
          this.states.get(nh * this.size + nw) === player) {
          count++;
        } else break;
      }
      for (let i = 1; i < n; i++) {
        const nh = h - i * dh, nw = w - i * dw;
        if (nh >= 0 && nh < this.size && nw >= 0 && nw < this.size &&
          this.states.get(nh * this.size + nw) === player) {
          count++;
        } else break;
      }
      if (count >= n) return [true, player];
    }
    return [false, -1];
  }

  gameEnd() {
    const [win, winner] = this.hasWinner();
    if (win) return [true, winner];
    if (this.availables.size === 0) return [true, -1];
    return [false, -1];
  }
}

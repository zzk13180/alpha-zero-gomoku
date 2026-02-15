import './style.css';
import { Board } from './board.js';
import { AlphaZeroPlayer } from './mcts.js';

const BOARD_SIZE = 9;
const CELL_SIZE = 44;

class GomokuGame {
    constructor() {
        this.boardSize = BOARD_SIZE;
        this.cellSize = CELL_SIZE;
        this.board = null;       // Board 实例（JS）
        this.uiBoard = [];       // 二维数组用于绘制
        this.history = [];
        this.currentPlayer = 1;  // 1: 玩家(黑), 2: AI(白)
        this.gameStarted = false;
        this.aiThinking = false;
        this.ai = null;          // AlphaZeroPlayer 实例
        this.playerPiece = 1;

        this.canvas = document.getElementById('board');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = this.cellSize * (this.boardSize + 1);
        this.canvas.height = this.cellSize * (this.boardSize + 1);

        this.initBoard();
        this.drawBoard();
        this.setupEventListeners();
        this.initEngine();
    }

    initBoard() {
        this.uiBoard = Array(this.boardSize).fill(null).map(() =>
            Array(this.boardSize).fill(0)
        );
        this.history = [];
        this.currentPlayer = 1;
        if (this.board) {
            this.board.init();
        }
    }

    setupEventListeners() {
        document.getElementById('startBtn').addEventListener('click', () => this.startGame());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetGame());
        document.getElementById('undoBtn').addEventListener('click', () => this.undoMove());

        document.getElementById('aiLevel').addEventListener('change', (e) => {
            if (this.ai) {
                this.ai.setNPlayout(parseInt(e.target.value));
            }
        });

        this.canvas.addEventListener('click', (e) => this.handleClick(e));
    }

    async initEngine() {
        const loadingDiv = document.getElementById('loading');
        const statusDiv = document.getElementById('status');

        loadingDiv.style.display = 'block';

        try {
            statusDiv.textContent = '加载 ONNX Runtime...';

            // 动态加载 ONNX Runtime Web (CDN)
            if (!window.ort) {
                await new Promise((resolve, reject) => {
                    const script = document.createElement('script');
                    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
                    script.onload = resolve;
                    script.onerror = reject;
                    document.head.appendChild(script);
                });
            }

            statusDiv.textContent = '加载神经网络模型...';
            const session = await ort.InferenceSession.create('model_fast.onnx');

            statusDiv.textContent = '初始化 AI...';
            const nPlayout = parseInt(document.getElementById('aiLevel').value);
            this.ai = new AlphaZeroPlayer(session, this.boardSize, nPlayout);
            this.board = new Board(this.boardSize, 5);

            loadingDiv.style.display = 'none';
            statusDiv.textContent = '就绪！点击开始游戏';
            document.getElementById('startBtn').disabled = false;
        } catch (error) {
            console.error('初始化失败:', error);
            statusDiv.textContent = '初始化失败: ' + error.message;
            loadingDiv.style.display = 'none';
        }
    }

    startGame() {
        if (!this.ai) { alert('系统尚未就绪'); return; }
        this.resetGame();
        this.gameStarted = true;
        document.getElementById('startBtn').disabled = true;

        const nPlayout = parseInt(document.getElementById('aiLevel').value);
        this.ai.setNPlayout(nPlayout);
        this.ai.reset();
        this.board.init();

        const aiFirst = document.getElementById('aiFirst').checked;
        if (aiFirst) {
            this.playerPiece = 2;
            this.currentPlayer = 2;
            this.updateStatus('AI 回合（黑棋）');
            setTimeout(() => this.aiMove(), 300);
        } else {
            this.playerPiece = 1;
            this.currentPlayer = 1;
            this.updateStatus('你的回合（黑棋）');
        }
    }

    resetGame() {
        this.initBoard();
        this.drawBoard();
        this.gameStarted = false;
        this.aiThinking = false;
        document.getElementById('startBtn').disabled = false;
        document.getElementById('history').innerHTML = '';
        document.getElementById('thinking').style.display = 'none';
        this.updateStatus('点击开始游戏');
    }

    handleClick(e) {
        if (!this.gameStarted || this.aiThinking || this.currentPlayer !== 1) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const col = Math.round((x - this.cellSize) / this.cellSize);
        const row = Math.round((y - this.cellSize) / this.cellSize);

        if (row >= 0 && row < this.boardSize && col >= 0 && col < this.boardSize) {
            if (this.uiBoard[row][col] === 0) {
                this.makeMove(row, col, 1);
            }
        }
    }

    makeMove(row, col, player) {
        const move = row * this.boardSize + col;

        this.uiBoard[row][col] = player;
        this.history.push({ row, col, player });

        // 同步更新 JS 棋盘
        this.board.doMove(move);
        if (this.ai) {
            this.ai.observeOpponentMove(move);
        }

        this.drawBoard();
        this.addMoveToHistory(row, col, player);

        const [gameEnd, winner] = this.board.gameEnd();
        if (gameEnd) {
            this.handleGameEnd(winner);
            return;
        }

        this.currentPlayer = player === 1 ? 2 : 1;
        if (this.currentPlayer === 2) {
            setTimeout(() => this.aiMove(), 100);
        } else {
            this.updateStatus('你的回合（黑棋）');
        }
    }

    async aiMove() {
        this.aiThinking = true;
        this.updateStatus('AI 思考中...');
        document.getElementById('thinking').style.display = 'flex';
        this.canvas.classList.add('disabled');

        try {
            // 让 UI 有时间渲染 "思考中" 状态
            await new Promise(r => setTimeout(r, 50));
            const move = await this.ai.getAction(this.board);

            if (move >= 0) {
                const row = Math.floor(move / this.boardSize);
                const col = move % this.boardSize;

                this.uiBoard[row][col] = 2;
                this.history.push({ row, col, player: 2 });
                // AI 自己的 observeOpponentMove 已在 getAction 中处理
                // 但需要同步棋盘状态（board.doMove 保持一致）
                // board 状态由 AI 内部的 copy 不修改，需要手动推进
                this.board.doMove(move);

                this.drawBoard();
                this.addMoveToHistory(row, col, 2);

                const [gameEnd, winner] = this.board.gameEnd();
                if (gameEnd) {
                    this.handleGameEnd(winner);
                    return;
                }
                this.currentPlayer = 1;
                this.updateStatus('你的回合（黑棋）');
            } else {
                this.updateStatus('AI 出错');
            }
        } catch (error) {
            console.error('AI 出错:', error);
            this.updateStatus('系统错误');
        } finally {
            this.aiThinking = false;
            document.getElementById('thinking').style.display = 'none';
            this.canvas.classList.remove('disabled');
        }
    }

    handleGameEnd(winner) {
        this.gameStarted = false;
        document.getElementById('startBtn').disabled = false;

        if (winner === this.playerPiece) {
            this.updateStatus('🎉 恭喜你获胜！');
        } else if (winner === -1) {
            this.updateStatus('🤝 平局');
        } else {
            this.updateStatus('😔 AI 获胜，再接再厉！');
        }
    }

    undoMove() {
        if (!this.gameStarted || this.aiThinking || this.history.length < 2) return;

        // 撤回两步（玩家 + AI）
        for (let i = 0; i < 2; i++) {
            if (this.history.length > 0) {
                const last = this.history.pop();
                this.uiBoard[last.row][last.col] = 0;
            }
        }

        // 从头重建棋盘状态
        this.board.init();
        this.ai.reset();
        for (const { row, col } of this.history) {
            const move = row * this.boardSize + col;
            this.board.doMove(move);
        }

        this.drawBoard();
        this.updateHistory();
        this.currentPlayer = 1;
        this.updateStatus('你的回合（黑棋）');
    }

    // ==================== 绘制 ====================

    drawBoard() {
        const ctx = this.ctx;
        const size = this.cellSize;

        ctx.fillStyle = '#dcb35c';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 3;
        ctx.lineCap = 'square';

        for (let i = 0; i < this.boardSize; i++) {
            ctx.beginPath();
            ctx.moveTo(size, size * (i + 1));
            ctx.lineTo(size * this.boardSize, size * (i + 1));
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(size * (i + 1), size);
            ctx.lineTo(size * (i + 1), size * this.boardSize);
            ctx.stroke();
        }

        // 星位
        const starPoints = [[2, 2], [2, 6], [4, 4], [6, 2], [6, 6]];
        ctx.fillStyle = '#2c3e50';
        for (const [row, col] of starPoints) {
            ctx.fillRect(size * (col + 1) - 4, size * (row + 1) - 4, 8, 8);
        }

        for (let row = 0; row < this.boardSize; row++) {
            for (let col = 0; col < this.boardSize; col++) {
                if (this.uiBoard[row][col] !== 0) {
                    this.drawPiece(row, col, this.uiBoard[row][col]);
                }
            }
        }

        if (this.history.length > 0) {
            const last = this.history[this.history.length - 1];
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 3;
            ctx.strokeRect(
                size * (last.col + 1) - 5,
                size * (last.row + 1) - 5,
                10, 10
            );
        }
    }

    drawPiece(row, col, player) {
        const ctx = this.ctx;
        const size = this.cellSize;
        const x = size * (col + 1);
        const y = size * (row + 1);
        const radius = size * 0.4;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);

        if (player === 1) {
            ctx.fillStyle = '#2c3e50';
            ctx.fill();
            ctx.fillStyle = 'rgba(255,255,255,0.2)';
            ctx.fillRect(x - radius / 2, y - radius / 2, radius / 2, radius / 2);
        } else {
            ctx.fillStyle = '#ecf0f1';
            ctx.fill();
            ctx.strokeStyle = '#2c3e50';
            ctx.lineWidth = 3;
            ctx.stroke();
        }
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }

    addMoveToHistory(row, col, player) {
        const history = document.getElementById('history');
        const moveNum = this.history.length;
        const playerName = player === 1 ? '玩家' : 'AI';
        const color = player === 1 ? '黑' : '白';

        const moveItem = document.createElement('div');
        moveItem.className = `move-item ${player === 1 ? 'player' : 'ai'}`;
        moveItem.textContent = `第${moveNum}步 ${playerName}(${color}): (${row}, ${col})`;
        history.appendChild(moveItem);
        history.scrollTop = history.scrollHeight;
    }

    updateHistory() {
        const history = document.getElementById('history');
        history.innerHTML = '';
        this.history.forEach((move, index) => {
            const playerName = move.player === 1 ? '玩家' : 'AI';
            const color = move.player === 1 ? '黑' : '白';
            const moveItem = document.createElement('div');
            moveItem.className = `move-item ${move.player === 1 ? 'player' : 'ai'}`;
            moveItem.textContent = `第${index + 1}步 ${playerName}(${color}): (${move.row}, ${move.col})`;
            history.appendChild(moveItem);
        });
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new GomokuGame();
});

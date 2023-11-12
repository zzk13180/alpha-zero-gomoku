import './style.css';
import mctsCode from './mcts.py?raw';

function loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

class GomokuGame {
    constructor() {
        // 模型是 9x9 训练的
        this.boardSize = 9;
        this.cellSize = 40;
        this.board = [];
        this.history = [];
        this.currentPlayer = 1; // 1: PLAYER(BLK), 2: AI(WHT)
        this.gameStarted = false;
        this.aiThinking = false;
        this.pyodideReady = false;
        this.ortSession = null;
        
        this.canvas = document.getElementById('board');
        this.ctx = this.canvas.getContext('2d');
        
        // 调整画布大小
        this.canvas.width = this.cellSize * (this.boardSize + 1);
        this.canvas.height = this.cellSize * (this.boardSize + 1);
        
        this.initBoard();
        this.drawBoard();
        this.setupEventListeners();
        this.initEngine();
    }

    initBoard() {
        this.board = Array(this.boardSize).fill(null).map(() => 
            Array(this.boardSize).fill(0)
        );
        this.history = [];
        this.currentPlayer = 1;
    }

    setupEventListeners() {
        const startBtn = document.getElementById('startBtn');
        const resetBtn = document.getElementById('resetBtn');
        const undoBtn = document.getElementById('undoBtn');
        const aiLevel = document.getElementById('aiLevel');
        
        startBtn.addEventListener('click', () => this.startGame());
        resetBtn.addEventListener('click', () => this.resetGame());
        undoBtn.addEventListener('click', () => this.undoMove());
        
        // 监听难度变化
        aiLevel.addEventListener('change', async (e) => {
            if (this.pyodideReady) {
                const level = e.target.value;
                await pyodide.runPythonAsync(`set_ai_level(${level})`);
            }
        });
        
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
    }

    async initEngine() {
        const loadingDiv = document.getElementById('loading');
        const statusDiv = document.getElementById('status');
        
        loadingDiv.style.display = 'block';
        
        try {
            // 1. 初始化 Pyodide 和 ONNX Runtime
            // 为了保证全局变量存在，这里手动加载 CDN (Vite 开发环境下)
            statusDiv.textContent = 'LOADING LIBRARIES...';
            // 检查全局变量是否已存在
            if (!window.loadPyodide) {
                await loadScript('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js');
            }
            if (!window.ort) {
                await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
            }

            statusDiv.textContent = 'SYSTEM INITIALIZING...';
            window.pyodide = await loadPyodide({
                indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
            });
            
            statusDiv.textContent = 'INSTALLING DEPENDENCIES...';
            await pyodide.loadPackage(['numpy', 'micropip']);
            
            // 2. 加载 ONNX 模型
            statusDiv.textContent = 'LOADING NEURAL NETWORK...';
            // 注意：Vite 把 public 下的文件直接放在根路径
            this.ortSession = await ort.InferenceSession.create('/model.onnx');
            
            // 3. 暴露预测函数给 Python
            window.predict = async (flatData) => {
                let inputData;
                if (flatData.toJs) {
                    inputData = flatData.toJs();
                } else {
                    inputData = flatData;
                }
                
                const tensor = new ort.Tensor('float32', inputData, [1, 4, 9, 9]);
                const feeds = { input: tensor };
                
                const results = await this.ortSession.run(feeds);
                
                // 返回 [log_probs, value]
                return [results.log_probs.data, results.value.data[0]];
            };

            statusDiv.textContent = 'COMPILING LOGIC...';
            await this.loadPythonCode();
            
            this.pyodideReady = true;
            loadingDiv.style.display = 'none';
            statusDiv.textContent = 'READY. PRESS START.';
            document.getElementById('startBtn').disabled = false;
            
        } catch (error) {
            console.error('初始化失败:', error);
            statusDiv.textContent = 'INIT FAILED: ' + error.message;
            loadingDiv.style.display = 'none';
        }
    }

    async loadPythonCode() {
        // 加载 AlphaZero MCTS Python 代码
        // 直接运行导入的字符串，无需担心缩进问题，因为原始文件是干净的
        await pyodide.runPythonAsync(mctsCode);
        console.log('Python AlphaZero 代码加载完成');
    }

    async startGame() {
        if (!this.pyodideReady) {
            alert('SYSTEM NOT READY');
            return;
        }

        this.resetGame();
        this.gameStarted = true;
        document.getElementById('startBtn').disabled = true;
        
        // 初始化游戏，传入参数
        const aiLevel = document.getElementById('aiLevel').value;
        await pyodide.runPythonAsync(`init_game(width=${this.boardSize}, height=${this.boardSize}, n_playout=${aiLevel})`);
        
        const aiFirst = document.getElementById('aiFirst').checked;
        
        if (aiFirst) {
            this.currentPlayer = 2;
            this.updateStatus('AI TURN (WHITE)');
            setTimeout(() => this.aiMove(), 500);
        } else {
            this.currentPlayer = 1;
            this.updateStatus('YOUR TURN (BLACK)');
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
        this.updateStatus('PRESS START TO PLAY');
    }

    handleClick(e) {
        if (!this.gameStarted || this.aiThinking || this.currentPlayer !== 1) {
            return;
        }

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const col = Math.round((x - this.cellSize) / this.cellSize);
        const row = Math.round((y - this.cellSize) / this.cellSize);

        if (row >= 0 && row < this.boardSize && col >= 0 && col < this.boardSize) {
            if (this.board[row][col] === 0) {
                this.makeMove(row, col, 1);
            }
        }
    }

    async makeMove(row, col, player) {
        const move = row * this.boardSize + col;
        
        this.board[row][col] = player;
        this.history.push({ row, col, player });
        
        await pyodide.runPythonAsync(`make_move(${move})`);
        
        this.drawBoard();
        this.addMoveToHistory(row, col, player);

        const result = await pyodide.runPythonAsync('check_game_end()');
        const gameEnd = result.toJs()[0];
        const winner = result.toJs()[1];

        if (gameEnd) {
            this.handleGameEnd(winner);
            return;
        }

        this.currentPlayer = player === 1 ? 2 : 1;

        if (this.currentPlayer === 2) {
            setTimeout(() => this.aiMove(), 100);
        } else {
            this.updateStatus('YOUR TURN (BLACK)');
        }
    }

    async aiMove() {
        this.aiThinking = true;
        this.updateStatus('AI IS THINKING...');
        document.getElementById('thinking').style.display = 'flex';
        this.canvas.classList.add('disabled');

        try {
            const move = await pyodide.runPythonAsync('await get_ai_move()');
            
            if (move !== null && move !== undefined && move !== -1) {
                const row = Math.floor(move / this.boardSize);
                const col = move % this.boardSize;
                
                await this.makeMove(row, col, 2);
            } else {
                console.error("AI returned invalid move:", move);
                this.updateStatus('AI ERROR');
            }
        } catch (error) {
            console.error('AI Move failed:', error);
            this.updateStatus('SYSTEM ERROR');
        } finally {
            this.aiThinking = false;
            document.getElementById('thinking').style.display = 'none';
            this.canvas.classList.remove('disabled');
        }
    }

    handleGameEnd(winner) {
        this.gameStarted = false;
        document.getElementById('startBtn').disabled = false;
        
        if (winner === 1) {
            this.updateStatus('YOU WIN!');
        } else if (winner === 2) {
            this.updateStatus('GAME OVER. AI WINS.');
        } else {
            this.updateStatus('DRAW GAME.');
        }
    }

    undoMove() {
        if (!this.gameStarted || this.aiThinking || this.history.length < 2) {
            return;
        }

        for (let i = 0; i < 2; i++) {
            if (this.history.length > 0) {
                const lastMove = this.history.pop();
                this.board[lastMove.row][lastMove.col] = 0;
            }
        }

        // 使用我们在 Python 端定义的函数
        const historyJson = JSON.stringify(this.history);
        pyodide.runPythonAsync(`
import json
restore_from_history(json.loads('${historyJson}'), width=${this.boardSize}, height=${this.boardSize})
        `);

        this.drawBoard();
        this.updateHistory();
        this.currentPlayer = 1;
        this.updateStatus('YOUR TURN (BLACK)');
    }

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

        let starPoints = [];
        if (this.boardSize === 15) {
            starPoints = [[3, 3], [3, 11], [7, 7], [11, 3], [11, 11]];
        } else if (this.boardSize === 9) {
            starPoints = [[2, 2], [2, 6], [4, 4], [6, 2], [6, 6]];
        }
        
        ctx.fillStyle = '#2c3e50';
        starPoints.forEach(([row, col]) => {
            const x = size * (col + 1) - 4;
            const y = size * (row + 1) - 4;
            ctx.fillRect(x, y, 8, 8);
        });

        for (let row = 0; row < this.boardSize; row++) {
            for (let col = 0; col < this.boardSize; col++) {
                if (this.board[row][col] !== 0) {
                    this.drawPiece(row, col, this.board[row][col]);
                }
            }
        }

        if (this.history.length > 0) {
            const last = this.history[this.history.length - 1];
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 3;
            const x = size * (last.col + 1);
            const y = size * (last.row + 1);
            ctx.strokeRect(x - 5, y - 5, 10, 10);
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
            ctx.fillRect(x - radius/2, y - radius/2, radius/2, radius/2);
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
        const playerName = player === 1 ? 'PLAYER' : 'AI';
        const color = player === 1 ? 'BLK' : 'WHT';
        
        const moveItem = document.createElement('div');
        moveItem.className = `move-item ${player === 1 ? 'player' : 'ai'}`;
        moveItem.textContent = `${moveNum}. ${playerName}(${color}): (${row}, ${col})`;
        
        history.appendChild(moveItem);
        history.scrollTop = history.scrollHeight;
    }

    updateHistory() {
        const history = document.getElementById('history');
        history.innerHTML = '';
        
        this.history.forEach((move, index) => {
            const playerName = move.player === 1 ? 'PLAYER' : 'AI';
            const color = move.player === 1 ? 'BLK' : 'WHT';
            
            const moveItem = document.createElement('div');
            moveItem.className = `move-item ${move.player === 1 ? 'player' : 'ai'}`;
            moveItem.textContent = `${index + 1}. ${playerName}(${color}): (${move.row}, ${move.col})`;
            
            history.appendChild(moveItem);
        });
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new GomokuGame();
});

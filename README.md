# AlphaZero 五子棋

基于 PyTorch 和 AlphaZero 算法实现的五子棋 AI。包含自定义策略价值网络、MCTS 搜索与自我对弈训练管线，从零开始训练强化学习模型。

## 🛠️ 环境安装

**依赖要求**：Python 3.10+，PyTorch 2.0+，NumPy 1.24+

```bash
# 克隆项目
git clone <repository_url>
cd alpha-zero-gomoku

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 训练模型

```bash
# 快速模式（默认）- 9x9 棋盘，约 30-60 分钟
python -m scripts.train

# 完整模式 - 15x15 标准棋盘，需要数小时
python -m scripts.train --mode full

# 加载已有模型继续训练
python -m scripts.train --mode fast --model outputs/model_fast_100.pth
```

#### 训练模式对比

| 参数 | fast 模式 | full 模式 |
|------|-----------|-----------|
| 棋盘大小 | 9×9 | 15×15 |
| MCTS 模拟次数 | 400 | 400 |
| 训练局数 | 500 | 1500 |
| 批量大小 | 512 | 512 |
| 缓冲区大小 | 10000 | 10000 |
| 预计时间 | 30-60 分钟 | 数小时 |

- **训练过程**：自动进行自我对弈收集数据，使用经验回放训练网络
- **模型保存**：每 50 局自动保存检查点到 `outputs/` 目录（如 `outputs/model_fast_50.pth`）
- **中断恢复**：按 `Ctrl+C` 中断时会自动保存当前模型

### 2. 人机对战

```bash
# 使用默认模型
python -m scripts.human_play

# 指定模型（自动在 outputs/ 目录查找）
python -m scripts.human_play --model model_fast_100.pth
```

- 输入格式：`行,列`（如 `4,4` 表示落子在第 4 行第 4 列）

### 3. 模型评估

对比两个模型的强弱：

```bash
# 两模型对战 20 局（自动在 outputs/ 目录查找）
python -m scripts.evaluate_models model_fast_50.pth model_fast_final.pth

# 更多对局和模拟次数（更准确）
python -m scripts.evaluate_models model_fast_50.pth model_fast_final.pth -n 30 -p 800
```

参数说明：
- `-n`: 对战局数（偶数，保证先手公平）
- `-p`: MCTS 模拟次数（越大越准确但越慢）
- `-s`: 棋盘大小（默认 9）

> **验证评估脚本**：让同一模型自我对战，胜率应接近 50%

## 📁 项目结构

```
alpha-zero-gomoku/
├── game/                       # 游戏核心模块
│   ├── __init__.py
│   ├── board.py               # 棋盘类：状态管理、胜负判定
│   ├── mcts.py                # MCTS 搜索 + AI 玩家
│   └── game_ui.py             # 自我对弈控制
├── model/                      # 神经网络模块
│   ├── __init__.py
│   ├── net.py                 # 策略价值网络结构 (CNN)
│   └── policy_value_net.py    # 网络封装：训练、推理、保存/加载
├── scripts/                    # 运行脚本
│   ├── train.py               # 训练入口
│   ├── human_play.py          # 人机对战
│   └── evaluate_models.py     # 模型评估对比
├── outputs/                    # 训练输出目录（自动生成）
│   └── *.pth                  # 保存的模型文件
├── pyproject.toml             # 项目配置 + Ruff 代码规范
├── requirements.txt           # 生产依赖
├── requirements-dev.txt       # 开发依赖
└── README.md
```

## 🧠 算法原理

### 核心思想

AlphaZero 将 **MCTS** 与 **深度神经网络** 结合：

- **策略价值网络**：同时输出落子概率 $P(s,a)$ 和局面评估 $V(s)$
- **MCTS 搜索**：使用网络指导搜索，通过 UCB 公式选择动作：
  $$UCB(s,a) = Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

### 网络结构

```
输入 (4, H, W)
    ├─ 通道 0: 当前玩家棋子位置
    ├─ 通道 1: 对手棋子位置
    ├─ 通道 2: 最后落子位置
    └─ 通道 3: 先手标记 (先手=1, 后手=0)
         │
         ▼
   共享卷积层 (Conv+ReLU) × 3
   [4→32→64→128 通道]
         │
    ┌────┴────┐
    ▼         ▼
 策略头      价值头
 Conv 1×1   Conv 1×1
    │         │
    ▼         ▼
 Softmax    Tanh
    │         │
    ▼         ▼
 落子概率   局面评估
 (H×W)     [-1, 1]
```

> **注意**：当前实现未使用 BatchNorm，实测表明在小规模自我对弈场景下，简单网络结构更稳定。

### 训练流程

1. **自我对弈 (Self-Play)**
   - 使用 MCTS + 当前网络进行对弈
   - 收集数据 $(s, \pi, z)$：状态、MCTS 概率、胜负结果
   - 添加 Dirichlet 噪声增加探索

2. **数据增强**
   - 利用棋盘对称性，将每局数据扩充 8 倍
   - 4 种旋转 × 2 种翻转

3. **网络训练**
   - 损失函数：$L = (z - v)^2 - \pi^T \log p + c\|\theta\|^2$
   - 使用 Adam 优化器 + 自适应学习率
   - 根据 KL 散度动态调整学习率

4. **循环迭代**
   - 重复以上步骤，网络逐渐变强

# AlphaZero Gomoku

[中文](README_zh-CN.md) | English

A Gomoku reinforcement learning system based on PyTorch and the AlphaZero algorithm. This project implements a custom Policy-Value Network, Monte Carlo Tree Search (MCTS), and a complete self-play training pipeline to train a reinforcement learning model from scratch.

## Requirements and Installation

**Prerequisites**: Python 3.10+

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/zzk13180/alpha-zero-gomoku.git
cd alpha-zero-gomoku
```

2. Create a virtual environment (Recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # UNIX/Linux/macOS
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Guide

### 1. Model Training

Start the self-play training process.

```bash
# Fast mode (9x9 board, suitable for demonstration)
python -m scripts.train

# Standard mode (15x15 board, computationally intensive)
python -m scripts.train --mode full

# Resume training from an existing model
python -m scripts.train --mode fast --model outputs/model_fast_100.pth
```

**Configuration**:

| Parameter | Fast Mode | Full Mode |
|---|---|---|
| Board Size | 9×9 | 15×15 |
| MCTS Simulations | 400 | 400 |
| Training Episodes | 500 | 1500 |
| Batch Size | 512 | 512 |
| Buffer Size | 10000 | 10000 |
| Estimated Time | 30-60 mins | Several hours |

*   **Auto-save**: Checkpoints are automatically saved to `outputs/` directory every 50 episodes (e.g., `outputs/model_fast_50.pth`).

### 2. Human-AI Play

Play against the trained model.

```bash
# Use the default model
python -m scripts.human_play

# Specify a model (looked up in outputs/ directory)
python -m scripts.human_play --model model_fast_100.pth
```

- Input format: `row,col` (e.g., `4,4` places a stone at row 4, column 4).

### 3. Model Evaluation

Compare the strength of two models:

```bash
# Battle between two models for 20 games (looked up in outputs/ directory)
python -m scripts.evaluate_models model_fast_50.pth model_fast_final.pth
```

**Optional Arguments**:
*   `-n`: Number of games (default 10)
*   `-p`: MCTS simulations per move (default 400)
*   `-s`: Board size (default 9)

## Project Structure

```
alpha-zero-gomoku/
├── game/                  # Core game logic
│   ├── board.py           # Board state and rules
│   ├── mcts.py            # MCTS implementation
│   └── game_ui.py         # Game flow control
├── model/                 # Neural Network Models
│   ├── net.py             # Policy-Value Network structure (CNN)
│   └── policy_value_net.py # Network interface wrapper
├── scripts/               # Execution scripts
│   ├── train.py           # Training entry point
│   ├── human_play.py      # Human vs AI entry point
│   └── evaluate_models.py # Model evaluation
├── outputs/               # Training outputs (auto-generated)
│   └── *.pth              # Saved model files
├── pyproject.toml         # Project config + Ruff linting rules
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
└── README.md
```

## Algorithm Principle

### Core Idea

AlphaZero combines **MCTS** with **Deep Neural Networks**:

- **Policy-Value Network**: Outputs both move probabilities $P(s,a)$ and position evaluation $V(s)$.
- **MCTS Search**: Uses the network to guide the search, selecting actions via the UCB formula:
  $$UCB(s,a) = Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

### Network Structure

```
Input (4, H, W)
    ├─ Channel 0: Current player's stones
    ├─ Channel 1: Opponent's stones
    ├─ Channel 2: Last move position
    └─ Channel 3: Player color (Start=1, Second=0)
         │
         ▼
   Shared Conv Layers (Conv+ReLU) × 3
   [4→32→64→128 Filters]
         │
    ┌────┴────┐
    ▼         ▼
 Policy Head   Value Head
 Conv 1×1      Conv 1×1
    │            │
    ▼            ▼
 Softmax        Tanh
    │            │
    ▼            ▼
 Move Probs     Evaluation
 (H×W)          [-1, 1]
```

> **Note**: Current implementation does not use BatchNorm. Experiments show simple network structures are more stable for small-scale self-play.

### Training Process

1. **Self-Play**
   - Play games using MCTS + current network.
   - Collect data $(s, \pi, z)$: State, MCTS probabilities, Game result.
   - Add Dirichlet noise for exploration.

2. **Data Augmentation**
   - Exploit board symmetry to expand data 8x.
   - 4 rotations × 2 flips.

3. **Network Training**
   - Loss function: $L = (z - v)^2 - \pi^T \log p + c\|\theta\|^2$
   - Adam optimizer + Adaptive learning rate.
   - Dynamic learning rate adjustment based on KL divergence.

4. **Iteration**
   - Repeat the steps above to improve the network continuously.

"""
AlphaZero训练脚本

训练流程:
1. 自我对弈收集数据
2. 数据增强（旋转、翻转）
3. 训练神经网络
4. 定期保存模型

使用方法:
    python -m scripts.train [--mode fast|full]

    --mode fast: 快速模式，小棋盘(9x9)，少量训练，用于学习和验证（约5-10分钟）
    --mode full: 完整模式，标准棋盘(15x15)，充分训练（数小时）

    默认为 fast 模式

模型输出:
    训练产生的模型文件将保存到 outputs/ 目录
"""

import argparse
import random
from collections import deque

import numpy as np

from game import Board, Game, MCTSPlayer
from model import PolicyValueNet

# ===== 训练模式配置 =====
TRAIN_CONFIGS = {
    "fast": {
        # 快速模式：小棋盘，高质量自对弈
        "board_width": 9,
        "board_height": 9,
        "n_in_row": 5,
        "n_playout": 600,  # 增加MCTS模拟次数，提升走棋质量
        "buffer_size": 10000,  # 缓冲区，避免过快遗忘
        "batch_size": 512,  # 批量
        "epochs": 5,  # 训练轮数
        "game_batch_num": 500,  # 训练局数，确保收敛
        "check_freq": 50,  # 保存频率
        "description": "快速模式 (9x9棋盘, ResNet+BN, 约30-60分钟)",
    },
    "full": {
        # 完整模式：标准棋盘，充分训练
        "board_width": 15,
        "board_height": 15,
        "n_in_row": 5,
        "n_playout": 600,  # 标准MCTS模拟次数
        "buffer_size": 10000,  # 大缓冲区
        "batch_size": 512,  # 大批量
        "epochs": 5,  # 更多训练轮数
        "game_batch_num": 1500,  # 完整训练局数
        "check_freq": 50,  # 保存频率
        "description": "完整模式 (15x15棋盘, 数小时)",
    },
}


class Trainer:
    """
    AlphaZero训练器

    核心参数说明:
        n_playout: MCTS模拟次数，越大越强但越慢
        buffer_size: 经验回放缓冲区大小
        batch_size: 每次训练的样本数
        epochs: 每批数据训练轮数
        kl_targ: KL散度目标值，用于自适应调整学习率
    """

    def __init__(self, model_file=None, mode="fast", start_epoch=0):
        # ===== 加载训练配置 =====
        if mode not in TRAIN_CONFIGS:
            raise ValueError(f"未知模式: {mode}, 可选: {list(TRAIN_CONFIGS.keys())}")

        config = TRAIN_CONFIGS[mode]
        self.mode = mode
        self.start_epoch = start_epoch

        # ===== 棋盘设置 =====
        self.board_width = config["board_width"]
        self.board_height = config["board_height"]
        self.n_in_row = config["n_in_row"]

        # ===== 训练参数 =====
        self.learn_rate = 2e-3  # 初始学习率
        self.lr_multiplier = 1.0  # 学习率乘数（自适应调整）
        self.temp = 1.0  # 温度参数（控制探索）
        self.n_playout = config["n_playout"]  # 每步MCTS模拟次数
        self.c_puct = 5  # MCTS探索常数
        self.buffer_size = config["buffer_size"]  # 经验缓冲区大小
        self.batch_size = config["batch_size"]  # 批量大小
        self.epochs = config["epochs"]  # 每次更新训练轮数
        self.kl_targ = 0.02  # KL散度目标
        self.check_freq = config["check_freq"]  # 保存模型频率
        self.game_batch_num = config["game_batch_num"]  # 总训练局数

        # ===== 初始化组件 =====
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # 经验回放缓冲区（FIFO队列）
        self.data_buffer = deque(maxlen=self.buffer_size)

        # 策略价值网络
        self.policy_net = PolicyValueNet(self.board_width, self.board_height, model_file)

        # MCTS玩家（自我对弈模式）
        self.mcts_player = MCTSPlayer(
            self.policy_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=True,
        )

    def augment_data(self, play_data):
        """
        数据增强 - 通过旋转和翻转扩充数据

        五子棋具有对称性，一局游戏可以生成8倍数据（本质上是等价的局面）:
        - 4种旋转（0°, 90°, 180°, 270°）
        - 每种旋转可水平翻转
        """
        augmented = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 旋转状态
                rot_state = np.array([np.rot90(s, i) for s in state])
                # 旋转概率分布
                rot_prob = np.rot90(
                    np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i
                )
                augmented.append((rot_state, np.flipud(rot_prob).flatten(), winner))

                # 水平翻转
                flip_state = np.array([np.fliplr(s) for s in rot_state])
                flip_prob = np.fliplr(rot_prob)
                augmented.append((flip_state, np.flipud(flip_prob).flatten(), winner))

        return augmented

    def collect_selfplay_data(self, n_games=1):
        """收集自我对弈数据"""
        for _ in range(n_games):
            winner, play_data = self.game.self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            # 数据增强后加入缓冲区
            augmented = self.augment_data(play_data)
            self.data_buffer.extend(augmented)

    def policy_update(self):
        """
        更新策略网络

        使用经验回放 + 自适应学习率:
        - 从缓冲区随机采样一批数据
        - 训练多个epoch
        - 根据KL散度调整学习率
        """
        # 随机采样
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        states = [d[0] for d in mini_batch]
        mcts_probs = [d[1] for d in mini_batch]
        winners = [d[2] for d in mini_batch]

        # 记录更新前的策略
        old_probs, _ = self.policy_net.get_policy_probs(states)

        # 训练多个epoch
        for _ in range(self.epochs):
            loss, entropy = self.policy_net.train_step(
                states, mcts_probs, winners, self.learn_rate * self.lr_multiplier
            )

            # 计算KL散度，判断是否需要提前停止
            new_probs, _ = self.policy_net.get_policy_probs(states)
            kl = np.mean(
                np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1)
            )
            if kl > self.kl_targ * 4:
                break  # KL散度过大，提前停止

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        return loss, entropy

    def run(self):
        """开始训练"""
        config = TRAIN_CONFIGS[self.mode]
        print("=" * 50)
        print(f"AlphaZero训练 - {config['description']}")
        print("=" * 50)
        print(f"棋盘: {self.board_width}x{self.board_height}, 连珠: {self.n_in_row}")
        print(f"MCTS模拟次数: {self.n_playout}, 批量大小: {self.batch_size}")
        print(f"训练局数: {self.game_batch_num}, 缓冲区: {self.buffer_size}")
        print("-" * 50)

        try:
            for i in range(self.start_epoch, self.game_batch_num):
                # 1. 自我对弈收集数据
                self.collect_selfplay_data(1)
                print(
                    f"第 {i + 1} 局完成, 步数: {self.episode_len}, 缓冲区: {len(self.data_buffer)}"
                )

                # 2. 数据足够时开始训练
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    print(
                        f"  -> 损失: {loss:.4f}, 熵: {entropy:.4f}, 学习率: {self.learn_rate * self.lr_multiplier:.6f}"
                    )

                # 3. 定期保存模型
                if (i + 1) % self.check_freq == 0:
                    model_name = f"model_{self.mode}_{i + 1}.pth"
                    self.policy_net.save_model(model_name)
                    print(f"  -> 模型已保存: outputs/{model_name}")

        except KeyboardInterrupt:
            print("\n训练被中断，正在保存模型...")
            self.policy_net.save_model(f"model_{self.mode}_interrupted.pth")
            print(f"模型已保存: outputs/model_{self.mode}_interrupted.pth")

        # 训练完成，保存最终模型
        else:
            final_model = f"model_{self.mode}_final.pth"
            self.policy_net.save_model(final_model)
            print("-" * 50)
            print(f"训练完成！最终模型已保存: outputs/{final_model}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AlphaZero五子棋训练")
    parser.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "full"],
        help="训练模式: fast(快速验证) 或 full(完整训练)",
    )
    parser.add_argument("--model", type=str, default=None, help="加载已有模型继续训练（可选）")
    parser.add_argument("--start-epoch", type=int, default=0, help="起始训练轮数（用于恢复训练）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n选择模式: {args.mode}")
    print(f"模式说明: {TRAIN_CONFIGS[args.mode]['description']}\n")

    trainer = Trainer(model_file=args.model, mode=args.mode, start_epoch=args.start_epoch)
    trainer.run()

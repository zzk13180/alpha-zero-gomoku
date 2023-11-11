import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .net import PolicyValueNet as Net


class PolicyValueNetWrapper:
    """策略价值网络包装类: 训练、推理、保存/加载模型"""

    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=1e-4, lr=2e-3)

        if model_file:
            self.net.load_state_dict(torch.load(model_file, map_location="cpu"))

    def policy_value_fn(self, board):
        """
        策略价值函数，供MCTS调用

        返回:
            action_probs: 合法动作及其概率的迭代器
            value: 当前局面的价值评估 [-1, 1]

        """
        legal_positions = board.availables
        state = board.current_state()
        state = np.ascontiguousarray(state.reshape(-1, 4, self.board_height, self.board_width))
        state_tensor = torch.FloatTensor(state)

        self.net.eval()
        with torch.no_grad():
            log_probs, value = self.net(state_tensor)
            probs = np.exp(log_probs.numpy().flatten())

        # 获取合法动作的概率并归一化
        # legal_probs = probs[legal_positions]
        # if legal_probs.sum() > 0:
        #     legal_probs = legal_probs / legal_probs.sum()
        # action_probs = zip(legal_positions, legal_probs, strict=True)

        action_probs = zip(legal_positions, probs[legal_positions], strict=True)
        return action_probs, value.item()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """执行一步训练，返回 (loss, entropy)"""
        states = torch.FloatTensor(np.array(state_batch))
        mcts_probs = torch.FloatTensor(np.array(mcts_probs))
        winners = torch.FloatTensor(np.array(winner_batch))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.net.train()
        self.optimizer.zero_grad()
        log_probs, values = self.net(states)

        value_loss = F.mse_loss(values.view(-1), winners)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_probs, dim=1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=1))
        return loss.item(), entropy.item()

    def get_policy_probs(self, state_batch):
        """批量获取策略概率（用于计算KL散度）"""
        states = torch.FloatTensor(np.array(state_batch))
        self.net.eval()
        with torch.no_grad():
            log_probs, values = self.net(states)
            probs = np.exp(log_probs.numpy())
        return probs, values.numpy()

    def save_model(self, model_file):
        """保存模型参数到 outputs 目录"""
        import os

        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        model_path = os.path.join(outputs_dir, model_file)
        torch.save(self.net.state_dict(), model_path)


# 为了保持向后兼容
PolicyValueNet = PolicyValueNetWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块: Conv-BN-ReLU-Conv-BN + 跳跃连接"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class PolicyValueNet(nn.Module):
    """
    策略价值网络 (ResNet 版本)

    网络结构:
        输入(4通道) -> Conv-BN-ReLU -> ResBlock×3 (64 filters)
                                          ├-> 策略头(Conv1×1-BN-FC) -> 落子概率
                                          └-> 价值头(Conv1×1-BN-FC-FC) -> 局面评估

    输入: (batch, 4, height, width)
        - 通道0: 当前玩家棋子位置
        - 通道1: 对手棋子位置
        - 通道2: 最后落子位置
        - 通道3: 先手标记（先手回合全1，后手回合全0）

    输出:
        - log_probs: (batch, h*w) 每个位置的落子对数概率
        - value: (batch, 1) 当前局面评估值 [-1, 1]
    """

    def __init__(self, board_width, board_height):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        n_filters = 64

        # 初始卷积层
        self.conv_init = nn.Conv2d(4, n_filters, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(n_filters)

        # 残差块（3个 -> 小棋盘足够，参数适中）
        self.res_blocks = nn.Sequential(
            ResBlock(n_filters),
            ResBlock(n_filters),
            ResBlock(n_filters),
        )

        # 策略头
        self.policy_conv = nn.Conv2d(n_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_width * board_height, board_width * board_height)

        # 价值头
        self.value_conv = nn.Conv2d(n_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_width * board_height, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """前向传播"""
        # 共享特征提取：初始卷积 + 残差块
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.res_blocks(x)

        # 策略头：计算每个位置的落子概率
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * self.board_width * self.board_height)
        log_probs = F.log_softmax(self.policy_fc(p), dim=1)

        # 价值头：评估当前局面的价值
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, self.board_width * self.board_height)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return log_probs, value

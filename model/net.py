import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    """
    策略价值网络: 共享卷积层 + 策略头 + 价值头

    网络结构:
        输入(4通道) -> Conv1(32) -> Conv2(64) -> Conv3(128)
                                                    ├-> 策略头 -> 落子概率
                                                    └-> 价值头 -> 局面评估

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

        # 共享特征提取层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)

        # 策略头
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        # self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # 价值头
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        # self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """前向传播"""
        # 共享特征提取：通过三层卷积提取棋盘特征
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))

        # 策略头：计算每个位置的落子概率
        p = F.relu(self.policy_conv(x))
        # p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 4 * self.board_width * self.board_height)
        log_probs = F.log_softmax(self.policy_fc(p), dim=1)

        # 价值头：评估当前局面的价值
        v = F.relu(self.value_conv(x))
        # v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 2 * self.board_width * self.board_height)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return log_probs, value

"""交通流量 ResNet：过去 12 步的两个特征 -> 未来 6 步流量。

在 PyCharm 中直接运行本文件，可用随机数据检查网络形状；不进行训练。
模型输入顺序为 [批大小, 时间步, 特征数]，特征约定为流量（Flow）、速度（Avg kph）。
"""

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    """一维残差块：卷积 -> ReLU -> 卷积 -> 与直接分支相加。"""

    def __init__(self, input_channels, num_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            input_channels, num_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            num_channels, num_channels, kernel_size=3, padding=1
        )

        if input_channels != num_channels:
            self.shortcut = nn.Conv1d(
                input_channels, num_channels, kernel_size=1
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, X):
        Y = F.relu(self.conv1(X))
        Y = self.conv2(Y)
        # 不在修正量上加 ReLU，允许正向和负向修正。
        return self.shortcut(X) + Y


class TrafficResNet(nn.Module):   #PyTorch 中，一个网络通常写成继承 nn.Module 的类
    """用一维残差网络直接进行多步车流量预测。
    __init__ 是准备计算工具,forward 是规定计算过程。"""

    def __init__(self, input_steps=12, predict_steps=6):
        super().__init__()
        self.input_steps = input_steps

        # 初步提取特征：[B, 2, 12] -> [B, 32, 12]。
        self.input_conv = nn.Conv1d(2, 32, kernel_size=3, padding=1)

        # 三个块结构相同，但各自拥有独立的参数。nn.Sequential 可以理解成：把里面的几个模块首尾接起来。
        self.residual_blocks = nn.Sequential( 
            Residual(32, 32),
            Residual(32, 32),
            Residual(32, 32),
        )

        # 预测头：384 -> 64 -> 6。
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(32 * input_steps, 64)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(64, predict_steps)

    def forward(self, X):
        if X.ndim != 3 or X.shape[1] != self.input_steps or X.shape[2] != 2:
            raise ValueError(
                f"输入形状应为 [批大小, {self.input_steps}, 2]，"
                f"实际收到 {tuple(X.shape)}"
            )

        # 数据表按时间排列；Conv1d 要求通道在时间之前。
        X = X.transpose(1, 2)                 # [B, 12, 2] -> [B, 2, 12]
        X = F.relu(self.input_conv(X))         # [B, 32, 12]
        X = self.residual_blocks(X)            # [B, 32, 12]
        X = self.flatten(X)                    # [B, 384]
        X = F.relu(self.fc1(X))                # [B, 64]
        X = self.dropout(X)                    # [B, 64]
        return self.fc2(X)                     # [B, 6]


if __name__ == "__main__":#只有直接运行这个文件时才会执行以下行
    # 随机输入只检查计算能否正常执行，不表示训练或预测效果。
    model = TrafficResNet(input_steps=12, predict_steps=6)
    model.eval()
    example_X = torch.randn(4, 12, 2)
    with torch.no_grad():
        example_Y = model(example_X)

    print(model)
    print("输入形状：", tuple(example_X.shape))
    print("输出形状：", tuple(example_Y.shape))
    print("参数总数：", sum(p.numel() for p in model.parameters()))
    assert example_Y.shape == (4, 6)

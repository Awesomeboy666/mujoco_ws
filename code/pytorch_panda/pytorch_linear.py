"""使用 PyTorch 简洁实现线性回归。

程序按照“生成数据、读取数据、定义模型、训练模型”的顺序，
演示如何利用 PyTorch 已封装好的组件完成一个完整的训练过程。
"""


# ==================== 模块一：导入所需工具 ====================

import torch
import random  # Python 随机数模块；当前示例中暂未使用，保留供后续实验使用。
from d2l import torch as d2l  # 《动手学深度学习》提供的 PyTorch 辅助工具。
import numpy as np  # 数值计算工具；当前示例中暂未使用。
from torch.utils import data  # 提供 TensorDataset 和 DataLoader 等数据读取工具。
from torch import nn  # 提供神经网络层、损失函数等组件。


# ==================== 模块二：生成模拟训练数据 ====================

# 设置生成数据时使用的真实权重和偏置。
# 数据将近似满足 y = 2*x1 - 3.4*x2 + 4.2。
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# synthetic_data(w, b, n) 的作用：生成 n 个满足 y = Xw + b + 随机噪声的样本。
# features 保存输入特征，形状为 (1000, 2)；labels 保存标签，形状为 (1000, 1)。就是样本的输入输出
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# ==================== 模块三：构造小批量数据迭代器 ====================

def load_array(data_arrays, batch_size, is_train=True):
    """把张量数据封装成可以按小批量读取的数据迭代器。

    参数：
        data_arrays: 包含特征和标签的元组，例如 (features, labels)。
        batch_size: 每个小批量包含的样本数量。
        is_train: 是否处于训练模式；为 True 时会随机打乱样本顺序。

    返回：
        DataLoader 对象。遍历该对象时，每次会得到一批特征和对应标签。
    """
    # TensorDataset 将多个张量按第 0 维一一对应地组合起来。
    # 星号 * 会把 data_arrays 中的特征和标签分别传入 TensorDataset。
    dataset = data.TensorDataset(*data_arrays)

    # DataLoader 负责分批读取数据。
    # shuffle=True 表示每轮训练前打乱数据，有助于减少样本顺序造成的影响。
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 每次训练取出 10 个样本。
batch_size = 10

# 调用 load_array，将全部特征和标签转换成小批量数据迭代器。
data_iter = load_array((features, labels), batch_size)

# iter(data_iter) 创建迭代器，next(...) 取出第一个批次。
# 在交互环境中可以查看该批次；脚本中这行仅用于演示数据读取方式。
next(iter(data_iter))


# ==================== 模块四：定义线性回归模型 ====================

# nn.Linear(2, 1) 创建一个全连接层：接收 2 个输入特征，输出 1 个预测值。
# nn.Sequential 将网络层按顺序组织起来；本例只有一个线性层。
net = nn.Sequential(nn.Linear(2, 1))

# normal_(0, 0.01) 将权重初始化为均值为 0、标准差为 0.01 的正态分布随机数。
net[0].weight.data.normal_(0, 0.01)

# fill_(0) 将线性层的偏置全部初始化为 0。
net[0].bias.data.fill_(0)


# ==================== 模块五：定义损失函数和优化器 ====================

# MSELoss() 创建均方误差损失函数，用于衡量预测值与真实标签之间的差异。
loss = nn.MSELoss()

# SGD 创建随机梯度下降优化器。
# net.parameters() 返回模型中所有待学习参数，lr=0.03 表示学习率为 0.03。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# ==================== 模块六：训练并评估模型 ====================

# epoch 表示完整遍历一次训练数据；这里总共训练 3 轮。
num_epochs = 3
for epoch in range(num_epochs):
    # 每次从 data_iter 中取出一个小批量的特征 X 和对应标签 y。
    for X, y in data_iter:
        # net(X) 执行前向传播得到预测值；loss(...) 计算本批次均方误差。
        l = loss(net(X), y)

        # 清除上一次反向传播留下的梯度，防止梯度被自动累加。
        trainer.zero_grad()

        # 根据当前损失执行反向传播，自动计算各模型参数的梯度。
        l.backward()

        # 优化器根据参数梯度和学习率更新模型的权重与偏置。
        trainer.step()

    # 一轮训练结束后，使用全部数据计算当前模型的整体损失。
    l = loss(net(features), labels)

    # 输出当前训练轮次以及损失值；损失越小，通常说明拟合效果越好。
    print(f'epoch {epoch + 1}, loss {l:f}')


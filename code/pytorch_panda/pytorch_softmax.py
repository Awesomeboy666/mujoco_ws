"""使用 PyTorch 简洁实现 Softmax 回归。

本程序使用 Fashion-MNIST 服装图像数据集完成十分类任务，流程为：
读取数据 -> 定义模型 -> 初始化参数 -> 定义损失函数与优化器
-> 训练模型 -> 在测试集上评估模型。

Fashion-MNIST 中每张图片的尺寸是 28×28，共有 10 个类别。
Softmax 回归会为每张图片输出 10 个分数，最大分数的位置就是预测类别。
"""


# ==================== 模块一：导入所需工具 ====================

import torch
from torch import nn
from d2l import torch as d2l


# ==================== 模块二：读取 Fashion-MNIST 数据 ====================

# batch_size 表示每次训练取出 256 张图片。
batch_size = 256

# train_iter 是训练集迭代器，test_iter 是测试集迭代器。
# X 的形状通常为 (batch_size, 1, 28, 28)，y 的形状为 (batch_size,)；
# y 中的每个元素都是一个 0～9 的类别编号。
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# ==================== 模块三：定义 Softmax 回归模型 ====================

# Flatten 将每张 28×28 图片展平成长度为 784 的向量。
# Linear 接收 784 个像素值并输出 10 个类别分数（logits）。
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10),
)


def init_weights(module):
    """初始化网络中线性层的权重。

    参数：
        module: net.apply() 遍历网络时传入的某一个网络层。

    作用：
        若当前层是 nn.Linear，就将权重初始化为均值 0、标准差 0.01
        的正态分布随机数。函数直接修改网络层，不需要返回值。
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)


# apply 会访问 net 中的每一层，并对每层调用 init_weights。
net.apply(init_weights)


# ==================== 模块四：定义损失函数和优化器 ====================

# CrossEntropyLoss 用于多分类任务，内部已经组合了 Softmax 和交叉熵。
# 因此模型末尾不能再添加 nn.Softmax，应直接传入原始类别分数 logits。
loss = nn.CrossEntropyLoss()

# SGD 使用随机梯度下降更新 net 的权重和偏置，学习率为 0.1。
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


# ==================== 模块五：定义统计和评估工具 ====================

class Accumulator:
    """在多个小批量之间累加若干个统计量。

    例如 Accumulator(3) 创建三个初值为 0 的位置，可以分别累计
    损失总和、预测正确的样本数和样本总数。
    """

    def __init__(self, n):
        """创建 n 个浮点数累加位置，并全部初始化为 0。"""
        self.data = [0.0] * n

    def add(self, *args):
        """把传入的各个数值分别加到对应的累加位置中。"""
        self.data = [
            old_value + float(new_value)
            for old_value, new_value in zip(self.data, args)
        ]

    def reset(self):
        """把全部统计量重新清零。"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, index):
        """支持使用 metric[index] 读取指定位置的统计量。"""
        return self.data[index]


def accuracy(y_hat, y):
    """计算一个批次中预测正确的样本数量。

    参数：
        y_hat: 模型输出，形状为 (批量大小, 10)。
        y: 真实类别，形状为 (批量大小,)。

    返回：
        当前批次预测正确的样本数，而不是正确率。
    """
    # 在每个样本的 10 个类别分数中寻找最大值所在的位置。
    predicted_classes = y_hat.argmax(dim=1)

    # 逐个比较预测类别与真实类别，得到由 True/False 组成的张量。
    correct_predictions = predicted_classes.to(y.dtype) == y

    # True 求和时作为 1，因此结果就是预测正确的样本数量。
    return float(correct_predictions.sum())


def evaluate_accuracy(net, data_iter):
    """计算模型在指定数据集上的分类准确率。

    参数：
        net: 要评估的神经网络模型。
        data_iter: 测试集或其他数据集的小批量迭代器。

    返回：
        正确预测数除以总样本数得到的准确率，范围为 0～1。
    """
    # 将模型切换到评估模式，为以后使用 Dropout 等层做好准备。
    net.eval()

    # 两个位置分别累计“预测正确数”和“样本总数”。
    metric = Accumulator(2)

    # 评估不需要计算梯度，这样可以节省内存并提高速度。
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), y.numel())

    return metric[0] / metric[1]


# ==================== 模块六：定义单轮训练函数 ====================

def train_epoch(net, train_iter, loss_function, optimizer):
    """使用全部训练数据训练模型一轮。

    参数：
        net: 要训练的神经网络模型。
        train_iter: 训练数据的小批量迭代器。
        loss_function: 计算分类损失的函数。
        optimizer: 根据梯度更新模型参数的优化器。

    返回：
        (本轮平均训练损失, 本轮训练准确率)。
    """
    # 将模型切换到训练模式。
    net.train()

    # 三个位置依次累计：损失总和、预测正确数、样本总数。
    metric = Accumulator(3)

    for X, y in train_iter:
        # 前向传播：根据图片 X 计算每个类别的分数。
        y_hat = net(X)

        # 计算当前批次的平均交叉熵损失。
        batch_loss = loss_function(y_hat, y)

        # PyTorch 默认累加梯度，所以反向传播前必须清除旧梯度。
        optimizer.zero_grad()

        # 反向传播，自动计算损失对每个模型参数的梯度。
        batch_loss.backward()

        # 根据梯度和学习率更新模型参数。
        optimizer.step()

        # batch_loss 是批次平均损失，乘以样本数后得到批次损失总和。
        # detach().item() 将统计值从计算图分离并转成 Python 数值。
        metric.add(
            batch_loss.detach().item() * y.numel(),
            accuracy(y_hat, y),
            y.numel(),
        )

    # 损失总和/样本数为平均损失；正确数/样本数为训练准确率。
    return metric[0] / metric[2], metric[1] / metric[2]


# ==================== 模块七：执行多轮训练并输出结果 ====================

def train(net, train_iter, test_iter, loss_function, optimizer, num_epochs):
    """训练模型多轮，并在每轮结束后输出训练和测试指标。

    参数：
        net: Softmax 回归模型。
        train_iter: 训练集迭代器。
        test_iter: 测试集迭代器。
        loss_function: 交叉熵损失函数。
        optimizer: 随机梯度下降优化器。
        num_epochs: 完整遍历训练集的次数。

    返回：
        无返回值；每一轮的结果会直接打印到终端。
    """
    for epoch in range(num_epochs):
        # 完成一轮训练，获得平均损失和训练准确率。
        train_loss, train_accuracy = train_epoch(
            net, train_iter, loss_function, optimizer
        )

        # 在测试集上检查模型对未参与训练数据的分类能力。
        test_accuracy = evaluate_accuracy(net, test_iter)

        # 准确率乘以 100 后以百分数显示。
        print(
            f"epoch {epoch + 1:2d}, "
            f"loss {train_loss:.4f}, "
            f"train acc {train_accuracy * 100:.2f}%, "
            f"test acc {test_accuracy * 100:.2f}%"
        )


# ==================== 模块八：程序入口 ====================

if __name__ == "__main__":
    # 训练 10 轮。程序入口可避免该文件被导入时自动开始训练。
    num_epochs = 10
    train(net, train_iter, test_iter, loss, trainer, num_epochs)

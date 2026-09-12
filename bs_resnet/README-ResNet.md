# 车流量 ResNet 学习版

直接运行 `modeltrain/myt12model_train.py`。脚本使用 CPU，输入过去12步的 `Flow` 和 `Avg kph`，预测未来6步流量，每步10分钟。网络定义在 `modeltrain/resnet_model.py`。

## 运行

从 Windows 复制来的 `.venv-resnet` 不能在 Linux 使用。选择安装了 numpy、pandas、matplotlib、torch 的 Linux Python 环境。本机已有的 `mujoco` 环境可以运行此学习版：

```bash
cd /home/xu/mujoco_ws/bs_resnet
/home/xu/miniconda3/envs/mujoco/bin/python modeltrain/myt12model_train.py
```

在 PyCharm 中也可以选择上述解释器，直接运行脚本。

## 阅读顺序

脚本按顺序完成：参数设置 → 读取数据 → 归一化 → 构造样本 → 训练和验证 → 测试及未来预测 → 保存结果和画图。

修改开头的 `EPOCHS`、`BATCH_SIZE`、`LR` 即可调整训练，默认100轮、批大小64、学习率0.001。不使用命令行参数。`PATIENCE = 15`：验证损失连续15轮未低于历史最佳值时提前停止，仍使用停止时最后一轮的模型进行测试和保存。

`USE_MY_MODEL = False` 时训练、评估并保存模型；改为 `True` 时加载 `outputs/resnet_simple/resnet.pt` 中的权重、归一化参数和步数，直接评估及预测。加载模式保留原模型和 `loss.csv`，更新指标、预测结果和图片，图片中隐藏损失子图。

默认读取 `Traffic_Data/Dongsanhuan.csv`，仅在末尾一行的流量和速度都为空时去掉该行。学习版直接使用其已整理好的连续10分钟数据，不包含通用缺失值处理。

按时间划分70%训练、10%验证、20%测试。归一化只使用训练段的最小值和最大值；每个样本的6步目标都属于同一个集合。验证和测试可以使用之前的历史输入。验证损失用于观察学习过程，测试使用最后一轮模型。

## 输出

结果保存到 `outputs/resnet_simple/`：

- `resnet.pt`：最后一轮权重、归一化参数和输入输出步数。
- `loss.csv`：每轮训练和验证损失。
- `metrics.csv`：第1步（10分钟后）和第6步（60分钟后）分别计算的 MAE、RMSE，每步在所有测试窗口上取平均。
- `test_predictions.csv`：每个测试窗口的6步真实流量和预测流量。
- `future_predictions.csv`：数据末尾之后的6步预测及时间。
- `training_and_predictions.png`：损失曲线和第1、6步预测对比。

流量预测及指标均已还原到原始尺度。旧版结果仍在 `outputs/resnet/`，与学习版的权重格式不同。

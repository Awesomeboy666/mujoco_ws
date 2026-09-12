"""CPU 训练 ResNet：过去12步的流量、速度，预测未来6步流量。"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 虚拟机中直接保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from resnet_model import TrafficResNet

# 1. 参数：学习时直接在这里修改
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Traffic_Data/Dongsanhuan.csv'
OUTPUT = ROOT / 'outputs/resnet_simple'
INPUT_STEPS = 12
PREDICT_STEPS = 6
EPOCHS = 100
PATIENCE = 15  # 验证损失连续15轮不改善就停止
BATCH_SIZE = 64
LR = 0.001
USE_MY_MODEL = False  # False：训练并保存；True：加载已有模型进行评估和预测

torch.manual_seed(42)
torch.set_num_threads(4)
OUTPUT.mkdir(parents=True, exist_ok=True)

# 2. 读取已整理好的连续10分钟数据，按时间划分70%训练、10%验证、20%测试
frame = pd.read_csv(DATA, parse_dates=['datetime']).sort_values('datetime')
# 仅在末尾流量和速度都为空时去掉最后一行
if frame[['Flow', 'Avg kph']].iloc[-1].isna().all():
    frame = frame.iloc[:-1]
values = frame[['Flow', 'Avg kph']].to_numpy(dtype=np.float32)
train_end = int(len(values) * 0.7)
val_end = int(len(values) * 0.8)

# 加载模型时沿用训练时的归一化参数和步数
if USE_MY_MODEL:
    saved = torch.load(OUTPUT / 'resnet.pt', map_location='cpu', weights_only=True)
    minimum = np.array(saved['minimum'], dtype=np.float32)
    span = np.array(saved['span'], dtype=np.float32)
    INPUT_STEPS = saved['input_steps']
    PREDICT_STEPS = saved['predict_steps']
else:
    minimum = values[:train_end].min(axis=0)
    span = values[:train_end].max(axis=0) - minimum
    span = np.where(span == 0, 1.0, span)

data = (values - minimum) / span


def make_samples(start, end):
    """i是预测的第一步：X取前12行，y取随后6行的流量。"""
    x, y = [], []
    for i in range(start, end - PREDICT_STEPS + 1):
        x.append(data[i - INPUT_STEPS:i])
        y.append(data[i:i + PREDICT_STEPS, 0])
    return torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))


# 每个样本的6步目标都在所属集合内；验证和测试可使用之前的历史输入
train_x, train_y = make_samples(INPUT_STEPS, train_end)
val_x, val_y = make_samples(train_end, val_end)
test_x, test_y = make_samples(val_end, len(data))
train_loader = DataLoader(TensorDataset(train_x, train_y),    #64组样本为一个batch
                          batch_size=BATCH_SIZE, shuffle=True)
print('训练、验证、测试样本数：', len(train_x), len(val_x), len(test_x))

# 3. 创建网络、损失函数和优化器，默认全部在CPU上计算
model = TrafficResNet(INPUT_STEPS, PREDICT_STEPS)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
#model.parameters() 是 PyTorch 通用的参数获取方式；训练整个网络时，把它传给优化器是标准写法。
history = []
if not USE_MY_MODEL:
    # 4. 训练：前向计算 → 清空梯度 → 反向传播 → 更新参数
    best_loss = float('inf')
    wait = 0
    for epoch in range(EPOCHS):
        model.train()       #设置模式
        total_loss = 0.0    #累计损失清零
        for x, y in train_loader:    #综合一个batch进行一次梯度更新
            prediction = model(x)
            loss = criterion(prediction, y)
            optimizer.zero_grad()   #清空上一次计算留下的梯度
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)

        train_loss = total_loss / len(train_x)
        model.eval()  # 关闭Dropout，验证集只计算损失，不更新参数
        with torch.no_grad():    #不记录用于反向传播的计算过程，减少内存和计算开销
            val_loss = criterion(model(val_x), val_y).item()    #这里没有 backward() 和 optimizer.step()，所以验证集不用于更新参数。
        history.append([train_loss, val_loss])
        print(f'Epoch {epoch + 1:03d}: train={train_loss:.6f}, val={val_loss:.6f}')
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            print(f'验证损失连续{PATIENCE}轮未改善，停止训练。')
            break
else:
    model.load_state_dict(saved['model_state'])


# 5. 测试与未来预测：使用最后一轮模型，并还原成实际流量
model.eval()
with torch.no_grad():
    prediction = model(test_x).numpy() * span[0] + minimum[0]  #model(text_x)使用的是最后一轮训练后的模型参数。,然后进行反归一化
    last_x = torch.from_numpy(data[-INPUT_STEPS:]).unsqueeze(0)
    future = model(last_x).numpy().ravel() * span[0] + minimum[0]
actual = test_y.numpy() * span[0] + minimum[0]
metrics = []
for step in [0, 5]:  # 第1步（10分钟后）、第6步（60分钟后）
    error = prediction[:, step] - actual[:, step]
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    metrics.append({'step': step + 1, 'MAE': mae, 'RMSE': rmse})
    print(f'测试集第{step + 1}步：MAE={mae:.3f}, RMSE={rmse:.3f}')

# 6. 保存权重、归一化参数、损失和预测结果
# minimum和span用于以后把新输入归一化、把预测值还原
if not USE_MY_MODEL:
    torch.save({'model_state': model.state_dict(),
                'minimum': minimum.tolist(), 'span': span.tolist(),
                'input_steps': INPUT_STEPS, 'predict_steps': PREDICT_STEPS},
               OUTPUT / 'resnet.pt')
    pd.DataFrame(history, columns=['train_mse', 'val_mse']).to_csv(
        OUTPUT / 'loss.csv', index=False)
pd.DataFrame(metrics).to_csv(
    OUTPUT / 'metrics.csv', index=False)
results = {}
for step in range(PREDICT_STEPS):
    results[f'actual_t+{step + 1}'] = actual[:, step]
    results[f'predicted_t+{step + 1}'] = prediction[:, step]
pd.DataFrame(results).to_csv(OUTPUT / 'test_predictions.csv', index=False)
pd.DataFrame({
    'datetime': pd.date_range(frame['datetime'].iloc[-1] + pd.Timedelta(minutes=10),
                              periods=PREDICT_STEPS, freq='10min'),
    'predicted_flow': future,
}).to_csv(OUTPUT / 'future_predictions.csv', index=False)

# 7. 画训练/验证损失，以及第1步和第6步的预测曲线
fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
if history:
    axes[0].plot(np.arange(1, len(history) + 1), history)
    axes[0].legend(['Train', 'Validation'])
    axes[0].set(xlabel='Epoch', ylabel='MSE (normalized)')
else:
    axes[0].set_visible(False)
for ax, step in zip(axes[1:], [0, PREDICT_STEPS - 1]):
    ax.plot(actual[:200, step], label='Actual')
    ax.plot(prediction[:200, step], label='ResNet')
    ax.set(title=f't+{step + 1}', xlabel='Test window', ylabel='Flow')
    ax.legend()
fig.savefig(OUTPUT / 'training_and_predictions.png', dpi=150)
plt.close(fig)
print('结果目录：', OUTPUT)

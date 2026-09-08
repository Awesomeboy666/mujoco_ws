# 车流量 ResNet：使用说明

## 在 PyCharm 中运行

本次已在项目中建立 `.venv-resnet` 环境，复用本机 Anaconda 的数据分析包并单独安装 CPU 版 PyTorch。
在 PyCharm 的项目解释器设置中选择：

`C:\Users\17323\Desktop\bs\.venv-resnet\Scripts\python.exe`

然后打开 `modeltrain/myt12model_train.py`，右键运行。此文件名沿用原项目，但现在是 **12步输入、6步输出的 ResNet 训练入口**。
若换电脑，可创建自己的虚拟环境并安装 `requirements-resnet.txt`。

## 输入和网络

默认数据为 `Traffic_Data/Dongsanhuan.csv`。输入列顺序固定为 `Flow, Avg kph`，速度单位 km/h。
每步10分钟，过去12步预测未来6步；Flow按时刻流量统计值理解，不标注成10分钟累计车辆数。
网络为：Conv1d(2→32) + ReLU → 3个残差块 → 展平384 → 全连接64 + ReLU + Dropout(0.1) → 全连接6。
残差块为 Conv1d → ReLU → Conv1d → 加回输入，无批归一化、无下采样。

## 数据处理

- 按时间排序并补齐时间索引，裁掉首尾完全空的记录；内部缺失不会被删除后拼接。
- 前70%训练，中间10%验证，后20%测试；归一化参数只由训练段拟合。
- 输入缺失只允许前向填充最多2步，禁止使用未来值。目标不填充，缺失目标窗口跳过。
- 跨集合的6步目标窗口丢弃。验证和测试可以使用预测时已知的历史，属于滚动预测评估。
- 当前CSV的历史加工来源没有确认，无法识别其中已经插值的值。当前测试指标仅针对这个CSV，不代表原始实测数据上的已验证精度。

## 训练和结果

默认最多100轮，批大小64，Adam学习率0.001，权重衰减0.0001，验证损失15轮不改善时早停。
只根据验证集选取最佳模型，训练结束后评估测试集。随机种子42；不同设备不保证完全一致。
结果保存到 `outputs/resnet/`：

- `resnet_best.pt`：最佳权重、归一化参数、特征顺序及配置。
- `loss.csv`：每轮训练/验证损失（归一化尺度）。
- `metrics.csv`：逐步及整体 MAE、RMSE、R2、MAPE，包含最后值持续预测的对照。
- `test_predictions.csv`：预测起点、目标时间、真实值、预测值及基线。
- `future_predictions.csv`：CSV末尾之后的6步预测，无真实未来标签。
- `training_and_predictions.png`：损失以及第1/6步预测曲线。
- `run_info.json`：数据路径、划分边界、训练配置与最佳轮次。

指标和导出流量都已经反归一化；不把预测取整或截断。MAPE仅排除真实零值，其他指标保留零值。
重复运行同一输出目录会更新结果，要保留不同实验请使用 `--output` 指定另一个目录。

在终端中也可以运行：

```powershell
.\.venv-resnet\Scripts\python.exe modeltrain/myt12model_train.py
```

仅加载已训练模型预测（不重新训练）：

```powershell
.\.venv-resnet\Scripts\python.exe modeltrain/myt12model_train.py --predict-only
```

## 原始数据清洗

`dataClean/mydataClean.py` 保留原脚本功能，固定km/h，同一10分钟格内取均值。
时间标签取右边界，避免把稍后的观测标记成更早已经可用；不按速度范围猜单位，不任意裁剪、取整或插值。
默认另存 `Traffic_Data/Dongsanhuan_cleaned.csv`，不会覆盖当前指定数据集。
原始记录稀疏，保留缺口后可能没有足够的连续6步真实目标，训练脚本会明确报错，而不会编造标签。
本次训练默认使用现有 `Dongsanhuan.csv`，不需要先运行清洗脚本。

## 修改前版本

三个修改前文件保存在 `work/before_resnet_update/`，用于与本次修改比较或恢复。
其他旧 LSTM/GRU/Attention 训练脚本未迁移，仍属于原实验，不是这次 ResNet 入口。
本次未初始化Git或创建提交；`.gitignore`已准备好，用于后续建立本地版本记录。

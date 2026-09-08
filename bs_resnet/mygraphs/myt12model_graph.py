import datetime
import numpy as np
import pandas as pd
import math
from matplotlib.dates import HourLocator, MinuteLocator
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
此代码功能为选择LSTM/GRU模型，绘制t+1/t+6/t+12的3天真实预测对比图 
'''


model_choice = "GRU"  #"LSTM" 或 "GRU"
# ============================================================================

# 数据归一化
def normalize_data(data, flow_scaler, avg_kph_scaler):
    data["Flow"] = flow_scaler.transform(data["Flow"].values.reshape(-1, 1))
    data["Avg kph"] = avg_kph_scaler.transform(data["Avg kph"].values.reshape(-1, 1))
    return data

# 反向归一化
def denormalize_data(data, scaler):
    if len(data.shape) == 1:
        data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(1, -1)[0]
    elif len(data.shape) == 2:
        data = scaler.inverse_transform(data).reshape(data.shape[0], data.shape[1])
    return data

# 添加时间滞后值适配t+1~t+12多步预测
def input_features(data):
    lag = 24  # 4小时历史（24*10分钟）
    train, test = [], []
    split_idx = int(len(data) * 0.8)
    predict_steps = 12  # 预测t+1~t+12共12步

    # 预留12步未来数据，避免索引越界
    for i in range(lag, len(data) - predict_steps):
        if i < split_idx:
            train.append(data.iloc[i - lag: i + predict_steps])
        else:
            test.append(data.iloc[i - lag: i + predict_steps])

    train = np.array(train)
    test = np.array(test)

    # 输入：lag步历史（Flow+Avg kph）；输出：12步未来Flow
    x_train = train[:, :-predict_steps, [0, 1]]
    y_train = train[:, -predict_steps:, 0]
    x_test = test[:, :-predict_steps, [0, 1]]
    y_test = test[:, -predict_steps:, 0]
    return x_train, y_train, x_test, y_test

# 绘制指定步长的3天对比图（仅绘制选中的模型）
def plot_step_3day(y_true, y_pred, test_dates, step_name, step_idx):

    # 提取指定步长数据
    y_true_step = y_true[:, step_idx].flatten()
    y_pred_step = y_pred[:, step_idx].flatten()

    # 取前3天数据（432个点 = 3天×24小时×6个10分钟）
    plot_points = 432
    x = test_dates[:plot_points]
    y_true_plot = y_true_step[:plot_points]
    y_pred_plot = y_pred_step[:plot_points]

    # 绘图配置
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y_true_plot, label='True Data', linewidth=2, color='#1f77b4')
    ax.plot(x, y_pred_plot, label=f'{model_choice} Prediction', linewidth=2, color='#ff7f0e')
    
    # 样式美化
    plt.legend(fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=17, rotation=45)
    plt.yticks(fontsize=17)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Flow', fontsize=18)
    plt.title(f'{model_choice} {step_name} Prediction (3 Days)', fontsize=20)
    
    # 时间格式化
    date_format = mpl.dates.DateFormatter("%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.ylim(0)
    
    # 保存图片
    save_name = f'{model_choice}_{step_name.replace("（","_").replace("）","").replace("：","_")}_3Day.png'
    fig.tight_layout()
    fig.savefig(save_name, dpi=300)
    plt.show()

if __name__ == '__main__':
    # 1. 数据读取与预处理
    data_path = "../Traffic_Data/Dongsanhuan.csv"
    df = pd.read_csv(data_path, parse_dates=["datetime"], index_col="datetime")

    # 归一化
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))
    df = normalize_data(df, flow_scaler, avg_kph_scaler)
    
    # 2. 构建多步预测数据集（t+1~t+12）
    x_train, y_train, x_test, y_test = input_features(df)
    
    # 3. 获取测试集日期索引（用于绘图x轴）
    split_idx = int(len(df) * 0.8)
    lag = 24
    predict_steps = 12
    test_dates_start = split_idx + lag
    test_dates = df.index[test_dates_start : test_dates_start + len(x_test)]

    # 4. 加载选中的模型（仅加载LSTM或GRU其中一个）
    model_path = f'../../webapp/model/{model_choice}_10min_t12.h5'
    model = load_model(model_path)
    print(f"成功加载 {model_choice} 模型：{model_path}")

    # 5. 模型预测 + 反归一化
    x_test_reshaped = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
    y_pred = model.predict(x_test_reshaped, verbose=0)
    # 反归一化真实值和预测值
    y_test_denorm = denormalize_data(y_test, flow_scaler)
    y_pred_denorm = denormalize_data(y_pred, flow_scaler)

    # 6. 绘制t+1/t+6/t+12的3天对比图（逐个绘制）
    step_configs = [
        ("t+1(10min)", 0),
        ("t+6(60min)", 5),
        ("t+12(120min)", 11)
    ]
    
    for step_name, step_idx in step_configs:
        print(f"\n正在绘制 {model_choice} - {step_name} 的3天预测对比图...")
        plot_step_3day(y_test_denorm, y_pred_denorm, test_dates, step_name, step_idx)

    print(f"\n{model_choice} 模型所有步长绘图完成！")
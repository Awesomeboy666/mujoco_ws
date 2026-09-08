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
此代码绘制在静态时间步的数据预测效果 
'''

# 数据归一化
def normalize_data(data, flow_scaler, avg_kph_scaler):
    data["Flow"] = flow_scaler.transform(data["Flow"].values.reshape(-1, 1))
    data["Avg kph"] = avg_kph_scaler.transform(data["Avg kph"].values.reshape(-1, 1))
    return data

# 反向归一化
def denormalize_data(data, scaler):
    data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(1, -1)[0]
    return data

# 添加时间滞后值以构建模型输入特征
def input_features(data):
    
    lag = 24  # 过去4小时 (24 * 10分钟)
    train, test = [], []
    
    # -------------------------- 动态划分数据集 --------------------------
    split_idx = int(len(data) * 0.8) 
    
    for i in range(lag, len(data)):
        if i < split_idx:
            train.append(data[i - lag: i + 1])
        else:
            test.append(data[i - lag: i + 1])

    train = np.array(train)
    test = np.array(test)

    # 设置模型的输入输出
    x_train = train[:, :-1, [0, 1]]  # select Flow and Avg kph as input features
    y_train = train[:, -1, [0]]  # select Flow as output
    x_test = test[:, :-1, [0, 1]]  # select Flow and Avg kph as input features
    y_test = test[:, -1, [0]]  # select Flow as output
    return x_train, y_train, x_test, y_test

#绘图
# 静态时间步
def plot_LSTM_threeDay(y_true, y_pred, test_dates):
    # 直接使用测试集的真实日期索引
    x = test_dates[:432]  # 432个点 = 3天 * 24小时 * 6个(10分钟/个)
    
    # 创建一个大小为 20x10 的画布
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    # 在子图中绘制真实值的折线
    ax.plot(x, y_true[:432], label='True Data', linewidth=2)
    
    # 绘制预测值
    for name, y_pred_item in zip(['LSTM'], y_preds):
        ax.plot(x, y_pred_item[:432], label=name, linestyle='-', linewidth=2)
    
    plt.legend(fontsize=17)
    plt.grid(True, alpha=0.3)

    plt.xticks(fontsize=17, rotation=45)
    plt.yticks(fontsize=17)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Flow', fontsize=18)
    
    # 格式化时间显示
    date_format = mpl.dates.DateFormatter("%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    plt.ylim(0)
    plt.title('AM+LSTM Static Prediction (3 Days)', fontsize=20)
    fig.tight_layout()
    fig.savefig('AM+LSTM_static_3Day.png', dpi=300)
    plt.show()

def plot_GRU_threeDay(y_true, y_pred, test_dates):
    # -------------------------- GRU绘图 --------------------------
    x = test_dates[:432]
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y_true[:432], label='True Data', linewidth=2)
    
    for name, y_pred_item in zip(['GRU'], y_preds):
        ax.plot(x, y_pred_item[:432], label=name, linestyle='-', linewidth=2)
    
    plt.legend(fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=17, rotation=45)
    plt.yticks(fontsize=17)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Flow', fontsize=18)
    
    date_format = mpl.dates.DateFormatter("%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    plt.ylim(0)
    plt.title('GRU Static Prediction (3 Days)', fontsize=20)
    fig.tight_layout()
    fig.savefig('GRU_static_3Day.png', dpi=300)
    plt.show()


if __name__ == '__main__':
   
    # 读入数据
    data_path = "../Traffic_Data/Dongsanhuan.csv"
    df = pd.read_csv(data_path, 
                     parse_dates=["datetime"],
                     index_col="datetime")

    # 分别对两个特征Flow、Avg kph使用不同的归一化器进行归一化
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))

    #使用定义好的归一化器归一化数据
    df = normalize_data(df, flow_scaler, avg_kph_scaler)
    
    # 添加时间滞后值以构建模型输入特征
    x_train, y_train, x_test, y_test = input_features(df)
    
    # 获取测试集对应的真实日期索引 (用于绘图)
    split_idx = int(len(df) * 0.8)
    lag = 24
    test_dates = df.index[split_idx + lag:]

    
    # 装载模型
    lstm = load_model('../../webapp/model/LSTM_10min.h5')
    gru = load_model('../../webapp/model/GRU_10min.h5')
    
    models = [lstm]  # 如果要同时画GRU，改成 [lstm, gru]
    names = ['LSTM'] # 改成 ['LSTM', 'GRU']
    
    y_test = denormalize_data(y_test, flow_scaler)
    y_preds = []
    for name, model in zip(names, models):
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
        predicted = model.predict(x_test)
        predicted = denormalize_data(predicted, flow_scaler)
        y_preds.append(predicted)
        print(name)

    # 绘图
    print("正在绘制静态3天预测图...")
    plot_LSTM_threeDay(y_test, y_preds, test_dates)
    
    #画GRU
    #plot_GRU_threeDay(y_test, y_preds, test_dates)
    

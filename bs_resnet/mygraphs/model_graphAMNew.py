import datetime
import numpy as np
import pandas as pd
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import Layer
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
加载 AM+LSTM(注意力机制) 模型，绘制5天静态时间步预测图
'''
# ===================== 核心：添加注意力机制层 =====================
class TimeSeriesAttention(Layer):
    def __init__(self, **kwargs):
        super(TimeSeriesAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="normal", trainable=True)
        self.V = self.add_weight(name="att_var", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        super(TimeSeriesAttention, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W))
        scores = tf.matmul(e, self.V)
        weights = tf.nn.softmax(scores, axis=1)
        output = x * weights
        return tf.reduce_sum(output, axis=1)

# 数据归一化
def normalize_data(data, flow_scaler, avg_kph_scaler):
    data["Flow"] = flow_scaler.transform(data["Flow"].values.reshape(-1, 1))
    data["Avg kph"] = avg_kph_scaler.transform(data["Avg kph"].values.reshape(-1, 1))
    return data

# 反向归一化
def denormalize_data(data, scaler):
    data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(1, -1)[0]
    return data

# 添加时间滞后值以构建模型输入特征（完全保留你的原始逻辑）
def input_features(data):
    lag = 48
    train, test = [], []
    for i in range(lag, len(data)):
        if i < 14109:
            train.append(data[i - lag: i + 1])
        else:
            test.append(data[i - lag: i + 1])

    train = np.array(train)
    test = np.array(test)

    x_train = train[:, :-1, [0, 1]]
    y_train = train[:, -1, [0]]
    x_test = test[:, :-1, [0, 1]]
    y_test = test[:, -1, [0]]
    return x_train, y_train, x_test, y_test

# 5天静态时间步绘图
def plot_AM_LSTM_5Day(y_true, y_pred):

    d = '2019-04-17 00:00'
    # 5天 = 1440个点（5分钟/个）
    x = pd.date_range(d, periods=1440, freq='5min')
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    ax.plot(x, y_true[:1440], label='True Data', linewidth=2)
    ax.plot(x, y_pred[:1440], label='AM-LSTM Prediction', linewidth=2, linestyle='-')
    
    plt.legend(fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Flow', fontsize=18)
    
    #月-日 时:分
    date_format = mpl.dates.DateFormatter("%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    
    plt.xlim(pd.Timestamp(d), pd.Timestamp('2019-04-22 00:00'))
    plt.ylim(0)
    
    fig.savefig('AM-LSTM_5Day_Static.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv("../Traffic_Data/Engldata.csv", parse_dates=["datetime (Veh/5 Minutes)"],
                     index_col="datetime (Veh/5 Minutes)")

    # 归一化
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))
    df = normalize_data(df, flow_scaler, avg_kph_scaler)
    
    # 构建数据集
    x_train, y_train, x_test, y_test = input_features(df)

    # ===================== 加载AM-LSTM模型（带自定义注意力层） =====================
    model = load_model(
        '../../webapp/model/AMLSTM.h5',
        custom_objects={'TimeSeriesAttention': TimeSeriesAttention}
    )
    print("✅ AM-LSTM 模型加载成功！")

    # 预测 + 反归一化
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
    y_pred = model.predict(x_test)
    y_pred = denormalize_data(y_pred, flow_scaler)
    y_test = denormalize_data(y_test, flow_scaler)

    # 绘制5天预测图
    plot_AM_LSTM_5Day(y_test, y_pred)
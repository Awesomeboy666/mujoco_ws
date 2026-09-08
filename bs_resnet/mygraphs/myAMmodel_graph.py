import datetime
import numpy as np
import pandas as pd
import math
import tensorflow as tf  #
from keras.layers import Layer  # 
from matplotlib.dates import HourLocator, MinuteLocator
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===================== 自定义时序注意力层 ====================
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

# 反归一化函数，和第一个文件保持一致
def denormalize_data(data, scaler):
    if len(data.shape) == 1:
        data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(1, -1)[0]
    elif len(data.shape) == 2:
        data = scaler.inverse_transform(data).reshape(data.shape[0], data.shape[1])
    return data

# 数据归一化
def normalize_data(data, flow_scaler, avg_kph_scaler):
    data["Flow"] = flow_scaler.transform(data["Flow"].values.reshape(-1, 1))
    data["Avg kph"] = avg_kph_scaler.transform(data["Avg kph"].values.reshape(-1, 1))
    return data

# 添加时间滞后值以构建模型输入特征
def input_features(data):
    lag = 24  # 过去4小时 
    train, test = [], []
    split_idx = int(len(data) * 0.8) 
    
    for i in range(lag, len(data)):
        if i < split_idx:
            train.append(data[i - lag: i + 1])
        else:
            test.append(data[i - lag: i + 1])

    train = np.array(train)
    test = np.array(test)

    # 设置模型的输入输出
    x_train = train[:, :-1, [0, 1]]  # Flow + Avg kph 作为输入特征
    y_train = train[:, -1, [0]]      # Flow 作为输出
    x_test = test[:, :-1, [0, 1]]    # Flow + Avg kph 作为输入特征
    y_test = test[:, -1, [0]]        # Flow 作为输出
    return x_train, y_train, x_test, y_test

# ===================== 绘图函数修正 =====================
# 静态时间步 - LSTM+AM模型绘图
def plot_LSTM_AM_threeDay(y_true, y_pred, test_dates):
    # 3天数据：3天 * 24小时 * 6个10分钟 = 432个点
    x = test_dates[:432]  
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    # 绘制真实值
    ax.plot(x, y_true[:432], label='True Data', linewidth=2)
    
    # 【修正】使用传入的y_pred参数，而非全局变量
    for name, y_pred_item in zip(['LSTM+AM'], y_pred):
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

# 静态时间步 - GRU模型绘图
def plot_GRU_threeDay(y_true, y_pred, test_dates):
    x = test_dates[:432]
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y_true[:432], label='True Data', linewidth=2)
    
    # 使用传入的y_pred参数
    for name, y_pred_item in zip(['GRU'], y_pred):
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
    # 1. 数据加载
    data_path = "../Traffic_Data/Dongsanhuan.csv"
    df = pd.read_csv(data_path, 
                     parse_dates=["datetime"],
                     index_col="datetime")

    # 2. 数据归一化
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))
    df = normalize_data(df, flow_scaler, avg_kph_scaler)
    
    # 3. 构建输入特征
    x_train, y_train, x_test, y_test = input_features(df)
    
    # 4. 获取测试集日期索引
    split_idx = int(len(df) * 0.8)
    lag = 24
    test_dates = df.index[split_idx + lag:]

    # 5.加载训练好的模型（适配LSTM+AM）
    # 加载LSTM+AM模型
    lstm_am_model = load_model('../../webapp/model/LSTM_AM_10min.h5', 
                               custom_objects={'TimeSeriesAttention': TimeSeriesAttention})
    # 加载GRU模型
    gru_model = load_model('../../webapp/model/GRU_10min.h5')
    
    # 可选：同时加载两个模型或单独加载
    models = [lstm_am_model]  # LSTM+AM + GRU
    names = ['LSTM+AM']           
    
    # 6. 反归一化真实值
    y_test = denormalize_data(y_test, flow_scaler)
    
    # 7. 模型预测
    y_preds = []
    for name, model in zip(names, models):
        predicted = model.predict(x_test, verbose=0)
        predicted = denormalize_data(predicted, flow_scaler)  # 反归一化预测值
        y_preds.append(predicted)
        print(f"{name} 预测完成")

    # 8. 绘制静态3天预测图
    print("正在绘制LSTM+AM静态3天预测图...")
    plot_LSTM_AM_threeDay(y_test, [y_preds[0]], test_dates)  # LSTM+AM预测结果
    
    #print("正在绘制GRU静态3天预测图...")
    #plot_GRU_threeDay(y_test, [y_preds[1]], test_dates)      # GRU预测结果
    
   
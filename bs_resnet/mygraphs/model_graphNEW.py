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
    lag = 48 #1h->4;24h->96 控制时间步
    train, test = [], []
    for i in range(lag, len(data)):
        if i < 14109:  # 0.8->14105-> 5.19 23:30
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

# 绘图
def plot_LSTM_fiveDay(y_true, y_preds, test_dates):
    # 起始时间 2019-04-17 00:00
    d = '2019-04-17 00:00'

    x = pd.date_range(d, periods=1440, freq='5min')
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    
    ax.plot(x, y_true[:1440], label='True Data', linewidth=2)
    for name, y_pred_item in zip(['LSTM Prediction'], y_preds):
        ax.plot(x, y_pred_item[:1440], label=name, linestyle='-', linewidth=2)

    plt.legend(fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=17, rotation=45)
    plt.yticks(fontsize=17)
    
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Flow', fontsize=18)

    #月-日 时:分
    date_format = mpl.dates.DateFormatter("%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.ylim(0)
    fig.tight_layout()
    
    plt.xlim(pd.Timestamp(d), pd.Timestamp('2019-04-22 00:00'))
    
    fig.savefig('LSTM_static_5Day.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv("../Traffic_Data/Engldata.csv", 
                     parse_dates=["datetime (Veh/5 Minutes)"],
                     index_col="datetime (Veh/5 Minutes)")
    
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))
    df = normalize_data(df, flow_scaler, avg_kph_scaler)
    
    x_train, y_train, x_test, y_test = input_features(df)
    
    split_idx = 14109
    lag = 48
    test_dates = df.index[split_idx + lag:]
    
    lstm = load_model('../../webapp/model/LSTM.h5')
    models = [lstm]
    names = ['LSTM']
    
    y_test = denormalize_data(y_test, flow_scaler)
    y_preds = []
    for name, model in zip(names, models):
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
        predicted = model.predict(x_test)
        predicted = denormalize_data(predicted, flow_scaler)
        y_preds.append(predicted)
        print(name)
    
    plot_LSTM_fiveDay(y_test, y_preds, test_dates)
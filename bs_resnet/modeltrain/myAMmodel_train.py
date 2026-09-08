import numpy as np
import pandas as pd
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

from keras import Input, Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, Dense, Dropout, Layer
from keras.callbacks import EarlyStopping
import sklearn.metrics as metrics
import tensorflow as tf

# ===================== 自定义时序注意力层（核心AM模块） =====================
class TimeSeriesAttention(Layer):
    def __init__(self, **kwargs):
        super(TimeSeriesAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 输入: [batch, timesteps, features]
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="normal", trainable=True)
        self.V = self.add_weight(name="att_var", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        super(TimeSeriesAttention, self).build(input_shape)

    def call(self, x):
        # 注意力得分计算 
        e = tf.tanh(tf.matmul(x, self.W))
        scores = tf.matmul(e, self.V)
        weights = tf.nn.softmax(scores, axis=1)
        
        # 加权求和 → 上下文向量
        output = x * weights
        return tf.reduce_sum(output, axis=1)

# ===================== 原数据处理逻辑 =====================
# 数据归一化
def normalize_data(data, flow_scaler, avg_kph_scaler):
    data["Flow"] = flow_scaler.transform(data["Flow"].values.reshape(-1, 1))
    data["Avg kph"] = avg_kph_scaler.transform(data["Avg kph"].values.reshape(-1, 1))
    return data

# 反向归一化（适配多维输入，增强鲁棒性）
def denormalize_data(data, scaler):
    if len(data.shape) == 1:
        data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(1, -1)[0]
    elif len(data.shape) == 2:
        data = scaler.inverse_transform(data).reshape(data.shape[0], data.shape[1])
    return data

# 添加时间滞后值以构建模型输入特征
def input_features(data):
    # 10分钟间隔：lag=24 对应4小时历史数据
    lag = 24  
    train, test = [], []
    # 动态划分训练集/测试集（8:2）
    split_idx = int(len(data) * 0.8)  
    
    for i in range(lag, len(data)):
        if i < split_idx:
            train.append(data[i - lag: i + 1])
        else:
            test.append(data[i - lag: i + 1])

    train = np.array(train)
    test = np.array(test)

    # 设置模型的输入输出（单步预测，Flow为输出）
    x_train = train[:, :-1, [0, 1]]  # Flow + Avg kph 作为输入特征
    y_train = train[:, -1, [0]]      # Flow 作为输出（单步）
    x_test = test[:, :-1, [0, 1]]    # Flow + Avg kph 作为输入特征
    y_test = test[:, -1, [0]]        # Flow 作为输出（单步）
    return x_train, y_train, x_test, y_test

# ===================== LSTM+Attention（AM）模型 =====================
def build_LSTM():
    global input_shape
    inputs = Input(shape=input_shape)

    lstm_out = LSTM(256, activation='tanh', return_sequences=True, kernel_regularizer=l2(1e-5))(inputs)
    lstm_out = Dropout(0.05)(lstm_out)  

    # 注意力层
    attention_out = TimeSeriesAttention()(lstm_out)

    # 全连接层
    dense = Dense(64, activation='relu')(attention_out)
    dense = Dropout(0.05)(dense)

    outputs = Dense(1)(dense)
    model = Model(inputs=inputs, outputs=outputs)
  
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse', metrics=['mape'])
    return model


# ===================== GRU模型 =====================
def build_GRU():
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(units=256, activation='relu', kernel_regularizer=l2(0.0001), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(GRU(units=16))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.0002)
    model.compile(loss="mse", optimizer=optimizer, metrics=['mape'])
    return model

# ===================== 评估模型 =====================
def evaluate_models(y_true, y_pred):
    y_true = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    # 计算MAPE
    sums = 0  
    for i in range(len(y_pred)):
        tmp = abs(y_true[i] - y_pred[i]) / y_true[i]
        sums += tmp
    mape = sums * (100 / len(y_pred))
    # 计算其他评估指标
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = metrics.r2_score(y_true, y_pred)
    
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % rmse)
    print('r2:%f' % r2)

# ===================== 主函数（仅适配LSTM+AM模型） =====================
if __name__ == '__main__':
    # 读入10分钟间隔的交通数据
    df = pd.read_csv("../Traffic_Data/Dongsanhuan.csv", 
                     parse_dates=["datetime"],
                     index_col="datetime")

    # 分别归一化Flow和Avg kph
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))
    df = normalize_data(df, flow_scaler, avg_kph_scaler)
    
    # 构建输入特征
    x_train, y_train, x_test, y_test = input_features(df)
    # 定义输入形状
    input_shape = (x_train.shape[1], x_train.shape[2])

    # 训练模型（LSTM+AM / GRU 可选）
    model_struct = "LSTM"  # 选择 LSTM 或 GRU
    if model_struct == "LSTM":
        model = build_LSTM()
        monitor = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)
        hist = model.fit(x_train, y_train, batch_size=128, epochs=300, callbacks=[monitor],
                         validation_split=0.05)
        model.save('../../webapp/model/LSTM_AM_10min.h5')  # 文件名标注AM（注意力）
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv('../../webapp/model/LSTM_AM_10min_loss.csv', encoding='utf-8', index=False)
    elif model_struct == "GRU":
        model = build_GRU()
        monitor = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)
        hist = model.fit(x_train, y_train, batch_size=128, epochs=300, callbacks=[monitor], validation_split=0.05)
        model.save('../../webapp/model/GRU_10min.h5')
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv('../../webapp/model/GRU_10min_loss.csv', encoding='utf-8', index=False)

    # 评估模型性能
    # 装载模型
    lstm_am_model = load_model('../../webapp/model/LSTM_AM_10min.h5', custom_objects={'TimeSeriesAttention': TimeSeriesAttention})
    gru_model = load_model('../../webapp/model/GRU_10min.h5')
    models = [lstm_am_model, gru_model]
    names = ['LSTM+AM', 'GRU']  # 标注LSTM+AM
    
    # 反归一化真实值
    y_test = denormalize_data(y_test, flow_scaler)
    
    # 评估每个模型
    for name, model in zip(names, models):
        predicted = model.predict(x_test, verbose=0)
        predicted = denormalize_data(predicted, flow_scaler)
        print(f'\n==================== {name} 评估指标 =====================')
        evaluate_models(y_test, predicted)


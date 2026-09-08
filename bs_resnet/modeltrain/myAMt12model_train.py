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
from keras.models import load_model
from keras.layers import LSTM, GRU, Dense, Dropout, Layer
from keras.callbacks import EarlyStopping
import sklearn.metrics as metrics
import tensorflow as tf

# ===================== 自定义时序注意力层 =====================
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

# ===================== 数据处理代码 =====================
def normalize_data(data, flow_scaler, avg_kph_scaler):
    data["Flow"] = flow_scaler.transform(data["Flow"].values.reshape(-1, 1))
    data["Avg kph"] = avg_kph_scaler.transform(data["Avg kph"].values.reshape(-1, 1))
    return data

def denormalize_data(data, scaler):
    if len(data.shape) == 1:
        data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(1, -1)[0]
    elif len(data.shape) == 2:
        data = scaler.inverse_transform(data).reshape(data.shape[0], data.shape[1])
    return data

def input_features(data):
    lag = 24
    train, test = [], []
    split_idx = int(len(data) * 0.8)
    predict_steps = 12

    for i in range(lag, len(data) - predict_steps):
        if i < split_idx:
            train.append(data.iloc[i - lag: i + predict_steps])
        else:
            test.append(data.iloc[i - lag: i + predict_steps])

    train = np.array(train)
    test = np.array(test)

    x_train = train[:, :-predict_steps, [0, 1]]
    y_train = train[:, -predict_steps:, 0]
    x_test = test[:, :-predict_steps, [0, 1]]
    y_test = test[:, -predict_steps:, 0]
    return x_train, y_train, x_test, y_test



def build_LSTM_with_attention():
    global input_shape
    inputs = Input(shape=input_shape)

    
    lstm_out = LSTM(256, activation='tanh', return_sequences=True, kernel_regularizer=l2(1e-5))(inputs)
    lstm_out = Dropout(0.05)(lstm_out)  

    # 注意力层
    attention_out = TimeSeriesAttention()(lstm_out)

    # 全连接层
    dense = Dense(64, activation='relu')(attention_out)
    dense = Dropout(0.05)(dense)

    outputs = Dense(12)(dense)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse', metrics=['mape'])
    return model



# ===================== 评估代码 =====================
def evaluate_models(y_true, y_pred, step_name, step_idx):
    y_true_step = y_true[:, step_idx].flatten()
    y_pred_step = y_pred[:, step_idx].flatten()
    
    y_true_step = [x for x in y_true_step if x > 0]
    y_pred_step = [y_pred_step[i] for i in range(len(y_true_step)) if y_true_step[i] > 0]

    if len(y_true_step) == 0:
        print(f'\n{step_name}：无有效样本')
        return

    sums = 0
    for i in range(len(y_pred_step)):
        tmp = abs(y_true_step[i] - y_pred_step[i]) / y_true_step[i]
        sums += tmp
    mape = sums * (100 / len(y_pred_step))
    vs = metrics.explained_variance_score(y_true_step, y_pred_step)
    mae = metrics.mean_absolute_error(y_true_step, y_pred_step)
    mse = metrics.mean_squared_error(y_true_step, y_pred_step)
    rmse = math.sqrt(mse)
    r2 = metrics.r2_score(y_true_step, y_pred_step)
    
    print(f'\n========== {step_name} 核心指标 ==========')
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % rmse)
    print('r2:%f' % r2)

# ===================== 主函数 =====================
if __name__ == '__main__':
    df = pd.read_csv("../Traffic_Data/Dongsanhuan.csv", 
                     parse_dates=["datetime"], index_col="datetime")

    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))
    df = normalize_data(df, flow_scaler, avg_kph_scaler)
    
    x_train, y_train, x_test, y_test = input_features(df)
    input_shape = (x_train.shape[1], x_train.shape[2])

    # 训练 Attention-LSTM
    model = build_LSTM_with_attention()
    monitor = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(x_train, y_train, batch_size=128, epochs=300, validation_split=0.1, callbacks=[monitor])
    model.save('../../webapp/model/LSTM_attention_final.h5')

    # 评估
    lstm_model = load_model('../../webapp/model/LSTM_attention_final.h5', custom_objects={'TimeSeriesAttention': TimeSeriesAttention})
    
    y_test = denormalize_data(y_test, flow_scaler)
    
    print('\n==================== Attention-LSTM（论文版） ====================')
    pred_lstm = lstm_model.predict(x_test, verbose=0)
    pred_lstm = denormalize_data(pred_lstm, flow_scaler)
    evaluate_models(y_test, pred_lstm, "t+1", 0)
    evaluate_models(y_test, pred_lstm, "t+6", 5)
    evaluate_models(y_test, pred_lstm, "t+12", 11)
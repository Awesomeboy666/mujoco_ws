import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers import Layer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# ===================== 【原模型一模一样】注意力层，无任何修改 =====================
class TimeSeriesAttention(Layer):
    def __init__(self, **kwargs):
        super(TimeSeriesAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(256, 256),
                                 initializer="normal", trainable=True)
        self.V = self.add_weight(name="att_var", shape=(256, 1),
                                 initializer="normal", trainable=True)
        super(TimeSeriesAttention, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W))
        scores = tf.matmul(e, self.V)
        att_weights = tf.nn.softmax(scores, axis=1)
        output = x * att_weights
        return tf.reduce_sum(output, axis=1)

# ===================== 数据预处理 =====================
def input_features(data):
    lag = 24
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# ===================== 核心：TF会话计算权重（解决所有兼容问题） =====================
if __name__ == '__main__':
    # 1. 加载数据（修改你的路径）
    df = pd.read_csv("../Traffic_Data/Dongsanhuan.csv", parse_dates=["datetime"], index_col="datetime")
    data = df[["Flow", "Avg kph"]].values
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    
    # 构建测试集
    X_test, y_test = input_features(data_scaled)
    sample = X_test[0:1]  # 取第一个样本

    # 2. 加载模型
    model = load_model("../../webapp/model/LSTM_AM_10min.h5", 
                       custom_objects={"TimeSeriesAttention": TimeSeriesAttention})

    # 3. 找到 LSTM输出层 + 注意力层（关键！）
    lstm_output_layer = model.layers[-3]  # LSTM的输出
    attention_layer = model.layers[-2]   # 注意力层

    # 4. 【核心】直接用模型权重，重新计算注意力权重（和原模型完全一致）
    lstm_out = lstm_output_layer.output
    W = attention_layer.W
    V = attention_layer.V
    
    e = tf.tanh(tf.matmul(lstm_out, W))
    scores = tf.matmul(e, V)
    att_weights = tf.nn.softmax(scores, axis=1)

    # 5. 【终极兼容】TF会话执行，获取权重数组
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        model.set_weights(model.get_weights())  # 加载训练好的权重
        weight_val = sess.run(att_weights, feed_dict={model.input: sample})
    
    att_weights_np = weight_val.squeeze()  # 转成numpy数组

    # 6. 绘制论文权重图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 5))
    
    time_steps = [f"T-{i}" for i in range(24, 0, -1)]
    plt.plot(time_steps, att_weights_np, marker='o', color='#2E86AB', linewidth=2, markersize=6)
    plt.title("LSTM+注意力机制 权重分配图", fontsize=14)
    plt.xlabel("历史时间步（10分钟/步）", fontsize=12)
    plt.ylabel("注意力权重", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 保存高清图
    plt.savefig("attention_weights.png", dpi=300)
    plt.show()
    print("✅ 注意力权重图生成成功！")
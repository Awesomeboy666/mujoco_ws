import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
from keras.layers import Layer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 【固定】你训练用的注意力层（必须一模一样）=====================
class TimeSeriesAttention(Layer):
    def __init__(self, **kwargs):
        super(TimeSeriesAttention, self).__init__(**kwargs)
        self.attention_weights = None  # 内部存储权重

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="normal", trainable=True)
        self.V = self.add_weight(name="att_var", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        super(TimeSeriesAttention, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W))
        scores = tf.matmul(e, self.V)
        self.attention_weights = tf.nn.softmax(scores, axis=1)  # 真正的权重
        output = x * self.attention_weights
        return tf.reduce_sum(output, axis=1)

# ===================== 数据处理 =====================
def normalize_data(data, flow_scaler, avg_kph_scaler):
    data["Flow"] = flow_scaler.transform(data["Flow"].values.reshape(-1, 1))
    data["Avg kph"] = avg_kph_scaler.transform(data["Avg kph"].values.reshape(-1, 1))
    return data

def get_test_data():
    df = pd.read_csv("../Traffic_Data/Dongsanhuan.csv", parse_dates=["datetime"], index_col="datetime")
    flow_scaler = MinMaxScaler().fit(df["Flow"].values.reshape(-1,1))
    avg_scaler = MinMaxScaler().fit(df["Avg kph"].values.reshape(-1,1))
    df = normalize_data(df, flow_scaler, avg_scaler)
    
    lag = 24
    x_test = []
    split_idx = int(len(df) * 0.8)
    for i in range(lag, len(df)):
        if i >= split_idx:
            x_test.append(df.iloc[i-lag:i, [0,1]].values)
    return np.array(x_test), flow_scaler

# ===================== 【修复】绘图函数 =====================
def plot_attention(att_weights):
    # 强制修正维度 → 解决所有形状报错
    att_weights = np.squeeze(att_weights)  # (样本数, 24)
    single = att_weights[0]                # (24,)
    multi = att_weights[:100]              # (100,24)
    avg = np.mean(multi, axis=0)           # (24,)
    steps = np.arange(1, 25)               # (24,)

    # 1. 柱状图
    plt.figure(figsize=(12,6))
    plt.bar(steps, single, color='#4285F4', alpha=0.8, label='单样本权重')
    plt.plot(steps, avg, color='#EA4335', linewidth=3, marker='o', label='平均权重')
    plt.xlabel('时间步(1-24,10分钟/步)')
    plt.ylabel('注意力权重')
    plt.title('注意力权重柱状图')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('../../webapp/model/attention_bar.png', dpi=300)
    plt.show()

    # 2. 热力图
    plt.figure(figsize=(12,6))
    plt.imshow(multi.T, cmap='Blues', aspect='auto')
    plt.colorbar(label='权重')
    plt.xlabel('测试样本')
    plt.ylabel('时间步')
    plt.title('注意力权重热力图')
    plt.savefig('../../webapp/model/attention_heatmap.png', dpi=300)
    plt.show()

# ===================== 主函数（核心修复）=====================
if __name__ == '__main__':
    x_test, _ = get_test_data()
    
    # 加载你的模型
    model = load_model(
        '../../webapp/model/LSTM_AM_10min.h5',
        custom_objects={'TimeSeriesAttention': TimeSeriesAttention}
    )

    # ✅【唯一正确方式】提取注意力权重（解决99%报错）
    att_layer = model.get_layer('time_series_attention')  # 找到注意力层
    # 构建权重提取模型
    weights_model = tf.keras.Model(
        inputs=model.input,
        outputs=att_layer.attention_weights  # 直接拿层内存储的权重
    )
    
    # 预测得到权重
    att_weights = weights_model.predict(x_test, verbose=0)
    
    # 绘图
    plot_attention(att_weights)
    print("✅ 绘图成功！图片已保存！")
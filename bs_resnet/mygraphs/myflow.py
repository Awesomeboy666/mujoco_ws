import datetime
import numpy as np
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

'''
此代码绘制交通流量真实数据走势图 
'''

# 数据归一化
def normalize_data(data, flow_scaler, avg_kph_scaler):
    data["Flow"] = flow_scaler.transform(data["Flow"].values.reshape(-1, 1))
    data["Avg kph"] = avg_kph_scaler.transform(data["Avg kph"].values.reshape(-1, 1))
    return data

# 反向归一化（还原Flow真实值）
def denormalize_data(data, scaler):
    data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(1, -1)[0]
    return data

# 绘制真实数据3天走势图（静态时间步）
def plot_true_data_threeDay(y_true, test_dates):
    # 取3天的真实数据点（432个点 = 3天 * 24小时 * 6个(10分钟/个)）
    x = test_dates[:432]
    
    # 创建画布
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    # 仅绘制真实值折线
    ax.plot(x, y_true[:432], label='True Data', linewidth=2, color='#1f77b4')
    
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
    plt.title('True Traffic Flow (3 Days, 10min Interval)', fontsize=20)
    fig.tight_layout()
    fig.savefig('True_Flow_3Day_10min.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # 1. 读取原始交通数据
    data_path = "../Traffic_Data/Dongsanhuan.csv"
    df = pd.read_csv(
        data_path, 
        parse_dates=["datetime"],
        index_col="datetime"
    )

    # 2. 初始化归一化器（用于还原真实Flow值）
    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Flow"].values.reshape(-1, 1))
    avg_kph_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df["Avg kph"].values.reshape(-1, 1))

    # 3. 数据归一化
    df_normalized = normalize_data(df.copy(), flow_scaler, avg_kph_scaler)
    
    # 4. 划分测试集
    split_idx = int(len(df) * 0.8)
    lag = 24  # 过去4小时的时间步（不影响真实数据提取，仅对齐原索引逻辑）
    test_dates = df.index[split_idx + lag:]  # 测试集时间索引
    y_test_normalized = df_normalized["Flow"].iloc[split_idx + lag:].values  # 归一化后的测试集Flow
    
    # 5. 还原为原始真实Flow值
    y_test_true = denormalize_data(y_test_normalized, flow_scaler)

    # 6. 绘制3天真实数据走势图
    print("正在绘制3天真实流量走势图...")
    plot_true_data_threeDay(y_test_true, test_dates)
    
   
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 核心配置=====================
# 你的损失文件路径：
LOSS_CSV_PATH = "../../webapp/model/LSTM_AM_10min_loss.csv"
# 图片保存路径
SAVE_PATH = "../../code/graphs/LSTM_AM_loss_curve.png"

# ===================== 读取Loss数据 =====================
def load_loss_data(csv_path):
    """读取LSTM_10min_loss.csv，返回训练集loss数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到Loss文件：{csv_path}，请检查路径！")
    
    # 读取你的CSV文件
    df_loss = pd.read_csv(csv_path, encoding='utf-8')
    # 生成训练轮次Epoch
    df_loss['Epoch'] = range(1, len(df_loss) + 1)
    return df_loss

# ===================== 只绘制 训练集Loss 曲线 =====================
def plot_loss_curve(df_loss, save_path):
    """仅绘制训练集MSE损失曲线，保存高清图片"""
    plt.figure(figsize=(12, 6), dpi=100)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 🔥 只画：训练集损失（loss列）
    plt.plot(
        df_loss['Epoch'], 
        df_loss['loss'], 
        label='Train Loss (MSE)', 
        linewidth=2.5, 
        color='#2E86AB',  # 蓝色
        marker='o', markersize=3, markevery=10
    )

    # 图表标题/坐标轴
    plt.title('AM+LSTM loss curve', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14, labelpad=10)
    plt.ylabel('Loss (Normalized MSE)', fontsize=14, labelpad=10)

    # 图例、刻度、网格
    plt.legend(fontsize=12, loc='upper right', frameon=True)
    plt.xticks(range(0, len(df_loss['Epoch']) + 1, 20), fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存并显示
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ 训练集Loss曲线已保存：{save_path}")
    plt.show()

# ===================== 主函数 =====================
if __name__ == '__main__':
    try:
        loss_data = load_loss_data(LOSS_CSV_PATH)
        print("📊 训练集Loss数据预览：")
        print(loss_data[['Epoch', 'loss']].head())
        plot_loss_curve(loss_data, SAVE_PATH)
    except Exception as e:
        print(f"❌ 绘图失败：{str(e)}")
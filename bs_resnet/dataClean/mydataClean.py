"""将原始记录整理为10分钟序列，速度单位为km/h；流量是时刻统计值。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def clean_data(input_path):
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    required = ['timestamp', 'vehicle_count', 'avg_speed']
    if not set(required).issubset(df.columns):
        raise ValueError(f'原始数据必须包含 {required}')
    if 'road_id' in df and df['road_id'].nunique() > 1:
        raise ValueError('存在多条道路，请先选择一条道路。')
    df = df[required].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    print('无效时间记录：', df['timestamp'].isna().sum())
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    if df.empty:
        raise ValueError('没有有效时间记录。')
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[~np.isfinite(df[col]) | (df[col] < 0), col] = np.nan
    # 右边界标记，时间格只包含标签时刻及之前的观测。
    # 时刻流量取平均，不解释成10分钟累计车辆数。
    result = df.set_index('timestamp').resample(
        '10min', closed='right', label='right'
    ).agg({'vehicle_count': 'mean', 'avg_speed': 'mean'})
    result = result.rename(columns={'vehicle_count': 'Flow', 'avg_speed': 'Avg kph'})
    result.index.name = 'datetime'
    # 保留缺口，不用未来观测插值，不猜单位，不截断或取整真实值。
    print('时间格：', len(result), '；缺失值：', result.isna().sum().to_dict())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'Traffic_Data/Dongsanhuanyuanshi.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'Traffic_Data/Dongsanhuan_cleaned.csv')
    args = parser.parse_args()
    result = clean_data(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, encoding='utf-8-sig', date_format='%Y-%m-%d %H:%M')
    print('已保存：', args.output)

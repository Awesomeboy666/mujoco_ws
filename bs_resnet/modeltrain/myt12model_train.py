"""在原多步训练脚本上改为 PyTorch ResNet：12步输入，6步输出。
PyCharm直接运行。默认使用用户指定CSV；路径不依赖工作目录。
"""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['Flow', 'Avg kph']
INPUT_STEPS, PREDICT_STEPS = 12, 6


def load_data(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if not {'datetime', *FEATURES}.issubset(df.columns):
        raise ValueError('CSV必须包含 datetime、Flow、Avg kph。')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='raise')
    if df['datetime'].isna().any() or df['datetime'].duplicated().any():
        raise ValueError('时间列有空值或重复值。')
    df = df.set_index('datetime').sort_index()[FEATURES]
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[~np.isfinite(df[col]) | (df[col] < 0), col] = np.nan
    # 裁去首尾完全空的记录，内部缺口保留，不能拼接不连续的时间。
    valid = df.notna().any(axis=1)
    if not valid.any():
        raise ValueError('没有有效观测。')
    df = df.loc[valid[valid].index[0]:valid[valid].index[-1]]
    grid = pd.date_range(df.index[0], df.index[-1], freq='10min')
    if not df.index.isin(grid).all():
        raise ValueError('存在不在10分钟时间格上的记录，请先清洗。')
    df = df.reindex(grid)
    df.index.name = 'datetime'
    return df


def normalize_data(data, flow_scaler, avg_kph_scaler):
    result = data.copy()
    for col, scaler in zip(FEATURES, [flow_scaler, avg_kph_scaler]):
        result[col] = scaler.transform(data[[col]].to_numpy()).ravel()
    return result


def denormalize_data(data, scaler):
    values = np.asarray(data)
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)


def input_features(data, train_end, val_end, lag=INPUT_STEPS, predict_steps=PREDICT_STEPS):
    """完整目标必须属于同一集合；验证/测试可以使用之前已知的历史。"""
    groups = {name: {'x': [], 'y': [], 'origins': [], 'times': []}
              for name in ['train', 'val', 'test']}
    # 仅输入可用过去观测填充最多2格；目标永不填充。
    history = data.ffill(limit=2).to_numpy(dtype=np.float32)
    flow = data['Flow'].to_numpy(dtype=np.float32)
    for i in range(lag, len(data)-predict_steps+1):
        end = i+predict_steps
        if end <= train_end:
            name = 'train'
        elif i >= train_end and end <= val_end:
            name = 'val'
        elif i >= val_end:
            name = 'test'
        else:
            continue
        x, y = history[i-lag:i], flow[i:end]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            continue
        g = groups[name]
        g['x'].append(x)
        g['y'].append(y)
        g['origins'].append(data.index[i-1])
        g['times'].append(data.index[i:end].to_numpy())
    for name, g in groups.items():
        if not g['x']:
            raise ValueError(f'{name}没有完整样本，请检查长度和缺失情况。')
        g['x'], g['y'], g['times'] = np.stack(g['x']), np.stack(g['y']), np.stack(g['times'])
    return groups


def prepare_data(path):
    raw = load_data(path)
    if len(raw) < 100:
        raise ValueError('数据过短，无法划分训练、验证和测试集。')
    train_end, val_end = int(len(raw)*0.7), int(len(raw)*0.8)
    scalers = []
    for col in FEATURES:
        values = raw.iloc[:train_end][[col]].dropna().to_numpy()
        if not len(values):
            raise ValueError(f'训练段的{col}没有有效观测。')
        scalers.append(MinMaxScaler().fit(values))
    normalized = normalize_data(raw, *scalers)
    groups = input_features(normalized, train_end, val_end)
    return raw, normalized, groups, scalers, (train_end, val_end)


def evaluate_models(y_true, y_pred):
    """仅MAPE排除真实零值，并同步过滤预测；其他指标保留零流量。"""
    rows = []
    for step in range(PREDICT_STEPS+1):
        a = y_true.ravel() if step == 0 else y_true[:, step-1]
        p = y_pred.ravel() if step == 0 else y_pred[:, step-1]
        mask = a != 0
        rows.append({
            'step': 'all' if step == 0 else f't+{step}',
            'MAE': metrics.mean_absolute_error(a, p),
            'RMSE': np.sqrt(metrics.mean_squared_error(a, p)),
            'R2': metrics.r2_score(a, p) if len(a) > 1 else np.nan,
            'MAPE_percent': np.mean(np.abs((a[mask]-p[mask])/a[mask]))*100 if mask.any() else np.nan,
            'MAPE_count': int(mask.sum())})
    return pd.DataFrame(rows)


def plot_results(history, actual, prediction, output):
    cache = ROOT/'work/matplotlib-cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot([r['epoch'] for r in history], [r['train_mse'] for r in history], label='Train')
    axes[0].plot([r['epoch'] for r in history], [r['val_mse'] for r in history], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='MSE (normalized)')
    for ax, step in zip(axes[1:], [0, 5]):
        ax.plot(actual[:200, step], label='Actual')
        ax.plot(prediction[:200, step], label='ResNet')
        ax.set(title=f't+{step+1}: first 200 test windows', xlabel='Test window', ylabel='Flow')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output/'training_and_predictions.png', dpi=150)
    plt.close(fig)


def train(args):
    # 延迟导入，检查数据处理时不要求安装PyTorch。
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    try:
        from .resnet_model import TrafficResNet
    except ImportError:
        from resnet_model import TrafficResNet

    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw, normalized, groups, scalers, boundaries = prepare_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    print('设备：', device)
    loaders = {}
    for name, g in groups.items():
        print(f"{name}: X={g['x'].shape}, Y={g['y'].shape}")
        loaders[name] = DataLoader(
            TensorDataset(torch.from_numpy(g['x']), torch.from_numpy(g['y'])),
            batch_size=args.batch_size, shuffle=(name == 'train'), num_workers=0)
    model = TrafficResNet(INPUT_STEPS, PREDICT_STEPS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best, stale, history = float('inf'), 0, []
    checkpoint = args.output/'resnet_best.pt'
    metadata = {
        'input_steps': INPUT_STEPS, 'predict_steps': PREDICT_STEPS,
        'features': FEATURES, 'interval_minutes': 10,
        'scalers': [{'scale': s.scale_.tolist(), 'min': s.min_.tolist()} for s in scalers],
        'data_path': str(args.data.resolve()), 'seed': args.seed,
        'learning_rate': args.lr, 'batch_size': args.batch_size, 'patience': args.patience,
        'train_end_exclusive': str(raw.index[boundaries[0]]),
        'val_end_exclusive': str(raw.index[boundaries[1]]),
        'counts': {name: len(g['x']) for name, g in groups.items()}}
    for epoch in range(1, args.epochs+1):
        losses = {}
        for name in ['train', 'val']:
            model.train(name == 'train')
            total = 0.0
            with torch.set_grad_enabled(name == 'train'):
                for x, y in loaders[name]:
                    x, y = x.to(device), y.to(device)
                    if name == 'train':
                        optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError('损失出现NaN或无穷值。')
                    if name == 'train':
                        loss.backward()
                        optimizer.step()
                    total += loss.item()*len(x)
            losses[name] = total/len(loaders[name].dataset)
        history.append({'epoch': epoch, 'train_mse': losses['train'], 'val_mse': losses['val']})
        print(f"Epoch {epoch:03d}: train={losses['train']:.6f}, val={losses['val']:.6f}", flush=True)
        if losses['val'] < best:
            best, stale = losses['val'], 0
            torch.save({'model_state': model.state_dict(), 'metadata': metadata,
                        'epoch': epoch, 'val_mse': best}, checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            print('验证损失连续未改善，提前停止。')
            break

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved['model_state'])
    model.eval()
    with torch.no_grad():
        prediction = np.concatenate([model(x.to(device)).cpu().numpy() for x, _ in loaders['test']])
    actual = denormalize_data(groups['test']['y'], scalers[0])
    prediction = denormalize_data(prediction, scalers[0])
    # 持续值基线：六个未来值全部等于最后一个已知流量。
    baseline = denormalize_data(
        np.repeat(groups['test']['x'][:, -1, 0:1], PREDICT_STEPS, axis=1), scalers[0])
    report = pd.concat([evaluate_models(actual, prediction).assign(model='ResNet'),
                        evaluate_models(actual, baseline).assign(model='Persistence')])
    report.to_csv(args.output/'metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(history).to_csv(args.output/'loss.csv', index=False)
    pd.DataFrame({
        'origin': np.repeat(np.array(groups['test']['origins']), PREDICT_STEPS),
        'target_time': groups['test']['times'].ravel(),
        'step': np.tile(np.arange(1, PREDICT_STEPS+1), len(actual)),
        'actual_flow': actual.ravel(), 'predicted_flow': prediction.ravel(),
        'baseline_flow': baseline.ravel(),
    }).to_csv(args.output/'test_predictions.csv', index=False)
    metadata.update({'best_epoch': saved['epoch'], 'best_val_mse': saved['val_mse']})
    (args.output/'run_info.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    # 对数据末尾之后的6步预测，没有真实标签，不参与测试指标。
    last_x = normalized.ffill(limit=2).iloc[-INPUT_STEPS:].to_numpy(dtype=np.float32)
    if np.isfinite(last_x).all():
        with torch.no_grad():
            future = model(torch.from_numpy(last_x[None]).to(device)).cpu().numpy()
        pd.DataFrame({
            'datetime': pd.date_range(raw.index[-1]+pd.Timedelta(minutes=10), periods=PREDICT_STEPS, freq='10min'),
            'predicted_flow': denormalize_data(future, scalers[0]).ravel(),
        }).to_csv(args.output/'future_predictions.csv', index=False)
    else:
        print('末尾缺失过多，未生成未来预测。')
    plot_results(history, actual, prediction, args.output)
    print(report.to_string(index=False))
    print('最佳轮次：', saved['epoch'], '；结果目录：', args.output)


def predict_from_checkpoint(data_path, checkpoint, output):
    """加载已有权重及训练时的归一化参数，不重新训练或重新拟合。"""
    import torch
    try:
        from .resnet_model import TrafficResNet
    except ImportError:
        from resnet_model import TrafficResNet
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    info = saved['metadata']
    if info['features'] != FEATURES:
        raise ValueError('模型输入特征顺序不一致。')
    raw = load_data(data_path)
    if len(raw) < info['input_steps']:
        raise ValueError(f"预测至少需要{info['input_steps']}步历史数据。")
    x = raw.ffill(limit=2).iloc[-info['input_steps']:].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('末尾历史缺失过多，不能预测。')
    for channel, scaler in enumerate(info['scalers']):
        x[:, channel] = x[:, channel]*scaler['scale'][0]+scaler['min'][0]
    model = TrafficResNet(info['input_steps'], info['predict_steps'])
    model.load_state_dict(saved['model_state'])
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(x[None])).numpy().ravel()
    flow_scaler = info['scalers'][0]
    y = (y-flow_scaler['min'][0])/flow_scaler['scale'][0]
    result = pd.DataFrame({
        'datetime': pd.date_range(raw.index[-1]+pd.Timedelta(minutes=10),
                                  periods=info['predict_steps'], freq='10min'),
        'predicted_flow': y})
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output/'future_predictions.csv', index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'Traffic_Data/Dongsanhuan.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/resnet')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--predict-only', action='store_true', help='只加载模型预测，不训练')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/resnet/resnet_best.pt')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.lr) <= 0:
        parser.error('epochs、batch-size、patience、lr必须为正数。')
    if args.predict_only:
        predict_from_checkpoint(args.data, args.checkpoint, args.output)
    else:
        train(args)

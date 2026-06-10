"""
纽约出租车分时段需求预测 - LSTM + TorchInspector v0.2.0
使用 2025 年 1 月黄色出租车数据，预测未来 1 小时全市订单量

用法:
    python taxi_rnn_demand.py [--epochs 30] [--seq-len 24] [--max-rows 500000]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchinspector import Inspector


# ── 数据 ──────────────────────────────────────────────────────────────────────
TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
DATA_DIR = Path(__file__).parent / "data"
PARQUET_PATH = DATA_DIR / "yellow_tripdata_2025-01.parquet"


def download_data():
    if PARQUET_PATH.exists():
        print(f"[INFO] 数据已存在: {PARQUET_PATH}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    print(f"[INFO] 正在下载: {TLC_URL}")
    import urllib.request
    urllib.request.urlretrieve(TLC_URL, PARQUET_PATH)
    print(f"[INFO] 下载完成: {PARQUET_PATH}")


# ── 构造小时级时序 ────────────────────────────────────────────────────────────
def build_hourly_series(df: pd.DataFrame, max_rows: int = None):
    """
    将原始行程数据聚合为每小时订单量时序
    返回: DataFrame，index=时间, columns=[demand, hour, weekday, is_peak]
    """
    if max_rows:
        df = df.head(max_rows)

    df = df.copy()
    df["pickup_dt"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_dt"])

    # 按小时聚合
    df["hour_bin"] = df["pickup_dt"].dt.floor("h")
    demand = df.groupby("hour_bin").size().reset_index(name="demand")
    demand = demand.sort_values("hour_bin").reset_index(drop=True)

    # 时间特征
    demand["hour"] = demand["hour_bin"].dt.hour
    demand["weekday"] = demand["hour_bin"].dt.weekday
    demand["is_peak"] = ((demand["hour"].between(7, 10)) | (demand["hour"].between(16, 20))).astype(float)
    # 周期性编码
    demand["hour_sin"] = np.sin(2 * np.pi * demand["hour"] / 24)
    demand["hour_cos"] = np.cos(2 * np.pi * demand["hour"] / 24)
    demand["weekday_sin"] = np.sin(2 * np.pi * demand["weekday"] / 7)
    demand["weekday_cos"] = np.cos(2 * np.pi * demand["weekday"] / 7)

    return demand


def normalize(series: pd.Series):
    """Z-score 标准化"""
    m, s = series.mean(), series.std()
    s = s if s > 0 else 1
    return (series - m) / s, m, s


# ── 序列构造 ──────────────────────────────────────────────────────────────────
def build_sequences(demand: pd.DataFrame, seq_len: int):
    """
    滑动窗口: 过去 seq_len 小时 → 预测下一小时需求
    特征: [demand_norm, hour_sin, hour_cos, weekday_sin, weekday_cos, is_peak]
    """
    feature_cols = ["demand_norm", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_peak"]
    features = demand[feature_cols].values
    targets = demand["demand_norm"].values

    X, y = [], []
    for i in range(len(demand) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y


# ── LSTM 模型 ─────────────────────────────────────────────────────────────────
class DemandLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── 训练 ──────────────────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_test, y_test, demand_mean, demand_std, args):
    model = DemandLSTM(input_dim=X_train.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.HuberLoss(delta=0.5)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(__file__).parent / "runs" / f"taxi_demand_{timestamp}"

    with Inspector(model, optimizer, str(log_dir),
                   log_interval=20, rnn_interval=20, health_report_interval=50) as ins:
        ins.watch(["head.0", "head.2", "lstm"])
        ins.log_graph(X_train[:1])

        print(f"\n{'='*65}")
        print(f" 分时段需求预测 | seq_len={args.seq_len} | epochs={args.epochs}")
        print(f" 训练集: {len(X_train)} | 测试集: {len(X_test)}")
        print(f" TensorBoard: {log_dir}")
        print(f"{'='*65}\n")

        for epoch in range(1, args.epochs + 1):
            # 训练
            model.train()
            t_loss, t_mae, t_cnt = 0, 0, 0
            for xb, yb in train_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                t_loss += loss.item() * len(xb)
                t_mae += (pred - yb).abs().mean().item() * len(xb)
                t_cnt += len(xb)
                ins.step(loss=loss.item())
            scheduler.step()

            # 验证
            model.eval()
            v_loss, v_mae, v_cnt = 0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    pred = model(xb)
                    v_loss += loss_fn(pred, yb).item() * len(xb)
                    v_mae += (pred - yb).abs().mean().item() * len(xb)
                    v_cnt += len(xb)

            # 反标准化 MAE → 真实订单数
            t_mae_real = (t_mae / t_cnt) * demand_std
            v_mae_real = (v_mae / v_cnt) * demand_std
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {t_loss/t_cnt:.4f} | MAE: {t_mae_real:.0f} 单/时 | "
                  f"Test Loss: {v_loss/v_cnt:.4f} | MAE: {v_mae_real:.0f} 单/时")

    print(f"\n[OK] TensorBoard: tensorboard --logdir=runs")
    return model


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LSTM 分时段需求预测")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seq-len", type=int, default=24, help="用过去 N 小时预测")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None, help="限制原始行数，None=用全部数据")
    args = parser.parse_args()

    download_data()

    print("[INFO] 加载数据...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[INFO] 原始行程: {len(df):,} 条")

    # 构造小时级时序
    demand = build_hourly_series(df, max_rows=args.max_rows)
    print(f"[INFO] 小时级时序: {len(demand)} 个小时")
    print(f"[INFO] 需求范围: {demand['demand'].min()} ~ {demand['demand'].max()} 单/时")

    # 标准化需求
    demand["demand_norm"], demand_mean, demand_std = normalize(demand["demand"])
    print(f"[INFO] 需求均值: {demand_mean:.0f}, 标准差: {demand_std:.0f}")

    # 构造序列
    X, y = build_sequences(demand, args.seq_len)
    print(f"[INFO] 序列: X={tuple(X.shape)}, y={tuple(y.shape)}")

    # 划分
    n = len(X)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_test, y_test = X[idx[split:]], y[idx[split:]]

    train_model(X_train, y_train, X_test, y_test, demand_mean, demand_std, args)


if __name__ == "__main__":
    main()

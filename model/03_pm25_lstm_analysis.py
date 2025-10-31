#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate, Flatten, RepeatVector
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
from pathlib import Path

# ---------------- 参数 ----------------
DATA_PATH = "data/processed_dataset.csv"
FEATURE_FILE = "results/selected_features_vif.csv"
RESULT_DIR = Path("lstm_results")
TIME_STEPS = 24
HORIZON = 6
EMBED_DIM = 4
EPOCHS = 50
BATCH_SIZE = 64

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- 数据准备 ----------------
print(" 读取数据...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.dropna(subset=["PM2_5"])

# 读取特征列表
selected_features = pd.read_csv(FEATURE_FILE)["feature"].tolist()
selected_features = [f for f in selected_features if f in df.columns]

# 编码 site_id
df["site_id"] = df["site_id"].astype(str)
le = LabelEncoder()
df["site_code"] = le.fit_transform(df["site_id"])
num_sites = len(le.classes_)
print(f" 检测到 {num_sites} 个监测站点")

# 特征矩阵与目标
X = df[selected_features]
y = df["PM2_5"].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
X_scaled = np.concatenate([X_scaled, df[["site_code"]].values], axis=1)

# ---------------- 构造时间序列 ----------------
def make_sequences(X, y, site_codes, time_steps=24, horizon=6):
    Xs, ys, site_ids = [], [], []
    for i in range(len(X) - time_steps - horizon + 1):
        Xs.append(X[i:(i + time_steps), :-1])
        ys.append(y[i + time_steps:i + time_steps + horizon, 0])
        site_ids.append(site_codes[i + time_steps])
    return np.array(Xs), np.array(ys), np.array(site_ids)

print(" 构造时间序列数据...")
X_seq, y_seq, site_seq = make_sequences(X_scaled, y_scaled, df["site_code"].values, TIME_STEPS, HORIZON)
split_idx = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
site_train, site_test = site_seq[:split_idx], site_seq[split_idx:]

# ---------------- 构建模型 ----------------
print(" 构建 LSTM + Embedding 模型...")
seq_input = Input(shape=(TIME_STEPS, X_train.shape[2]), name="seq_input")
site_input = Input(shape=(1,), name="site_input")

site_embed = Embedding(input_dim=num_sites, output_dim=EMBED_DIM)(site_input)
site_embed = Flatten()(site_embed)
site_embed_repeated = RepeatVector(TIME_STEPS)(site_embed)
combined = Concatenate(axis=-1)([seq_input, site_embed_repeated])

x = LSTM(64)(combined)
x = Dropout(0.2)(x)
output = Dense(HORIZON, activation="linear")(x)

model = Model([seq_input, site_input], output)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="mse")

print(model.summary())

# ---------------- 训练 ----------------
print(" 开始训练模型...")
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

model.fit([X_train, site_train], y_train, validation_split=0.1,
          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# ---------------- 预测 ----------------
print(" 生成预测结果...")
y_pred_train_scaled = np.nan_to_num(model.predict([X_train, site_train]))
y_pred_test_scaled = np.nan_to_num(model.predict([X_test, site_test]))

y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).reshape(y_pred_train_scaled.shape)
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).reshape(y_pred_test_scaled.shape)
y_true_train = scaler_y.inverse_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_true_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# ---------------- 计算指标 ----------------
print(" 计算模型指标（训练集 + 测试集）...")
def compute_metrics(y_true, y_pred):
    metrics = {}
    for h in range(HORIZON):
        rmse = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
        r2 = r2_score(y_true[:, h], y_pred[:, h])
        mae = mean_absolute_error(y_true[:, h], y_pred[:, h])
        metrics[f"h{h+1}"] = {"RMSE": rmse, "R2": r2, "MAE": mae}
    return metrics

metrics_train = compute_metrics(y_true_train, y_pred_train)
metrics_test = compute_metrics(y_true_test, y_pred_test)

df_train = pd.DataFrame(metrics_train).T.reset_index().rename(columns={"index": "horizon"})
df_train["dataset"] = "train"
df_test = pd.DataFrame(metrics_test).T.reset_index().rename(columns={"index": "horizon"})
df_test["dataset"] = "test"

metrics_combined = pd.concat([df_train, df_test], axis=0)[["dataset", "horizon", "RMSE", "R2", "MAE"]]
metrics_combined.to_csv(RESULT_DIR / "LSTM_metrics_embedding.csv", index=False, encoding="utf-8")

print(" 模型训练与测试指标已保存：LSTM_metrics_embedding.csv")

# ---------------- 保存预测结果 ----------------
print(" 保存预测结果（仅测试集）...")
pred_cols = [f"PM2_5_pred_h{h+1}" for h in range(HORIZON)]
true_cols = [f"PM2_5_true_h{h+1}" for h in range(HORIZON)]
pred_df = pd.DataFrame(y_pred_test, columns=pred_cols)
true_df = pd.DataFrame(y_true_test, columns=true_cols)
combined_df = pd.concat([true_df, pred_df], axis=1)
combined_df.to_csv(RESULT_DIR / "LSTM_predictions_embedding.csv", index=False, encoding="utf-8")

# ---------------- 保存参数 ----------------
params = {
    "time_steps": TIME_STEPS,
    "horizon": HORIZON,
    "embed_dim": EMBED_DIM,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_sites": num_sites,
    "features": selected_features
}
with open(RESULT_DIR / "LSTM_best_params_embedding.json", "w", encoding="utf-8") as f:
    json.dump(params, f, indent=4, ensure_ascii=False)

print("\\n 完成：LSTM (embedding版) 训练与测试指标合并输出，文件保存在 lstm_results/ 文件夹中。")

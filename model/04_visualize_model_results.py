#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_model_results_v3.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------- 基础设置 ----------------
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style='whitegrid', context='talk')

RESULT_DIR = Path("visualization_results_v3")
RESULT_DIR.mkdir(exist_ok=True)

# ---------------- 1️  模型性能对比 ----------------
try:
    perf_df = pd.read_csv("optimized_full_results/optimized_model_performance.csv")
    metrics = ["train_rmse", "test_rmse", "train_r2", "test_r2", "train_mae", "test_mae"]
    melt_df = perf_df.melt(id_vars="model", value_vars=metrics, var_name="Metric", value_name="Score")
    plt.figure(figsize=(12,6))
    sns.barplot(data=melt_df, x="Metric", y="Score", hue="model", palette="Set2")
    plt.title("model_performance (RMSE, R², MAE)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "1_model_performance_comparison.png", dpi=300)
    plt.close()
    print(" 图1: 模型性能对比图已生成。")
except Exception as e:
    print(f" 模型性能图生成失败: {e}")

# ---------------- 2️ 特征重要性 ----------------
try:
    feat_df = pd.read_csv("optimized_full_results/Feature_Importance.csv")
    top_feats = feat_df.groupby("feature")["importance"].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_feats.values, y=top_feats.index, palette="viridis")
    plt.title("Feature_Importance Top 15")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "2_feature_importance_top15.png", dpi=300)
    plt.close()
    print(" 图2: 特征重要性Top15图已生成。")
except Exception as e:
    print(f" 特征重要性图生成失败: {e}")

# ---------------- 3️ LSTM 多步预测性能对比 ----------------
try:
    lstm_df = pd.read_csv("lstm_results/LSTM_metrics_embedding.csv")
    if {"dataset", "horizon", "RMSE", "R2", "MAE"}.issubset(lstm_df.columns):
        plt.figure(figsize=(12,6))
        for metric in ["RMSE", "R2", "MAE"]:
            sns.lineplot(data=lstm_df, x="horizon", y=metric, hue="dataset", marker="o")
        plt.title("LSTM performance over Multi-step Predictions")
        plt.xlabel("prediction Horizon (steps)")
        plt.ylabel("Metric Value")
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "3_lstm_multi_step_train_test_metrics.png", dpi=300)
        plt.close()
        print(" 图3: LSTM 训练/测试多步指标对比图已生成。")
except Exception as e:
    print(f" LSTM 多步指标图生成失败: {e}")



print("\n 所有图表已生成，保存在 visualization_results_v3/ 文件夹中。")

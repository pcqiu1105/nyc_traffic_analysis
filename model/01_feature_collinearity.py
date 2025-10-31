#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

# -----------------------
# 参数设置
# -----------------------
DATA_PATH = "data/processed_dataset.csv"
OUT_DIR = Path("results")
VIF_THRESHOLD = 10
CORR_THRESHOLD = 0.95
MISSING_THRESHOLD = 0.5
TARGET_COL = "PM2_5"
TIME_COL = "timestamp"

# -----------------------
# 主流程
# -----------------------
def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # 1️ 读取数据
    df = pd.read_csv(DATA_PATH)
    print(f"\n 数据维度：{df.shape}")

    # 2️ 删除非特征列（时间与目标）
    drop_cols = [TIME_COL, TARGET_COL]
    non_num_cols = [c for c in ["site_id", "borough", "borough_area"] if c in df.columns]
    drop_cols += non_num_cols
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    print(f" 可分析特征数（数值型）：{len(X.columns)}")

    # 3️ 缺失率检测
    missing_ratio = X.isnull().mean()
    high_missing = missing_ratio[missing_ratio > MISSING_THRESHOLD].index.tolist()
    if high_missing:
        print(f"\n以下特征缺失率过高 (> {MISSING_THRESHOLD*100:.0f}%)，将跳过共线性检测：")
        for c in high_missing:
            print(f"  - {c}: {missing_ratio[c]*100:.1f}% 缺失")

    X_valid = X.drop(columns=high_missing, errors="ignore").dropna()
    print(f"\n 参与共线性检测的特征数：{len(X_valid.columns)}")

    if len(X_valid.columns) == 0:
        print(" 没有足够特征用于检测。")
        return

    # 4️ 计算 Pearson 相关矩阵
    corr = X_valid.corr().abs()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

    # 找出高相关特征对
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_pairs = [(upper.index[i], upper.columns[j])
                       for i, j in zip(*np.where(upper > CORR_THRESHOLD)) if i != j]
    print(f"\n 检测到高相关特征对：{len(high_corr_pairs)} 对")

    # --- 构建高相关组 ---
    group_dict = {}
    group_id = 0
    for f1, f2 in high_corr_pairs:
        existing_groups = [gid for gid, members in group_dict.items() if f1 in members or f2 in members]
        if existing_groups:
            group_dict[existing_groups[0]].update([f1, f2])
        else:
            group_id += 1
            group_dict[group_id] = set([f1, f2])

    # 输出高相关组结果
    group_records = []
    for gid, members in group_dict.items():
        for f in members:
            group_records.append({"correlation_group": gid, "feature": f})

    group_df = pd.DataFrame(group_records)
    group_df.to_csv(OUT_DIR / "feature_correlation_groups.csv", index=False, encoding="utf-8")

    print(f" 高相关组数量：{len(group_dict)}（详见 feature_correlation_groups.csv）")

    # 5️ 计算方差膨胀因子（VIF）
    vif_df = pd.DataFrame()
    vif_df["feature"] = X_valid.columns
    vif_df["VIF"] = [variance_inflation_factor(X_valid.values, i)
                     for i in range(X_valid.shape[1])]

    # 合并组信息
    vif_merged = pd.merge(vif_df, group_df, on="feature", how="left").fillna({"correlation_group": 0}).astype({"correlation_group": int})
    vif_merged.to_csv(OUT_DIR / "feature_vif_with_groups.csv", index=False, encoding="utf-8")

    print("\n 共线性检测完成")
    print(f" - 输出文件：{OUT_DIR / 'feature_vif_with_groups.csv'}")
    print(f" - 相关组文件：{OUT_DIR / 'feature_correlation_groups.csv'}")
    print(f" - 热力图：{OUT_DIR / 'correlation_heatmap.png'}")

if __name__ == "__main__":
    main()

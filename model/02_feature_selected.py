#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_verify_selected_features_vif.py
===================================
目标：
验证人工筛选后的特征是否满足共线性要求（VIF < 10）。

输入：
  - data/processed_dataset.csv          清洗后的数据
  - 用户提供的特征列表

输出：
  - results/selected_features_vif.csv
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

# -----------------------
# 参数设置
# -----------------------
DATA_PATH = "data/processed_dataset.csv"
OUT_DIR = Path("results")
VIF_THRESHOLD = 10
TIME_COL = "timestamp"
TARGET_COL = "PM2_5"

#  手动在此处输入最终保留的特征列表
SELECTED_FEATURES = ["trip_count", "total_distance","avg_speed", "temperature" ,"PM2_5_lag_1","sin_hour","cos_hour","sin_day","cos_day","humidity","wind_speed","wind_direction","precipitation","weather_severity"]


# -----------------------
# 主流程
# -----------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1️ 读取数据
    df = pd.read_csv(DATA_PATH)
    print(f" 数据维度：{df.shape}")

    # 2️ 清洗数据，仅保留需要的特征
    if not SELECTED_FEATURES:
        print(" 请在脚本顶部填写 SELECTED_FEATURES 列表或导入特征文件。")
        return

    cols_to_use = [c for c in SELECTED_FEATURES if c in df.columns]
    X = df[cols_to_use].select_dtypes(include=[np.number]).dropna()

    print(f" 参与验证的特征数：{len(X.columns)}")

    # 3️ 计算VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # 4️ 输出结果
    out_path = OUT_DIR / "selected_features_vif.csv"
    vif_data.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n VIF结果：")
    print(vif_data)

    # 5️ 验证结论
    high_vif = vif_data[vif_data["VIF"] > VIF_THRESHOLD]
    if len(high_vif) == 0:
        print(f"\n 所有特征共线性良好（VIF < {VIF_THRESHOLD}）")
    else:
        print(f"\n 以下特征VIF过高（>{VIF_THRESHOLD}）：")
        print(high_vif)

    print(f"\n 结果已保存至：{out_path}")

if __name__ == "__main__":
    main()

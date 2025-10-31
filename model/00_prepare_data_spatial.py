#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_prepare_data_spatial.py
==========================
目标：
建立考虑空间结构（borough / borough_area / site_id）的基础建模数据。

逻辑：
  1. 读取原始数据；
  2. 删除 PM2.5 缺失行；
  3. 按 site_id + timestamp 排序；
  4. 汇总各站点的样本数与时间范围；
  5. 输出干净数据和空间元信息。

输出：
  data/processed_dataset.csv
  data/metadata.json
"""

import argparse
import json
from pathlib import Path
import pandas as pd


# -----------------------
# 基础列定义
# -----------------------
TIME_COL_NAME = "timestamp"
TARGET_COL = "PM2_5"
SITE_COL = "site_id"
BOROUGH_COL = "borough"
AREA_COL = "borough_area"


# -----------------------
# 主流程
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="PM2.5 数据预处理（含空间结构）")
    parser.add_argument("--data", type=str, default="final_complete_dataset.csv", help="原始数据路径")
    parser.add_argument("--outdir", type=str, default="data", help="输出目录")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # 1️ 读取数据
    df = pd.read_csv(args.data)
    required_cols = [TIME_COL_NAME, TARGET_COL, SITE_COL]
    for col in required_cols:
        assert col in df.columns, f"缺少关键列：{col}"

    # 2️ 删除 PM2.5 缺失值
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    after = len(df)
    print(f"已删除 {before - after} 条 PM2.5 缺失记录，保留 {after} 条有效样本。")

    # 3️ 时间格式化与排序
    df[TIME_COL_NAME] = pd.to_datetime(df[TIME_COL_NAME])
    df = df.sort_values([SITE_COL, TIME_COL_NAME]).reset_index(drop=True)

    # 4️ 按站点汇总统计
    site_summary = (
        df.groupby([SITE_COL])
        .agg(
            start_time=(TIME_COL_NAME, "min"),
            end_time=(TIME_COL_NAME, "max"),
            samples=("PM2_5", "count"),
        )
        .reset_index()
    )

    # 如果存在 borough / borough_area，则一并统计
    if BOROUGH_COL in df.columns or AREA_COL in df.columns:
        spatial_summary = (
            df.groupby([BOROUGH_COL, AREA_COL])[SITE_COL]
            .nunique()
            .reset_index()
            .rename(columns={SITE_COL: "num_sites"})
        )
    else:
        spatial_summary = None

    print("\n===== 各站点样本统计 =====")
    print(site_summary.head())

    # 5️ 输出结果文件
    processed_path = outdir / "processed_dataset.csv"
    meta_path = outdir / "metadata.json"
    site_summary_path = outdir / "site_summary.csv"

    df.to_csv(processed_path, index=False, encoding="utf-8")
    site_summary.to_csv(site_summary_path, index=False, encoding="utf-8")

    metadata = {
        "total_rows": int(len(df)),
        "num_sites": int(df[SITE_COL].nunique()),
        "num_boroughs": int(df[BOROUGH_COL].nunique()) if BOROUGH_COL in df.columns else None,
        "num_areas": int(df[AREA_COL].nunique()) if AREA_COL in df.columns else None,
        "columns": list(df.columns),
        "notes": {
            "target": TARGET_COL,
            "time_ordering": "按 site_id + timestamp 排序，确保时序一致。",
            "missing_PM2_5": "已删除所有 PM2.5 缺失样本。",
            "spatial_features": "borough, borough_area, site_id 均被保留以用于空间分析或嵌入。",
            "next_step": "传统模型将 borough/area/site_id 编码后作为类别特征；LSTM 模型按 site_id 分组建模。"
        },
    }

    if spatial_summary is not None:
        metadata["spatial_summary"] = spatial_summary.to_dict(orient="records")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n数据准备完成")
    print(f" - 清洗后数据：{processed_path}")
    print(f" - 站点统计表：{site_summary_path}")
    print(f" - 元信息文件：{meta_path}")
    print("\n下一步建议：对交通与气象特征进行全局重要性分析。")


if __name__ == "__main__":
    main()

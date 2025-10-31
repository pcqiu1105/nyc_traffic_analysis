import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE_FILE_PATH = '/mnt/d/nyc_traffic_analysis'

def process_air_quality_data(air_quality_raw):
    """
    处理原始空气质量数据，准备与交通数据合并
    """
    print("=== 处理空气质量数据 ===")
    
    if isinstance(air_quality_raw, str):
        air_df = pd.read_csv(air_quality_raw)
    else:
        air_df = air_quality_raw.copy()
    
    print(f"原始数据形状: {air_df.shape}")
    
    # 时间处理 - 移除时区信息以统一格式
    air_df['obs_time_utc'] = pd.to_datetime(air_df['ObservationTimeUTC'])
    air_df['obs_time_ny'] = air_df['obs_time_utc'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    air_df['hour'] = air_df['obs_time_ny'].dt.tz_localize(None)

    air_df = air_df.rename(columns={'Value': 'PM2_5'})
    
    # 过滤异常值
    initial_count = len(air_df)
    air_df = air_df[(air_df['PM2_5'] >= 0) & (air_df['PM2_5'] <= 200)]
    print(f"过滤后: {len(air_df)} 行")
    
    # 按站点和时间聚合
    air_hourly = air_df.groupby(['SiteID', 'hour']).agg({
        'PM2_5': 'mean'
    }).reset_index()
    
    print(f"聚合后形状: {air_hourly.shape}")
    print(f"时间范围: {air_hourly['hour'].min()} 到 {air_hourly['hour'].max()}")
    print(f"覆盖站点数: {air_hourly['SiteID'].nunique()}")
    
    print(f"PM2.5浓度统计:")
    print(f"  平均值: {air_hourly['PM2_5'].mean():.2f} μg/m³")
    print(f"  中位数: {air_hourly['PM2_5'].median():.2f} μg/m³")
    
    return air_hourly

def merge_traffic_air_quality(traffic_features, air_quality_hourly):
    """
    合并交通数据和空气质量数据
    """
    print("\n=== 合并交通和空气质量数据 ===")
    
    traffic_features['timestamp'] = pd.to_datetime(traffic_features['timestamp'])
    air_quality_hourly['hour'] = pd.to_datetime(air_quality_hourly['hour'])
    
    # 合并数据
    merged_df = pd.merge(
        traffic_features,
        air_quality_hourly,
        left_on=['site_id', 'timestamp'],
        right_on=['SiteID', 'hour'],
        how='left'
    )
    
    merged_df = merged_df.drop(['SiteID', 'hour'], axis=1, errors='ignore')
    
    # 数据合并统计
    total_records = len(merged_df)
    records_with_pm25 = merged_df['PM2_5'].notna().sum()
    coverage_rate = records_with_pm25 / total_records * 100
    
    print(f"合并后数据形状: {merged_df.shape}")
    print(f"包含PM2.5数据的记录: {records_with_pm25}/{total_records} ({coverage_rate:.1f}%)")
    
    return merged_df

def analyze_merged_data(merged_df):
    """
    分析合并后的数据质量
    """
    print("\n=== 合并数据质量分析 ===")
    
    print("数据概览:")
    print(f"时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
    print(f"覆盖站点数: {merged_df['site_id'].nunique()}")
    
    # 数据完整性分析
    completeness = pd.DataFrame({
        '总记录数': len(merged_df),
        '有PM2.5数据': merged_df['PM2_5'].notna().sum(),
        '有交通数据': (merged_df['trip_count'] > 0).sum(),
        '完整记录': merged_df[(merged_df['PM2_5'].notna()) & (merged_df['trip_count'] > 0)].shape[0]
    }, index=[0])
    
    print("\n数据完整性:")
    print(completeness)
    
    return completeness

def create_time_series_features(merged_df):
    """
    创建时间序列特征
    """
    print("\n=== 创建时间序列特征 ===")
    
    df = merged_df.copy()
    
    # 时间特征
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['month'] = df['timestamp'].dt.month
    
    # 时间周期特征
    df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # 按站点排序数据
    df = df.sort_values(['site_id', 'timestamp'])
    
    # 创建滞后特征
    lag_periods = [1, 3, 6]
    for lag in lag_periods:
        df[f'PM2_5_lag_{lag}'] = df.groupby('site_id')['PM2_5'].shift(lag)
        df[f'traffic_lag_{lag}'] = df.groupby('site_id')['trip_count'].shift(lag)
    
    # 移动平均特征
    for window in [3, 6, 12]:
        df[f'PM2_5_ma_{window}'] = df.groupby('site_id')['PM2_5'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'traffic_ma_{window}'] = df.groupby('site_id')['trip_count'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    print(f"添加时间序列特征后数据形状: {df.shape}")
    
    return df

def correct_traffic_site_ids(traffic_features):
    """
    修正交通数据中的站点ID以匹配空气质量数据
    """
    site_id_correction = {
        '36005NY11534': '360050080',
        '36005NY11790': '360050110', 
        '36047NY07974': '360470052',
        '36061NY08552': '360610115',
        '36061NY08653': '360610135',
        '36061NY12380': '360850111',
        '36081NY07615': '360810120',
        '36081NY08198': '360810124',
        '36085NY03820': '360470118',
        '36085NY04805': '360050112'
    }
    
    traffic_features_corrected = traffic_features.copy()
    traffic_features_corrected['site_id'] = traffic_features_corrected['site_id'].map(
        lambda x: site_id_correction.get(x, x)
    )
    
    print("站点ID修正完成")
    print(f"修正后唯一站点数: {traffic_features_corrected['site_id'].nunique()}")
    
    return traffic_features_corrected

def save_final_dataset(merged_df, output_path):
    """
    保存最终数据集
    """
    print(f"\n=== 保存最终数据集 ===")
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    merged_df.to_csv(output_path, index=False)
    
    print(f"数据集保存至: {output_path}")
    print(f"最终数据形状: {merged_df.shape}")
    
    complete_records = merged_df[merged_df['PM2_5'].notna()].shape[0]
    print(f"完整记录数: {complete_records}")

def main():
    """
    主函数：整合交通和空气质量数据
    """
    print("开始整合交通和空气质量数据...")
    
    # 读取数据
    traffic_features_tmp = pd.read_csv(f'{BASE_FILE_PATH}/data/processed/traffic_features_by_borough.csv')
    traffic_features = correct_traffic_site_ids(traffic_features_tmp)
    air_quality_raw = f'{BASE_FILE_PATH}/data/raw/AirQuality2401_2403.csv'
    
    # 处理空气质量数据
    air_quality_hourly = process_air_quality_data(air_quality_raw)
    
    # 合并数据
    merged_data = merge_traffic_air_quality(traffic_features, air_quality_hourly)
    
    # 数据分析
    completeness = analyze_merged_data(merged_data)
    
    # 创建时间序列特征
    final_data = create_time_series_features(merged_data)
    
    # 保存最终数据集
    output_path = f'{BASE_FILE_PATH}/data/processed/final_traffic_pm25_dataset.csv'
    save_final_dataset(final_data, output_path)

    print(f"\n=== 整合完成 ===")
    print(f"最终数据集包含 {len(final_data)} 条记录")
    print(f"数据覆盖 {final_data['site_id'].nunique()} 个站点")
    
    return final_data

if __name__ == "__main__":
    final_dataset = main()
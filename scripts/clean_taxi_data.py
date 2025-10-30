import pandas as pd
import numpy as np

BASE_FILE_PATH = '/mnt/d/nyc_traffic_analysis'

def clean_taxi_data(taxi_df, zones_df):
    """
    彻底清洗出租车数据，过滤异常值并按行政区聚合
    """
    print("=== 开始出租车数据清洗 ===")
    print(f"原始数据行数: {len(taxi_df):,}")
    
    df = taxi_df.copy()
    
    def filter_by_time_range(df, start_date='2024-01-01', end_date='2024-04-01'):
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        mask = (df['tpep_pickup_datetime'] >= start_date) & (df['tpep_pickup_datetime'] < end_date)
        return df[mask]

    df = filter_by_time_range(df)

    print(f"时间范围: {df['tpep_pickup_datetime'].min()} 到 {df['tpep_pickup_datetime'].max()}")
    
    # 处理缺失值
    initial_rows = len(df)
    essential_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'trip_distance']
    df = df.dropna(subset=essential_cols)
    print(f"移除关键字段空值后: {len(df):,} 行")
    
    passenger_median = df['passenger_count'].median()
    df['passenger_count'] = df['passenger_count'].fillna(passenger_median)
    
    # 计算行程时间
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    
    # 异常值过滤
    distance_mask = (df['trip_distance'] >= 0.1) & (df['trip_distance'] <= 30)
    before_distance = len(df)
    df = df[distance_mask]
    print(f"距离过滤: 保留 {len(df):,} 行")
    
    time_mask = (df['trip_duration'] >= 0.0167) & (df['trip_duration'] <= 3)
    before_time = len(df)
    df = df[time_mask]
    print(f"时间过滤: 保留 {len(df):,} 行")
    
    # 乘客数量修正和过滤
    df.loc[df['passenger_count'] == 0, 'passenger_count'] = 1
    passenger_mask = (df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)
    before_passenger = len(df)
    df = df[passenger_mask]
    print(f"乘客过滤: 保留 {len(df):,} 行")
    
    # 计算车速并过滤
    df['avg_speed'] = df['trip_distance'] / df['trip_duration']
    speed_mask = (df['avg_speed'] >= 1) & (df['avg_speed'] <= 60)
    before_speed = len(df)
    df = df[speed_mask]
    print(f"车速过滤: 保留 {len(df):,} 行")
    
    # 添加行政区信息
    if zones_df is not None:
        zones_borough = zones_df[['LocationID', 'Borough']].drop_duplicates()

        # 添加上车行政区
        df = df.merge(zones_borough.rename(columns={'LocationID': 'PULocationID', 'Borough': 'PU_Borough'}), 
                    on='PULocationID', how='left')
        # 添加下车行政区  
        df = df.merge(zones_borough.rename(columns={'LocationID': 'DOLocationID', 'Borough': 'DO_Borough'}), 
                    on='DOLocationID', how='left')
        
        # 使用上车行政区作为主要行政区
        df['Borough'] = df['PU_Borough']
        
        print(f"行政区分布: {df['PU_Borough'].value_counts().to_dict()}")
    else:
        print("警告：未提供区域信息文件")
        df['Borough'] = 'Unknown'
        df['PU_Borough'] = 'Unknown'
        df['DO_Borough'] = 'Unknown'
        
    # 提取时间信息
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('H')
    df['hour_of_day'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # 数据质量报告
    print(f"最终保留数据: {len(df):,} 行 ({len(df)/initial_rows*100:.1f}% 原始数据)")
    print(f"平均行程距离: {df['trip_distance'].mean():.2f} 英里")
    print(f"平均行程时间: {df['trip_duration'].mean():.2f} 小时")
    print(f"平均车速: {df['avg_speed'].mean():.2f} mph")
    
    return df

def aggregate_taxi_by_borough_hour(cleaned_taxi_df):
    """
    按行政区和小时聚合出租车数据
    """
    print("\n=== 按行政区-小时聚合数据 ===")
    
    if 'Borough' not in cleaned_taxi_df.columns:
        print("错误：数据中缺少行政区信息")
        return None
    
    # 按行政区和小时聚合
    borough_hourly = cleaned_taxi_df.groupby(['Borough', 'pickup_hour']).agg({
        'trip_distance': ['sum', 'mean'],   # 总行驶里程和平均里程
        'PULocationID': 'count',            # 行程数量
        'passenger_count': ['sum', 'mean'], # 总乘客数和平均乘客
        'avg_speed': 'mean',                # 平均车速
        'trip_duration': 'mean',            # 平均行程时长
        'total_amount': ['sum', 'mean']     # 总金额和平均金额
    }).reset_index()
    
    # 重命名列
    borough_hourly = borough_hourly.rename(columns={
        'PULocationID': 'trip_count',
        'pickup_hour': 'hour'
    })
    
    print(f"聚合后数据形状: {borough_hourly.shape}")
    print(f"时间范围: {borough_hourly['hour'].min()} 到 {borough_hourly['hour'].max()}")
    
    return borough_hourly

if __name__ == "__main__":
    # 读取数据
    # taxi_df = pd.read_parquet(f'{BASE_FILE_PATH}/data/raw/yellow_tripdata_2024-01.parquet')
    taxi_df = pd.read_parquet(f'{BASE_FILE_PATH}/data/raw/combined_data.parquet')
    zones_df = pd.read_csv(f'{BASE_FILE_PATH}/data/raw/taxi_zone_lookup.csv')

    
    # 清洗数据
    cleaned_taxi = clean_taxi_data(taxi_df, zones_df)
    
    # 按行政区聚合
    borough_hourly = aggregate_taxi_by_borough_hour(cleaned_taxi)
    
    # 保存清洗后的数据
    cleaned_taxi.to_csv(f'{BASE_FILE_PATH}/data/processed/cleaned_taxi_data_2024_01_03.csv', index=False)
    borough_hourly.to_csv(f'{BASE_FILE_PATH}/data/processed/taxi_borough_hourly_2024_01_03.csv', index=False)
    
    print(f"\n清洗完成！")
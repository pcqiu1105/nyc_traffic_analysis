import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

BASE_FILE_PATH = '/mnt/d/nyc_traffic_analysis'

def prepare_spatial_infrastructure(air_quality_sites, buffer_distances=[300, 500, 1000]):
    """
    步骤1: 建立空间基础设施 - 为每个空气质量站点创建缓冲区
    """
    print("=== 建立空间基础设施 ===")
    air_quality_sites = air_quality_sites[air_quality_sites['SiteID'].notna() & (air_quality_sites['SiteID'] != '')] 
    
    geometry = [Point(lon, lat) for lon, lat in zip(air_quality_sites['Longitude'], air_quality_sites['Latitude'])]
    air_sites_gdf = gpd.GeoDataFrame(
        air_quality_sites,
        geometry=geometry,
        crs="EPSG:4326"
    )
    
    air_sites_gdf = air_sites_gdf.to_crs("EPSG:32618")
    
    print(f"为 {len(air_sites_gdf)} 个空气质量站点创建了缓冲区")
    print(f"缓冲区半径: {buffer_distances} 米")
    
    return air_sites_gdf

def prepare_taxi_data(taxi_df, shapefile_path):
    """
    步骤2: 准备出租车数据 - 转换为空间数据格式
    """
    print("\n=== 准备出租车数据 ===")
    
    zones_gdf = gpd.read_file(shapefile_path)
    
    print("计算区域质心...")
    zones_gdf['centroid'] = zones_gdf.geometry.centroid
    zones_gdf_wgs84 = zones_gdf.to_crs("EPSG:4326")
    zones_gdf_wgs84['centroid_wgs84'] = zones_gdf_wgs84.geometry.centroid
    
    zone_coords = {}
    for _, zone in zones_gdf_wgs84.iterrows():
        location_id = zone['LocationID']
        centroid = zone['centroid_wgs84']
        zone_coords[location_id] = (centroid.x, centroid.y)
    
    print(f"成功计算 {len(zone_coords)} 个区域的中心坐标")

    df = taxi_df.copy()
    
    df['pickup_lon'] = df['PULocationID'].map(lambda x: zone_coords.get(x, (0, 0))[0])
    df['pickup_lat'] = df['PULocationID'].map(lambda x: zone_coords.get(x, (0, 0))[1])
    
    original_count = len(df)
    valid_coords_mask = (df['pickup_lon'] != 0) & (df['pickup_lat'] != 0)
    df_valid = df[valid_coords_mask]
    
    print(f"坐标映射结果: {len(df_valid)}/{original_count} 行成功映射")
    
    geometry = [Point(lon, lat) for lon, lat in zip(df_valid['pickup_lon'], df_valid['pickup_lat'])]
    taxi_gdf = gpd.GeoDataFrame(df_valid, geometry=geometry, crs="EPSG:4326")
    taxi_gdf_utm = taxi_gdf.to_crs("EPSG:32618")
    
    print(f"出租车数据空间化完成: {len(taxi_gdf_utm)} 行")
    print(f"覆盖行政区: {taxi_gdf_utm['Borough'].nunique()} 个")
    
    if 'pickup_hour' not in taxi_gdf_utm.columns:
        taxi_gdf_utm['pickup_hour'] = taxi_gdf_utm['tpep_pickup_datetime'].dt.floor('h')
    
    print(f"时间范围: {taxi_gdf_utm['pickup_hour'].min()} 到 {taxi_gdf_utm['pickup_hour'].max()}")
    
    return taxi_gdf_utm

def fill_missing_records(results_df, site_to_borough):
    """填充缺失的时间点-站点组合"""
    print("填充缺失记录...")
    
    all_timestamps = results_df['timestamp'].unique()
    all_sites = list(site_to_borough.keys())
    
    from itertools import product
    full_index = pd.MultiIndex.from_product([all_timestamps, all_sites], 
                                        names=['timestamp', 'site_id'])
    
    results_df_full = results_df.set_index(['timestamp', 'site_id']).reindex(full_index).reset_index()
    
    results_df_full['borough'] = results_df_full['site_id'].map(site_to_borough)
    
    numeric_cols = ['trip_count', 'total_distance', 'avg_trip_distance', 'avg_speed', 
                'total_passengers', 'total_revenue', 'avg_duration']
    for col in numeric_cols:
        if col in results_df_full.columns:
            results_df_full[col] = results_df_full[col].fillna(0)
    
    filled_count = len(results_df_full) - len(results_df)
    print(f"填充了 {filled_count} 条缺失记录")
    
    return results_df_full

def borough_based_aggregation_complete(taxi_df, air_sites_gdf):
    """
    完整的基于行政区的聚合方法 - 确保100%覆盖
    """
    print("=== 使用行政区聚合方法（确保完整覆盖） ===")
    
    site_to_borough = {
        '36061NY08454': 'Manhattan', 
        '36061NY08552': 'Manhattan',
        '36061NY08653': 'Manhattan', 
        '36061NY09734': 'Manhattan',
        '36061NY09929': 'Manhattan', 
        '36061NY10130': 'Manhattan',
        '36061NY12380': 'Manhattan',
        '36005NY11534': 'Bronx', 
        '36005NY11790': 'Bronx', 
        '36005NY12387': 'Bronx',
        '36047NY07974': 'Brooklyn',
        '36081NY07615': 'Queens', 
        '36081NY08198': 'Queens', 
        '36081NY09285': 'Queens',
        '36085NY03820': 'Staten Island', 
        '36085NY04805': 'Staten Island'
    }
    
    all_air_sites = set(air_sites_gdf['SiteID'])
    mapped_sites = set(site_to_borough.keys())
    
    unmapped_sites = all_air_sites - mapped_sites
    if unmapped_sites:
        print(f"警告: 以下站点未在映射中: {unmapped_sites}")
        for site_id in unmapped_sites:
            site_to_borough[site_id] = 'Manhattan'
    
    print(f"站点-行政区映射: {len(site_to_borough)} 个站点")
    
    if 'Borough' not in taxi_df.columns:
        print("错误: 出租车数据缺少行政区信息")
        return None
    
    print("按行政区和小时聚合交通数据...")
    borough_hourly = taxi_df.groupby(['Borough', 'pickup_hour']).agg({
        'trip_distance': ['count', 'sum', 'mean'],
        'avg_speed': 'mean',
        'passenger_count': 'sum',
        'total_amount': 'sum',
        'trip_duration': 'mean'
    }).reset_index()
    
    borough_hourly.columns = [
        'Borough', 'timestamp', 'trip_count', 'total_distance', 'avg_trip_distance',
        'avg_speed', 'total_passengers', 'total_revenue', 'avg_duration'
    ]
    borough_hourly['timestamp'] = pd.to_datetime(borough_hourly['timestamp'])

    print(f"行政区聚合完成: {len(borough_hourly)} 条记录")
    
    print("将交通数据分配到各站点...")
    results = []
    
    for _, row in borough_hourly.iterrows():
        borough = row['Borough']
        sites_in_borough = [site for site, b in site_to_borough.items() if b == borough]
        
        if sites_in_borough:
            num_sites = len(sites_in_borough)
            
            for site_id in sites_in_borough:
                results.append({
                    'timestamp': row['timestamp'],
                    'site_id': site_id,
                    'borough': borough,
                    'trip_count': row['trip_count'] / num_sites,
                    'total_distance': row['total_distance'] / num_sites,
                    'avg_trip_distance': row['avg_trip_distance'],
                    'avg_speed': row['avg_speed'],
                    'total_passengers': row['total_passengers'] / num_sites,
                    'total_revenue': row['total_revenue'] / num_sites,
                    'avg_duration': row['avg_duration']
                })
        else:
            print(f"警告: 行政区 {borough} 没有对应的空气质量站点")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n=== 数据质量报告 ===")
    print(f"总记录数: {len(results_df):,}")
    print(f"覆盖站点数: {results_df['site_id'].nunique()}")
    print(f"时间范围: {results_df['timestamp'].min()} 到 {results_df['timestamp'].max()}")
    
    # 检查完整性并填充缺失记录
    expected_records = len(site_to_borough) * results_df['timestamp'].nunique()
    actual_records = len(results_df)
    coverage_rate = actual_records / expected_records * 100
    
    print(f"覆盖完整性: {actual_records}/{expected_records} ({coverage_rate:.1f}%)")
    
    if coverage_rate < 100:
        print("覆盖不完整，填充缺失记录...")
        results_df = fill_missing_records(results_df, site_to_borough)

    return results_df

def create_final_features_complete(aggregated_df):
    """
    为完整的聚合数据创建最终特征
    """
    print("\n=== 创建最终特征表 ===")
    
    df = aggregated_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    borough_areas = {
        'Manhattan': 59.1,
        'Brooklyn': 183.4,
        'Queens': 281.5,
        'Bronx': 109.0,
        'Staten Island': 151.5,
        'EWR': 30.0
    }
    
    df['borough_area'] = df['borough'].map(borough_areas)
    df['trip_density'] = df['trip_count'] / df['borough_area']
    df['passenger_density'] = df['total_passengers'] / df['borough_area']
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 9, 17, 18, 19])
    
    df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df = df.sort_values(['site_id', 'timestamp'])
    
    print("创建滞后特征...")
    lag_periods = [1, 3, 6]
    
    for lag in lag_periods:
        df[f'trip_count_lag_{lag}'] = df.groupby('site_id')['trip_count'].shift(lag)
        df[f'avg_speed_lag_{lag}'] = df.groupby('site_id')['avg_speed'].shift(lag)
        df[f'passenger_density_lag_{lag}'] = df.groupby('site_id')['passenger_density'].shift(lag)
    
    print("创建移动平均特征...")
    for window in [3, 6, 12]:
        df[f'trip_count_ma_{window}'] = df.groupby('site_id')['trip_count'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'avg_speed_ma_{window}'] = df.groupby('site_id')['avg_speed'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    print("处理缺失值...")
    lag_cols = [col for col in df.columns if 'lag' in col]
    for col in lag_cols:
        df[col] = df.groupby('site_id')[col].fillna(method='ffill')
    
    ma_cols = [col for col in df.columns if 'ma_' in col]
    for col in ma_cols:
        df[col] = df.groupby('site_id')[col].transform(
            lambda x: x.fillna(x.mean())
        )
    
    df = df.fillna(0)
    
    print(f"最终特征表形状: {df.shape}")
    
    return df

def main():
    """
    主函数：执行完整的出租车数据缓冲区聚合流程
    """
    print("开始出租车数据缓冲区聚合流程...")
    
    air_quality_sites = pd.read_csv(f'{BASE_FILE_PATH}/data/raw/station-info.csv')
    taxi_df = pd.read_csv(f'{BASE_FILE_PATH}/data/processed/cleaned_taxi_data_2024_01_03.csv')
    shapefile_path = f'{BASE_FILE_PATH}/data/raw/taxi_zones/taxi_zones.shp'
    
    taxi_df['tpep_pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'])
    
    air_sites_gdf = prepare_spatial_infrastructure(air_quality_sites)
    
    taxi_gdf = prepare_taxi_data(taxi_df, shapefile_path)
    
    aggregated_results = borough_based_aggregation_complete(taxi_gdf, air_sites_gdf)
    
    if len(aggregated_results) > 0:
        traffic_features = create_final_features_complete(aggregated_results)
        
        output_dir = f'{BASE_FILE_PATH}/data/processed/'
        traffic_features.to_csv(f'{output_dir}traffic_features_by_borough.csv', index=False)
        
        print(f"\n=== 流程完成 ===")
        print(f"特征表保存至: traffic_features_by_borough.csv")
        print(f"记录数: {len(traffic_features)}")
        
        print(f"\n统计信息:")
        print(f"总记录数: {len(traffic_features)}")
        print(f"时间范围: {traffic_features['timestamp'].min()} 到 {traffic_features['timestamp'].max()}")
        print(f"覆盖站点数: {traffic_features['site_id'].nunique()}")
        
        return traffic_features
    else:
        print("错误: 空间聚合未产生任何结果，无法创建特征表")
        return None

if __name__ == "__main__":
    traffic_features = main()
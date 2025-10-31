import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import json

BASE_FILE_PATH = '/mnt/d/nyc_traffic_analysis'

def load_osm_data_geopandas(shapefile_path):
    """
    读取OSM道路数据
    """
    print("=== 读取OSM道路数据 ===")
    
    try:
        roads_gdf = gpd.read_file(shapefile_path)
        print(f"成功读取OSM数据: {len(roads_gdf)} 条道路")
        
        return roads_gdf
        
    except Exception as e:
        print(f"OSM数据读取失败: {e}")
        return None

def load_manual_weather_data(json_file_path):
    """
    加载手动下载的气象数据
    """
    print("=== 从JSON文件加载气象数据 ===")
    
    try:
        with open(json_file_path, 'r') as f:
            weather_data = json.load(f)
        
        # 提取小时数据
        hourly_data = weather_data['hourly']
        
        # 转换为DataFrame
        weather_df = pd.DataFrame({
            'timestamp': hourly_data['time'],
            'temperature': hourly_data['temperature_2m'],
            'humidity': hourly_data['relative_humidity_2m'],
            'pressure': hourly_data['pressure_msl'],
            'wind_speed': hourly_data['wind_speed_10m'],
            'wind_direction': hourly_data['wind_direction_10m'],
            'precipitation': hourly_data['precipitation']
        })
        
        # 转换时间列
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        
        # 单位转换：风速从km/h转为m/s
        weather_df['wind_speed'] = weather_df['wind_speed'] / 3.6
        
        print(f"气象数据处理完成: {len(weather_df)} 小时记录")
        print(f"时间范围: {weather_df['timestamp'].min()} 到 {weather_df['timestamp'].max()}")
        
        return weather_df
        
    except Exception as e:
        print(f"JSON气象数据加载失败: {e}")

def improved_calculate_road_features(air_sites_gdf, roads_gdf, buffer_distances=[300, 500, 1000]):
    """
    改进版：为每个空气质量站点计算道路特征
    """
    print("\n=== 计算站点道路特征 ===")
    
    # 统一坐标系
    target_crs = "EPSG:32618"
    air_sites_utm = air_sites_gdf.to_crs(target_crs)
    roads_utm = roads_gdf.to_crs(target_crs)
    
    # 确定长度列
    length_column = None
    possible_length_cols = ['length', 'Length', 'LENGTH', 'Shape_Leng', 'SHAPE_Leng']
    
    for col in possible_length_cols:
        if col in roads_utm.columns:
            length_column = col
            break
    
    if length_column is None:
        roads_utm = roads_utm.copy()
        roads_utm['calc_length'] = roads_utm.geometry.length
        length_column = 'calc_length'
    
    print(f"使用长度列: {length_column}")
    
    # 构建空间索引以提高性能
    roads_sindex = roads_utm.sindex
    
    road_features = []
    
    for i, site in air_sites_utm.iterrows():
        site_id = site['SiteID']
        if pd.isna(site_id):
            continue
            
        print(f"处理站点 {i+1}/{len(air_sites_utm)}: {site_id}")
        
        site_road_features = {'site_id': site_id}
        
        for buffer_dist in buffer_distances:
            buffer_geom = site.geometry.buffer(buffer_dist)
            buffer_area = buffer_geom.area
            
            # 使用空间索引快速查询
            possible_indices = list(roads_sindex.intersection(buffer_geom.bounds))
            
            if possible_indices:
                possible_roads = roads_utm.iloc[possible_indices]
                roads_in_buffer = possible_roads[possible_roads.intersects(buffer_geom)]
                
                if len(roads_in_buffer) > 0:
                    total_road_length = roads_in_buffer[length_column].sum()
                    road_density = total_road_length / (buffer_area / 1000000)
                    
                    # 道路类型统计
                    if 'fclass' in roads_in_buffer.columns:
                        major_road_types = ['motorway', 'trunk', 'primary', 'secondary']
                        major_roads = roads_in_buffer[roads_in_buffer['fclass'].isin(major_road_types)]
                        major_road_ratio = major_roads[length_column].sum() / total_road_length if total_road_length > 0 else 0
                    else:
                        major_road_ratio = 0
                    
                    intersection_density = len(roads_in_buffer) / (buffer_area / 1000000)
                    
                else:
                    total_road_length = 0
                    road_density = 0
                    major_road_ratio = 0
                    intersection_density = 0
            else:
                total_road_length = 0
                road_density = 0
                major_road_ratio = 0
                intersection_density = 0
            
            # 存储特征
            site_road_features[f'road_density_{buffer_dist}m'] = road_density
            site_road_features[f'total_road_length_{buffer_dist}m'] = total_road_length
            site_road_features[f'major_road_ratio_{buffer_dist}m'] = major_road_ratio
            site_road_features[f'intersection_density_{buffer_dist}m'] = intersection_density
        
        road_features.append(site_road_features)
    
    road_features_df = pd.DataFrame(road_features)
    print(f"道路特征计算完成: {len(road_features_df)} 个站点")
    
    return road_features_df

def validate_data_integration(final_df):
    """
    验证数据整合质量
    """
    print("\n=== 数据整合质量验证 ===")
    
    print(f"最终数据集形状: {final_df.shape}")
    print(f"时间范围: {final_df['timestamp'].min()} 到 {final_df['timestamp'].max()}")
    print(f"覆盖站点数: {final_df['site_id'].nunique()}")
    
    # 缺失值分析
    missing_stats = final_df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]
    
    if len(missing_stats) > 0:
        print("\n缺失值统计:")
        for col, count in missing_stats.items():
            percent = count / len(final_df) * 100
            print(f"  {col}: {count} 行 ({percent:.1f}%)")
    else:
        print("无缺失值!")
    
    return len(missing_stats) == 0

def create_weather_features(weather_df):
    """
    创建气象衍生特征
    """
    print("\n=== 创建气象衍生特征 ===")
    
    df = weather_df.copy()
    
    # 风向分类
    def wind_direction_category(wind_deg):
        if wind_deg >= 337.5 or wind_deg < 22.5:
            return 'N'
        elif 22.5 <= wind_deg < 67.5:
            return 'NE'
        elif 67.5 <= wind_deg < 112.5:
            return 'E'
        elif 112.5 <= wind_deg < 157.5:
            return 'SE'
        elif 157.5 <= wind_deg < 202.5:
            return 'S'
        elif 202.5 <= wind_deg < 247.5:
            return 'SW'
        elif 247.5 <= wind_deg < 292.5:
            return 'W'
        else:
            return 'NW'
    
    df['wind_direction_cat'] = df['wind_direction'].apply(wind_direction_category)
    
    # 风速分类
    df['wind_speed_cat'] = pd.cut(df['wind_speed'], 
                                 bins=[0, 2, 5, 10, float('inf')],
                                 labels=['Calm', 'Light', 'Moderate', 'Strong'])
    
    # 温度分类
    df['temperature_cat'] = pd.cut(df['temperature'],
                                  bins=[-float('inf'), 0, 5, 10, float('inf')],
                                  labels=['Freezing', 'Cold', 'Cool', 'Mild'])
    
    # 气象条件综合指标
    df['weather_severity'] = (
        (df['wind_speed'] > 8).astype(int) +
        (df['precipitation'] > 1).astype(int) +
        (df['temperature'] < 0).astype(int)
    )
    
    # 时间滞后特征
    for lag in [1, 3, 6]:
        df[f'temperature_lag_{lag}'] = df['temperature'].shift(lag)
        df[f'wind_speed_lag_{lag}'] = df['wind_speed'].shift(lag)
    
    print(f"气象特征创建完成: {df.shape[1]} 个特征")
    
    return df

def integrate_all_data_sources_fixed():
    """
    整合所有数据源
    """
    print("开始整合所有数据源...")
    
    # 1. 读取现有数据
    print("1. 读取现有数据...")
    traffic_pm25_df = pd.read_csv(f'{BASE_FILE_PATH}/data/processed/final_traffic_pm25_dataset.csv')
    traffic_pm25_df['timestamp'] = pd.to_datetime(traffic_pm25_df['timestamp'])
    
    air_sites = pd.read_csv(f'{BASE_FILE_PATH}/data/raw/station-info.csv')
    air_sites = air_sites[air_sites['SiteID'].notna() & (air_sites['SiteID'] != '')] 

    # 创建空气质量站点的GeoDataFrame
    lat_col, lon_col = None, None
    for col in air_sites.columns:
        if col.lower() in ['latitude', 'lat']:
            lat_col = col
        elif col.lower() in ['longitude', 'lon']:
            lon_col = col
    
    if lat_col and lon_col:
        geometry = [Point(lon, lat) for lon, lat in zip(air_sites[lon_col], air_sites[lat_col])]
        air_sites_gdf = gpd.GeoDataFrame(air_sites, geometry=geometry, crs="EPSG:4326")
        print(f"创建了 {len(air_sites_gdf)} 个站点的空间数据")
    else:
        print("错误: 无法找到经纬度列")
        return None
    
    # 2. 获取OSM道路数据
    print("2. 获取OSM道路数据...")
    osm_shapefile_path = f'{BASE_FILE_PATH}/data/raw/osm/gis_osm_roads_free_1.shp'
    roads_gdf = load_osm_data_geopandas(osm_shapefile_path)
    
    if roads_gdf is not None:
        road_features_df = improved_calculate_road_features(air_sites_gdf, roads_gdf)
    else:
        print("OSM数据加载失败，跳过道路特征")
        road_features_df = pd.DataFrame({'site_id': air_sites_gdf['SiteID'].unique()})
    
    # 3. 获取气象数据
    print("3. 获取气象数据...")
    weather_csv_path = f'{BASE_FILE_PATH}/data/raw/weather_nyc_2024_0103.json'
    weather_df = load_manual_weather_data(weather_csv_path)
    weather_features_df = create_weather_features(weather_df)
    
    # 4. 合并所有数据
    print("4. 合并所有数据...")
    
    # 合并道路特征
    if not road_features_df.empty:
        final_df = pd.merge(traffic_pm25_df, road_features_df, on='site_id', how='left')
        print(f"合并道路特征后: {final_df.shape}")
    else:
        final_df = traffic_pm25_df.copy()
    
    # 合并气象特征
    final_df = pd.merge(final_df, weather_features_df, on='timestamp', how='left')
    print(f"合并气象特征后: {final_df.shape}")
    
    # 5. 数据验证
    is_valid = validate_data_integration(final_df)
    
    # 6. 保存最终数据集
    print("5. 保存最终数据集...")
    output_path = f'{BASE_FILE_PATH}/outputs/final_complete_dataset.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"\n=== 数据整合完成 ===")
    print(f"数据集保存至: {output_path}")
    
    return final_df

if __name__ == "__main__":
    final_dataset = integrate_all_data_sources_fixed()

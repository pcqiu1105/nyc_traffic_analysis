import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import json
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# 添加模块路径到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merge_osm_and_climate import (
    load_osm_data_geopandas,
    load_manual_weather_data,
    improved_calculate_road_features,
    validate_data_integration,
    create_weather_features,
    integrate_all_data_sources_fixed
)


class TestOSMDataLoading:
    """测试OSM道路数据加载功能"""
    
    def create_sample_osm_data(self):
        """创建测试用的OSM道路数据"""
        # 创建一些简单的线几何图形作为道路
        lines = [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
            LineString([(0, 1), (1, 0)]),
            LineString([(2, 0), (3, 1)])
        ]
        
        data = {
            'geometry': lines,
            'fclass': ['motorway', 'primary', 'secondary', 'residential'],
            'name': ['Road A', 'Road B', 'Road C', 'Road D'],
            'length': [1414.2, 1414.2, 1414.2, 1414.2]  # 近似长度
        }
        
        return gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    def test_load_osm_data_geopandas_success(self):
        """测试成功加载OSM数据"""
        # 创建临时shapefile
        gdf = self.create_sample_osm_data()
        with tempfile.NamedTemporaryFile(suffix='.shp', delete=False) as f:
            temp_file_path = f.name
        
        try:
            gdf.to_file(temp_file_path)
            
            result = load_osm_data_geopandas(temp_file_path)
            
            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 4
            assert 'geometry' in result.columns
        finally:
            # 清理临时文件
            for ext in ['.shp', '.shx', '.dbf', '.prj']:
                file_path = temp_file_path.replace('.shp', ext)
                if os.path.exists(file_path):
                    os.unlink(file_path)
    
    def test_load_osm_data_geopandas_file_not_found(self):
        """测试文件不存在的情况"""
        result = load_osm_data_geopandas('/nonexistent/path/file.shp')
        
        assert result is None
    
    @patch('geopandas.read_file')
    def test_load_osm_data_geopandas_exception_handling(self, mock_read_file):
        """测试异常处理"""
        mock_read_file.side_effect = Exception("Test error")
        
        result = load_osm_data_geopandas('/some/path/file.shp')
        
        assert result is None


class TestWeatherDataLoading:
    """测试气象数据加载功能"""
    
    def create_sample_weather_json(self):
        """创建测试用的气象JSON数据"""
        base_time = "2024-01-01T00:00"
        times = [pd.Timestamp(base_time) + pd.Timedelta(hours=i) for i in range(24)]
        
        weather_data = {
            'hourly': {
                'time': [t.isoformat() for t in times],
                'temperature_2m': np.random.uniform(-5, 25, 24).tolist(),
                'relative_humidity_2m': np.random.randint(30, 90, 24).tolist(),
                'pressure_msl': np.random.uniform(1010, 1025, 24).tolist(),
                'wind_speed_10m': np.random.uniform(0, 15, 24).tolist(),
                'wind_direction_10m': np.random.uniform(0, 360, 24).tolist(),
                'precipitation': np.random.uniform(0, 5, 24).tolist()
            }
        }
        
        return weather_data
    
    def test_load_manual_weather_data_success(self):
        """测试成功加载气象数据"""
        weather_data = self.create_sample_weather_json()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(weather_data, f)
            temp_file_path = f.name
        
        try:
            result = load_manual_weather_data(temp_file_path)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 24
            assert 'timestamp' in result.columns
            assert 'temperature' in result.columns
            assert 'wind_speed' in result.columns
            
            # 检查风速单位转换 (km/h to m/s)
            assert result['wind_speed'].max() < 5  # 转换后应该在合理范围内
            
            # 检查时间转换
            assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
        finally:
            os.unlink(temp_file_path)
    
    def test_load_manual_weather_data_file_not_found(self):
        """测试气象数据文件不存在的情况"""
        result = load_manual_weather_data('/nonexistent/path/weather.json')
        
        assert result is None
    
    def test_load_manual_weather_data_invalid_json(self):
        """测试无效JSON文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file_path = f.name
        
        try:
            result = load_manual_weather_data(temp_file_path)
            
            assert result is None
        finally:
            os.unlink(temp_file_path)


class TestRoadFeaturesCalculation:
    """测试道路特征计算功能"""
    
    def create_sample_air_sites_gdf(self):
        """创建测试用的空气质量站点数据"""
        data = {
            'SiteID': ['360050080', '360610115', '360470052'],
            'Latitude': [40.7128, 40.7282, 40.6782],
            'Longitude': [-74.0060, -73.9626, -73.9442]
        }
        
        geometry = [Point(lon, lat) for lon, lat in zip(data['Longitude'], data['Latitude'])]
        gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")
        
        return gdf
    
    def create_sample_roads_gdf(self):
        """创建测试用的道路数据"""
        # 在站点周围创建一些道路
        lines = [
            LineString([(-74.01, 40.71), (-74.00, 40.71)]),  # 靠近第一个站点
            LineString([(-73.96, 40.73), (-73.96, 40.72)]),  # 靠近第二个站点
            LineString([(-73.94, 40.68), (-73.95, 40.67)]),  # 靠近第三个站点
            LineString([(-74.10, 40.80), (-74.10, 40.70)])   # 远离所有站点
        ]
        
        data = {
            'geometry': lines,
            'fclass': ['motorway', 'primary', 'secondary', 'motorway'],
            'length': [1000, 800, 600, 2000]
        }
        
        return gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    def test_improved_calculate_road_features_basic(self):
        """测试基本道路特征计算"""
        air_sites_gdf = self.create_sample_air_sites_gdf()
        roads_gdf = self.create_sample_roads_gdf()
        
        result = improved_calculate_road_features(air_sites_gdf, roads_gdf)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 3个站点
        assert 'site_id' in result.columns
        
        # 检查是否创建了预期的特征列
        expected_columns = [
            'road_density_300m', 'total_road_length_300m', 
            'major_road_ratio_300m', 'intersection_density_300m'
        ]
        for col in expected_columns:
            assert col in result.columns
    
    def test_improved_calculate_road_features_different_buffer_sizes(self):
        """测试不同缓冲区大小的特征计算"""
        air_sites_gdf = self.create_sample_air_sites_gdf()
        roads_gdf = self.create_sample_roads_gdf()
        
        custom_buffers = [200, 400, 800]
        result = improved_calculate_road_features(air_sites_gdf, roads_gdf, custom_buffers)
        
        # 检查自定义缓冲区特征
        for buffer_size in custom_buffers:
            assert f'road_density_{buffer_size}m' in result.columns
            assert f'total_road_length_{buffer_size}m' in result.columns
    
    def test_improved_calculate_road_features_no_roads(self):
        """测试没有道路的情况"""
        air_sites_gdf = self.create_sample_air_sites_gdf()
        
        # 创建空的道路数据
        empty_roads = gpd.GeoDataFrame({
            'geometry': [],
            'fclass': [],
            'length': []
        }, crs="EPSG:4326")
        
        result = improved_calculate_road_features(air_sites_gdf, empty_roads)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # 检查所有道路特征应该为0
        for col in result.columns:
            if col != 'site_id' and 'ratio' not in col:
                assert all(result[col] == 0)
    
    def test_improved_calculate_road_features_length_column_detection(self):
        """测试长度列自动检测"""
        air_sites_gdf = self.create_sample_air_sites_gdf()
        roads_gdf = self.create_sample_roads_gdf()
        
        # 重命名长度列
        roads_gdf = roads_gdf.rename(columns={'length': 'Shape_Leng'})
        
        result = improved_calculate_road_features(air_sites_gdf, roads_gdf)
        
        assert isinstance(result, pd.DataFrame)
        # 应该成功计算而不报错


class TestDataIntegrationValidation:
    """测试数据整合验证功能"""
    
    def create_sample_final_dataset(self):
        """创建测试用的最终数据集"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(24)]
        site_ids = ['360050080', '360610115']
        
        data = []
        for timestamp in timestamps:
            for site_id in site_ids:
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'trip_count': np.random.randint(1, 100),
                    'PM2_5': np.random.uniform(5, 50),
                    'road_density_300m': np.random.uniform(0, 10),
                    'temperature': np.random.uniform(-5, 25),
                    'wind_speed': np.random.uniform(0, 10)
                })
        
        df = pd.DataFrame(data)
        
        # 添加一些缺失值用于测试
        df.loc[0:5, 'PM2_5'] = np.nan
        df.loc[10:15, 'road_density_300m'] = np.nan
        
        return df
    
    def test_validate_data_integration_basic(self):
        """测试基本数据验证"""
        final_df = self.create_sample_final_dataset()
        
        result = validate_data_integration(final_df)
        
        # 应该返回布尔值
        assert isinstance(result, bool)
    
    def test_validate_data_integration_complete_data(self):
        """测试完整数据的验证"""
        final_df = self.create_sample_final_dataset()
        # 移除所有缺失值
        final_df = final_df.dropna()
        
        result = validate_data_integration(final_df)
        
        # 完整数据应该通过验证
        assert result is True
    
    def test_validate_data_integration_missing_data(self):
        """测试有缺失数据的验证"""
        final_df = self.create_sample_final_dataset()
        
        result = validate_data_integration(final_df)
        
        # 有缺失数据应该返回False
        assert result is False


class TestWeatherFeaturesCreation:
    """测试气象特征创建功能"""
    
    def create_sample_weather_data(self):
        """创建测试用的气象数据"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(48)]
        
        data = {
            'timestamp': timestamps,
            'temperature': np.random.uniform(-10, 30, 48),
            'humidity': np.random.randint(20, 95, 48),
            'pressure': np.random.uniform(1000, 1030, 48),
            'wind_speed': np.random.uniform(0, 15, 48),
            'wind_direction': np.random.uniform(0, 360, 48),
            'precipitation': np.random.uniform(0, 8, 48)
        }
        
        return pd.DataFrame(data)
    
    def test_create_weather_features_basic(self):
        """测试基本气象特征创建"""
        weather_df = self.create_sample_weather_data()
        
        result = create_weather_features(weather_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(weather_df)
        
        # 检查新创建的特征
        expected_features = [
            'wind_direction_cat', 'wind_speed_cat', 'temperature_cat',
            'weather_severity'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_create_weather_features_wind_direction_categories(self):
        """测试风向分类"""
        weather_df = self.create_sample_weather_data()
        
        result = create_weather_features(weather_df)
        
        # 检查风向分类值
        valid_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        assert all(result['wind_direction_cat'].isin(valid_directions))
    
    def test_create_weather_features_wind_speed_categories(self):
        """测试风速分类"""
        weather_df = self.create_sample_weather_data()
        
        result = create_weather_features(weather_df)
        
        # 检查风速分类值
        valid_categories = ['Calm', 'Light', 'Moderate', 'Strong']
        assert all(result['wind_speed_cat'].isin(valid_categories))
    
    def test_create_weather_features_temperature_categories(self):
        """测试温度分类"""
        weather_df = self.create_sample_weather_data()
        
        result = create_weather_features(weather_df)
        
        # 检查温度分类值
        valid_categories = ['Freezing', 'Cold', 'Cool', 'Mild']
        assert all(result['temperature_cat'].isin(valid_categories))
    
    def test_create_weather_features_lag_features(self):
        """测试滞后特征"""
        weather_df = self.create_sample_weather_data()
        
        result = create_weather_features(weather_df)
        
        # 检查滞后特征
        for lag in [1, 3, 6]:
            assert f'temperature_lag_{lag}' in result.columns
            assert f'wind_speed_lag_{lag}' in result.columns
        
        # 检查滞后特征的缺失值处理
        assert result['temperature_lag_1'].isna().sum() == 1  # 第一个值应该是NaN
    
    def test_create_weather_features_weather_severity(self):
        """测试气象严重程度指标"""
        weather_df = self.create_sample_weather_data()
        
        result = create_weather_features(weather_df)
        
        # 检查严重程度指标范围
        assert result['weather_severity'].min() >= 0
        assert result['weather_severity'].max() <= 3


class TestIntegrationAllDataSources:
    """测试所有数据源整合功能"""
    
    def create_sample_traffic_pm25_data(self):
        """创建测试用的交通和PM2.5数据"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(24)]
        site_ids = ['360050080', '360610115']
        
        data = []
        for timestamp in timestamps:
            for site_id in site_ids:
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'trip_count': np.random.randint(1, 100),
                    'PM2_5': np.random.uniform(5, 50),
                    'avg_speed': np.random.uniform(10, 40)
                })
        
        return pd.DataFrame(data)
    
    def create_sample_station_info(self):
        """创建测试用的站点信息"""
        data = {
            'SiteID': ['360050080', '360610115', '360470052'],
            'Latitude': [40.7128, 40.7282, 40.6782],
            'Longitude': [-74.0060, -73.9626, -73.9442],
            'SiteName': ['Site1', 'Site2', 'Site3']
        }
        return pd.DataFrame(data)
    
    @patch('your_module_name.pd.read_csv')
    @patch('your_module_name.load_osm_data_geopandas')
    @patch('your_module_name.load_manual_weather_data')
    @patch('your_module_name.improved_calculate_road_features')
    @patch('your_module_name.create_weather_features')
    def test_integrate_all_data_sources_fixed_success(
        self, mock_create_weather, mock_calc_roads, 
        mock_load_weather, mock_load_osm, mock_read_csv
    ):
        """测试成功的数据整合"""
        # 设置mock返回值
        mock_read_csv.side_effect = [
            self.create_sample_traffic_pm25_data(),  # traffic_pm25_df
            self.create_sample_station_info()        # air_sites
        ]
        
        mock_load_osm.return_value = TestRoadFeaturesCalculation().create_sample_roads_gdf()
        mock_load_weather.return_value = TestWeatherFeaturesCreation().create_sample_weather_data()
        mock_calc_roads.return_value = pd.DataFrame({
            'site_id': ['360050080', '360610115'],
            'road_density_300m': [5.2, 3.8]
        })
        mock_create_weather.return_value = TestWeatherFeaturesCreation().create_sample_weather_data()
        
        result = integrate_all_data_sources_fixed()
        
        assert isinstance(result, pd.DataFrame)
        assert 'trip_count' in result.columns
        assert 'PM2_5' in result.columns
        assert 'road_density_300m' in result.columns
        assert 'temperature' in result.columns
    
    @patch('your_module_name.pd.read_csv')
    def test_integrate_all_data_sources_fixed_missing_coordinates(self, mock_read_csv):
        """测试缺少经纬度信息的情况"""
        # 创建没有经纬度列的站点信息
        invalid_sites = pd.DataFrame({
            'SiteID': ['360050080', '360610115'],
            'SiteName': ['Site1', 'Site2']
            # 缺少Latitude和Longitude列
        })
        
        mock_read_csv.side_effect = [
            self.create_sample_traffic_pm25_data(),  # traffic_pm25_df
            invalid_sites                            # air_sites
        ]
        
        result = integrate_all_data_sources_fixed()
        
        # 应该返回None因为无法创建空间数据
        assert result is None


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
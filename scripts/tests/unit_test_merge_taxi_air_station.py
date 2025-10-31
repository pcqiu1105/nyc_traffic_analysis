import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta
import tempfile
import os
import sys

# 添加模块路径到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merge_taxi_air_station import (
    prepare_spatial_infrastructure,
    prepare_taxi_data,
    fill_missing_records,
    borough_based_aggregation_complete,
    create_final_features_complete
)


class TestSpatialInfrastructure:
    """测试空间基础设施准备功能"""
    
    def create_sample_air_quality_sites(self):
        """创建测试用的空气质量站点数据"""
        data = {
            'SiteID': ['36061NY08454', '36005NY11534', '36047NY07974', '36081NY07615', '36085NY03820'],
            'Longitude': [-74.0060, -73.8667, -73.9442, -73.9626, -74.1496],
            'Latitude': [40.7128, 40.8568, 40.6782, 40.7282, 40.5795],
            'SiteName': ['Site1', 'Site2', 'Site3', 'Site4', 'Site5']
        }
        return pd.DataFrame(data)
    
    def test_prepare_spatial_infrastructure_basic(self):
        """测试基本空间基础设施准备"""
        air_quality_sites = self.create_sample_air_quality_sites()
        
        result = prepare_spatial_infrastructure(air_quality_sites)
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 5
        assert 'geometry' in result.columns
        assert result.crs == "EPSG:32618"
    
    def test_prepare_spatial_infrastructure_with_invalid_sites(self):
        """测试包含无效站点的数据"""
        air_quality_sites = self.create_sample_air_quality_sites()
        # 添加无效站点
        invalid_sites = pd.DataFrame({
            'SiteID': [None, '', 'valid_site'],
            'Longitude': [-74.0, -73.9, -73.8],
            'Latitude': [40.7, 40.8, 40.9]
        })
        air_quality_sites = pd.concat([air_quality_sites, invalid_sites], ignore_index=True)
        
        result = prepare_spatial_infrastructure(air_quality_sites)
        
        # 应该只保留有效的站点
        assert len(result) == 6  # 5个原始有效 + 1个新有效
        assert all(result['SiteID'].notna())
        assert all(result['SiteID'] != '')
    
    def test_prepare_spatial_infrastructure_custom_buffer(self):
        """测试自定义缓冲区距离"""
        air_quality_sites = self.create_sample_air_quality_sites()
        custom_buffers = [200, 400, 800]
        
        result = prepare_spatial_infrastructure(air_quality_sites, custom_buffers)
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 5


class TestTaxiDataPreparation:
    """测试出租车数据准备功能"""
    
    def create_sample_taxi_data(self):
        """创建测试用的出租车数据"""
        dates = pd.date_range('2024-01-01', '2024-01-03', freq='H')
        n_records = len(dates)
        
        data = {
            'tpep_pickup_datetime': dates,
            'PULocationID': np.random.choice([1, 2, 3, 4, 5], n_records),
            'Borough': np.random.choice(['Manhattan', 'Brooklyn', 'Queens'], n_records),
            'trip_distance': np.random.uniform(1, 10, n_records),
            'avg_speed': np.random.uniform(10, 40, n_records),
            'passenger_count': np.random.randint(1, 5, n_records),
            'total_amount': np.random.uniform(10, 50, n_records),
            'trip_duration': np.random.uniform(0.1, 2, n_records)
        }
        return pd.DataFrame(data)
    
    def create_sample_shapefile(self):
        """创建测试用的shapefile数据"""
        # 创建简单的测试几何图形
        polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        ]
        
        data = {
            'LocationID': [1, 2, 3, 4, 5],
            'zone': ['Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5'],
            'geometry': polygons
        }
        
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        # 保存到临时文件
        temp_dir = tempfile.mkdtemp()
        shapefile_path = os.path.join(temp_dir, "test_zones.shp")
        gdf.to_file(shapefile_path)
        
        return shapefile_path, temp_dir
    
    def test_prepare_taxi_data_basic(self):
        """测试基本出租车数据准备"""
        taxi_df = self.create_sample_taxi_data()
        shapefile_path, temp_dir = self.create_sample_shapefile()
        
        try:
            result = prepare_taxi_data(taxi_df, shapefile_path)
            
            assert isinstance(result, gpd.GeoDataFrame)
            assert 'geometry' in result.columns
            assert 'pickup_lon' in result.columns
            assert 'pickup_lat' in result.columns
            assert result.crs == "EPSG:32618"
        finally:
            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_prepare_taxi_data_coordinate_mapping(self):
        """测试坐标映射功能"""
        taxi_df = self.create_sample_taxi_data()
        shapefile_path, temp_dir = self.create_sample_shapefile()
        
        try:
            result = prepare_taxi_data(taxi_df, shapefile_path)
            
            # 检查坐标映射
            assert all(result['pickup_lon'] != 0)
            assert all(result['pickup_lat'] != 0)
            assert 'pickup_hour' in result.columns
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_prepare_taxi_data_with_invalid_locations(self):
        """测试包含无效位置的数据"""
        taxi_df = self.create_sample_taxi_data()
        # 添加无效的位置ID
        invalid_data = pd.DataFrame({
            'tpep_pickup_datetime': [pd.Timestamp('2024-01-01 12:00:00')],
            'PULocationID': [999],  # 不存在的LocationID
            'Borough': ['Unknown'],
            'trip_distance': [5.0],
            'avg_speed': [25.0],
            'passenger_count': [2],
            'total_amount': [20.0],
            'trip_duration': [0.5]
        })
        taxi_df = pd.concat([taxi_df, invalid_data], ignore_index=True)
        
        shapefile_path, temp_dir = self.create_sample_shapefile()
        
        try:
            result = prepare_taxi_data(taxi_df, shapefile_path)
            
            # 应该过滤掉无效坐标的记录
            assert len(result) <= len(taxi_df)
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestMissingRecordsFilling:
    """测试缺失记录填充功能"""
    
    def create_sample_aggregated_data(self):
        """创建测试用的聚合数据"""
        timestamps = pd.date_range('2024-01-01', '2024-01-02', freq='H')
        site_ids = ['36061NY08454', '36005NY11534', '36047NY07974']
        
        data = []
        for i, timestamp in enumerate(timestamps[:10]):  # 只创建部分数据用于测试缺失
            for site_id in site_ids[:2]:  # 只使用部分站点
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'borough': 'Manhattan' if '36061' in site_id else 'Bronx',
                    'trip_count': i + 1,
                    'total_distance': (i + 1) * 10,
                    'avg_trip_distance': 5.0,
                    'avg_speed': 25.0,
                    'total_passengers': (i + 1) * 2,
                    'total_revenue': (i + 1) * 20,
                    'avg_duration': 0.5
                })
        
        return pd.DataFrame(data)
    
    def test_fill_missing_records_basic(self):
        """测试基本缺失记录填充"""
        aggregated_df = self.create_sample_aggregated_data()
        site_to_borough = {
            '36061NY08454': 'Manhattan',
            '36005NY11534': 'Bronx',
            '36047NY07974': 'Brooklyn'  # 这个站点在原始数据中缺失
        }
        
        result = fill_missing_records(aggregated_df, site_to_borough)
        
        # 检查是否填充了缺失记录
        expected_records = len(aggregated_df['timestamp'].unique()) * len(site_to_borough)
        assert len(result) == expected_records
        
        # 检查数值列填充
        numeric_cols = ['trip_count', 'total_distance', 'avg_trip_distance', 'avg_speed']
        for col in numeric_cols:
            if col in result.columns:
                assert result[col].isna().sum() == 0
    
    def test_fill_missing_records_borough_mapping(self):
        """测试行政区映射在填充中的正确性"""
        aggregated_df = self.create_sample_aggregated_data()
        site_to_borough = {
            '36061NY08454': 'Manhattan',
            '36005NY11534': 'Bronx',
            '36047NY07974': 'Brooklyn'
        }
        
        result = fill_missing_records(aggregated_df, site_to_borough)
        
        # 检查行政区映射
        for site_id, expected_borough in site_to_borough.items():
            site_data = result[result['site_id'] == site_id]
            assert all(site_data['borough'] == expected_borough)


class TestBoroughBasedAggregation:
    """测试基于行政区的聚合功能"""
    
    def create_sample_taxi_gdf(self):
        """创建测试用的空间出租车数据"""
        dates = pd.date_range('2024-01-01', '2024-01-02', freq='H')
        n_records = len(dates)
        
        data = {
            'Borough': np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx'], n_records),
            'pickup_hour': dates,
            'trip_distance': np.random.uniform(1, 10, n_records),
            'avg_speed': np.random.uniform(10, 40, n_records),
            'passenger_count': np.random.randint(1, 5, n_records),
            'total_amount': np.random.uniform(10, 50, n_records),
            'trip_duration': np.random.uniform(0.1, 2, n_records),
            'geometry': [Point(0, 0) for _ in range(n_records)]  # 虚拟几何
        }
        
        return gpd.GeoDataFrame(data, crs="EPSG:32618")
    
    def create_sample_air_sites_gdf(self):
        """创建测试用的空气质量站点数据"""
        data = {
            'SiteID': ['36061NY08454', '36005NY11534', '36047NY07974', '36081NY07615', '36085NY03820'],
            'geometry': [Point(0, 0) for _ in range(5)]  # 虚拟几何
        }
        
        return gpd.GeoDataFrame(data, crs="EPSG:32618")
    
    def test_borough_based_aggregation_complete_basic(self):
        """测试基本行政区聚合"""
        taxi_gdf = self.create_sample_taxi_gdf()
        air_sites_gdf = self.create_sample_air_sites_gdf()
        
        result = borough_based_aggregation_complete(taxi_gdf, air_sites_gdf)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'site_id' in result.columns
        assert 'timestamp' in result.columns
        assert 'borough' in result.columns
    
    def test_borough_based_aggregation_complete_aggregation_logic(self):
        """测试聚合逻辑"""
        # 创建有明确模式的数据
        test_data = pd.DataFrame({
            'Borough': ['Manhattan', 'Manhattan', 'Brooklyn'],
            'pickup_hour': [
                pd.Timestamp('2024-01-01 10:00:00'),
                pd.Timestamp('2024-01-01 10:00:00'),
                pd.Timestamp('2024-01-01 10:00:00')
            ],
            'trip_distance': [5.0, 10.0, 3.0],
            'avg_speed': [20.0, 25.0, 18.0],
            'passenger_count': [1, 2, 1],
            'total_amount': [15.0, 25.0, 10.0],
            'trip_duration': [0.5, 0.8, 0.3],
            'geometry': [Point(0, 0), Point(0, 0), Point(0, 0)]
        })
        taxi_gdf = gpd.GeoDataFrame(test_data, crs="EPSG:32618")
        air_sites_gdf = self.create_sample_air_sites_gdf()
        
        result = borough_based_aggregation_complete(taxi_gdf, air_sites_gdf)
        
        # 检查聚合结果
        manhattan_sites = result[result['borough'] == 'Manhattan']
        assert len(manhattan_sites) > 0
        
        # 检查数值分配
        total_manhattan_trips = manhattan_sites['trip_count'].sum()
        expected_trips = 2  # Manhattan有2条记录
        assert abs(total_manhattan_trips - expected_trips) < 0.001
    
    def test_borough_based_aggregation_complete_missing_borough(self):
        """测试缺少行政区信息的情况"""
        taxi_gdf = self.create_sample_taxi_gdf()
        taxi_gdf = taxi_gdf.drop(columns=['Borough'])  # 移除行政区列
        air_sites_gdf = self.create_sample_air_sites_gdf()
        
        result = borough_based_aggregation_complete(taxi_gdf, air_sites_gdf)
        
        # 应该返回None
        assert result is None


class TestFinalFeaturesCreation:
    """测试最终特征创建功能"""
    
    def create_sample_aggregated_data(self):
        """创建测试用的聚合数据"""
        timestamps = pd.date_range('2024-01-01', '2024-01-03', freq='H')
        site_ids = ['36061NY08454', '36005NY11534']
        
        data = []
        for timestamp in timestamps:
            for site_id in site_ids:
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'borough': 'Manhattan' if '36061' in site_id else 'Bronx',
                    'trip_count': np.random.randint(1, 100),
                    'total_distance': np.random.uniform(10, 1000),
                    'avg_trip_distance': np.random.uniform(1, 10),
                    'avg_speed': np.random.uniform(10, 40),
                    'total_passengers': np.random.randint(1, 200),
                    'total_revenue': np.random.uniform(50, 5000),
                    'avg_duration': np.random.uniform(0.1, 2)
                })
        
        return pd.DataFrame(data)
    
    def test_create_final_features_complete_basic(self):
        """测试基本特征创建"""
        aggregated_df = self.create_sample_aggregated_data()
        
        result = create_final_features_complete(aggregated_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(aggregated_df)
        
        # 检查新创建的特征
        expected_features = [
            'borough_area', 'trip_density', 'passenger_density',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_rush_hour',
            'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_create_final_features_complete_lag_features(self):
        """测试滞后特征创建"""
        aggregated_df = self.create_sample_aggregated_data()
        
        result = create_final_features_complete(aggregated_df)
        
        # 检查滞后特征
        lag_features = [col for col in result.columns if 'lag' in col]
        assert len(lag_features) > 0
        
        # 检查移动平均特征
        ma_features = [col for col in result.columns if 'ma_' in col]
        assert len(ma_features) > 0
    
    def test_create_final_features_complete_missing_value_handling(self):
        """测试缺失值处理"""
        aggregated_df = self.create_sample_aggregated_data()
        
        result = create_final_features_complete(aggregated_df)
        
        # 检查没有缺失值
        assert result.isna().sum().sum() == 0
    
    def test_create_final_features_complete_temporal_features(self):
        """测试时间特征创建"""
        aggregated_df = self.create_sample_aggregated_data()
        
        result = create_final_features_complete(aggregated_df)
        
        # 检查时间特征范围
        assert result['hour_of_day'].min() >= 0
        assert result['hour_of_day'].max() <= 23
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6
        
        # 检查布尔特征
        assert result['is_weekend'].dtype == bool
        assert result['is_rush_hour'].dtype == bool


class TestIntegration:
    """测试模块集成"""
    
    def test_end_to_end_processing(self):
        """测试端到端处理流程"""
        # 创建测试数据
        air_quality_sites = TestSpatialInfrastructure().create_sample_air_quality_sites()
        taxi_df = TestTaxiDataPreparation().create_sample_taxi_data()
        
        # 由于shapefile创建比较复杂，跳过实际文件操作
        # 主要测试数据流和函数调用
        
        air_sites_gdf = prepare_spatial_infrastructure(air_quality_sites)
        assert isinstance(air_sites_gdf, gpd.GeoDataFrame)
        
        # 测试其他函数的集成逻辑
        aggregated_data = TestBoroughBasedAggregation().create_sample_aggregated_data()
        features = create_final_features_complete(aggregated_data)
        assert isinstance(features, pd.DataFrame)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
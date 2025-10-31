import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加模块路径到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clean_taxi_data import clean_taxi_data, aggregate_taxi_by_borough_hour


class TestTaxiDataCleaning:
    """测试出租车数据清洗功能"""
    
    def create_sample_taxi_data(self):
        """创建测试用的出租车数据"""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='H')
        n_records = len(dates)
        
        data = {
            'tpep_pickup_datetime': dates,
            'tpep_dropoff_datetime': [date + timedelta(hours=0.5) for date in dates],
            'PULocationID': np.random.randint(1, 10, n_records),
            'DOLocationID': np.random.randint(1, 10, n_records),
            'trip_distance': np.random.uniform(0.1, 25, n_records),
            'passenger_count': np.random.choice([1, 2, 3, 4, 5, 6, np.nan], n_records, p=[0.2, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05]),
            'total_amount': np.random.uniform(5, 50, n_records)
        }
        
        # 添加一些异常值用于测试过滤
        data['trip_distance'][:10] = 50  # 过长距离
        data['trip_distance'][10:20] = 0  # 过短距离
        
        return pd.DataFrame(data)
    
    def create_sample_zones_data(self):
        """创建测试用的区域数据"""
        data = {
            'LocationID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'Borough': ['Manhattan', 'Manhattan', 'Brooklyn', 'Brooklyn', 'Queens', 'Queens', 'Bronx', 'Bronx', 'Staten Island']
        }
        return pd.DataFrame(data)
    
    def test_clean_taxi_data_basic_functionality(self):
        """测试基本数据清洗功能"""
        # 准备测试数据
        taxi_df = self.create_sample_taxi_data()
        zones_df = self.create_sample_zones_data()
        
        # 执行清洗
        cleaned_df = clean_taxi_data(taxi_df, zones_df)
        
        # 断言检查
        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) > 0
        assert 'Borough' in cleaned_df.columns
        assert 'trip_duration' in cleaned_df.columns
        assert 'avg_speed' in cleaned_df.columns
    
    def test_clean_taxi_data_time_filtering(self):
        """测试时间范围过滤"""
        taxi_df = self.create_sample_taxi_data()
        zones_df = self.create_sample_zones_data()
        
        cleaned_df = clean_taxi_data(taxi_df, zones_df)
        
        # 检查时间范围过滤
        min_date = cleaned_df['tpep_pickup_datetime'].min()
        max_date = cleaned_df['tpep_pickup_datetime'].max()
        
        assert min_date >= pd.Timestamp('2024-01-01')
        assert max_date < pd.Timestamp('2024-04-01')
    
    def test_clean_taxi_data_distance_filtering(self):
        """测试距离过滤"""
        taxi_df = self.create_sample_taxi_data()
        zones_df = self.create_sample_zones_data()
        
        cleaned_df = clean_taxi_data(taxi_df, zones_df)
        
        # 检查距离范围
        assert cleaned_df['trip_distance'].min() >= 0.1
        assert cleaned_df['trip_distance'].max() <= 30
    
    def test_clean_taxi_data_duration_filtering(self):
        """测试行程时间过滤"""
        taxi_df = self.create_sample_taxi_data()
        zones_df = self.create_sample_zones_data()
        
        cleaned_df = clean_taxi_data(taxi_df, zones_df)
        
        # 检查行程时间范围
        assert cleaned_df['trip_duration'].min() >= 0.0167  # 1分钟
        assert cleaned_df['trip_duration'].max() <= 3  # 3小时
    
    def test_clean_taxi_data_passenger_filtering(self):
        """测试乘客数量过滤和填充"""
        taxi_df = self.create_sample_taxi_data()
        zones_df = self.create_sample_zones_data()
        
        cleaned_df = clean_taxi_data(taxi_df, zones_df)
        
        # 检查乘客数量范围
        assert cleaned_df['passenger_count'].min() >= 1
        assert cleaned_df['passenger_count'].max() <= 6
        assert cleaned_df['passenger_count'].isna().sum() == 0
    
    def test_clean_taxi_data_speed_filtering(self):
        """测试车速过滤"""
        taxi_df = self.create_sample_taxi_data()
        zones_df = self.create_sample_zones_data()
        
        cleaned_df = clean_taxi_data(taxi_df, zones_df)
        
        # 检查车速范围
        assert cleaned_df['avg_speed'].min() >= 1
        assert cleaned_df['avg_speed'].max() <= 60
    
    def test_clean_taxi_data_without_zones(self):
        """测试没有区域数据的情况"""
        taxi_df = self.create_sample_taxi_data()
        
        cleaned_df = clean_taxi_data(taxi_df, None)
        
        # 检查默认行政区设置
        assert 'Borough' in cleaned_df.columns
        assert all(cleaned_df['Borough'] == 'Unknown')
    
    def test_clean_taxi_data_time_features(self):
        """测试时间特征提取"""
        taxi_df = self.create_sample_taxi_data()
        zones_df = self.create_sample_zones_data()
        
        cleaned_df = clean_taxi_data(taxi_df, zones_df)
        
        # 检查时间特征
        assert 'pickup_hour' in cleaned_df.columns
        assert 'hour_of_day' in cleaned_df.columns
        assert 'day_of_week' in cleaned_df.columns
        assert 'is_weekend' in cleaned_df.columns
        
        # 检查时间特征范围
        assert cleaned_df['hour_of_day'].min() >= 0
        assert cleaned_df['hour_of_day'].max() <= 23
        assert cleaned_df['day_of_week'].min() >= 0
        assert cleaned_df['day_of_week'].max() <= 6


class TestTaxiDataAggregation:
    """测试出租车数据聚合功能"""
    
    def create_sample_cleaned_data(self):
        """创建测试用的清洗后数据"""
        dates = pd.date_range('2024-01-01', '2024-01-07', freq='H')
        n_records = len(dates)
        
        data = {
            'Borough': np.random.choice(['Manhattan', 'Brooklyn', 'Queens'], n_records),
            'pickup_hour': dates,
            'trip_distance': np.random.uniform(1, 10, n_records),
            'PULocationID': np.random.randint(1, 100, n_records),
            'passenger_count': np.random.randint(1, 6, n_records),
            'avg_speed': np.random.uniform(10, 40, n_records),
            'trip_duration': np.random.uniform(0.1, 2, n_records),
            'total_amount': np.random.uniform(10, 50, n_records)
        }
        
        return pd.DataFrame(data)
    
    def test_aggregate_taxi_by_borough_hour_basic(self):
        """测试基本聚合功能"""
        cleaned_df = self.create_sample_cleaned_data()
        
        aggregated_df = aggregate_taxi_by_borough_hour(cleaned_df)
        
        # 断言检查
        assert isinstance(aggregated_df, pd.DataFrame)
        assert len(aggregated_df) > 0
        assert 'Borough' in aggregated_df.columns
        assert 'hour' in aggregated_df.columns
    
    def test_aggregate_taxi_by_borough_hour_columns(self):
        """测试聚合结果的列"""
        cleaned_df = self.create_sample_cleaned_data()
        
        aggregated_df = aggregate_taxi_by_borough_hour(cleaned_df)
        
        # 检查聚合列是否存在
        expected_columns = ['Borough', 'hour']
        assert all(col in aggregated_df.columns for col in expected_columns)
    
    def test_aggregate_taxi_by_borough_hour_without_borough(self):
        """测试没有行政区信息的情况"""
        cleaned_df = self.create_sample_cleaned_data()
        cleaned_df = cleaned_df.drop(columns=['Borough'])
        
        aggregated_df = aggregate_taxi_by_borough_hour(cleaned_df)
        
        # 应该返回None
        assert aggregated_df is None
    
    def test_aggregate_taxi_by_borough_hour_aggregation_logic(self):
        """测试聚合逻辑"""
        # 创建有明确模式的数据用于测试聚合逻辑
        test_data = pd.DataFrame({
            'Borough': ['Manhattan', 'Manhattan', 'Brooklyn'],
            'pickup_hour': [
                pd.Timestamp('2024-01-01 10:00:00'),
                pd.Timestamp('2024-01-01 10:00:00'),
                pd.Timestamp('2024-01-01 10:00:00')
            ],
            'trip_distance': [5.0, 10.0, 3.0],
            'PULocationID': [1, 2, 3],
            'passenger_count': [1, 2, 1],
            'avg_speed': [20.0, 25.0, 18.0],
            'trip_duration': [0.5, 0.8, 0.3],
            'total_amount': [15.0, 25.0, 10.0]
        })
        
        aggregated_df = aggregate_taxi_by_borough_hour(test_data)
        
        # 检查聚合结果
        manhattan_data = aggregated_df[aggregated_df['Borough'] == 'Manhattan'].iloc[0]
        
        # 检查行程计数
        assert manhattan_data[('PULocationID', 'count')] == 2
        # 检查总距离
        assert manhattan_data[('trip_distance', 'sum')] == 15.0
        # 检查平均距离
        assert manhattan_data[('trip_distance', 'mean')] == 7.5


class TestDataQuality:
    """测试数据质量"""
    
    def test_data_integrity_after_cleaning(self):
        """测试清洗后数据完整性"""
        taxi_df = TestTaxiDataCleaning().create_sample_taxi_data()
        zones_df = TestTaxiDataCleaning().create_sample_zones_data()
        
        cleaned_df = clean_taxi_data(taxi_df, zones_df)
        
        # 检查没有空值在关键字段
        essential_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID', 'trip_distance']
        for col in essential_cols:
            assert cleaned_df[col].isna().sum() == 0
    
    def test_aggregation_data_quality(self):
        """测试聚合数据质量"""
        cleaned_df = TestTaxiDataAggregation().create_sample_cleaned_data()
        
        aggregated_df = aggregate_taxi_by_borough_hour(cleaned_df)
        
        if aggregated_df is not None:
            # 检查聚合数据没有空值
            assert not aggregated_df.isna().any().any()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
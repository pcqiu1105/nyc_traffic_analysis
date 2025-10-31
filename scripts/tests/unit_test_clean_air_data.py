import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# 添加模块路径到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clean_air_data import (
    process_air_quality_data,
    merge_traffic_air_quality,
    analyze_merged_data,
    create_time_series_features,
    correct_traffic_site_ids,
    save_final_dataset
)


class TestAirQualityDataProcessing:
    """测试空气质量数据处理功能"""
    
    def create_sample_air_quality_data(self):
        """创建测试用的空气质量数据"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(100)]
        
        data = {
            'ObservationTimeUTC': [ts.strftime('%Y-%m-%d %H:%M:%S+00:00') for ts in timestamps],
            'SiteID': ['360050080', '360610115', '360470052'] * 34,  # 循环使用3个站点
            'Value': np.random.uniform(5, 50, 100)
        }
        
        # 添加一些异常值用于测试过滤
        data['Value'][:5] = -1  # 负值
        data['Value'][5:10] = 300  # 超出范围的值
        
        return pd.DataFrame(data)
    
    def test_process_air_quality_data_basic(self):
        """测试基本空气质量数据处理"""
        air_quality_raw = self.create_sample_air_quality_data()
        
        result = process_air_quality_data(air_quality_raw)
        
        assert isinstance(result, pd.DataFrame)
        assert 'PM2_5' in result.columns
        assert 'hour' in result.columns
        assert 'SiteID' in result.columns
        
        # 检查异常值过滤
        assert result['PM2_5'].min() >= 0
        assert result['PM2_5'].max() <= 200
    
    def test_process_air_quality_data_from_file(self):
        """测试从文件读取空气质量数据"""
        # 创建临时文件
        air_quality_raw = self.create_sample_air_quality_data()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            air_quality_raw.to_csv(f.name, index=False)
            temp_file_path = f.name
        
        try:
            result = process_air_quality_data(temp_file_path)
            
            assert isinstance(result, pd.DataFrame)
            assert 'PM2_5' in result.columns
        finally:
            os.unlink(temp_file_path)
    
    def test_process_air_quality_data_time_processing(self):
        """测试时间处理逻辑"""
        air_quality_raw = self.create_sample_air_quality_data()
        
        result = process_air_quality_data(air_quality_raw)
        
        # 检查时间列存在
        assert 'hour' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['hour'])
        
        # 检查时间范围
        assert result['hour'].min() >= pd.Timestamp('2024-01-01')
    
    def test_process_air_quality_data_aggregation(self):
        """测试数据聚合逻辑"""
        # 创建有重复时间的数据
        base_date = datetime(2024, 1, 1, 10, 0, 0)
        data = {
            'ObservationTimeUTC': [
                base_date.strftime('%Y-%m-%d %H:%M:%S+00:00'),
                base_date.strftime('%Y-%m-%d %H:%M:%S+00:00'),
                base_date.strftime('%Y-%m-%d %H:%M:%S+00:00')
            ],
            'SiteID': ['360050080', '360050080', '360050080'],
            'Value': [10.0, 20.0, 30.0]
        }
        air_quality_raw = pd.DataFrame(data)
        
        result = process_air_quality_data(air_quality_raw)
        
        # 检查聚合结果（应该计算平均值）
        assert len(result) == 1
        assert result.iloc[0]['PM2_5'] == 20.0  # (10+20+30)/3 = 20


class TestTrafficAirQualityMerge:
    """测试交通和空气质量数据合并功能"""
    
    def create_sample_traffic_features(self):
        """创建测试用的交通特征数据"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(50)]
        site_ids = ['360050080', '360610115', '360470052']
        
        data = []
        for timestamp in timestamps:
            for site_id in site_ids:
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'trip_count': np.random.randint(1, 100),
                    'avg_speed': np.random.uniform(10, 40),
                    'total_distance': np.random.uniform(100, 1000)
                })
        
        return pd.DataFrame(data)
    
    def create_sample_air_quality_hourly(self):
        """创建测试用的空气质量小时数据"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(50)]
        site_ids = ['360050080', '360610115']  # 故意少一个站点测试合并
        
        data = []
        for timestamp in timestamps:
            for site_id in site_ids:
                data.append({
                    'hour': timestamp,
                    'SiteID': site_id,
                    'PM2_5': np.random.uniform(5, 50)
                })
        
        return pd.DataFrame(data)
    
    def test_merge_traffic_air_quality_basic(self):
        """测试基本数据合并"""
        traffic_features = self.create_sample_traffic_features()
        air_quality_hourly = self.create_sample_air_quality_hourly()
        
        result = merge_traffic_air_quality(traffic_features, air_quality_hourly)
        
        assert isinstance(result, pd.DataFrame)
        assert 'PM2_5' in result.columns
        assert 'trip_count' in result.columns
        
        # 检查列是否正确移除
        assert 'SiteID' not in result.columns
        assert 'hour' not in result.columns
    
    def test_merge_traffic_air_quality_coverage(self):
        """测试数据覆盖统计"""
        traffic_features = self.create_sample_traffic_features()
        air_quality_hourly = self.create_sample_air_quality_hourly()
        
        result = merge_traffic_air_quality(traffic_features, air_quality_hourly)
        
        # 检查覆盖统计
        total_records = len(result)
        records_with_pm25 = result['PM2_5'].notna().sum()
        
        assert records_with_pm25 > 0
        assert records_with_pm25 <= total_records
    
    def test_merge_traffic_air_quality_time_alignment(self):
        """测试时间对齐"""
        traffic_features = self.create_sample_traffic_features()
        air_quality_hourly = self.create_sample_air_quality_hourly()
        
        result = merge_traffic_air_quality(traffic_features, air_quality_hourly)
        
        # 检查时间列类型
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
        
        # 检查时间范围一致性
        traffic_min_time = traffic_features['timestamp'].min()
        traffic_max_time = traffic_features['timestamp'].max()
        result_min_time = result['timestamp'].min()
        result_max_time = result['timestamp'].max()
        
        assert result_min_time == traffic_min_time
        assert result_max_time == traffic_max_time


class TestMergedDataAnalysis:
    """测试合并数据分析功能"""
    
    def create_sample_merged_data(self):
        """创建测试用的合并数据"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(24)]
        site_ids = ['360050080', '360610115']
        
        data = []
        for timestamp in timestamps:
            for site_id in site_ids:
                has_pm25 = np.random.choice([True, False], p=[0.7, 0.3])
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'trip_count': np.random.randint(1, 100),
                    'PM2_5': np.random.uniform(5, 50) if has_pm25 else np.nan
                })
        
        return pd.DataFrame(data)
    
    def test_analyze_merged_data_basic(self):
        """测试基本数据分析"""
        merged_df = self.create_sample_merged_data()
        
        result = analyze_merged_data(merged_df)
        
        assert isinstance(result, pd.DataFrame)
        assert '总记录数' in result.columns
        assert '有PM2.5数据' in result.columns
        assert '有交通数据' in result.columns
        assert '完整记录' in result.columns
    
    def test_analyze_merged_data_completeness_calculation(self):
        """测试完整性计算"""
        # 创建有明确模式的数据
        data = {
            'timestamp': [datetime(2024, 1, 1, 10, 0, 0)] * 4,
            'site_id': ['site1', 'site2', 'site3', 'site4'],
            'trip_count': [10, 20, 0, 30],
            'PM2_5': [15.0, np.nan, 25.0, 35.0]
        }
        merged_df = pd.DataFrame(data)
        
        result = analyze_merged_data(merged_df)
        
        # 检查统计计算
        stats = result.iloc[0]
        assert stats['总记录数'] == 4
        assert stats['有PM2.5数据'] == 3
        assert stats['有交通数据'] == 3  # trip_count > 0
        assert stats['完整记录'] == 2  # 既有PM2.5又有交通数据


class TestTimeSeriesFeatures:
    """测试时间序列特征创建功能"""
    
    def create_sample_merged_data_for_features(self):
        """创建用于特征创建的测试数据"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(48)]  # 2天的数据
        site_ids = ['360050080', '360610115']
        
        data = []
        for timestamp in timestamps:
            for site_id in site_ids:
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'trip_count': np.random.randint(1, 100),
                    'PM2_5': np.random.uniform(5, 50)
                })
        
        return pd.DataFrame(data)
    
    def test_create_time_series_features_basic(self):
        """测试基本时间序列特征创建"""
        merged_df = self.create_sample_merged_data_for_features()
        
        result = create_time_series_features(merged_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(merged_df)
        
        # 检查基础时间特征
        expected_features = [
            'hour_of_day', 'day_of_week', 'is_weekend', 'month',
            'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_create_time_series_features_lag_features(self):
        """测试滞后特征创建"""
        merged_df = self.create_sample_merged_data_for_features()
        
        result = create_time_series_features(merged_df)
        
        # 检查滞后特征
        lag_periods = [1, 3, 6]
        for lag in lag_periods:
            assert f'PM2_5_lag_{lag}' in result.columns
            assert f'traffic_lag_{lag}' in result.columns
        
        # 检查滞后特征的正确性（第一个时间点应该有NaN）
        first_row = result.iloc[0]
        for lag in lag_periods:
            assert pd.isna(first_row[f'PM2_5_lag_{lag}'])
            assert pd.isna(first_row[f'traffic_lag_{lag}'])
    
    def test_create_time_series_features_moving_average(self):
        """测试移动平均特征"""
        merged_df = self.create_sample_merged_data_for_features()
        
        result = create_time_series_features(merged_df)
        
        # 检查移动平均特征
        windows = [3, 6, 12]
        for window in windows:
            assert f'PM2_5_ma_{window}' in result.columns
            assert f'traffic_ma_{window}' in result.columns
        
        # 检查移动平均计算（第一个时间点应该等于自身）
        first_row = result.iloc[0]
        for window in windows:
            assert not pd.isna(first_row[f'PM2_5_ma_{window}'])
            assert not pd.isna(first_row[f'traffic_ma_{window}'])
    
    def test_create_time_series_features_temporal_ranges(self):
        """测试时间特征范围"""
        merged_df = self.create_sample_merged_data_for_features()
        
        result = create_time_series_features(merged_df)
        
        # 检查时间特征范围
        assert result['hour_of_day'].min() >= 0
        assert result['hour_of_day'].max() <= 23
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6
        assert result['month'].min() >= 1
        assert result['month'].max() <= 12
        
        # 检查布尔特征类型
        assert result['is_weekend'].dtype == bool


class TestSiteIdCorrection:
    """测试站点ID修正功能"""
    
    def create_sample_traffic_features_with_old_ids(self):
        """创建包含旧站点ID的交通特征数据"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(10)]
        
        # 混合新旧站点ID
        old_site_ids = ['36005NY11534', '36047NY07974', '36061NY08552', 'unknown_site']
        new_site_ids = ['360050080', '360470052', '360610115']  # 对应的新ID
        
        data = []
        for timestamp in timestamps:
            for site_id in old_site_ids:
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'trip_count': np.random.randint(1, 100)
                })
        
        return pd.DataFrame(data)
    
    def test_correct_traffic_site_ids_basic(self):
        """测试基本站点ID修正"""
        traffic_features = self.create_sample_traffic_features_with_old_ids()
        
        result = correct_traffic_site_ids(traffic_features)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(traffic_features)
        
        # 检查已知站点的修正
        assert '360050080' in result['site_id'].values
        assert '360470052' in result['site_id'].values
        assert '360610115' in result['site_id'].values
        
        # 检查未知站点保持不变
        assert 'unknown_site' in result['site_id'].values
    
    def test_correct_traffic_site_ids_mapping_completeness(self):
        """测试映射完整性"""
        traffic_features = self.create_sample_traffic_features_with_old_ids()
        
        result = correct_traffic_site_ids(traffic_features)
        
        # 检查唯一站点数
        unique_sites_before = traffic_features['site_id'].nunique()
        unique_sites_after = result['site_id'].nunique()
        
        # 修正后唯一站点数应该相同或更少（如果合并了重复映射）
        assert unique_sites_after <= unique_sites_before


class TestFinalDatasetSaving:
    """测试最终数据集保存功能"""
    
    def create_sample_final_data(self):
        """创建测试用的最终数据"""
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i) for i in range(10)]
        site_ids = ['360050080', '360610115']
        
        data = []
        for timestamp in timestamps:
            for site_id in site_ids:
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'trip_count': np.random.randint(1, 100),
                    'PM2_5': np.random.uniform(5, 50),
                    'hour_of_day': timestamp.hour,
                    'day_of_week': timestamp.weekday()
                })
        
        return pd.DataFrame(data)
    
    def test_save_final_dataset(self):
        """测试数据集保存功能"""
        final_data = self.create_sample_final_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file_path = f.name
        
        try:
            save_final_dataset(final_data, temp_file_path)
            
            # 检查文件是否创建
            assert os.path.exists(temp_file_path)
            
            # 检查保存的数据是否正确
            loaded_data = pd.read_csv(temp_file_path)
            assert len(loaded_data) == len(final_data)
            assert 'trip_count' in loaded_data.columns
            assert 'PM2_5' in loaded_data.columns
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_save_final_dataset_directory_creation(self):
        """测试目录自动创建"""
        final_data = self.create_sample_final_data()
        
        # 创建不存在的目录路径
        temp_dir = tempfile.mkdtemp()
        non_existent_subdir = os.path.join(temp_dir, 'nonexistent', 'subdir')
        output_path = os.path.join(non_existent_subdir, 'test_data.csv')
        
        try:
            save_final_dataset(final_data, output_path)
            
            # 检查目录是否被创建
            assert os.path.exists(non_existent_subdir)
            assert os.path.exists(output_path)
            
        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestIntegration:
    """测试模块集成"""
    
    def test_end_to_end_processing(self):
        """测试端到端处理流程"""
        # 创建测试数据
        traffic_features = TestTrafficAirQualityMerge().create_sample_traffic_features()
        air_quality_raw = TestAirQualityDataProcessing().create_sample_air_quality_data()
        
        # 测试数据处理流程
        air_quality_hourly = process_air_quality_data(air_quality_raw)
        assert isinstance(air_quality_hourly, pd.DataFrame)
        
        merged_data = merge_traffic_air_quality(traffic_features, air_quality_hourly)
        assert isinstance(merged_data, pd.DataFrame)
        
        analysis_result = analyze_merged_data(merged_data)
        assert isinstance(analysis_result, pd.DataFrame)
        
        final_features = create_time_series_features(merged_data)
        assert isinstance(final_features, pd.DataFrame)
        
        # 测试站点ID修正
        corrected_features = correct_traffic_site_ids(traffic_features)
        assert isinstance(corrected_features, pd.DataFrame)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
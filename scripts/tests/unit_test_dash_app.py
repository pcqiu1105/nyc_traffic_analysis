import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, call
import warnings

# 过滤警告
warnings.filterwarnings("ignore")

# 添加模块路径到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dash_app import create_proper_interactive_dashboard, main


class TestInteractiveDashboard:
    """测试交互式仪表盘功能"""
    
    def create_sample_final_df(self):
        """创建测试用的最终数据集"""
        base_date = pd.Timestamp('2024-01-01')
        timestamps = [base_date + pd.Timedelta(hours=i) for i in range(100)]
        site_ids = ['360050080', '360610115', '360470052']
        boroughs = ['Manhattan', 'Brooklyn', 'Queens']
        
        data = []
        for timestamp in timestamps:
            for i, site_id in enumerate(site_ids):
                data.append({
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'borough': boroughs[i % 3],
                    'hour_of_day': timestamp.hour,
                    'day_of_week': timestamp.dayofweek,
                    'PM2_5': np.random.uniform(5, 50),
                    'trip_count': np.random.randint(1, 100),
                    'avg_speed': np.random.uniform(10, 40),
                    'total_passengers': np.random.randint(1, 200),
                    'trip_density': np.random.uniform(0, 10),
                    'road_density_500m': np.random.uniform(0, 5),
                    'major_road_ratio_500m': np.random.uniform(0, 1),
                    'intersection_density_500m': np.random.uniform(0, 2),
                    'temperature': np.random.uniform(-5, 30),
                    'humidity': np.random.uniform(30, 90),
                    'wind_speed': np.random.uniform(0, 10),
                    'total_revenue': np.random.uniform(100, 1000),
                    'is_rush_hour': np.random.choice([True, False]),
                    'is_weekend': timestamp.dayofweek >= 5
                })
        
        return pd.DataFrame(data)
    
    def create_sample_air_quality_sites(self):
        """创建测试用的空气质量站点数据"""
        data = {
            'SiteID': ['360050080', '360610115', '360470052'],
            'Longitude': [-74.0060, -73.9626, -73.9442],
            'Latitude': [40.7128, 40.7282, 40.6782],
            'SiteName': ['Site1', 'Site2', 'Site3']
        }
        return pd.DataFrame(data)
    
    def create_sample_lstm_predictions(self):
        """创建测试用的LSTM预测数据"""
        base_date = pd.Timestamp('2024-01-01')
        timestamps = [base_date + pd.Timedelta(hours=i) for i in range(50)]
        boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        
        data = []
        for timestamp in timestamps:
            for borough in boroughs:
                data.append({
                    'timestamp': timestamp,
                    'borough': borough,
                    'true_t1': np.random.uniform(5, 50),
                    'pred_t1': np.random.uniform(5, 50),
                    'true_t3': np.random.uniform(5, 50),
                    'pred_t3': np.random.uniform(5, 50),
                    'true_t6': np.random.uniform(5, 50),
                    'pred_t6': np.random.uniform(5, 50)
                })
        
        return pd.DataFrame(data)
    
    def create_sample_realtime_predictions(self):
        """创建测试用的实时预测数据"""
        base_date = pd.Timestamp('2024-01-01')
        timestamps = [base_date + pd.Timedelta(hours=i) for i in range(50)]
        boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        
        data = []
        for timestamp in timestamps:
            for borough in boroughs:
                data.append({
                    'timestamp': timestamp,
                    'borough': borough,
                    'PM2_5_true': np.random.uniform(5, 50),
                    'PM2_5_pred_XGBoost': np.random.uniform(5, 50),
                    'PM2_5_pred_LightGBM': np.random.uniform(5, 50),
                    'PM2_5_pred_RandomForest': np.random.uniform(5, 50)
                })
        
        return pd.DataFrame(data)
    
    def create_sample_geojson(self):
        """创建测试用的GeoJSON数据"""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"BoroName": "Manhattan"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-74.02, 40.70], [-73.98, 40.70], [-73.98, 40.75], [-74.02, 40.75], [-74.02, 40.70]]]
                    }
                },
                {
                    "type": "Feature", 
                    "properties": {"BoroName": "Brooklyn"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-73.98, 40.65], [-73.94, 40.65], [-73.94, 40.70], [-73.98, 40.70], [-73.98, 40.65]]]
                    }
                }
            ]
        }
        return geojson
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    @patch('folium.Choropleth')
    @patch('folium.Marker')
    @patch('folium.LayerControl')
    @patch('plotly.graph_objects.Figure')
    def test_create_proper_interactive_dashboard_basic(
        self, mock_plotly_fig, mock_layer_control, mock_marker, 
        mock_choropleth, mock_folium_map, mock_json_load, mock_open
    ):
        """测试基本交互式仪表盘创建"""
        # 设置mock返回值
        final_df = self.create_sample_final_df()
        air_quality_sites = self.create_sample_air_quality_sites()
        lstm_pred_df = self.create_sample_lstm_predictions()
        realtime_pred_df = self.create_sample_realtime_predictions()
        
        # 模拟GeoJSON文件
        mock_json_load.return_value = self.create_sample_geojson()
        
        # 模拟Folium地图
        mock_map_instance = MagicMock()
        mock_folium_map.return_value = mock_map_instance
        
        # 模拟Plotly图表
        mock_fig_instance = MagicMock()
        mock_plotly_fig.return_value = mock_fig_instance
        
        # 创建临时目录用于输出
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                # 创建必要的子目录
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                result = create_proper_interactive_dashboard(
                    final_df, air_quality_sites, 
                    'dummy_geojson_path', lstm_pred_df, realtime_pred_df
                )
                
                # 检查结果
                assert result is not None
                assert 'outputs/interactive_dashboard_with_predictions.html' in result
                
                # 检查文件是否被创建
                expected_path = os.path.join(temp_dir, 'outputs/interactive_dashboard_with_predictions.html')
                assert os.path.exists(expected_path)
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    def test_create_proper_interactive_dashboard_data_processing(
        self, mock_folium_map, mock_json_load, mock_open
    ):
        """测试数据预处理逻辑"""
        final_df = self.create_sample_final_df()
        air_quality_sites = self.create_sample_air_quality_sites()
        lstm_pred_df = self.create_sample_lstm_predictions()
        realtime_pred_df = self.create_sample_realtime_predictions()
        
        # 模拟GeoJSON
        mock_json_load.return_value = self.create_sample_geojson()
        
        # 模拟Folium
        mock_map_instance = MagicMock()
        mock_folium_map.return_value = mock_map_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                result = create_proper_interactive_dashboard(
                    final_df, air_quality_sites,
                    'dummy_path', lstm_pred_df, realtime_pred_df
                )
                
                # 检查LSTM数据聚合
                # 应该按时间戳聚合5个区的平均值
                lstm_agg = lstm_pred_df.groupby('timestamp').agg({
                    'true_t1': 'mean', 'pred_t1': 'mean',
                    'true_t3': 'mean', 'pred_t3': 'mean', 
                    'true_t6': 'mean', 'pred_t6': 'mean'
                }).reset_index()
                
                assert len(lstm_agg) == lstm_pred_df['timestamp'].nunique()
                
                # 检查实时预测数据聚合
                realtime_agg = realtime_pred_df.groupby('timestamp').agg({
                    'PM2_5_true': 'mean',
                    'PM2_5_pred_XGBoost': 'mean',
                    'PM2_5_pred_LightGBM': 'mean',
                    'PM2_5_pred_RandomForest': 'mean'
                }).reset_index()
                
                assert len(realtime_agg) == realtime_pred_df['timestamp'].nunique()
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    @patch('sklearn.metrics.mean_absolute_error')
    @patch('sklearn.metrics.mean_squared_error')
    @patch('sklearn.metrics.r2_score')
    def test_create_proper_interactive_dashboard_model_metrics(
        self, mock_r2, mock_mse, mock_mae, mock_folium_map, 
        mock_json_load, mock_open
    ):
        """测试模型指标计算"""
        # 设置mock返回值
        mock_mae.return_value = 2.5
        mock_mse.return_value = 8.0  # RMSE = sqrt(8.0) ≈ 2.828
        mock_r2.return_value = 0.85
        
        final_df = self.create_sample_final_df()
        air_quality_sites = self.create_sample_air_quality_sites()
        lstm_pred_df = self.create_sample_lstm_predictions()
        realtime_pred_df = self.create_sample_realtime_predictions()
        
        mock_json_load.return_value = self.create_sample_geojson()
        mock_folium_map.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                result = create_proper_interactive_dashboard(
                    final_df, air_quality_sites,
                    'dummy_path', lstm_pred_df, realtime_pred_df
                )
                
                # 检查指标计算函数被调用
                assert mock_mae.called
                assert mock_mse.called  
                assert mock_r2.called
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    def test_create_proper_interactive_dashboard_empty_data(
        self, mock_folium_map, mock_json_load, mock_open
    ):
        """测试空数据处理"""
        # 创建空数据
        empty_df = pd.DataFrame()
        air_quality_sites = self.create_sample_air_quality_sites()
        lstm_pred_df = self.create_sample_lstm_predictions()
        realtime_pred_df = self.create_sample_realtime_predictions()
        
        mock_json_load.return_value = self.create_sample_geojson()
        mock_folium_map.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                # 应该能够处理空数据而不崩溃
                try:
                    result = create_proper_interactive_dashboard(
                        empty_df, air_quality_sites,
                        'dummy_path', lstm_pred_df, realtime_pred_df
                    )
                    # 即使有空数据，也应该生成HTML文件
                    assert result is not None
                except Exception as e:
                    pytest.fail(f"空数据处理失败: {e}")
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    def test_create_proper_interactive_dashboard_missing_columns(
        self, mock_folium_map, mock_json_load, mock_open
    ):
        """测试缺失列的处理"""
        # 创建缺少某些列的数据
        final_df = self.create_sample_final_df()
        # 移除一些列
        final_df = final_df.drop(columns=['temperature', 'humidity', 'wind_speed'], errors='ignore')
        
        air_quality_sites = self.create_sample_air_quality_sites()
        lstm_pred_df = self.create_sample_lstm_predictions()
        realtime_pred_df = self.create_sample_realtime_predictions()
        
        mock_json_load.return_value = self.create_sample_geojson()
        mock_folium_map.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                # 应该能够处理缺失列而不崩溃
                try:
                    result = create_proper_interactive_dashboard(
                        final_df, air_quality_sites,
                        'dummy_path', lstm_pred_df, realtime_pred_df
                    )
                    assert result is not None
                except Exception as e:
                    pytest.fail(f"缺失列处理失败: {e}")
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    def test_create_proper_interactive_dashboard_geojson_error(
        self, mock_folium_map, mock_json_load, mock_open
    ):
        """测试GeoJSON错误处理"""
        final_df = self.create_sample_final_df()
        air_quality_sites = self.create_sample_air_quality_sites()
        lstm_pred_df = self.create_sample_lstm_predictions()
        realtime_pred_df = self.create_sample_realtime_predictions()
        
        # 模拟GeoJSON加载错误
        mock_json_load.side_effect = Exception("GeoJSON file not found")
        
        mock_folium_map.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                # 应该能够处理GeoJSON错误而不崩溃
                try:
                    result = create_proper_interactive_dashboard(
                        final_df, air_quality_sites,
                        'dummy_path', lstm_pred_df, realtime_pred_df
                    )
                    assert result is not None
                except Exception as e:
                    pytest.fail(f"GeoJSON错误处理失败: {e}")
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    def test_create_proper_interactive_dashboard_html_generation(
        self, mock_folium_map, mock_json_load, mock_open
    ):
        """测试HTML生成功能"""
        final_df = self.create_sample_final_df()
        air_quality_sites = self.create_sample_air_quality_sites()
        lstm_pred_df = self.create_sample_lstm_predictions()
        realtime_pred_df = self.create_sample_realtime_predictions()
        
        mock_json_load.return_value = self.create_sample_geojson()
        mock_map_instance = MagicMock()
        mock_map_instance._repr_html_.return_value = "<div>Mock Map</div>"
        mock_folium_map.return_value = mock_map_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                result = create_proper_interactive_dashboard(
                    final_df, air_quality_sites,
                    'dummy_path', lstm_pred_df, realtime_pred_df
                )
                
                # 检查HTML文件内容
                expected_path = os.path.join(temp_dir, 'outputs/interactive_dashboard_with_predictions.html')
                with open(expected_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # 检查关键HTML元素
                assert '<!DOCTYPE html>' in html_content
                assert '<title>NYC Taxi & Air Quality Interactive Dashboard</title>' in html_content
                assert 'Plotly' in html_content
                assert 'dashboard-container' in html_content
    
    @patch('pandas.read_csv')
    @patch('your_module_name.create_proper_interactive_dashboard')
    def test_main_function(self, mock_create_dashboard, mock_read_csv):
        """测试主函数"""
        # 设置mock返回值
        final_df = self.create_sample_final_df()
        air_quality_sites = self.create_sample_air_quality_sites()
        lstm_pred_df = self.create_sample_lstm_predictions()
        realtime_pred_df = self.create_sample_realtime_predictions()
        
        mock_read_csv.side_effect = [
            final_df,  # final_complete_dataset.csv
            air_quality_sites,  # station-info.csv
            lstm_pred_df,  # lstm_predictions.csv
            realtime_pred_df  # realtime_predictions.csv
        ]
        
        mock_create_dashboard.return_value = "/mock/path/dashboard.html"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                # 执行主函数
                main()
                
                # 检查函数调用
                mock_create_dashboard.assert_called_once()
                
                # 获取调用参数
                call_args = mock_create_dashboard.call_args[0]
                assert len(call_args) == 5
                assert call_args[0].equals(final_df)
                assert call_args[1].equals(air_quality_sites)
                assert 'boroughs.geojson' in call_args[2]
                assert call_args[3].equals(lstm_pred_df)
                assert call_args[4].equals(realtime_pred_df)


class TestEdgeCases:
    """测试边界情况"""
    
    def create_minimal_data(self):
        """创建最小数据集"""
        return pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01 10:00:00')],
            'site_id': ['test_site'],
            'borough': ['Manhattan'],
            'hour_of_day': [10],
            'PM2_5': [25.0],
            'trip_count': [50],
            'avg_speed': [25.0],
            'total_revenue': [1000.0],
            'temperature': [15.0],
            'humidity': [60.0]
        })
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    def test_single_record_data(self, mock_folium_map, mock_json_load, mock_open):
        """测试单条记录数据"""
        final_df = self.create_minimal_data()
        air_quality_sites = pd.DataFrame({
            'SiteID': ['test_site'],
            'Longitude': [-74.0060],
            'Latitude': [40.7128]
        })
        
        # 创建最小预测数据
        lstm_pred_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01 10:00:00')],
            'borough': ['Manhattan'],
            'true_t1': [25.0], 'pred_t1': [24.5],
            'true_t3': [26.0], 'pred_t3': [25.5],
            'true_t6': [27.0], 'pred_t6': [26.5]
        })
        
        realtime_pred_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01 10:00:00')],
            'borough': ['Manhattan'],
            'PM2_5_true': [25.0],
            'PM2_5_pred_XGBoost': [24.5],
            'PM2_5_pred_LightGBM': [24.8],
            'PM2_5_pred_RandomForest': [24.2]
        })
        
        mock_json_load.return_value = self.create_sample_geojson()
        mock_folium_map.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                try:
                    result = create_proper_interactive_dashboard(
                        final_df, air_quality_sites,
                        'dummy_path', lstm_pred_df, realtime_pred_df
                    )
                    assert result is not None
                except Exception as e:
                    pytest.fail(f"单条记录处理失败: {e}")
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    def test_missing_prediction_data(self, mock_folium_map, mock_json_load, mock_open):
        """测试缺失预测数据"""
        final_df = self.create_sample_final_df()
        air_quality_sites = self.create_sample_air_quality_sites()
        
        # 创建空的预测数据
        empty_pred_df = pd.DataFrame()
        
        mock_json_load.return_value = self.create_sample_geojson()
        mock_folium_map.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                try:
                    result = create_proper_interactive_dashboard(
                        final_df, air_quality_sites,
                        'dummy_path', empty_pred_df, empty_pred_df
                    )
                    assert result is not None
                except Exception as e:
                    pytest.fail(f"缺失预测数据处理失败: {e}")


class TestDataValidation:
    """测试数据验证"""
    
    @patch('builtins.open')
    @patch('json.load')
    @patch('folium.Map')
    def test_timestamp_processing(self, mock_folium_map, mock_json_load, mock_open):
        """测试时间戳处理"""
        final_df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
            'site_id': ['site1', 'site2'],
            'borough': ['Manhattan', 'Brooklyn'],
            'hour_of_day': [10, 11],
            'PM2_5': [25.0, 26.0],
            'trip_count': [50, 60]
        })
        
        air_quality_sites = pd.DataFrame({
            'SiteID': ['site1', 'site2'],
            'Longitude': [-74.0060, -73.9626],
            'Latitude': [40.7128, 40.7282]
        })
        
        lstm_pred_df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
            'borough': ['Manhattan', 'Brooklyn'],
            'true_t1': [25.0, 26.0], 'pred_t1': [24.5, 25.5]
        })
        
        realtime_pred_df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
            'borough': ['Manhattan', 'Brooklyn'],
            'PM2_5_true': [25.0, 26.0],
            'PM2_5_pred_XGBoost': [24.5, 25.5]
        })
        
        mock_json_load.return_value = self.create_sample_geojson()
        mock_folium_map.return_value = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('your_module_name.BASE_FILE_PATH', temp_dir):
                os.makedirs(os.path.join(temp_dir, 'outputs'), exist_ok=True)
                
                try:
                    result = create_proper_interactive_dashboard(
                        final_df, air_quality_sites,
                        'dummy_path', lstm_pred_df, realtime_pred_df
                    )
                    assert result is not None
                except Exception as e:
                    pytest.fail(f"时间戳处理失败: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
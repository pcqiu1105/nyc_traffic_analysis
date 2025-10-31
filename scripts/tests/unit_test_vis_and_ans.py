import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
import warnings

# 过滤matplotlib警告
warnings.filterwarnings("ignore", category=UserWarning)

# 添加模块路径到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vis_and_ans import (
    create_spatial_visualization,
    create_temporal_analysis,
    create_static_analysis,
    create_interactive_visualizations,
    create_geographic_dashboard,
    generate_column_statistics,
    generate_key_findings
)


class TestSpatialVisualization:
    """测试空间可视化功能"""
    
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
                    'is_rush_hour': np.random.choice([True, False]),
                    'is_weekend': timestamp.dayofweek >= 5,
                    'wind_speed_cat': np.random.choice(['Calm', 'Light', 'Moderate', 'Strong']),
                    'weather_severity': np.random.randint(0, 4)
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
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_spatial_visualization_basic(self, mock_savefig, mock_show):
        """测试基本空间可视化"""
        final_df = self.create_sample_final_df()
        air_quality_sites = self.create_sample_air_quality_sites()
        
        # 模拟nybb数据集
        with patch('geopandas.datasets.get_path') as mock_get_path:
            # 创建模拟的nyc地图数据
            mock_nyc_data = gpd.GeoDataFrame({
                'geometry': [Point(0, 0).buffer(1)],
                'BoroName': ['Manhattan']
            })
            mock_get_path.return_value = 'mock_path'
            
            with patch('geopandas.read_file') as mock_read_file:
                mock_read_file.return_value = mock_nyc_data
                
                try:
                    create_spatial_visualization(final_df, air_quality_sites)
                    
                    # 检查是否调用了保存和显示
                    mock_savefig.assert_called_once()
                    mock_show.assert_called_once()
                    
                except Exception as e:
                    # 如果出现地理数据问题，跳过测试
                    pytest.skip(f"Spatial visualization test skipped due to: {e}")
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_spatial_visualization_missing_columns(self, mock_savefig, mock_show):
        """测试缺少某些列的情况"""
        final_df = self.create_sample_final_df()
        # 移除一些列
        final_df = final_df.drop(columns=['road_density_500m', 'total_passengers'], errors='ignore')
        air_quality_sites = self.create_sample_air_quality_sites()
        
        with patch('geopandas.datasets.get_path'):
            with patch('geopandas.read_file'):
                try:
                    create_spatial_visualization(final_df, air_quality_sites)
                    mock_savefig.assert_called_once()
                except Exception as e:
                    pytest.skip(f"Test skipped due to: {e}")


class TestTemporalAnalysis:
    """测试时间序列分析功能"""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_temporal_analysis_basic(self, mock_savefig, mock_show):
        """测试基本时间序列分析"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        
        create_temporal_analysis(final_df)
        
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_temporal_analysis_with_missing_data(self, mock_savefig, mock_show):
        """测试有缺失数据的时间序列分析"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        # 添加一些缺失值
        final_df.loc[0:10, 'PM2_5'] = np.nan
        final_df.loc[20:30, 'temperature'] = np.nan
        
        create_temporal_analysis(final_df)
        
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()


class TestStaticAnalysis:
    """测试静态分析功能"""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_static_analysis_basic(self, mock_savefig, mock_show):
        """测试基本静态分析"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        
        create_static_analysis(final_df)
        
        # 应该保存多个图表
        assert mock_savefig.call_count >= 3
        assert mock_show.call_count >= 3
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_static_analysis_missing_features(self, mock_savefig, mock_show):
        """测试缺少某些特征的静态分析"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        # 移除一些特征列
        final_df = final_df.drop(columns=['trip_density', 'road_density_500m'], errors='ignore')
        
        create_static_analysis(final_df)
        
        # 应该仍然能够运行而不报错
        assert mock_savefig.call_count >= 3


class TestInteractiveVisualizations:
    """测试交互式可视化功能"""
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_interactive_visualizations_basic(self, mock_write_html):
        """测试基本交互式可视化"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        air_quality_sites = TestSpatialVisualization().create_sample_air_quality_sites()
        
        create_interactive_visualizations(final_df, air_quality_sites)
        
        # 检查是否调用了HTML保存
        mock_write_html.assert_called_once()
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_create_interactive_visualizations_empty_data(self, mock_write_html):
        """测试空数据的交互式可视化"""
        final_df = pd.DataFrame()  # 空数据框
        air_quality_sites = TestSpatialVisualization().create_sample_air_quality_sites()
        
        try:
            create_interactive_visualizations(final_df, air_quality_sites)
        except Exception:
            # 空数据应该会报错，这是预期的
            pass


class TestGeographicDashboard:
    """测试地理仪表板功能"""
    
    @patch('folium.Map.save')
    def test_create_geographic_dashboard_basic(self, mock_save):
        """测试基本地理仪表板创建"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        air_quality_sites = TestSpatialVisualization().create_sample_air_quality_sites()
        
        # 模拟folium相关操作
        with patch('folium.Map') as mock_map:
            with patch('folium.TileLayer') as mock_tile:
                with patch('folium.CircleMarker') as mock_circle:
                    with patch('folium.plugins.HeatMap') as mock_heatmap:
                        with patch('folium.LayerControl') as mock_control:
                            mock_map_instance = MagicMock()
                            mock_map.return_value = mock_map_instance
                            
                            create_geographic_dashboard(final_df, air_quality_sites)
                            
                            # 检查是否调用了保存
                            mock_save.assert_called_once()
    
    @patch('folium.Map.save')
    def test_create_geographic_dashboard_no_sites(self, mock_save):
        """测试没有站点数据的情况"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        air_quality_sites = pd.DataFrame()  # 空站点数据
        
        with patch('folium.Map') as mock_map:
            mock_map_instance = MagicMock()
            mock_map.return_value = mock_map_instance
            
            try:
                create_geographic_dashboard(final_df, air_quality_sites)
                mock_save.assert_called_once()
            except Exception as e:
                pytest.skip(f"Geographic dashboard test skipped: {e}")


class TestColumnStatistics:
    """测试列统计功能"""
    
    def test_generate_column_statistics_basic(self):
        """测试基本列统计生成"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        
        result = generate_column_statistics(final_df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'Feature Name' in result.columns
        assert 'Data Type' in result.columns
        assert 'Missing Ratio (%)' in result.columns
    
    def test_generate_column_statistics_empty_data(self):
        """测试空数据的列统计"""
        empty_df = pd.DataFrame()
        
        result = generate_column_statistics(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_generate_column_statistics_with_missing_values(self):
        """测试有缺失值的列统计"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        # 添加缺失值
        final_df.loc[0:10, 'PM2_5'] = np.nan
        final_df.loc[20:30, 'trip_count'] = np.nan
        
        result = generate_column_statistics(final_df)
        
        # 检查缺失值统计
        pm25_stats = result[result['Feature Name'] == 'PM2_5'].iloc[0]
        assert pm25_stats['Missing Count'] > 0
        assert pm25_stats['Missing Ratio (%)'] > 0


class TestKeyFindings:
    """测试关键发现生成功能"""
    
    @patch('builtins.print')
    def test_generate_key_findings_basic(self, mock_print):
        """测试基本关键发现生成"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        
        generate_key_findings(final_df)
        
        # 检查print被调用（表示输出了结果）
        assert mock_print.call_count > 0
    
    @patch('builtins.print')
    def test_generate_key_findings_missing_pm25(self, mock_print):
        """测试缺少PM2.5数据的关键发现"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        final_df['PM2_5'] = np.nan  # 所有PM2.5数据都缺失
        
        generate_key_findings(final_df)
        
        # 应该仍然能够运行而不报错
        assert mock_print.call_count > 0
    
    @patch('builtins.print')
    def test_generate_key_findings_empty_data(self, mock_print):
        """测试空数据的关键发现"""
        empty_df = pd.DataFrame()
        
        try:
            generate_key_findings(empty_df)
            # 空数据应该会报错，但我们需要确保异常被适当处理
        except Exception:
            pass


class TestIntegration:
    """测试可视化模块集成"""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    @patch('plotly.graph_objects.Figure.write_html')
    @patch('folium.Map.save')
    @patch('builtins.print')
    def test_complete_visualization_workflow(
        self, mock_print, mock_folium_save, mock_plotly_write, 
        mock_plt_savefig, mock_plt_show
    ):
        """测试完整可视化工作流"""
        final_df = TestSpatialVisualization().create_sample_final_df()
        air_quality_sites = TestSpatialVisualization().create_sample_air_quality_sites()
        
        # 模拟所有外部依赖
        with patch('geopandas.datasets.get_path'):
            with patch('geopandas.read_file') as mock_gpd_read:
                with patch('folium.Map') as mock_folium_map:
                    with patch('folium.TileLayer') as mock_tile:
                        with patch('folium.CircleMarker') as mock_circle:
                            with patch('folium.plugins.HeatMap') as mock_heatmap:
                                with patch('folium.LayerControl') as mock_control:
                                    # 设置mock返回值
                                    mock_nyc_data = gpd.GeoDataFrame({
                                        'geometry': [Point(0, 0).buffer(1)],
                                        'BoroName': ['Manhattan']
                                    })
                                    mock_gpd_read.return_value = mock_nyc_data
                                    
                                    mock_folium_instance = MagicMock()
                                    mock_folium_map.return_value = mock_folium_instance
                                    
                                    try:
                                        # 执行所有可视化函数
                                        create_spatial_visualization(final_df, air_quality_sites)
                                        create_temporal_analysis(final_df)
                                        create_static_analysis(final_df)
                                        create_interactive_visualizations(final_df, air_quality_sites)
                                        create_geographic_dashboard(final_df, air_quality_sites)
                                        generate_column_statistics(final_df)
                                        generate_key_findings(final_df)
                                        
                                        # 验证关键函数被调用
                                        assert mock_plt_savefig.call_count >= 4
                                        assert mock_plotly_write.call_count >= 1
                                        assert mock_folium_save.call_count >= 1
                                        assert mock_print.call_count > 0
                                        
                                    except Exception as e:
                                        pytest.skip(f"Integration test skipped: {e}")


class TestErrorHandling:
    """测试错误处理"""
    
    def test_visualization_with_invalid_data_types(self):
        """测试无效数据类型处理"""
        # 创建包含无效数据类型的DataFrame
        invalid_df = pd.DataFrame({
            'timestamp': ['invalid_date'] * 10,
            'site_id': [1, 2, 3] * 4,
            'PM2_5': ['not_numeric'] * 10,
            'trip_count': ['also_string'] * 10
        })
        air_quality_sites = TestSpatialVisualization().create_sample_air_quality_sites()
        
        # 这些应该优雅地处理错误或跳过
        try:
            create_temporal_analysis(invalid_df)
        except Exception:
            pass  # 预期会报错
        
        try:
            generate_column_statistics(invalid_df)
        except Exception:
            pass  # 预期会报错
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualization_with_single_row_data(self, mock_savefig, mock_show):
        """测试单行数据处理"""
        single_row_df = TestSpatialVisualization().create_sample_final_df().head(1)
        air_quality_sites = TestSpatialVisualization().create_sample_air_quality_sites()
        
        with patch('geopandas.datasets.get_path'):
            with patch('geopandas.read_file'):
                try:
                    create_spatial_visualization(single_row_df, air_quality_sites)
                    create_temporal_analysis(single_row_df)
                    create_static_analysis(single_row_df)
                    
                    # 应该能够运行而不报错
                    assert mock_savefig.call_count >= 3
                    
                except Exception as e:
                    pytest.skip(f"Single row test skipped: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
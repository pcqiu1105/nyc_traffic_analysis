import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_FILE_PATH = '/mnt/d/nyc_traffic_analysis'

def create_proper_interactive_dashboard(final_df, air_quality_sites, geojson_path, lstm_pred_df, realtime_pred_df):
    """创建修复版的交互式仪表盘"""
    
    print("Creating interactive dashboard...")
    print(f"Input data shape: {final_df.shape}")
    print(f"LSTM predictions shape: {lstm_pred_df.shape}")
    print(f"Realtime predictions shape: {realtime_pred_df.shape}")
    
    # 数据预处理
    final_df = final_df.copy()
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
    
    # 处理LSTM预测数据 - 按时间戳聚合5个区的平均值
    lstm_pred_df['timestamp'] = pd.to_datetime(lstm_pred_df['timestamp'])
    lstm_agg = lstm_pred_df.groupby('timestamp').agg({
        'true_t1': 'mean',
        'true_t3': 'mean',
        'true_t6': 'mean',
        'pred_t1': 'mean',
        'pred_t3': 'mean',
        'pred_t6': 'mean'
    }).reset_index()
    lstm_agg['timestamp'] = lstm_agg['timestamp'].astype(str)
    
    print(f"LSTM aggregated data shape: {lstm_agg.shape}")
    
    # 处理实时预测数据 - 按时间戳聚合5个区的平均值
    realtime_pred_df['timestamp'] = pd.to_datetime(realtime_pred_df['timestamp'])
    realtime_agg = realtime_pred_df.groupby('timestamp').agg({
        'PM2_5_true': 'mean',
        'PM2_5_pred_XGBoost': 'mean',
        'PM2_5_pred_LightGBM': 'mean',
        'PM2_5_pred_RandomForest': 'mean'
    }).reset_index()
    
    print(f"Realtime aggregated data shape: {realtime_agg.shape}")
    
    # 计算三个模型的评估指标
    model_metrics = {}
    for model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
        col_name = f'PM2_5_pred_{model_name}'
        y_true = realtime_agg['PM2_5_true'].values
        y_pred = realtime_agg[col_name].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        model_metrics[model_name] = {
            'MAE': round(mae, 3),
            'RMSE': round(float(rmse), 3),
            'R2': round(r2, 3)
        }
    
    print(f"Model metrics calculated: {model_metrics}")
    
    # 打印数据基本信息
    print(f"Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
    
    # 创建基础统计数据
    stats_data = {
        'start_date': final_df['timestamp'].min().strftime('%Y-%m-%d'),
        'end_date': final_df['timestamp'].max().strftime('%Y-%m-%d'),
        'total_records': f"{len(final_df):,}",
        'boroughs': ', '.join(sorted(final_df['borough'].unique())),
        'avg_pm25': f"{final_df['PM2_5'].mean():.2f}",
        'total_trips': f"{final_df['trip_count'].sum():,}",
        'avg_speed': f"{final_df['avg_speed'].mean():.1f}"
    }

    # 准备完整的原始数据供前端筛选
    print("Preparing full dataset for frontend filtering...")
    
    # 每日数据聚合
    full_daily_data = final_df.groupby(final_df['timestamp'].dt.date).agg({
        'PM2_5': 'mean',
        'trip_count': 'sum',
        'avg_speed': 'mean',
        'temperature': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    full_daily_data.columns = ['date', 'PM2_5', 'trip_count', 'avg_speed', 'temperature', 'wind_speed']
    full_daily_data['date'] = full_daily_data['date'].astype(str)
    
    # 每小时数据聚合
    full_hourly_data = final_df.groupby('hour_of_day').agg({
        'trip_count': 'mean',
        'PM2_5': 'mean',
        'avg_speed': 'mean'
    }).reset_index()
    
    # 行政区数据聚合
    full_borough_data = final_df.groupby(['borough', final_df['timestamp'].dt.date]).agg({
        'PM2_5': 'mean',
        'trip_count': 'sum',
        'avg_speed': 'mean'
    }).reset_index()
    full_borough_data.columns = ['borough', 'date', 'PM2_5', 'trip_count', 'avg_speed']
    full_borough_data['date'] = full_borough_data['date'].astype(str)
    
    lstm_agg['timestamp'] = lstm_agg['timestamp'].astype(str)
    realtime_agg['timestamp'] = realtime_agg['timestamp'].astype(str)

    # 准备基础数据（立即加载）
    full_data_json = json.dumps({
        'daily': full_daily_data.to_dict('records'),
        'hourly': full_hourly_data.to_dict('records'),
        'borough': full_borough_data.to_dict('records'),
        'date_range': {
            'min': full_daily_data['date'].min(),
            'max': full_daily_data['date'].max()
        }
    })

    # 预测数据单独序列化（延迟加载）
    lstm_json = json.dumps(lstm_agg.to_dict('records'))
    realtime_json = json.dumps(realtime_agg.to_dict('records'))
    
    # 创建地图可视化
    print("Creating Folium map...")
    try:
        import folium
        
        # 读取 GeoJSON
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        # 聚合行政区数据
        borough_agg = final_df.groupby('borough').agg({
            'PM2_5': 'mean',
            'trip_count': 'sum',
            'avg_speed': 'mean',
            'total_revenue': 'sum',
            'temperature': 'mean',
            'humidity': 'mean'
        }).reset_index()
        
        print(f"Borough aggregation for map:\n{borough_agg}")
        
        # 创建 Folium 地图
        m = folium.Map(
            location=[40.7128, -74.0060],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # 创建 choropleth 图层
        folium.Choropleth(
            geo_data=geojson_data,
            name='choropleth',
            data=borough_agg,
            columns=['borough', 'PM2_5'],
            key_on='feature.properties.BoroName',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=1.0,
            line_weight=3,
            line_color='navy',
            legend_name='PM2.5 (μg/m³)',
            highlight=True
        ).add_to(m)
        
        # 添加详细信息弹窗
        for _, row in borough_agg.iterrows():
            # 找到对应的 GeoJSON feature
            for feature in geojson_data['features']:
                if feature['properties']['BoroName'] == row['borough']:
                    # 计算中心点
                    coords = feature['geometry']['coordinates']
                    if feature['geometry']['type'] == 'Polygon':
                        lons = [c[0] for c in coords[0]]
                        lats = [c[1] for c in coords[0]]
                    else:  # MultiPolygon
                        lons = [c[0] for polygon in coords for c in polygon[0]]
                        lats = [c[1] for polygon in coords for c in polygon[0]]
                    
                    center_lat = sum(lats) / len(lats)
                    center_lon = sum(lons) / len(lons)
                    
                    # 创建弹窗内容
                    popup_html = f"""
                    <div style="font-family: Arial; width: 250px;">
                        <h3 style="color: #2c3e50; margin-bottom: 10px;">{row['borough']}</h3>
                        <hr style="margin: 5px 0;">
                        <p style="margin: 5px 0;"><b>Air Quality:</b></p>
                        <p style="margin: 2px 0 2px 15px;">PM2.5: <b>{row['PM2_5']:.2f}</b> μg/m³</p>
                        <p style="margin: 5px 0;"><b>Traffic:</b></p>
                        <p style="margin: 2px 0 2px 15px;">Total Trips: <b>{row['trip_count']:,.0f}</b></p>
                        <p style="margin: 2px 0 2px 15px;">Avg Speed: <b>{row['avg_speed']:.1f}</b> mph</p>
                        <p style="margin: 5px 0;"><b>Revenue:</b></p>
                        <p style="margin: 2px 0 2px 15px;">Total: <b>${row['total_revenue']/1000000:.2f}M</b></p>
                        <p style="margin: 5px 0;"><b>Weather:</b></p>
                        <p style="margin: 2px 0 2px 15px;">Temp: <b>{row['temperature']:.1f}°C</b></p>
                        <p style="margin: 2px 0 2px 15px;">Humidity: <b>{row['humidity']:.0f}%</b></p>
                    </div>
                    """
                    
                    # 添加标记
                    folium.Marker(
                        location=[center_lat, center_lon],
                        popup=folium.Popup(popup_html, max_width=300),
                        icon=folium.Icon(color='red', icon='info-sign'),
                        tooltip=f"{row['borough']}: PM2.5 {row['PM2_5']:.1f}"
                    ).add_to(m)
                    
                    break
        
        # 添加图层控制
        folium.LayerControl().add_to(m)
        
        # 转换为 HTML iframe
        map_html = m._repr_html_()
        
        print("Folium map created successfully")
        
    except Exception as e:
        print(f"Folium map creation failed: {e}")
        import traceback
        traceback.print_exc()
        map_html = "<p style='color: red; text-align: center; padding: 50px;'>地图加载失败</p>"

    # 创建时间序列图表
    print("Creating time series...")
    try:
        daily_data = final_df.groupby(final_df['timestamp'].dt.date).agg({
            'PM2_5': 'mean',
            'trip_count': 'sum',
            'avg_speed': 'mean'
        }).reset_index()
        daily_data.columns = ['date', 'PM2_5', 'trip_count', 'avg_speed']
        
        # 转换日期为字符串格式
        daily_data['date'] = daily_data['date'].astype(str)
        
        print(f"Daily data shape: {daily_data.shape}")
        
        time_fig = go.Figure()
        
        # PM2.5趋势线
        time_fig.add_trace(go.Scatter(
            x=daily_data['date'].tolist(),
            y=daily_data['PM2_5'].tolist(),
            name='PM2.5',
            line=dict(color='red', width=3),
            mode='lines+markers'
        ))
        
        # 行程数柱状图
        time_fig.add_trace(go.Bar(
            x=daily_data['date'].tolist(),
            y=daily_data['trip_count'].tolist(),
            name='Trip Count',
            marker_color='lightblue',
            opacity=0.6,
            yaxis='y2'
        ))
        
        time_fig.update_layout(
            title="Daily PM2.5 and Traffic Trends",
            xaxis=dict(
                title='Date',
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1周", step="day", stepmode="backward"),
                        dict(count=1, label="1月", step="month", stepmode="backward"),
                        dict(count=2, label="2月", step="month", stepmode="backward"),
                        dict(step="all", label="全部")
                    ]),
                    bgcolor='lightblue',
                    activecolor='orange'
                ),
                rangeslider=dict(visible=True, thickness=0.05),
                type='date'
            ),
            yaxis=dict(
                title='PM2.5 (μg/m³)',
                tickfont=dict(color='red'),
                side='left'
            ),
            yaxis2=dict(
                title='Trip Count',
                tickfont=dict(color='blue'),
                overlaying='y',
                side='right'
            ),
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
    except Exception as e:
        print(f"Time series failed: {e}")
        import traceback
        traceback.print_exc()
        time_fig = go.Figure()
        time_fig.add_annotation(text=f"Time series error: {str(e)}", showarrow=False)
        time_fig.update_layout(title="Daily Trends", height=400)
    
    # 创建交通模式分析图表
    print("Creating traffic patterns...")
    try:
        hourly_data = final_df.groupby('hour_of_day').agg({
            'trip_count': 'mean',
            'PM2_5': 'mean',
            'avg_speed': 'mean'
        }).reset_index()
        
        print(f"Hourly data shape: {hourly_data.shape}")
        
        traffic_fig = go.Figure()
        
        # 行程数柱状图
        traffic_fig.add_trace(go.Bar(
            x=hourly_data['hour_of_day'].tolist(),
            y=hourly_data['trip_count'].tolist(),
            name='Average Trips',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # PM2.5折线图
        traffic_fig.add_trace(go.Scatter(
            x=hourly_data['hour_of_day'].tolist(),
            y=hourly_data['PM2_5'].tolist(),
            name='PM2.5',
            line=dict(color='red', width=3),
            mode='lines+markers',
            yaxis='y2'
        ))
        
        traffic_fig.update_layout(
            title="Daily Traffic and Pollution Patterns",
            xaxis=dict(title='Hour of Day', tickmode='linear'),
            yaxis=dict(
                title='Average Trip Count',
                side='left'
            ),
            yaxis2=dict(
                title='PM2.5 (μg/m³)',
                side='right',
                overlaying='y'
            ),
            height=400,
            showlegend=True
        )
        
    except Exception as e:
        print(f"Traffic patterns failed: {e}")
        import traceback
        traceback.print_exc()
        traffic_fig = go.Figure()
        traffic_fig.add_annotation(text=f"Traffic pattern error: {str(e)}", showarrow=False)
        traffic_fig.update_layout(title="Traffic Patterns", height=400)
    
    # 创建气象影响分析图表
    print("Creating weather impact analysis...")
    try:
        if 'temperature' not in final_df.columns:
            raise ValueError("Temperature column not found")
        if 'wind_speed' not in final_df.columns:
            raise ValueError("Wind speed column not found")
            
        # 抽样数据以避免过多点
        sample_size = min(2000, len(final_df))
        sample_df = final_df.sample(sample_size, random_state=42).copy()
        
        print(f"Weather sample size: {len(sample_df)}")
        print(f"Temperature range: {sample_df['temperature'].min():.1f} to {sample_df['temperature'].max():.1f}")
        print(f"PM2.5 range: {sample_df['PM2_5'].min():.1f} to {sample_df['PM2_5'].max():.1f}")
        
        weather_fig = go.Figure(data=[
            go.Scatter(
                x=sample_df['temperature'].tolist(),
                y=sample_df['PM2_5'].tolist(),
                mode='markers',
                marker=dict(
                    size=(sample_df['trip_count'] / sample_df['trip_count'].max() * 30 + 5).tolist(),
                    color=sample_df['wind_speed'].tolist(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Wind Speed<br>(m/s)"),
                    line=dict(width=0.5, color='white')
                ),
                text=sample_df['borough'].tolist(),
                hovertemplate='<b>Temperature:</b> %{x:.1f}°C<br>' +
                            '<b>PM2.5:</b> %{y:.1f} μg/m³<br>' +
                            '<b>Wind Speed:</b> %{marker.color:.1f} m/s<br>' +
                            '<b>Borough:</b> %{text}<br>' +
                            '<b>Hour:</b> %{customdata[0]:.0f}<br>' +
                            '<b>Trips:</b> %{customdata[1]:,.0f}' +
                            '<extra></extra>',
                customdata=np.column_stack((
                    sample_df['hour_of_day'].tolist(),
                    sample_df['trip_count'].tolist()
                )).tolist()
            )
        ])
        
        weather_fig.update_layout(
            title="Temperature vs PM2.5 Concentration",
            xaxis_title="Temperature (°C)",
            yaxis_title="PM2.5 (μg/m³)",
            height=400,
            hovermode='closest'
        )
        
    except Exception as e:
        print(f"Weather impact failed: {e}")
        import traceback
        traceback.print_exc()
        weather_fig = go.Figure()
        weather_fig.add_annotation(text=f"Weather error: {str(e)}", showarrow=False)
        weather_fig.update_layout(title="Weather Impact", height=400)

    
    # 新增: LSTM多步预测图表
    print("Creating LSTM multi-step prediction chart...")
    try:
        lstm_fig = go.Figure()
        
        # t+1
        lstm_fig.add_trace(go.Scatter(
            x=lstm_agg['timestamp'].tolist(),
            y=lstm_agg['true_t1'].tolist(),
            name='True t+1',
            line=dict(color='#2c3e50', width=2),
            mode='lines'
        ))
        lstm_fig.add_trace(go.Scatter(
            x=lstm_agg['timestamp'].tolist(),
            y=lstm_agg['pred_t1'].tolist(),
            name='Pred t+1',
            line=dict(color='#3498db', width=2, dash='dash'),
            mode='lines'
        ))
        
        # t+3
        lstm_fig.add_trace(go.Scatter(
            x=lstm_agg['timestamp'].tolist(),
            y=lstm_agg['true_t3'].tolist(),
            name='True t+3',
            line=dict(color='#27ae60', width=2),
            mode='lines'
        ))
        lstm_fig.add_trace(go.Scatter(
            x=lstm_agg['timestamp'].tolist(),
            y=lstm_agg['pred_t3'].tolist(),
            name='Pred t+3',
            line=dict(color='#2ecc71', width=2, dash='dash'),
            mode='lines'
        ))
        
        # t+6
        lstm_fig.add_trace(go.Scatter(
            x=lstm_agg['timestamp'].tolist(),
            y=lstm_agg['true_t6'].tolist(),
            name='True t+6',
            line=dict(color='#c0392b', width=2),
            mode='lines'
        ))
        lstm_fig.add_trace(go.Scatter(
            x=lstm_agg['timestamp'].tolist(),
            y=lstm_agg['pred_t6'].tolist(),
            name='Pred t+6',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            mode='lines'
        ))
        
        lstm_fig.update_layout(
            title="LSTM Multi-step PM2.5 Prediction (NYC Average)",
            xaxis_title="Timestamp",
            yaxis_title="PM2.5 (μg/m³)",
            height=450,
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        print("LSTM prediction chart created")
        
    except Exception as e:
        print(f"LSTM prediction chart failed: {e}")
        import traceback
        traceback.print_exc()
        lstm_fig = go.Figure()
        lstm_fig.add_annotation(text=f"LSTM chart error: {str(e)}", showarrow=False)
        lstm_fig.update_layout(title="LSTM Prediction", height=400)

    # 新增: 三模型实时预测对比图表
    print("Creating real-time model comparison chart...")
    try:
        realtime_agg['timestamp_str'] = realtime_agg['timestamp'].astype(str)
        
        realtime_fig = go.Figure()
        
        # 真实值
        realtime_fig.add_trace(go.Scatter(
            x=realtime_agg['timestamp_str'].tolist(),
            y=realtime_agg['PM2_5_true'].tolist(),
            name='True PM2.5',
            line=dict(color='#2c3e50', width=3),
            mode='lines'
        ))
        
        # XGBoost预测
        realtime_fig.add_trace(go.Scatter(
            x=realtime_agg['timestamp_str'].tolist(),
            y=realtime_agg['PM2_5_pred_XGBoost'].tolist(),
            name='XGBoost',
            line=dict(color='#3498db', width=2, dash='dash'),
            mode='lines'
        ))
        
        # LightGBM预测
        realtime_fig.add_trace(go.Scatter(
            x=realtime_agg['timestamp_str'].tolist(),
            y=realtime_agg['PM2_5_pred_LightGBM'].tolist(),
            name='LightGBM',
            line=dict(color='#2ecc71', width=2, dash='dash'),
            mode='lines'
        ))
        
        # RandomForest预测
        realtime_fig.add_trace(go.Scatter(
            x=realtime_agg['timestamp_str'].tolist(),
            y=realtime_agg['PM2_5_pred_RandomForest'].tolist(),
            name='RandomForest',
            line=dict(color='#e67e22', width=2, dash='dash'),
            mode='lines'
        ))
        
        realtime_fig.update_layout(
            title="Real-time PM2.5 Prediction Comparison (NYC Average)",
            xaxis_title="Timestamp",
            yaxis_title="PM2.5 (μg/m³)",
            height=450,
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        print("Real-time comparison chart created")
        
    except Exception as e:
        print(f"Real-time comparison chart failed: {e}")
        import traceback
        traceback.print_exc()
        realtime_fig = go.Figure()
        realtime_fig.add_annotation(text=f"Real-time chart error: {str(e)}", showarrow=False)
        realtime_fig.update_layout(title="Real-time Prediction", height=400)
    
    # 将图表转换为JSON格式
    print("Converting figures to JSON...")
    try:
        
        # 使用正确的序列化方法
        def serialize_figure(fig):
            return json.loads(fig.to_json())
        
        time_data = serialize_figure(time_fig)
        traffic_data = serialize_figure(traffic_fig)
        weather_data = serialize_figure(weather_fig)
        lstm_data = serialize_figure(lstm_fig)
        realtime_data = serialize_figure(realtime_fig)
        
        charts_data = {
            'time': time_data,
            'traffic': traffic_data,
            'weather': weather_data,
            'lstm': lstm_data,
            'realtime': realtime_data
        }
        
        for name, data in charts_data.items():
            print(f"  {name}: {len(str(data))} chars")
        
    except Exception as e:
        print(f"JSON conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 生成HTML仪表盘
    print("Generating HTML dashboard...")
    
    # 在生成HTML的部分，修改JavaScript代码，添加详细的错误处理：

    # 在生成HTML的部分，修改JavaScript代码，添加详细的错误处理：

    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NYC Taxi & Air Quality Interactive Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }}
            .dashboard-container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header {{
                background: #2c3e50;
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.2em;
            }}
            .control-panel {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 15px rgba(0,0,0,0.1);
                margin: 25px auto;
                max-width: 1200px;
            }}
            .filter-section {{
                display: flex;
                flex-wrap: wrap;
                gap: 40px;
                align-items: start;
                justify-content: center;
                margin-bottom: 25px;
            }}
            .filter-group {{
                display: flex;
                flex-direction: column;
                gap: 12px;
                min-width: 220px;
                text-align: center;
                align-items: center;
            }}
            .filter-group label {{
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }}
            .date-filter {{
                display: flex;
                gap: 10px;
                align-items: center;
                justify-content: center;
            }}
            .date-filter input[type="date"] {{
                padding: 8px 12px;
                border: 2px solid #3498db;
                border-radius: 5px;
                font-size: 14px;
            }}
            .checkbox-group {{
                display: flex;
                flex-direction: column;
                gap: 8px;
                max-height: 160px;
                overflow-y: auto;
                border: 2px solid #3498db;
                border-radius: 6px;
                padding: 12px;
                background: white;
                width: 100%;
            }}
            .checkbox-item {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .checkbox-item input[type="checkbox"] {{
                width: 16px;
                height: 16px;
            }}
            .export-group {{
                display: flex;
                flex-direction: column;
                gap: 12px;
                min-width: 220px;
                text-align: center;
                align-items: center;
            }}
            .export-btn {{
                padding: 12px 20px;
                background: #27ae60;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
                font-size: 14px;
                transition: background 0.3s;
                text-align: center;
                width: 100%;
                max-width: 250px;
            }}
            .export-btn:hover {{
                background: #229954;
            }}
            .export-btn.advanced {{
                background: #95a5a6;
            }}
            .export-btn.advanced:hover {{
                background: #7f8c8d;
            }}
            .action-buttons {{
                display: flex;
                gap: 12px;
                margin-top: 15px;
                justify-content: center;
            }}
            .action-btn {{
                padding: 10px 20px;
                background: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
            }}
            .action-btn:hover {{
                background: #2980b9;
            }}
            .action-btn.reset {{
                background: #95a5a6;
            }}
            .action-btn.reset:hover {{
                background: #7f8c8d;
            }}
            #advancedExportPanel {{
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
                display: none;
                grid-column: 1 / -1;
                text-align: center;
            }}
            .export-options {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                justify-items: center;
            }}
            .option-group {{
                background: white;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
                width: 100%;
                max-width: 300px;
                text-align: left;
            }}
            .option-group h4 {{
                margin: 0 0 8px 0;
                color: #2c3e50;
                font-size: 13px;
                border-bottom: 1px solid #3498db;
                padding-bottom: 5px;
            }}
            .option-group label {{
                display: block;
                margin: 6px 0;
                font-size: 12px;
                color: #2c3e50;
                cursor: pointer;
            }}
            .mini-btn {{
                padding: 4px 8px;
                background: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                font-size: 11px;
                margin-right: 5px;
                margin-bottom: 5px;
            }}
            .export-btn-large {{
                width: 100%;
                max-width: 300px;
                padding: 12px;
                background: #e67e22;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
                margin: 15px auto 0;
                display: block;
            }}
            .charts-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 25px;
                padding: 0 25px 25px 25px;
                max-width: 1200px;
                margin: 0 auto;
            }}
            .chart-container {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 15px rgba(0,0,0,0.1);
                min-height: 450px;
                margin: 0 auto 25px auto;
                width: 100%;
                position: relative;
                z-index: 1;
            }}
            .chart-container > div {{
                width: 100%;
                height: 100%;
            }}
            .full-width {{
                grid-column: 1 / -1;
            }}
            .metrics-container {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin: 20px 0;
                padding: 25px;
                background: #ecf0f1;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                min-height: 200px;
                position: relative;
                z-index: 2;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                text-align: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}
            .metric-card h4 {{
                margin: 0 0 15px 0;
                color: #2c3e50;
                font-size: 1.2em;
                border-bottom: 3px solid #3498db;
                padding-bottom: 8px;
            }}
            .metric-row {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #ecf0f1;
                align-items: center;
            }}
            .metric-row:last-child {{
                border-bottom: none;
            }}
            .metric-label {{
                font-weight: bold;
                color: #7f8c8d;
                font-size: 14px;
            }}
            .metric-value {{
                color: #2c3e50;
                font-weight: bold;
                font-size: 15px;
            }}
            #filterInfo {{
                color: #7f8c8d;
                font-size: 14px;
                margin-top: 10px;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
            }}
            .loading-message {{
                text-align: center;
                padding: 20px;
                color: #7f8c8d;
                font-style: italic;
            }}
            .error-message {{
                text-align: center;
                padding: 20px;
                color: #e74c3c;
                background: #fdf2f2;
                border: 1px solid #e74c3c;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <!-- 调试信息面板 -->
        <div class="debug-info" id="debugInfo"></div>

        <div class="dashboard-container">
            <div class="header">
                <h1>NYC Taxi & Air Quality Dashboard</h1>
                <p>Interactive Analysis of Traffic Patterns and Pollution Levels with ML Predictions</p>
            </div>
            
            <!-- 控制面板 -->
            <div class="control-panel">
                <div class="filter-section">
                    <!-- 日期筛选 -->
                    <div class="filter-group">
                        <label>日期范围</label>
                        <div class="date-filter">
                            <input type="date" id="startDate" value="2024-01-01" min="2024-01-01" max="2024-03-31">
                            <span>至</span>
                            <input type="date" id="endDate" value="2024-03-31" min="2024-01-01" max="2024-03-31">
                        </div>
                    </div>
                    
                    <!-- 行政区筛选 -->
                    <div class="filter-group">
                        <label>行政区选择</label>
                        <div class="checkbox-group">
                            <div class="checkbox-item">
                                <input type="checkbox" id="boroughBronx" checked>
                                <label for="boroughBronx">Bronx</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="boroughBrooklyn" checked>
                                <label for="boroughBrooklyn">Brooklyn</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="boroughManhattan" checked>
                                <label for="boroughManhattan">Manhattan</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="boroughQueens" checked>
                                <label for="boroughQueens">Queens</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="boroughStatenIsland" checked>
                                <label for="boroughStatenIsland">Staten Island</label>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 操作按钮 -->
                    <div class="filter-group">
                        <label>数据操作</label>
                        <div class="action-buttons">
                            <button class="action-btn" onclick="applyFilters()">应用筛选</button>
                            <button class="action-btn reset" onclick="resetFilters()">重置全部</button>
                        </div>
                    </div>
                    
                    <!-- 导出按钮组 -->
                    <div class="export-group">
                        <label>数据导出</label>
                        <button class="export-btn" onclick="quickExport()">快速导出预测数据 (CSV)</button>
                        <button class="export-btn advanced" onclick="toggleAdvancedExport()">高级选项 <span id="advancedArrow">▼</span></button>
                    </div>
                </div>
                
                <!-- 筛选信息显示 -->
                <div id="filterInfo"></div>
                
                <!-- 高级导出选项 -->
                <div id="advancedExportPanel">
                    <div class="export-options">
                        <div class="option-group">
                            <h4>选择数据集:</h4>
                            <label><input type="checkbox" id="exportLSTM" checked> LSTM多步预测</label>
                            <label><input type="checkbox" id="exportRealtime" checked> 实时预测对比</label>
                            <label><input type="checkbox" id="exportDaily"> 每日汇总数据</label>
                            <label><input type="checkbox" id="exportHourly"> 每小时数据</label>
                            <label><input type="checkbox" id="exportBorough"> 行政区详细数据</label>
                        </div>
                        
                        <div class="option-group">
                            <h4>LSTM预测字段:</h4>
                            <div>
                                <button class="mini-btn" onclick="selectAllLSTM()">全选</button>
                                <button class="mini-btn" onclick="deselectAllLSTM()">取消全选</button>
                            </div>
                            <label><input type="checkbox" class="lstm-field" value="timestamp" checked> timestamp</label>
                            <label><input type="checkbox" class="lstm-field" value="true_t1" checked> true_t1</label>
                            <label><input type="checkbox" class="lstm-field" value="pred_t1" checked> pred_t1</label>
                            <label><input type="checkbox" class="lstm-field" value="true_t3" checked> true_t3</label>
                            <label><input type="checkbox" class="lstm-field" value="pred_t3" checked> pred_t3</label>
                            <label><input type="checkbox" class="lstm-field" value="true_t6" checked> true_t6</label>
                            <label><input type="checkbox" class="lstm-field" value="pred_t6" checked> pred_t6</label>
                        </div>
                        
                        <div class="option-group">
                            <h4>实时预测字段:</h4>
                            <div>
                                <button class="mini-btn" onclick="selectAllRealtime()">全选</button>
                                <button class="mini-btn" onclick="deselectAllRealtime()">取消全选</button>
                            </div>
                            <label><input type="checkbox" class="realtime-field" value="timestamp" checked> timestamp</label>
                            <label><input type="checkbox" class="realtime-field" value="PM2_5_true" checked> PM2_5_true</label>
                            <label><input type="checkbox" class="realtime-field" value="PM2_5_pred_XGBoost" checked> PM2_5_pred_XGBoost</label>
                            <label><input type="checkbox" class="realtime-field" value="PM2_5_pred_LightGBM" checked> PM2_5_pred_LightGBM</label>
                            <label><input type="checkbox" class="realtime-field" value="PM2_5_pred_RandomForest" checked> PM2_5_pred_RandomForest</label>
                        </div>
                        
                        <div class="option-group">
                            <h4>文件格式:</h4>
                            <label><input type="radio" name="exportFormat" value="csv" checked> CSV</label>
                            <label><input type="radio" name="exportFormat" value="json"> JSON</label>
                        </div>
                    </div>
                    <button class="export-btn-large" onclick="advancedExport()">导出所选数据</button>
                </div>
            </div>
            
            <!-- 地图 -->
            <div class="chart-container full-width" style="padding: 0; margin: 20px;">
                <h3 style="text-align: center; color: #2c3e50; margin: 10px 0;">NYC Air Quality & Traffic Overview by Borough</h3>
                <div style="width: 100%; height: 500px; border: 2px solid navy; overflow: hidden;">
                    {map_html}
                </div>
            </div>
            
            <div class="charts-grid">    
                <!-- 时间序列图 -->
                <div class="chart-container full-width">
                    <div id="time-chart">
                        <div class="loading-message">加载时间序列图表...</div>
                    </div>
                </div>
                
                <!-- 交通模式图 -->
                <div class="chart-container">
                    <div id="traffic-chart">
                        <div class="loading-message">加载交通模式图表...</div>
                    </div>
                </div>
                
                <!-- 气象影响 -->
                <div class="chart-container">
                    <div id="weather-chart">
                        <div class="loading-message">加载气象影响图表...</div>
                    </div>
                </div>
                
                <!-- LSTM多步预测 -->
                <div class="chart-container full-width">
                    <div id="lstm-chart">
                        <div class="loading-message">加载LSTM预测图表...</div>
                    </div>
                </div>
                
                <!-- 实时预测对比 -->
                <div class="chart-container full-width">
                    <div id="realtime-chart">
                        <div class="loading-message">加载实时预测图表...</div>
                    </div>
                </div>
                <!-- 模型指标卡片 -->
                <div class="metrics-container">
                    <div class="metric-card">
                        <h4>XGBoost</h4>
                        <div class="metric-row">
                            <span class="metric-label">MAE:</span>
                            <span class="metric-value">{model_metrics['XGBoost']['MAE']}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">RMSE:</span>
                            <span class="metric-value">{model_metrics['XGBoost']['RMSE']}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">R²:</span>
                            <span class="metric-value">{model_metrics['XGBoost']['R2']}</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>LightGBM</h4>
                        <div class="metric-row">
                            <span class="metric-label">MAE:</span>
                            <span class="metric-value">{model_metrics['LightGBM']['MAE']}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">RMSE:</span>
                            <span class="metric-value">{model_metrics['LightGBM']['RMSE']}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">R²:</span>
                            <span class="metric-value">{model_metrics['LightGBM']['R2']}</span>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h4>Random Forest</h4>
                        <div class="metric-row">
                            <span class="metric-label">MAE:</span>
                            <span class="metric-value">{model_metrics['RandomForest']['MAE']}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">RMSE:</span>
                            <span class="metric-value">{model_metrics['RandomForest']['RMSE']}</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">R²:</span>
                            <span class="metric-value">{model_metrics['RandomForest']['R2']}</span>
                        </div>
                    </div>
                </div>
            </div>
            <!-- 页脚 -->
            <footer style="
                background: #2c3e50;
                color: white;
                text-align: center;
                padding: 20px;
                margin-top: 30px;
                font-family: Arial, sans-serif;
                border-top: 3px solid #3498db;
            ">
                <div style="max-width: 1200px; margin: 0 auto;">
                    <div style="margin-bottom: 10px; font-size: 16px; font-weight: bold;">
                        NYC Traffic & Air Quality Analysis Platform
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 5px; font-size: 14px; margin-bottom: 10px;">
                        <div>
                            <span>2025</span>
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            <span>邱鹏程</span>
                        </div>
                        <div>
                            <a href="mailto:231820201@smail.nju.edu.cn" style="color: white; text-decoration: none;">
                                231820201@smail.nju.edu.cn
                            </a>
                        </div>
                    </div>
                    <div style="margin-top: 15px; font-size: 12px; color: #bdc3c7;">
                        © 2025 NYC Traffic Analysis Project. All rights reserved.
                    </div>
                </div>
            </footer>   
        </div>
        <script>
            // ========== 调试功能 ==========
            let debugMode = false;
            
            function toggleDebug() {{
                debugMode = !debugMode;
                const debugInfo = document.getElementById('debugInfo');
                debugInfo.style.display = debugMode ? 'block' : 'none';
                updateDebugInfo('调试模式: ' + (debugMode ? '开启' : '关闭'));
            }}
            
            function updateDebugInfo(message) {{
                if (!debugMode) return;
                const debugInfo = document.getElementById('debugInfo');
                const timestamp = new Date().toLocaleTimeString();
                debugInfo.innerHTML += `[${{timestamp}}] ${{message}}<br>`;
                debugInfo.scrollTop = debugInfo.scrollHeight;
            }}
            
            function logChartData(chartName, data) {{
                if (!debugMode) return;
                updateDebugInfo(`${{chartName}} - data: ${{data?.data?.length || 0}} traces, layout: ${{!!data?.layout}}`);
            }}
            
            // ========== 全局变量声明 ==========
            let fullData = null;
            let originalCharts = {{}};
            let lstmData = null;
            let realtimeData = null;
            let exportInitialized = false;

            // 预测数据JSON字符串
            const lstmDataJSON = `{lstm_json}`;
            const realtimeDataJSON = `{realtime_json}`;

            // ========== 初始化函数 ==========
            function initializeDashboard() {{
                console.log('🚀 Dashboard initializing...');
                updateDebugInfo('开始初始化仪表盘');
                
                // 检查Plotly是否加载
                if (typeof Plotly === 'undefined') {{
                    const errorMsg = '❌ Plotly库未加载，请检查网络连接';
                    console.error(errorMsg);
                    updateDebugInfo(errorMsg);
                    showChartError('所有图表', 'Plotly库加载失败');
                    return;
                }}
                
                updateDebugInfo('✅ Plotly库已加载');
                
                // 检查DOM元素
                const chartContainers = ['time-chart', 'traffic-chart', 'weather-chart', 'lstm-chart', 'realtime-chart'];
                chartContainers.forEach(containerId => {{
                    const container = document.getElementById(containerId);
                    if (!container) {{
                        updateDebugInfo(`❌ 图表容器未找到: ${{containerId}}`);
                    }} else {{
                        updateDebugInfo(`✅ 图表容器已找到: ${{containerId}}`);
                    }}
                }});
                
                // 加载基础数据
                try {{
                    fullData = {full_data_json};
                    updateDebugInfo('✅ 基础数据加载成功');
                    console.log('Basic data loaded:', fullData);
                }} catch(e) {{
                    const errorMsg = `❌ 基础数据加载失败: ${{e.message}}`;
                    console.error(errorMsg);
                    updateDebugInfo(errorMsg);
                }}

                // 加载所有图表
                loadAllCharts();
            }}
            
            // ========== 图表加载函数 ==========
            function loadAllCharts() {{
                console.log('📊 Loading all charts...');
                updateDebugInfo('开始加载所有图表');
                
                try {{
                    // 加载基础图表数据
                    const time_data = {json.dumps(time_data)};
                    const traffic_data = {json.dumps(traffic_data)};
                    const weather_data = {json.dumps(weather_data)};
                    const lstm_data = {json.dumps(lstm_data)};
                    const realtime_data = {json.dumps(realtime_data)};
                    
                    updateDebugInfo('✅ 图表数据解析成功');
                    logChartData('时间序列', time_data);
                    logChartData('交通模式', traffic_data);
                    logChartData('气象影响', weather_data);
                    logChartData('LSTM预测', lstm_data);
                    logChartData('实时预测', realtime_data);
                    
                    // 保存原始配置
                    originalCharts = {{
                        time: time_data,
                        traffic: traffic_data,
                        weather: weather_data,
                        lstm: lstm_data,
                        realtime: realtime_data
                    }};
                    
                    updateDebugInfo('✅ 图表配置保存成功');
                    
                    // 渲染基础图表
                    renderChart('time-chart', time_data, '时间序列');
                    renderChart('traffic-chart', traffic_data, '交通模式');
                    renderChart('weather-chart', weather_data, '气象影响');
                    
                    updateDebugInfo('✅ 基础图表渲染完成');
                    
                    // 延迟加载预测数据
                    setTimeout(() => {{
                        try {{
                            updateDebugInfo('开始加载预测数据');
                            lstmData = JSON.parse(lstmDataJSON);
                            realtimeData = JSON.parse(realtimeDataJSON);
                            updateDebugInfo('✅ 预测数据加载成功');
                            
                            // 渲染预测图表
                            renderChart('lstm-chart', lstm_data, 'LSTM预测');
                            renderChart('realtime-chart', realtime_data, '实时预测');
                            
                            updateDebugInfo('🎉 所有图表加载完成！');
                            console.log('✅ All charts rendered successfully!');
                        }} catch(e) {{
                            const errorMsg = `❌ 预测图表加载失败: ${{e.message}}`;
                            console.error(errorMsg);
                            updateDebugInfo(errorMsg);
                            showChartError('预测图表', e.message);
                        }}
                    }}, 1000);
                    
                }} catch(e) {{
                    const errorMsg = `❌ 图表数据加载失败: ${{e.message}}`;
                    console.error(errorMsg);
                    updateDebugInfo(errorMsg);
                    showChartError('所有图表', e.message);
                }}
            }}
            
            function renderChart(containerId, chartData, chartName) {{
                try {{
                    const container = document.getElementById(containerId);
                    if (!container) {{
                        throw new Error(`图表容器 ${{containerId}} 未找到`);
                    }}
                    
                    if (!chartData || !chartData.data) {{
                        throw new Error(`${{chartName}} 数据格式错误`);
                    }}

                    container.innerHTML = '';
                     
                    Plotly.newPlot(containerId, chartData.data, chartData.layout, {{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false
                    }}).then(() => {{
                        updateDebugInfo(`✅ ${{chartName}} 渲染成功`);
                    }}).catch(plotlyError => {{
                        throw new Error(`Plotly渲染错误: ${{plotlyError.message}}`);
                    }});
                    
                }} catch(e) {{
                    console.error(`❌ ${{chartName}} 渲染失败:`, e);
                    updateDebugInfo(`❌ ${{chartName}} 渲染失败: ${{e.message}}`);
                    showChartError(chartName, e.message);
                }}
            }}
            
            function showChartError(chartName, errorMessage) {{
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.innerHTML = `
                    <h4>${{chartName}} 加载失败</h4>
                    <p>错误信息: ${{errorMessage}}</p>
                    <button class="mini-btn" onclick="retryLoadCharts()">重试加载</button>
                `;
                
                // 找到对应的图表容器
                const chartIds = {{
                    '时间序列': 'time-chart',
                    '交通模式': 'traffic-chart', 
                    '气象影响': 'weather-chart',
                    'LSTM预测': 'lstm-chart',
                    '实时预测': 'realtime-chart',
                    '所有图表': 'time-chart' // 默认显示在第一个图表容器
                }};
                
                const containerId = chartIds[chartName] || 'time-chart';
                const container = document.getElementById(containerId);
                if (container) {{
                    container.innerHTML = '';
                    container.appendChild(errorDiv);
                }}
            }}
            
            function retryLoadCharts() {{
                updateDebugInfo('🔄 重新加载图表...');
                loadAllCharts();
            }}
            
            // ========== 筛选功能 ==========
            function getSelectedBoroughs() {{
                const boroughs = [];
                if (document.getElementById('boroughBronx').checked) boroughs.push('Bronx');
                if (document.getElementById('boroughBrooklyn').checked) boroughs.push('Brooklyn');
                if (document.getElementById('boroughManhattan').checked) boroughs.push('Manhattan');
                if (document.getElementById('boroughQueens').checked) boroughs.push('Queens');
                if (document.getElementById('boroughStatenIsland').checked) boroughs.push('Staten Island');
                return boroughs;
            }}
            
            function selectAllBoroughs() {{
                document.getElementById('boroughBronx').checked = true;
                document.getElementById('boroughBrooklyn').checked = true;
                document.getElementById('boroughManhattan').checked = true;
                document.getElementById('boroughQueens').checked = true;
                document.getElementById('boroughStatenIsland').checked = true;
                updateDebugInfo('✅ 全选所有行政区');
            }}
            
            function deselectAllBoroughs() {{
                document.getElementById('boroughBronx').checked = false;
                document.getElementById('boroughBrooklyn').checked = false;
                document.getElementById('boroughManhattan').checked = false;
                document.getElementById('boroughQueens').checked = false;
                document.getElementById('boroughStatenIsland').checked = false;
                updateDebugInfo('✅ 取消选择所有行政区');
            }}
            
            function applyFilters() {{
                if (!fullData) {{
                    alert('数据未加载完成，请稍后重试');
                    return;
                }}
                
                const startDate = new Date(document.getElementById('startDate').value);
                const endDate = new Date(document.getElementById('endDate').value);
                const selectedBoroughs = getSelectedBoroughs();
                
                if (startDate > endDate) {{
                    alert('开始日期不能晚于结束日期！');
                    return;
                }}
                
                if (selectedBoroughs.length === 0) {{
                    alert('请至少选择一个行政区！');
                    return;
                }}
                
                updateDebugInfo(`应用筛选: ${{startDate.toLocaleDateString()}} - ${{endDate.toLocaleDateString()}}, 行政区: ${{selectedBoroughs.join(', ')}}`);
                
                try {{
                    // 筛选每日数据
                    const filteredDaily = fullData.daily.filter(item => {{
                        const date = new Date(item.date);
                        return date >= startDate && date <= endDate;
                    }});
                    
                    // 更新基础图表
                    updateTimeSeries(filteredDaily);
                    
                    // 如果预测数据已加载，也更新预测图表
                    if (lstmData) {{
                        const filteredLSTM = lstmData.filter(item => {{
                            const date = new Date(item.timestamp);
                            return date >= startDate && date <= endDate;
                        }});
                        updateLSTMPrediction(filteredLSTM);
                    }}
                    
                    if (realtimeData) {{
                        const filteredRealtime = realtimeData.filter(item => {{
                            const date = new Date(item.timestamp);
                            return date >= startDate && date <= endDate;
                        }});
                        updateRealtimePrediction(filteredRealtime);
                    }}
                    
                    document.getElementById('filterInfo').innerHTML = 
                        `<b>已筛选:</b> ${{filteredDaily.length}} 天数据 | 
                        行政区: ${{selectedBoroughs.join(', ')}} | 
                        时间: ${{startDate.toLocaleDateString()}} - ${{endDate.toLocaleDateString()}}`;
                        
                    updateDebugInfo(`✅ 筛选完成: ${{filteredDaily.length}} 条数据`);
                }} catch(e) {{
                    console.error('筛选出错:', e);
                    updateDebugInfo(`❌ 筛选失败: ${{e.message}}`);
                    alert('筛选失败: ' + e.message);
                }}
            }}
            
            function resetFilters() {{
                if (!fullData || !fullData.date_range) {{
                    location.reload();
                    return;
                }}
                
                document.getElementById('startDate').value = fullData.date_range.min;
                document.getElementById('endDate').value = fullData.date_range.max;
                selectAllBoroughs();
                
                const config = {{responsive: true}};
                Plotly.react('time-chart', originalCharts.time.data, originalCharts.time.layout, config);
                Plotly.react('traffic-chart', originalCharts.traffic.data, originalCharts.traffic.layout, config);
                Plotly.react('weather-chart', originalCharts.weather.data, originalCharts.weather.layout, config);
                
                if (lstmData) {{
                    Plotly.react('lstm-chart', originalCharts.lstm.data, originalCharts.lstm.layout, config);
                }}
                
                if (realtimeData) {{
                    Plotly.react('realtime-chart', originalCharts.realtime.data, originalCharts.realtime.layout, config);
                }}
                
                document.getElementById('filterInfo').innerHTML = '';
                updateDebugInfo('✅ 筛选条件已重置');
            }}
            
            function updateTimeSeries(data) {{
                if (!data || data.length === 0) {{
                    updateDebugInfo('❌ 时间序列数据为空');
                    return;
                }}
                
                const dates = data.map(d => d.date);
                const pm25 = data.map(d => d.PM2_5);
                const trips = data.map(d => d.trip_count);
                
                Plotly.react('time-chart', [
                    {{
                        x: dates, y: pm25, name: 'PM2.5', type: 'scatter',
                        mode: 'lines+markers', line: {{color: 'red', width: 3}}
                    }},
                    {{
                        x: dates, y: trips, name: 'Trip Count', type: 'bar',
                        marker: {{color: 'lightblue'}}, opacity: 0.6, yaxis: 'y2'
                    }}
                ], {{
                    title: 'Daily PM2.5 and Traffic Trends (Filtered)',
                    xaxis: {{title: 'Date'}},
                    yaxis: {{title: 'PM2.5 (μg/m³)', titlefont: {{color: 'red'}}, tickfont: {{color: 'red'}}, side: 'left'}},
                    yaxis2: {{title: 'Trip Count', titlefont: {{color: 'blue'}}, tickfont: {{color: 'blue'}}, overlaying: 'y', side: 'right'}},
                    height: 500, showlegend: true, hovermode: 'x unified'
                }}, {{responsive: true}});
                
                updateDebugInfo(`✅ 时间序列已更新: ${{data.length}} 个数据点`);
            }}
            
            function updateLSTMPrediction(data) {{
                if (!data || data.length === 0 || !lstmData) {{
                    updateDebugInfo('❌ LSTM预测数据为空');
                    return;
                }}
                
                const timestamps = data.map(d => d.timestamp);
                
                Plotly.react('lstm-chart', [
                    {{
                        x: timestamps, y: data.map(d => d.true_t1), name: 'True t+1',
                        line: {{color: '#2c3e50', width: 2}}, mode: 'lines'
                    }},
                    {{
                        x: timestamps, y: data.map(d => d.pred_t1), name: 'Pred t+1',
                        line: {{color: '#3498db', width: 2, dash: 'dash'}}, mode: 'lines'
                    }},
                    {{
                        x: timestamps, y: data.map(d => d.true_t3), name: 'True t+3',
                        line: {{color: '#27ae60', width: 2}}, mode: 'lines'
                    }},
                    {{
                        x: timestamps, y: data.map(d => d.pred_t3), name: 'Pred t+3',
                        line: {{color: '#2ecc71', width: 2, dash: 'dash'}}, mode: 'lines'
                    }},
                    {{
                        x: timestamps, y: data.map(d => d.true_t6), name: 'True t+6',
                        line: {{color: '#c0392b', width: 2}}, mode: 'lines'
                    }},
                    {{
                        x: timestamps, y: data.map(d => d.pred_t6), name: 'Pred t+6',
                        line: {{color: '#e74c3c', width: 2, dash: 'dash'}}, mode: 'lines'
                    }}
                ], {{
                    title: 'LSTM Multi-step PM2.5 Prediction (NYC Average - Filtered)',
                    xaxis_title: 'Timestamp',
                    yaxis_title: 'PM2.5 (μg/m³)',
                    height: 450, showlegend: true, hovermode: 'x unified',
                    legend: {{orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1}}
                }}, {{responsive: true}});
                
                updateDebugInfo(`✅ LSTM预测已更新: ${{data.length}} 个数据点`);
            }}

            function updateRealtimePrediction(data) {{
                if (!data || data.length === 0 || !realtimeData) {{
                    updateDebugInfo('❌ 实时预测数据为空');
                    return;
                }}
                
                const timestamps = data.map(d => d.timestamp);
                
                Plotly.react('realtime-chart', [
                    {{
                        x: timestamps, y: data.map(d => d.PM2_5_true), name: 'True PM2.5',
                        line: {{color: '#2c3e50', width: 3}}, mode: 'lines'
                    }},
                    {{
                        x: timestamps, y: data.map(d => d.PM2_5_pred_XGBoost), name: 'XGBoost',
                        line: {{color: '#3498db', width: 2, dash: 'dash'}}, mode: 'lines'
                    }},
                    {{
                        x: timestamps, y: data.map(d => d.PM2_5_pred_LightGBM), name: 'LightGBM',
                        line: {{color: '#2ecc71', width: 2, dash: 'dash'}}, mode: 'lines'
                    }},
                    {{
                        x: timestamps, y: data.map(d => d.PM2_5_pred_RandomForest), name: 'RandomForest',
                        line: {{color: '#e67e22', width: 2, dash: 'dash'}}, mode: 'lines'
                    }}
                ], {{
                    title: 'Real-time PM2.5 Prediction Comparison (NYC Average - Filtered)',
                    xaxis_title: 'Timestamp',
                    yaxis_title: 'PM2.5 (μg/m³)',
                    height: 450, showlegend: true, hovermode: 'x unified',
                    legend: {{orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1}}
                }}, {{responsive: true}});
                
                updateDebugInfo(`✅ 实时预测已更新: ${{data.length}} 个数据点`);
            }}
            
            // ========== 数据导出功能 ==========
            function toggleAdvancedExport() {{
                const panel = document.getElementById('advancedExportPanel');
                const arrow = document.getElementById('advancedArrow');
                if (panel.style.display === 'none') {{
                    panel.style.display = 'block';
                    arrow.textContent = '▲';
                    updateDebugInfo('📊 高级导出选项已展开');
                }} else {{
                    panel.style.display = 'none';
                    arrow.textContent = '▼';
                    updateDebugInfo('📊 高级导出选项已收起');
                }}
            }}
            
            function selectAllLSTM() {{
                document.querySelectorAll('.lstm-field').forEach(cb => cb.checked = true);
                updateDebugInfo('✅ 全选LSTM字段');
            }}
            
            function deselectAllLSTM() {{
                document.querySelectorAll('.lstm-field').forEach(cb => cb.checked = false);
                updateDebugInfo('✅ 取消选择LSTM字段');
            }}
            
            function selectAllRealtime() {{
                document.querySelectorAll('.realtime-field').forEach(cb => cb.checked = true);
                updateDebugInfo('✅ 全选实时预测字段');
            }}
            
            function deselectAllRealtime() {{
                document.querySelectorAll('.realtime-field').forEach(cb => cb.checked = false);
                updateDebugInfo('✅ 取消选择实时预测字段');
            }}
            
            function quickExport() {{
                if (!fullData || !lstmData || !realtimeData) {{
                    alert('数据正在加载中，请稍后再试...');
                    updateDebugInfo('❌ 快速导出失败: 数据未加载完成');
                    return;
                }}
                
                const startDate = new Date(document.getElementById('startDate').value);
                const endDate = new Date(document.getElementById('endDate').value);
                
                // 筛选LSTM数据
                const filteredLSTM = lstmData.filter(item => {{
                    const date = new Date(item.timestamp);
                    return date >= startDate && date <= endDate;
                }});
                
                // 筛选实时预测数据
                const filteredRealtime = realtimeData.filter(item => {{
                    const date = new Date(item.timestamp);
                    return date >= startDate && date <= endDate;
                }});
                
                // 合并数据
                const combinedData = filteredLSTM.map((lstm, index) => {{
                    const realtime = filteredRealtime[index] || {{}};
                    return {{
                        timestamp: lstm.timestamp,
                        // LSTM数据
                        lstm_true_t1: lstm.true_t1,
                        lstm_pred_t1: lstm.pred_t1,
                        lstm_true_t3: lstm.true_t3,
                        lstm_pred_t3: lstm.pred_t3,
                        lstm_true_t6: lstm.true_t6,
                        lstm_pred_t6: lstm.pred_t6,
                        // 实时预测数据
                        PM2_5_true: realtime.PM2_5_true,
                        PM2_5_pred_XGBoost: realtime.PM2_5_pred_XGBoost,
                        PM2_5_pred_LightGBM: realtime.PM2_5_pred_LightGBM,
                        PM2_5_pred_RandomForest: realtime.PM2_5_pred_RandomForest
                    }};
                }});
                
                // 生成文件名
                const startStr = startDate.toISOString().split('T')[0];
                const endStr = endDate.toISOString().split('T')[0];
                const filename = `nyc_predictions_${{startStr}}_to_${{endStr}}.csv`;
                
                // 导出CSV
                const csv = convertToCSV(combinedData);
                downloadFile(csv, filename, 'text/csv');
                
                updateDebugInfo(`📥 快速导出完成: ${{combinedData.length}} 条记录`);
                console.log(`快速导出: ${{combinedData.length}} 条记录`);
                alert(`成功导出 ${{combinedData.length}} 条记录！`);
            }}
            
            function advancedExport() {{
                if (!fullData) {{
                    alert('基础数据未加载完成，请稍后再试');
                    updateDebugInfo('❌ 高级导出失败: 基础数据未加载');
                    return;
                }}
                
                const startDate = new Date(document.getElementById('startDate').value);
                const endDate = new Date(document.getElementById('endDate').value);
                
                // 获取选中的数据集
                const exportLSTM = document.getElementById('exportLSTM').checked;
                const exportRealtime = document.getElementById('exportRealtime').checked;
                const exportDaily = document.getElementById('exportDaily').checked;
                const exportHourly = document.getElementById('exportHourly').checked;
                const exportBorough = document.getElementById('exportBorough').checked;
                
                if (!exportLSTM && !exportRealtime && !exportDaily && !exportHourly && !exportBorough) {{
                    alert('请至少选择一个数据集！');
                    updateDebugInfo('❌ 高级导出失败: 未选择任何数据集');
                    return;
                }}
                
                // 检查预测数据是否已加载
                if (exportLSTM && !lstmData) {{
                    alert('LSTM预测数据未加载，请稍后再试');
                    updateDebugInfo('❌ 高级导出失败: LSTM数据未加载');
                    return;
                }}
                
                if (exportRealtime && !realtimeData) {{
                    alert('实时预测数据未加载，请稍后再试');
                    updateDebugInfo('❌ 高级导出失败: 实时预测数据未加载');
                    return;
                }}
                
                // 获取文件格式
                const format = document.querySelector('input[name="exportFormat"]:checked').value;
                
                let combinedData = [];
                
                // 处理LSTM数据
                if (exportLSTM) {{
                    const selectedFields = Array.from(document.querySelectorAll('.lstm-field:checked')).map(cb => cb.value);
                    if (selectedFields.length === 0) {{
                        alert('请至少选择一个LSTM字段！');
                        updateDebugInfo('❌ 高级导出失败: 未选择LSTM字段');
                        return;
                    }}
                    
                    const filteredLSTM = lstmData.filter(item => {{
                        const date = new Date(item.timestamp);
                        return date >= startDate && date <= endDate;
                    }});
                    
                    filteredLSTM.forEach((item, index) => {{
                        if (!combinedData[index]) combinedData[index] = {{}};
                        selectedFields.forEach(field => {{
                            combinedData[index][`lstm_${{field}}`] = item[field];
                        }});
                    }});
                    
                    updateDebugInfo(`✅ 已添加LSTM数据: ${{filteredLSTM.length}} 条`);
                }}
                
                // 处理实时预测数据
                if (exportRealtime) {{
                    const selectedFields = Array.from(document.querySelectorAll('.realtime-field:checked')).map(cb => cb.value);
                    if (selectedFields.length === 0) {{
                        alert('请至少选择一个实时预测字段！');
                        updateDebugInfo('❌ 高级导出失败: 未选择实时预测字段');
                        return;
                    }}
                    
                    const filteredRealtime = realtimeData.filter(item => {{
                        const date = new Date(item.timestamp);
                        return date >= startDate && date <= endDate;
                    }});
                    
                    filteredRealtime.forEach((item, index) => {{
                        if (!combinedData[index]) combinedData[index] = {{}};
                        selectedFields.forEach(field => {{
                            if (field === 'timestamp' && combinedData[index]['lstm_timestamp']) {{
                                // 避免重复timestamp
                                return;
                            }}
                            combinedData[index][`realtime_${{field}}`] = item[field];
                        }});
                    }});
                    
                    updateDebugInfo(`✅ 已添加实时预测数据: ${{filteredRealtime.length}} 条`);
                }}
                
                // 处理每日数据
                if (exportDaily) {{
                    const filteredDaily = fullData.daily.filter(item => {{
                        const date = new Date(item.date);
                        return date >= startDate && date <= endDate;
                    }});
                    
                    filteredDaily.forEach((item, index) => {{
                        if (!combinedData[index]) combinedData[index] = {{}};
                        combinedData[index].daily_date = item.date;
                        combinedData[index].daily_PM2_5 = item.PM2_5;
                        combinedData[index].daily_trip_count = item.trip_count;
                        combinedData[index].daily_avg_speed = item.avg_speed;
                    }});
                    
                    updateDebugInfo(`✅ 已添加每日数据: ${{filteredDaily.length}} 条`);
                }}
                
                // 处理每小时数据
                if (exportHourly) {{
                    combinedData = fullData.hourly.map(item => ({{
                        hour_of_day: item.hour_of_day,
                        hourly_trip_count: item.trip_count,
                        hourly_PM2_5: item.PM2_5,
                        hourly_avg_speed: item.avg_speed
                    }}));
                    updateDebugInfo(`✅ 已添加每小时数据: ${{combinedData.length}} 条`);
                }}
                
                // 处理行政区数据
                if (exportBorough) {{
                    const selectedBoroughs = getSelectedBoroughs();
                    const filteredBorough = fullData.borough.filter(item => {{
                        const date = new Date(item.date);
                        return selectedBoroughs.includes(item.borough) && 
                            date >= startDate && date <= endDate;
                    }});
                    
                    // 如果只导出行政区数据，直接使用
                    if (!exportLSTM && !exportRealtime && !exportDaily && !exportHourly) {{
                        combinedData = filteredBorough;
                    }}
                    
                    updateDebugInfo(`✅ 已添加行政区数据: ${{filteredBorough.length}} 条`);
                }}
                
                if (combinedData.length === 0) {{
                    alert('筛选后无数据可导出！');
                    updateDebugInfo('❌ 高级导出失败: 筛选后无数据');
                    return;
                }}
                
                // 生成文件名
                const startStr = startDate.toISOString().split('T')[0];
                const endStr = endDate.toISOString().split('T')[0];
                const filename = `nyc_custom_export_${{startStr}}_to_${{endStr}}.${{format}}`;
                
                // 导出文件
                if (format === 'csv') {{
                    const csv = convertToCSV(combinedData);
                    downloadFile(csv, filename, 'text/csv');
                }} else {{
                    const json = JSON.stringify(combinedData, null, 2);
                    downloadFile(json, filename, 'application/json');
                }}
                
                updateDebugInfo(`📥 高级导出完成 (${{format.toUpperCase()}}): ${{combinedData.length}} 条记录`);
                console.log(`高级导出 (${{format.toUpperCase()}}): ${{combinedData.length}} 条记录`);
                alert(`成功导出 ${{combinedData.length}} 条记录！`);
            }}
            
            function convertToCSV(data) {{
                if (!data || data.length === 0) return '';
                
                const headers = Object.keys(data[0]);
                const csvRows = [headers.join(',')];
                
                data.forEach(row => {{
                    const values = headers.map(header => {{
                        const value = row[header];
                        // 处理包含逗号的值
                        if (typeof value === 'string' && value.includes(',')) {{
                            return `"${{value}}"`;
                        }}
                        return value !== undefined && value !== null ? value : '';
                    }});
                    csvRows.push(values.join(','));
                }});
                
                // 添加UTF-8 BOM，让Excel正确识别编码
                return '\uFEFF' + csvRows.join('\\n');
            }}
            
            function downloadFile(content, filename, mimeType) {{
                const blob = new Blob([content], {{ type: `${{mimeType}};charset=utf-8;` }});
                const link = document.createElement('a');
                
                if (navigator.msSaveBlob) {{
                    navigator.msSaveBlob(blob, filename);
                }} else {{
                    link.href = URL.createObjectURL(blob);
                    link.download = filename;
                    link.style.display = 'none';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    setTimeout(() => URL.revokeObjectURL(link.href), 100);
                }}
            }}
            
            // ========== 页面加载完成后的初始化 ==========
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('🏁 DOM加载完成，开始初始化...');
                initializeDashboard();
            }});
            
            // 如果DOM已经加载完成，直接初始化
            if (document.readyState === 'complete' || document.readyState === 'interactive') {{
                setTimeout(initializeDashboard, 100);
            }}
        </script>
    </body>
    </html>
    """

        # 保存HTML文件
    output_path = f"{BASE_FILE_PATH}/outputs/interactive_dashboard_with_predictions.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"Interactive dashboard saved: {output_path}")
    return output_path

# 主执行函数
def main():
    """主执行函数"""
    # 加载数据
    final_df = pd.read_csv(f'{BASE_FILE_PATH}/outputs/final_complete_dataset.csv', low_memory=False)
    air_quality_sites = pd.read_csv(f'{BASE_FILE_PATH}/data/raw/station-info.csv')
    geojson_path = f'{BASE_FILE_PATH}/data/raw/boroughs.geojson'
    
    # 加载预测数据
    lstm_pred_df = pd.read_csv(f'{BASE_FILE_PATH}/data/processed/lstm_predictions.csv')
    realtime_pred_df = pd.read_csv(f'{BASE_FILE_PATH}/data/processed/realtime_predictions.csv')
    
    # 过滤有效站点
    air_quality_sites = air_quality_sites[air_quality_sites['SiteID'].notna() & (air_quality_sites['SiteID'] != '')]
    
    print("Starting comprehensive visualization analysis...")
    print(f"Final dataset: {final_df.shape}")
    print(f"LSTM predictions: {lstm_pred_df.shape}")
    print(f"Realtime predictions: {realtime_pred_df.shape}")
    
    # 创建交互式仪表盘
    dashboard_path = create_proper_interactive_dashboard(
        final_df, 
        air_quality_sites, 
        geojson_path,
        lstm_pred_df,
        realtime_pred_df
    )
    
    print("All visualizations completed!")
    print("Output files:")
    print(f"   - Interactive dashboard: {dashboard_path}")

if __name__ == "__main__":
    main()
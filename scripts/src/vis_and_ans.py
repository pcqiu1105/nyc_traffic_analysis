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

BASE_FILE_PATH = '/mnt/d/nyc_traffic_analysis'

def create_spatial_visualization(final_df, air_quality_sites):
    """Create spatial distribution visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    geometry = [Point(lon, lat) for lon, lat in zip(air_quality_sites['Longitude'], air_quality_sites['Latitude'])]
    air_sites_gdf = gpd.GeoDataFrame(
        air_quality_sites,
        geometry=geometry,
        crs="EPSG:4326"
    )
    
    air_sites_gdf = air_sites_gdf.to_crs("EPSG:32618")
    
    # 1. Site distribution map
    ax1 = axes[0, 0]
    nyc_map = gpd.read_file(gpd.datasets.get_path('nybb'))
    nyc_map = nyc_map.to_crs("EPSG:4326")
    
    nyc_map.plot(ax=ax1, alpha=0.3, edgecolor='black')
    air_sites_gdf.plot(ax=ax1, color='red', markersize=50, alpha=0.7)
    
    for idx, row in air_sites_gdf.iterrows():
        ax1.annotate(row['SiteID'], (row.geometry.x, row.geometry.y), 
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    
    ax1.set_title('NYC Air Quality Monitoring Sites Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # 2. Traffic activity heatmap by borough
    ax2 = axes[0, 1]
    heatmap_data = final_df.pivot_table(
        index='borough', 
        columns='hour_of_day', 
        values='trip_count', 
        aggfunc='mean'
    ).fillna(0)
    
    sns.heatmap(heatmap_data, ax=ax2, cmap='YlOrRd', cbar_kws={'label': 'Average Trip Count'})
    ax2.set_title('Hourly Traffic Patterns by Borough Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Borough')
    
    # 3. PM2.5 concentration distribution
    ax3 = axes[1, 0]
    site_pm25 = final_df.groupby('site_id')['PM2_5'].mean().sort_values(ascending=False)
    
    bars = ax3.bar(range(len(site_pm25)), site_pm25.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(site_pm25))))
    ax3.set_title('Average PM2.5 Concentration by Site', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Site ID')
    ax3.set_ylabel('PM2.5 Concentration (μg/m³)')
    ax3.set_xticks(range(len(site_pm25)))
    ax3.set_xticklabels(site_pm25.index, rotation=45, ha='right')
    
    ax3.axhline(y=15, color='green', linestyle='--', alpha=0.7, label='WHO Guideline')
    ax3.axhline(y=35, color='orange', linestyle='--', alpha=0.7, label='EPA Standard')
    ax3.legend()
    
    # 4. Feature correlation heatmap
    ax4 = axes[1, 1]
    numeric_cols = ['PM2_5', 'trip_count', 'avg_speed', 'total_passengers', 
                   'temperature', 'humidity', 'wind_speed', 'road_density_500m']
    numeric_cols = [col for col in numeric_cols if col in final_df.columns]
    
    correlation = final_df[numeric_cols].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', 
                center=0, ax=ax4, fmt='.2f')
    ax4.set_title('Main Features Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{BASE_FILE_PATH}/outputs/data_statistics_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_temporal_analysis(final_df):
    """Create temporal analysis charts"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Daily pattern of PM2.5
    ax1 = axes[0, 0]
    hourly_pm25 = final_df.groupby('hour_of_day')['PM2_5'].mean()
    hourly_traffic = final_df.groupby('hour_of_day')['trip_count'].mean()
    
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(hourly_pm25.index, hourly_pm25.values, 'r-', linewidth=2, label='PM2.5')
    line2 = ax1_twin.plot(hourly_traffic.index, hourly_traffic.values, 'b-', linewidth=2, label='Traffic Volume')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('PM2.5 Concentration (μg/m³)', color='red')
    ax1_twin.set_ylabel('Average Trip Count', color='blue')
    ax1.set_title('Daily Pattern of PM2.5 and Traffic Volume', fontsize=12, fontweight='bold')
    
    # 2. Weekly pattern
    ax2 = axes[0, 1]
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekday_pm25 = final_df.groupby('day_of_week')['PM2_5'].mean()
    
    ax2.bar(weekday_names, weekday_pm25.values, 
            color=['blue', 'blue', 'blue', 'blue', 'blue', 'red', 'red'])
    ax2.set_title('Weekly Pattern of PM2.5 Concentration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PM2.5 Concentration (μg/m³)')
    
    # 3. Missing values distribution
    ax3 = axes[1, 0]
    missing_stats = final_df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
    
    if len(missing_stats) > 15:
        missing_stats = missing_stats.head(15)
    
    bars = ax3.barh(range(len(missing_stats)), missing_stats.values)
    ax3.set_yticks(range(len(missing_stats)))
    ax3.set_yticklabels(missing_stats.index, fontsize=8)
    ax3.set_xlabel('Missing Values Count')
    ax3.set_title('Feature Missing Values Distribution (Top 15)', fontsize=12, fontweight='bold')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percent = (width / len(final_df)) * 100
        ax3.text(width + 10, bar.get_y() + bar.get_height()/2, 
                f'{percent:.1f}%', ha='left', va='center', fontsize=8)
    
    # 4. Weather conditions vs PM2.5
    ax4 = axes[1, 1]
    scatter = ax4.scatter(final_df['temperature'], final_df['PM2_5'], 
                         c=final_df['wind_speed'], alpha=0.6, cmap='viridis')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('PM2.5 Concentration (μg/m³)')
    ax4.set_title('Relationship between Temperature, Wind Speed and PM2.5', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Wind Speed (m/s)')
    
    plt.tight_layout()
    plt.savefig(f'{BASE_FILE_PATH}/outputs/temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_static_analysis(final_df):
    """Create static analysis charts"""
    print("Generating static analysis charts...")
    
    # 1. Borough Comparison Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1.1 PM2.5 Distribution by Borough
    sns.boxplot(data=final_df, x='borough', y='PM2_5', ax=axes[0,0])
    axes[0,0].set_title('PM2.5 Concentration Distribution by Borough', fontweight='bold')
    axes[0,0].set_xlabel('Borough')
    axes[0,0].set_ylabel('PM2.5 Concentration (μg/m³)')
    
    # 1.2 Average Trip Density by Borough
    trip_density_avg = final_df.groupby('borough')['trip_density'].mean().sort_values(ascending=False)
    sns.barplot(x=trip_density_avg.index, y=trip_density_avg.values, ax=axes[0,1], palette='rocket')
    axes[0,1].set_title('Average Taxi Trip Density by Borough', fontweight='bold')
    axes[0,1].set_ylabel('Trip Density (trips/km²)')
    
    # 1.3 Road Density vs PM2.5 Relationship
    sns.scatterplot(data=final_df, x='road_density_500m', y='PM2_5', hue='borough', ax=axes[1,0], alpha=0.6)
    axes[1,0].set_title('Road Density vs PM2.5 Concentration', fontweight='bold')
    axes[1,0].set_xlabel('Road Density (500m radius)')
    axes[1,0].set_ylabel('PM2.5 Concentration (μg/m³)')
    
    # 1.4 Average Speed vs PM2.5 Relationship
    sns.scatterplot(data=final_df, x='avg_speed', y='PM2_5', hue='is_rush_hour', ax=axes[1,1], alpha=0.6)
    axes[1,1].set_title('Taxi Speed vs PM2.5 Concentration', fontweight='bold')
    axes[1,1].set_xlabel('Average Speed (km/h)')
    axes[1,1].set_ylabel('PM2.5 Concentration (μg/m³)')
    
    plt.tight_layout()
    plt.savefig(f'{BASE_FILE_PATH}/outputs/static_borough_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Road Network Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 2.1 Major Road Ratio vs PM2.5
    sns.regplot(data=final_df, x='major_road_ratio_500m', y='PM2_5', ax=axes[0,0], scatter_kws={'alpha':0.5})
    axes[0,0].set_title('Major Road Ratio vs PM2.5 Concentration', fontweight='bold')
    axes[0,0].set_xlabel('Major Road Ratio (500m radius)')
    axes[0,0].set_ylabel('PM2.5 Concentration (μg/m³)')
    
    # 2.2 Intersection Density vs PM2.5
    sns.regplot(data=final_df, x='intersection_density_500m', y='PM2_5', ax=axes[0,1], scatter_kws={'alpha':0.5})
    axes[0,1].set_title('Intersection Density vs PM2.5 Concentration', fontweight='bold')
    axes[0,1].set_xlabel('Intersection Density (500m radius)')
    axes[0,1].set_ylabel('PM2.5 Concentration (μg/m³)')
    
    # 2.3 Rush Hour vs Non-Rush Hour Comparison
    rush_hour_data = final_df.groupby('is_rush_hour')['PM2_5'].mean()
    axes[1,0].bar(['Non-Rush Hour', 'Rush Hour'], rush_hour_data.values, color=['lightblue', 'coral'])
    axes[1,0].set_ylabel('Average PM2.5 Concentration (μg/m³)')
    axes[1,0].set_title('PM2.5: Rush Hour vs Non-Rush Hour', fontweight='bold')
    
    # 2.4 Weekday vs Weekend Comparison
    weekend_data = final_df.groupby('is_weekend')['PM2_5'].mean()
    axes[1,1].bar(['Weekday', 'Weekend'], weekend_data.values, color=['lightgreen', 'orange'])
    axes[1,1].set_ylabel('Average PM2.5 Concentration (μg/m³)')
    axes[1,1].set_title('PM2.5: Weekday vs Weekend', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{BASE_FILE_PATH}/outputs/static_road_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Weather Impact Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 3.1 Temperature vs PM2.5
    sns.scatterplot(data=final_df, x='temperature', y='PM2_5', hue='borough', ax=axes[0,0], alpha=0.6)
    axes[0,0].set_title('Temperature vs PM2.5 Concentration', fontweight='bold')
    axes[0,0].set_xlabel('Temperature (°C)')
    
    # 3.2 Humidity vs PM2.5
    sns.scatterplot(data=final_df, x='humidity', y='PM2_5', ax=axes[0,1], alpha=0.6)
    axes[0,1].set_title('Humidity vs PM2.5 Concentration', fontweight='bold')
    axes[0,1].set_xlabel('Humidity (%)')
    
    # 3.3 Wind Speed Category vs PM2.5
    sns.boxplot(data=final_df, x='wind_speed_cat', y='PM2_5', ax=axes[1,0])
    axes[1,0].set_title('Wind Speed Category vs PM2.5 Concentration', fontweight='bold')
    axes[1,0].set_xlabel('Wind Speed Category')
    
    # 3.4 Weather Severity vs PM2.5
    sns.boxplot(data=final_df, x='weather_severity', y='PM2_5', ax=axes[1,1])
    axes[1,1].set_title('Weather Severity vs PM2.5 Concentration', fontweight='bold')
    axes[1,1].set_xlabel('Weather Severity')
    
    plt.tight_layout()
    plt.savefig(f'{BASE_FILE_PATH}/outputs/static_weather_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_interactive_visualizations(final_df, air_quality_sites):
    """Create interactive visualizations"""
    print("Generating interactive visualizations...")
    
    # Ensure timestamp format is correct
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
    
    # 1. Enhanced Time Series Dashboard
    daily_data = final_df.groupby(final_df['timestamp'].dt.date).agg({
        'PM2_5': 'mean',
        'trip_count': 'sum',
        'avg_speed': 'mean',
        'trip_density': 'mean',
        'temperature': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    
    time_series_fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('PM2.5 Concentration Trend', 'Taxi Trip Volume', 
                       'Average Speed Trend', 'Trip Density Trend',
                       'Temperature Trend', 'Wind Speed Trend'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.08
    )
    
    # PM2.5 Trend
    time_series_fig.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['PM2_5'], 
                  name='PM2.5', line=dict(color='red')),
        row=1, col=1
    )
    
    # Trip Count Trend
    time_series_fig.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['trip_count'],
                  name='Trip Count', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Average Speed Trend
    time_series_fig.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['avg_speed'],
                  name='Avg Speed', line=dict(color='green')),
        row=2, col=1
    )
    
    # Trip Density Trend
    time_series_fig.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['trip_density'],
                  name='Trip Density', line=dict(color='purple')),
        row=2, col=2
    )
    
    # Temperature Trend
    time_series_fig.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['temperature'],
                  name='Temperature', line=dict(color='orange')),
        row=3, col=1
    )
    
    # Wind Speed Trend
    time_series_fig.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['wind_speed'],
                  name='Wind Speed', line=dict(color='brown')),
        row=3, col=2
    )
    
    time_series_fig.update_layout(
        height=900, 
        title_text="NYC Comprehensive Time Series Dashboard",
        showlegend=False
    )
    time_series_fig.write_html(f"{BASE_FILE_PATH}/outputs/interactive_time_series.html")

def create_geographic_dashboard(final_df, air_quality_sites):
    """Create geographic dashboard with proper map tiles"""
    # Create base map with proper tile layer
    nyc_center = [40.7128, -74.0060]
    m = folium.Map(location=nyc_center, zoom_start=11, tiles='OpenStreetMap')
    
    # Add multiple tile layers for better map display
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter').add_to(m)
    
    # Add air quality monitoring sites
    for idx, site in air_quality_sites.iterrows():
        # Get average PM2.5 for this site
        site_data = final_df[final_df['site_id'] == site['SiteID']]
        if not site_data.empty:
            site_pm25 = site_data['PM2_5'].mean()
            site_trips = site_data['trip_count'].mean()
            
            # Set color based on PM2.5 concentration
            if site_pm25 <= 12:
                color = 'green'
            elif site_pm25 <= 35:
                color = 'orange'
            else:
                color = 'red'
            
            # Create popup content
            popup_content = f"""
            <b>Site: {site['SiteID']}</b><br>
            Avg PM2.5: {site_pm25:.2f} μg/m³<br>
            Avg Trips: {site_trips:.0f}<br>
            Coordinates: {site['Latitude']:.4f}, {site['Longitude']:.4f}
            """
            
            folium.CircleMarker(
                location=[site['Latitude'], site['Longitude']],
                radius=10,
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Site {site['SiteID']} - PM2.5: {site_pm25:.2f}",
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    # Prepare heatmap data (using trip density)
    heat_data = []
    for idx, row in final_df.iterrows():
        site_info = air_quality_sites[air_quality_sites['SiteID'] == row['site_id']]
        if not site_info.empty:
            lat = site_info['Latitude'].iloc[0]
            lon = site_info['Longitude'].iloc[0]
            # Use trip density for heatmap intensity
            intensity = min(row['trip_density'] / 100, 1.0)  # Normalize
            heat_data.append([lat, lon, intensity])
    
    # Add heatmap layer if we have data
    if heat_data:
        HeatMap(heat_data, 
                name='Trip Density Heatmap', 
                min_opacity=0.2, 
                max_opacity=0.8,
                radius=15,
                blur=10,
                gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)
    
    # Add borough boundaries (if available)
    try:
        nyc_boroughs = gpd.read_file(gpd.datasets.get_path('nybb'))
        for idx, row in nyc_boroughs.iterrows():
            borough_geo = row['geometry']
            if hasattr(borough_geo, '__iter__'):
                for poly in borough_geo:
                    folium.GeoJson(
                        poly.__geo_interface__,
                        style_function=lambda x: {
                            'fillColor': 'transparent',
                            'color': 'black',
                            'weight': 2,
                            'fillOpacity': 0.1
                        }
                    ).add_to(m)
    except:
        print("Note: Could not load borough boundaries")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = '''
                 <h3 align="center" style="font-size:16px"><b>NYC Air Quality Monitoring Sites & Traffic Density</b></h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    m.save(f"{BASE_FILE_PATH}/outputs/interactive_nyc_map.html")
    print("Geographic dashboard created with multiple map layers")

def generate_column_statistics(final_df):
    """Generate detailed column statistics"""
    stats_list = []
    
    for col in final_df.columns:
        col_stats = {
            'Feature Name': col,
            'Data Type': final_df[col].dtype,
            'Total Count': len(final_df),
            'Non-Null Count': final_df[col].count(),
            'Missing Count': final_df[col].isnull().sum(),
            'Missing Ratio (%)': (final_df[col].isnull().sum() / len(final_df) * 100)
        }
        
        if pd.api.types.is_numeric_dtype(final_df[col]):
            col_stats.update({
                'Min': final_df[col].min(),
                'Max': final_df[col].max(),
                'Mean': final_df[col].mean(),
                'Median': final_df[col].median(),
                'Std': final_df[col].std()
            })
        else:
            col_stats.update({
                'Unique Count': final_df[col].nunique(),
                'Most Common': final_df[col].mode().iloc[0] if not final_df[col].mode().empty else 'N/A'
            })
        
        stats_list.append(col_stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    print("Complete Dataset Column Statistics Report")
    
    feature_categories = {
        'Air Quality Features': [col for col in final_df.columns if 'PM2_5' in col],
        'Traffic Features': [col for col in final_df.columns if any(x in col for x in ['trip', 'speed', 'passenger', 'distance', 'duration', 'revenue'])],
        'Road Features': [col for col in final_df.columns if any(x in col for x in ['road', 'intersection'])],
        'Weather Features': [col for col in final_df.columns if any(x in col for x in ['temperature', 'humidity', 'wind', 'pressure', 'precipitation'])],
        'Time Features': [col for col in final_df.columns if any(x in col for x in ['hour', 'day', 'weekend', 'rush', 'sin', 'cos'])],
        'Identifier Features': [col for col in final_df.columns if col in ['timestamp', 'site_id', 'site_name', 'borough']]
    }
    
    for category, features in feature_categories.items():
        if features:
            category_stats = stats_df[stats_df['Feature Name'].isin(features)]
            print(f"\n{category} ({len(features)} features)")
            
            display_cols = ['Feature Name', 'Data Type', 'Missing Ratio (%)']
            if category in ['Air Quality Features', 'Traffic Features', 'Road Features', 'Weather Features']:
                display_cols.extend(['Min', 'Max', 'Mean'])
            elif category == 'Identifier Features':
                display_cols.extend(['Unique Count', 'Most Common'])
            
            print(category_stats[display_cols].round(3).to_string(index=False))
    
    return stats_df

def generate_key_findings(final_df):
    """Generate key findings summary"""
    print("Key Data Findings Summary")
    
    # 1. PM2.5 statistics
    pm25_stats = final_df['PM2_5'].describe()
    print(f"\nPM2.5 Concentration Statistics:")
    print(f"   Mean: {pm25_stats['mean']:.2f} μg/m³")
    print(f"   Median: {pm25_stats['50%']:.2f} μg/m³") 
    print(f"   Std: {pm25_stats['std']:.2f} μg/m³")
    
    who_standard = 15
    epa_standard = 35
    above_who = (final_df['PM2_5'] > who_standard).sum() / len(final_df) * 100
    above_epa = (final_df['PM2_5'] > epa_standard).sum() / len(final_df) * 100
    
    print(f"   Above WHO Standard: {above_who:.1f}% of records")
    print(f"   Above EPA Standard: {above_epa:.1f}% of records")
    
    # 2. Traffic activity statistics
    print(f"\nTraffic Activity Statistics:")
    print(f"   Average Hourly Trips: {final_df['trip_count'].mean():.1f}")
    print(f"   Total Passengers: {final_df['total_passengers'].sum():,.0f}")
    
    # 3. Spatial distribution
    print(f"\nSpatial Distribution:")
    borough_summary = final_df.groupby('borough').agg({
        'site_id': 'nunique',
        'trip_count': 'sum',
        'PM2_5': 'mean'
    })
    
    for borough in borough_summary.index:
        sites = borough_summary.loc[borough, 'site_id']
        trips = borough_summary.loc[borough, 'trip_count']
        pm25 = borough_summary.loc[borough, 'PM2_5']
        print(f"   {borough}: {sites} sites, {trips:,.0f} trips, PM2.5: {pm25:.2f} μg/m³")
    
    # 4. Data quality assessment
    print(f"\nData Quality Assessment:")
    total_missing = final_df.isnull().sum().sum()
    total_cells = final_df.shape[0] * final_df.shape[1]
    missing_percent = (total_missing / total_cells) * 100
    
    print(f"   Overall Missing Rate: {missing_percent:.2f}%")
    print(f"   Available Modeling Records: {final_df.dropna(subset=['PM2_5']).shape[0]:,} rows")

# Execute complete visualization analysis
final_df = pd.read_csv(f'{BASE_FILE_PATH}/outputs/final_complete_dataset.csv')
air_quality_sites = pd.read_csv(f'{BASE_FILE_PATH}/data/raw/station-info.csv')
air_quality_sites = air_quality_sites[air_quality_sites['SiteID'].notna() & (air_quality_sites['SiteID'] != '')] 

# Execute original analysis
create_spatial_visualization(final_df, air_quality_sites)
create_temporal_analysis(final_df)

# New static and interactive analysis
create_static_analysis(final_df)
create_interactive_visualizations(final_df, air_quality_sites)

# Statistical reports
generate_column_statistics(final_df)
generate_key_findings(final_df)

print("All visualization analysis completed!")
print("Static charts saved in outputs/ directory")
print("Interactive charts saved as HTML files, open in browser to view")
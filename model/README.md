# PM2.5 短时预测与交通特征重要性分析

本项目旨在基于交通与气象数据，量化交通特征对空气质量的全局影响，并利用 LSTM 模型实现 PM2.5 的多步短时预测。

## 项目结构

| 文件名 | 功能说明 |
|--------|-----------|
| 00_prepare_data_spatial.py | 数据预处理与空间字段准备 |
| 01_feature_collinearity.py | 特征共线性检测 |
| 02_feature_selected.py | 特征筛选与验证 |
| 03_optimized_analysis.py | XGBoost / LightGBM / RandomForest 模型调参与性能比较 |
| 03_pm25_lstm_analysis.py | LSTM 模型预测（输出训练与测试指标） |
| 04_visualize_model_results.py | 可视化模型性能与预测结果 |

## 环境配置

### 使用 pip
```bash
pip install -r requirements.txt
```

### 使用 conda
```bash
conda env create -f environment.yml
conda activate pm25_forecast
```

## 运行流程

依次运行以下脚本：
```bash
python 00_prepare_data_spatial.py
python 01_feature_collinearity.py
python 02_feature_selected.py
python 03_optimized_analysis.py
python 03_pm25_lstm_analysis.py
python 04_visualize_model_results.py
```





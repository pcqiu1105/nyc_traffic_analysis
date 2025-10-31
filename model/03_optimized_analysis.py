"""
optimized_analysis

"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============ 快速调参函数 ============
def evaluate_model(model, param_grid, X_train, X_val, y_train, y_val):
    """轻量自动调参：遍历参数组合，选出验证集R²最优"""
    best_model, best_params, best_r2 = None, None, -1e9
    for params in param_grid:
        model.set_params(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = r2_score(y_val, preds)
        if score > best_r2:
            best_model = model
            best_params = params
            best_r2 = score
    return best_model, best_params, best_r2

# ============ 主函数 ============
def main():
    data_path = "data/processed_dataset.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    target = 'PM2_5'
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # 从 selected_features_vif.csv 中读取最终特征
    selected_features = pd.read_csv("results/selected_features_vif.csv")["feature"].tolist()
    feature_cols = [c for c in df.columns if c in selected_features]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # === 检测潜在泄漏特征 ===
    leakage_keywords = ['PM2_5_ma']
    leakage_cols = [c for c in X.columns if any(k in c for k in leakage_keywords)]
    check_report = []

    if leakage_cols:
        check_report.append("检测到潜在泄漏特征（含当前PM2.5信息）:\n" + ", ".join(leakage_cols))
        X = X.drop(columns=leakage_cols)
    else:
        check_report.append("未检测到明显泄漏特征。")

    check_report.append(f"\n最终用于训练的特征数量: {X.shape[1]}")
    result_dir = "optimized_full_results"
    os.makedirs(result_dir, exist_ok=True)
    report_path = os.path.join(result_dir, "Feature_Check_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(check_report))
    print("\n".join(check_report))

    # 填充缺失值
    X_numeric = X.select_dtypes(include=[np.number])
    X[X_numeric.columns] = X_numeric.fillna(X_numeric.median())

    # 分类变量编码
    categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 时序划分 (7:1:2)
    train_size = int(0.7 * len(X_scaled))
    val_size = int(0.1 * len(X_scaled))
    X_train, X_val, X_test = (
        X_scaled[:train_size],
        X_scaled[train_size:train_size+val_size],
        X_scaled[train_size+val_size:]
    )
    y_train, y_val, y_test = (
        y[:train_size],
        y[train_size:train_size+val_size],
        y[train_size+val_size:]
    )

    print("\n=== 轻量级自动调参 ===")
    best_params_report = []

    # ===== XGBoost =====
    param_xgb = [
        {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300},
        {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 400},
        {"max_depth": 10, "learning_rate": 0.1, "n_estimators": 500}
    ]
    print("\n调参中: XGBoost")
    best_xgb, xgb_params, xgb_r2 = evaluate_model(
        xgb.XGBRegressor(random_state=42, n_jobs=-1), param_xgb,
        X_train, X_val, y_train, y_val
    )
    best_params_report.append(f"XGBoost 最优参数: {xgb_params}, 验证R²={xgb_r2:.4f}")

    # ===== LightGBM =====
    param_lgb = [
        {"max_depth": 8, "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 300},
        {"max_depth": 10, "num_leaves": 63, "learning_rate": 0.05, "n_estimators": 400},
        {"max_depth": 12, "num_leaves": 127, "learning_rate": 0.1, "n_estimators": 500}
    ]
    print("\n调参中: LightGBM")
    best_lgb, lgb_params, lgb_r2 = evaluate_model(
        lgb.LGBMRegressor(random_state=42, n_jobs=-1), param_lgb,
        X_train, X_val, y_train, y_val
    )
    best_params_report.append(f"LightGBM 最优参数: {lgb_params}, 验证R²={lgb_r2:.4f}")

    # ===== Random Forest =====
    param_rf = [
        {"n_estimators": 300, "max_depth": 15, "min_samples_split": 5},
        {"n_estimators": 400, "max_depth": 20, "min_samples_split": 2},
        {"n_estimators": 500, "max_depth": None, "min_samples_split": 10}
    ]
    print("\n调参中: Random Forest")
    best_rf, rf_params, rf_r2 = evaluate_model(
        RandomForestRegressor(random_state=42, n_jobs=-1), param_rf,
        X_train, X_val, y_train, y_val
    )
    best_params_report.append(f"Random Forest 最优参数: {rf_params}, 验证R²={rf_r2:.4f}")

    # 保存最优参数报告
    best_param_path = os.path.join(result_dir, "Best_Params_Report.txt")
    with open(best_param_path, "w", encoding="utf-8") as f:
        f.write("\n".join(best_params_report))
    print(f"\n最优参数已保存至: {best_param_path}")

    # ===== 使用最优模型预测 =====
    y_pred_xgb = best_xgb.predict(X_test)
    y_pred_lgb = best_lgb.predict(X_test)
    y_pred_rf = best_rf.predict(X_test)

    # ===== 性能指标计算 =====
    def metrics(model_name, y_true, y_pred, y_train_true, y_train_pred):
        return {
            'model': model_name,
            'train_rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'train_r2': r2_score(y_train_true, y_train_pred),
            'test_r2': r2_score(y_true, y_pred),
            'train_mae': mean_absolute_error(y_train_true, y_train_pred),
            'test_mae': mean_absolute_error(y_true, y_pred)
        }

    results = [
        metrics("XGBoost", y_test, y_pred_xgb, y_train, best_xgb.predict(X_train)),
        metrics("LightGBM", y_test, y_pred_lgb, y_train, best_lgb.predict(X_train)),
        metrics("Random Forest", y_test, y_pred_rf, y_train, best_rf.predict(X_train))
    ]
    metrics_df = pd.DataFrame(results)

    # ===== 保存性能指标 =====
    metrics_path = os.path.join(result_dir, "optimized_model_performance.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    # ===== 合并预测输出 =====
    meta_test = df.iloc[-len(y_test):].reset_index(drop=True)
    meta_test["PM2_5_true"] = y_test.values
    meta_test["PM2_5_pred_XGBoost"] = y_pred_xgb
    meta_test["PM2_5_pred_LightGBM"] = y_pred_lgb
    meta_test["PM2_5_pred_RandomForest"] = y_pred_rf
    meta_test.to_csv(os.path.join(result_dir, "AllModels_full_predictions.csv"), index=False, encoding="utf-8")

    # ===== 输出特征重要性 =====
    print("\n=== 输出特征重要性 ===")
    fi_all = []

    def add_importance(model, model_name):
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        })
        fi["model"] = model_name
        return fi.sort_values("importance", ascending=False)

    fi_all.append(add_importance(best_xgb, "XGBoost"))
    fi_all.append(add_importance(best_lgb, "LightGBM"))
    fi_all.append(add_importance(best_rf, "Random Forest"))

    feature_importance_df = pd.concat(fi_all, ignore_index=True)
    feature_importance_df = feature_importance_df.sort_values(["model", "importance"], ascending=[True, False])
    feature_importance_path = os.path.join(result_dir, "Feature_Importance.csv")
    feature_importance_df.to_csv(feature_importance_path, index=False, encoding="utf-8")

    print(f"\n性能指标已保存至: {metrics_path}")
    print(f"预测结果已保存至: {os.path.join(result_dir, 'AllModels_full_predictions.csv')}")
    print(f"特征重要性已保存至: {feature_importance_path}")

    # ===== 输出最佳模型 =====
    best_model_name = metrics_df.loc[metrics_df["test_r2"].idxmax(), "model"]
    print(f"\n最佳模型为: {best_model_name}")
    best_pred_col = f"PM2_5_pred_{best_model_name.replace(' ', '')}"
    best_pred_path = os.path.join(result_dir, "BestModel_full_predictions.csv")
    meta_test[["timestamp", "PM2_5_true", best_pred_col] +
              [c for c in meta_test.columns if c not in ['timestamp', 'PM2_5_true', best_pred_col]]
              ].to_csv(best_pred_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()

"""train.py
Train regression models and log experiments to MLflow. Produces a saved model file in models/
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
import numpy as np


def load_data(p: Path):
    return pd.read_csv(p)


def prepare_Xy(df: pd.DataFrame):
    # simple feature set
    feats = ['Agent_Age', 'Agent_Rating', 'distance_km', 'pickup_delay_min', 'order_hour']
    X = df[feats].fillna(0)
    y = df['Delivery_Time']
    return X, y


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def main():
    proc = Path(__file__).parent / 'data' / 'processed' / 'amazon_delivery_processed.csv'
    if not proc.exists():
        raise SystemExit('Processed data not found. Run data_prep.py first.')
    df = load_data(proc)
    X, y = prepare_Xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlflow.set_experiment('amazon_delivery_regression')
    best_model = None
    best_rmse = float('inf')
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)

    # 1) Linear Regression (baseline)
    with mlflow.start_run(run_name='LinearRegression'):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        metrics = evaluate(lr, X_test, y_test)
        mlflow.log_param('model', 'LinearRegression')
        mlflow.log_metrics(metrics)
        # input example
        input_example = X_test.iloc[:3].to_dict(orient='records')
        mlflow.sklearn.log_model(lr, artifact_path='model', input_example=input_example)
        print('LinearRegression', metrics)
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_model = lr
            joblib.dump(lr, models_dir / 'best_model.pkl')

    # 2) RandomForest with RandomizedSearchCV
    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=3, cv=2, random_state=42, n_jobs=-1)
    with mlflow.start_run(run_name='RandomForest_Search'):
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        metrics = evaluate(best_rf, X_test, y_test)
        mlflow.log_param('model', 'RandomForest')
        mlflow.log_param('best_params', rf_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_rf, artifact_path='model', input_example=input_example)
        print('RandomForest', metrics, 'best_params=', rf_search.best_params_)
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_model = best_rf
            joblib.dump(best_rf, models_dir / 'best_model.pkl')

    # 3) XGBoost with RandomizedSearchCV
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)
    xgb_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 8],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    xgb_search = RandomizedSearchCV(xgb_reg, xgb_params, n_iter=3, cv=2, random_state=42, n_jobs=-1)
    with mlflow.start_run(run_name='XGBoost_Search'):
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        metrics = evaluate(best_xgb, X_test, y_test)
        mlflow.log_param('model', 'XGBoost')
        mlflow.log_param('best_params', xgb_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_xgb, artifact_path='model', input_example=input_example)
        print('XGBoost', metrics, 'best_params=', xgb_search.best_params_)
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_model = best_xgb
            joblib.dump(best_xgb, models_dir / 'best_model.pkl')

    print('Best RMSE:', best_rmse)


if __name__ == '__main__':
    main()

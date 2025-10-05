"""data_prep.py
Functions to load, clean and feature-engineer the amazon delivery dataset.
Saves a processed CSV to data/processed/amazon_delivery_processed.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def haversine_km(lat1, lon1, lat2, lon2):
    # vectorized haversine (works with pandas Series)
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # strip whitespace in string columns
    str_cols = df.select_dtypes(include=[object]).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()

    # Replace empty strings back to NaN where appropriate
    df.replace({'': np.nan, 'nan': np.nan}, inplace=True)

    # Agent_Rating has some missing values -> fill with median
    if 'Agent_Rating' in df.columns:
        df['Agent_Rating'] = pd.to_numeric(df['Agent_Rating'], errors='coerce')
        df['Agent_Rating'].fillna(df['Agent_Rating'].median(), inplace=True)

    # Weather: fill missing with 'Unknown'
    if 'Weather' in df.columns:
        df['Weather'] = df['Weather'].fillna('Unknown')

    # Traffic: normalize values (trim & capitalize)
    if 'Traffic' in df.columns:
        df['Traffic'] = df['Traffic'].astype(str).str.strip().str.title()

    # Remove rows with invalid coordinates (both store lat/lon zero)
    if set(['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']).issubset(df.columns):
        mask_bad_coords = (
            (df['Store_Latitude'].abs() < 1e-6) & (df['Store_Longitude'].abs() < 1e-6)
        )
        if mask_bad_coords.any():
            df = df.loc[~mask_bad_coords].reset_index(drop=True)

    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Distance in km
    df['distance_km'] = haversine_km(df['Store_Latitude'], df['Store_Longitude'], df['Drop_Latitude'], df['Drop_Longitude'])

    # Parse date/time columns
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.time
    df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce').dt.time

    # create datetime for order and pickup to compute pickup delay
    df['order_dt'] = pd.to_datetime(df['Order_Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Order_Time'].astype(str), errors='coerce')
    df['pickup_dt'] = pd.to_datetime(df['Order_Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Pickup_Time'].astype(str), errors='coerce')
    df['pickup_delay_min'] = (df['pickup_dt'] - df['order_dt']).dt.total_seconds() / 60.0
    df['pickup_delay_min'] = df['pickup_delay_min'].fillna(0).clip(lower=0)

    # time features
    df['order_hour'] = df['order_dt'].dt.hour.fillna(0).astype(int)
    df['order_dow'] = df['order_dt'].dt.dayofweek.fillna(0).astype(int)  # 0=Mon

    # Normalize Area values
    if 'Area' in df.columns:
        df['Area'] = df['Area'].str.title()

    # Trim Vehicle and Category
    for c in ['Vehicle', 'Category']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()

    return df


def save_processed(df: pd.DataFrame, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved processed data to {out}")


def main():
    src = Path(__file__).parent / 'amazon_delivery.csv'
    if not src.exists():
        # try workspace root path
        src = Path(r"c:\Users\aswin\Documents\Labmentix\7. Amazon Delivery Time Prediction\amazon_delivery.csv")
    df = load_data(str(src))
    print('Raw shape:', df.shape)
    df = clean_data(df)
    print('After clean shape:', df.shape)
    df = feature_engineer(df)
    save_processed(df, Path(__file__).parent / 'data' / 'processed' / 'amazon_delivery_processed.csv')


if __name__ == '__main__':
    main()

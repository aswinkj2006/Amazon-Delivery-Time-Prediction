"""eda.py
Generates basic EDA outputs and saves plots to outputs/figs
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_processed(path: str):
    return pd.read_csv(path)


def basic_eda(df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # distribution of delivery times
    plt.figure(figsize=(8,4))
    sns.histplot(df['Delivery_Time'], bins=50)
    plt.title('Delivery Time Distribution')
    plt.xlabel('Delivery Time (minutes)')
    plt.savefig(out_dir / 'delivery_time_dist.png')
    plt.close()

    # delivery time by category (top 10)
    top_cat = df.groupby('Category')['Delivery_Time'].median().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_cat.values, y=top_cat.index)
    plt.title('Median Delivery Time by Category (top 10)')
    plt.xlabel('Median Delivery Time (minutes)')
    plt.savefig(out_dir / 'median_delivery_by_category.png')
    plt.close()

    # distance vs delivery time scatter
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df['distance_km'], y=df['Delivery_Time'], alpha=0.3)
    plt.title('Distance vs Delivery Time')
    plt.xlabel('Distance (km)')
    plt.ylabel('Delivery Time (minutes)')
    plt.savefig(out_dir / 'distance_vs_delivery.png')
    plt.close()

    print('Saved EDA figures to', out_dir)


if __name__ == '__main__':
    proc = Path(__file__).parent / 'data' / 'processed' / 'amazon_delivery_processed.csv'
    if not proc.exists():
        print('Processed data not found. Run data_prep.py first.')
    else:
        df = load_processed(proc)
        basic_eda(df, Path(__file__).parent / 'outputs' / 'figs')

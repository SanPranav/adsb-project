import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import folium
import os
from datetime import datetime

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_sessions():
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not all_files:
        print("No session files found in /data")
        return None
    
    dfs = []
    for f in all_files:
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined = combined.dropna(subset=['lat', 'lon'])
    combined = combined[combined['lat'] != '']
    combined['lat'] = pd.to_numeric(combined['lat'], errors='coerce')
    combined['lon'] = pd.to_numeric(combined['lon'], errors='coerce')
    combined['altitude_ft'] = pd.to_numeric(combined['altitude_ft'], errors='coerce')
    combined['speed_kts'] = pd.to_numeric(combined['speed_kts'], errors='coerce')
    combined['rssi'] = pd.to_numeric(combined['rssi'], errors='coerce')
    combined = combined.dropna(subset=['lat', 'lon'])
    
    print(f"Loaded {len(combined)} records from {len(all_files)} sessions")
    print(f"Unique aircraft: {combined['icao'].nunique()}")
    print(f"Time range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    
    return combined

def plot_altitude_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    df['altitude_ft'].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='white')
    ax.set_xlabel('Altitude (ft)')
    ax.set_ylabel('Message count')
    ax.set_title('Altitude distribution of received ADS-B messages')
    ax.axvline(x=10000, color='red', linestyle='--', alpha=0.5, label='10,000 ft')
    ax.axvline(x=30000, color='orange', linestyle='--', alpha=0.5, label='30,000 ft')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'altitude_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_rssi_vs_altitude(df):
    sample = df[['altitude_ft', 'rssi']].dropna().sample(min(5000, len(df)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(sample['altitude_ft'], sample['rssi'],
               alpha=0.3, s=5, color='steelblue')
    ax.set_xlabel('Altitude (ft)')
    ax.set_ylabel('RSSI (dBFS)')
    ax.set_title('Signal strength vs altitude')
    m, b = np.polyfit(sample['altitude_ft'].fillna(0),
                      sample['rssi'].fillna(0), 1)
    x_line = np.linspace(sample['altitude_ft'].min(),
                         sample['altitude_ft'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='red',
            linewidth=2, label=f'Trend (slope={m:.6f})')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'rssi_vs_altitude.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_traffic_by_hour(df):
    df['hour'] = df['timestamp'].dt.hour
    hourly = df.groupby('hour')['icao'].nunique()
    fig, ax = plt.subplots(figsize=(10, 4))
    hourly.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_xlabel('Hour of day (local)')
    ax.set_ylabel('Unique aircraft count')
    ax.set_title('Air traffic volume by hour — San Diego airspace')
    ax.set_xticklabels([f"{h:02d}:00" for h in hourly.index], rotation=45)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'traffic_by_hour.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_speed_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    df['speed_kts'].dropna().hist(bins=40, ax=ax,
                                   color='steelblue', edgecolor='white')
    ax.set_xlabel('Ground speed (knots)')
    ax.set_ylabel('Message count')
    ax.set_title('Aircraft speed distribution')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'speed_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def generate_flight_map(df):
    center_lat = df['lat'].median()
    center_lon = df['lon'].median()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    for icao, group in df.groupby('icao'):
        group = group.sort_values('timestamp')
        coords = list(zip(group['lat'], group['lon']))
        if len(coords) < 2:
            continue
        folium.PolyLine(
            coords,
            weight=1.5,
            opacity=0.6,
            color='steelblue'
        ).add_to(m)
    
    path = os.path.join(OUTPUT_DIR, 'flight_map.html')
    m.save(path)
    print(f"Saved interactive map: {path}")

def generate_summary_stats(df):
    stats = {
        "total_messages": len(df),
        "unique_aircraft": df['icao'].nunique(),
        "unique_flights": df['flight'].nunique(),
        "avg_altitude_ft": round(df['altitude_ft'].mean(), 0),
        "max_altitude_ft": df['altitude_ft'].max(),
        "avg_speed_kts": round(df['speed_kts'].mean(), 1),
        "avg_rssi_dbfs": round(df['rssi'].mean(), 2),
        "collection_hours": round(
            (df['timestamp'].max() - df['timestamp'].min())
            .total_seconds() / 3600, 2
        )
    }
    
    print("\n=== SUMMARY STATS (copy into paper) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    import json
    with open(os.path.join(OUTPUT_DIR, 'summary_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    return stats

def run_all():
    df = load_all_sessions()
    if df is None:
        return
    
    print("\nGenerating figures...")
    plot_altitude_distribution(df)
    plot_rssi_vs_altitude(df)
    plot_traffic_by_hour(df)
    plot_speed_distribution(df)
    generate_flight_map(df)
    generate_summary_stats(df)
    print(f"\nAll outputs saved to /{OUTPUT_DIR}/")

if __name__ == "__main__":
    run_all()
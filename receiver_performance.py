import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

YOUR_LAT = 33.0157816  # Del Norte area — update with your exact home coords
YOUR_LON = -117.1187338

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_range(df):
    R = 6371
    lat1 = np.radians(YOUR_LAT)
    lat2 = np.radians(df['lat'])
    dlat = np.radians(df['lat'] - YOUR_LAT)
    dlon = np.radians(df['lon'] - YOUR_LON)
    a = (np.sin(dlat/2)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    df['range_km'] = R * c
    df['range_nmi'] = df['range_km'] * 0.539957
    return df

def plot_rssi_vs_range(df):
    sample = df[['range_nmi', 'rssi']].dropna()
    sample = sample[sample['range_nmi'] < 300]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(sample['range_nmi'], sample['rssi'],
               alpha=0.2, s=4, color='steelblue')
    
    bins = np.arange(0, 300, 10)
    sample['range_bin'] = pd.cut(sample['range_nmi'], bins=bins)
    median_rssi = sample.groupby('range_bin')['rssi'].median()
    bin_centers = [b.mid for b in median_rssi.index]
    ax.plot(bin_centers, median_rssi.values, color='red',
            linewidth=2, label='Median RSSI per 10 nmi bin')
    
    ax.set_xlabel('Range (nautical miles)')
    ax.set_ylabel('RSSI (dBFS)')
    ax.set_title('Receiver signal strength vs range — RTL-SDR V3 + FlightAware antenna')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'rssi_vs_range.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_detection_range_map(df):
    max_range = df.groupby('icao')['range_nmi'].max()
    print(f"\nMax detection range: {max_range.max():.1f} nmi")
    print(f"Median max range per aircraft: {max_range.median():.1f} nmi")
    print(f"Aircraft detected beyond 100 nmi: {(max_range > 100).sum()}")
    print(f"Aircraft detected beyond 200 nmi: {(max_range > 200).sum()}")

def plot_message_rate_vs_altitude(df):
    df['alt_bin'] = pd.cut(df['altitude_ft'],
                            bins=range(0, 45000, 2000))
    msg_rate = df.groupby('alt_bin').size()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    msg_rate.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_xlabel('Altitude band (ft)')
    ax.set_ylabel('Message count')
    ax.set_title('ADS-B message volume by altitude band')
    ax.set_xticklabels(
        [f"{int(b.left/1000)}k" if hasattr(b, 'left') else ''
         for b in msg_rate.index],
        rotation=45
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'messages_by_altitude.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def run_all():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files:
        print("No data files yet")
        return
    
    df = pd.concat([pd.read_csv(os.path.join(DATA_DIR, f))
                    for f in files], ignore_index=True)
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['altitude_ft'] = pd.to_numeric(df['altitude_ft'], errors='coerce')
    df['rssi'] = pd.to_numeric(df['rssi'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    
    df = compute_range(df)
    
    print("Running receiver performance analysis...")
    plot_rssi_vs_range(df)
    plot_detection_range_map(df)
    plot_message_rate_vs_altitude(df)

if __name__ == "__main__":
    run_all()
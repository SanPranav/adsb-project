import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import os
import json
import traceback

FILES = [
    "data/session_20260728_130550.csv",
    "data/session_20260728_193409.csv",
    "data/session_20260728_213344.csv",
    "data/session_20260729_085503.csv",
    "data/session_20260730_221201.csv",
    "data/session_20260809_221548.csv",
    "data/session_20260809_231939.csv",
    "data/session_20260810_093921.csv",
    "data/session_20260810_112041.csv"
]
    

YOUR_LAT = 32.9175
YOUR_LON = -117.1476
OUTPUT = 'output'
os.makedirs(OUTPUT, exist_ok=True)

try:
    print("Loading files...")
    dfs = []
    for f in FILES:
        df = pd.read_csv(f)
        df['session'] = f
        dfs.append(df)
        print(f"  Loaded {f}: {len(df)} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined['rssi'] = pd.to_numeric(combined['rssi'], errors='coerce')
    combined['lat'] = pd.to_numeric(combined['lat'], errors='coerce')
    combined['lon'] = pd.to_numeric(combined['lon'], errors='coerce')
    print(f"Total rows: {len(combined)}")

    # Compute range
    valid = combined.dropna(subset=['lat','lon']).copy()
    R = 6371
    lat1 = np.radians(YOUR_LAT)
    lat2 = np.radians(valid['lat'])
    dlat = np.radians(valid['lat'] - YOUR_LAT)
    dlon = np.radians(valid['lon'] - YOUR_LON)
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    valid['range_nmi'] = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) * 0.539957
    print(f"Range computed for {len(valid)} rows")

    # Figure 1: RSSI vs Range
    print("Generating Figure 1...")
    fig, ax = plt.subplots(figsize=(10, 5))
    sample = valid.dropna(subset=['rssi','range_nmi'])
    ax.scatter(sample['range_nmi'], sample['rssi'],
               alpha=0.15, s=4, color='steelblue', label='Individual message')
    bins = np.arange(0, 95, 5)
    sample2 = sample.copy()
    sample2['bin'] = pd.cut(sample2['range_nmi'], bins=bins)
    median_rssi = sample2.groupby('bin', observed=True)['rssi'].median()
    centers = [b.mid for b in median_rssi.index]
    ax.plot(centers, median_rssi.values, color='red',
            linewidth=2, label='Median RSSI per 5 nmi bin')
    m, b = np.polyfit(sample['range_nmi'], sample['rssi'], 1)
    xl = np.linspace(0, 90, 100)
    ax.plot(xl, m*xl+b, color='orange', linestyle='--',
            linewidth=1.5, label=f'Linear trend (slope={m:.3f} dBFS/nmi)')
    ax.set_xlabel('Slant range (nautical miles)')
    ax.set_ylabel('RSSI (dBFS)')
    ax.set_title('Figure 1: Signal strength vs range - RTL-SDR V3, San Diego')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig1_rssi_vs_range.png', dpi=150)
    plt.close()
    print("  Saved Figure 1")

    # Figure 2: Traffic by hour
    print("Generating Figure 2...")
    fig, ax = plt.subplots(figsize=(10, 4))
    combined['hour'] = combined['timestamp'].dt.hour
    hourly = combined.groupby('hour')['icao'].nunique().reindex(range(24), fill_value=0)
    colors = ['coral' if h in [14,15,16,17] else 'steelblue' for h in range(24)]
    hourly.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.set_xlabel('Hour of day (UTC)')
    ax.set_ylabel('Unique aircraft observed')
    ax.set_title('Figure 2: Air traffic volume by hour - San Diego, Jul 28-31, Aug 9-10 2026')
    ax.set_xticklabels([f'{h:02d}' for h in range(24)], rotation=0, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig2_traffic_by_hour.png', dpi=150)
    plt.close()
    print("  Saved Figure 2")

    # Figure 3: Daily count
    print("Generating Figure 3...")
    fig, ax = plt.subplots(figsize=(8, 4))
    combined['date'] = combined['timestamp'].dt.date
    daily = combined.groupby('date')['icao'].nunique()
    daily.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_xlabel('Date')
    ax.set_ylabel('Unique aircraft')
    ax.set_title('Figure 3: Daily unique aircraft count')
    ax.set_xticklabels([str(d) for d in daily.index], rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig3_daily_count.png', dpi=150)
    plt.close()
    print("  Saved Figure 3")

    # Figure 4: RSSI distribution
    print("Generating Figure 4...")
    fig, ax = plt.subplots(figsize=(8, 4))
    combined['rssi'].dropna().hist(bins=40, ax=ax,
                                    color='steelblue', edgecolor='white')
    ax.axvline(combined['rssi'].mean(), color='red', linestyle='--',
               label=f'Mean: {combined["rssi"].mean():.1f} dBFS')
    ax.set_xlabel('RSSI (dBFS)')
    ax.set_ylabel('Message count')
    ax.set_title('Figure 4: Received signal strength distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig4_rssi_distribution.png', dpi=150)
    plt.close()
    print("  Saved Figure 4")

    # Figure 5: Range distribution
    print("Generating Figure 5...")
    fig, ax = plt.subplots(figsize=(8, 4))
    valid['range_nmi'].hist(bins=40, ax=ax,
                             color='steelblue', edgecolor='white')
    ax.axvline(valid['range_nmi'].mean(), color='red', linestyle='--',
               label=f'Mean: {valid["range_nmi"].mean():.1f} nmi')
    ax.axvline(valid['range_nmi'].max(), color='orange', linestyle='--',
               label=f'Max: {valid["range_nmi"].max():.1f} nmi')
    ax.set_xlabel('Range (nautical miles)')
    ax.set_ylabel('Message count')
    ax.set_title('Figure 5: Aircraft detection range distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig5_range_distribution.png', dpi=150)
    plt.close()
    print("  Saved Figure 5")

    # Flight map
    print("Generating flight map...")
    m = folium.Map(location=[YOUR_LAT, YOUR_LON], zoom_start=8,
                   tiles='CartoDB positron')
    folium.Marker([YOUR_LAT, YOUR_LON],
                  popup='Receiver location',
                  icon=folium.Icon(color='red', icon='star')).add_to(m)
    count = 0
    for icao, group in valid.groupby('icao'):
        group = group.sort_values('timestamp')
        coords = list(zip(group['lat'], group['lon']))
        if len(coords) < 2:
            continue
        folium.PolyLine(coords, weight=1, opacity=0.5,
                        color='steelblue').add_to(m)
        count += 1
    m.save(f'{OUTPUT}/flight_map.html')
    print(f"  Saved flight map ({count} flight paths)")

    # Summary stats
    stats = {
        "total_messages": int(len(combined)),
        "unique_aircraft": int(combined['icao'].nunique()),
        "unique_flights": int(combined['flight'].nunique()),
        "mean_rssi_dbfs": round(float(combined['rssi'].mean()), 2),
        "best_rssi_dbfs": round(float(combined['rssi'].max()), 2),
        "max_range_nmi": round(float(valid['range_nmi'].max()), 1),
        "mean_range_nmi": round(float(valid['range_nmi'].mean()), 1),
        "rssi_range_correlation": round(float(
            valid.dropna(subset=['rssi','range_nmi'])['rssi'].corr(
            valid.dropna(subset=['rssi','range_nmi'])['range_nmi'])), 4)
    }
    with open(f'{OUTPUT}/summary_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\nDone. Files in /output:")
    for f in os.listdir(OUTPUT):
        print(f"  {f}")

except Exception as e:
    print(f"\nERROR: {e}")
    traceback.print_exc()   
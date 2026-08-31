import requests
import pandas as pd
import time
import datetime
import os
import json

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DUMP1090_URL = "http://localhost:8080/data/aircraft.json"

def fetch_aircraft():
    try:
        response = requests.get(DUMP1090_URL, timeout=2)
        data = response.json()
        return data.get("aircraft", [])
    except Exception as e:
        print(f"Fetch error: {e}")
        return []

def log_session():
    session_start = datetime.datetime.now()
    filename = os.path.join(
        OUTPUT_DIR,
        f"session_{session_start.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    
    print(f"Logging to {filename}")
    print("Press Ctrl+C to stop")
    
    rows = []
    
    try:
        while True:
            timestamp = datetime.datetime.now().isoformat()
            aircraft_list = fetch_aircraft()
            
            for ac in aircraft_list:
                row = {
                    "timestamp": timestamp,
                    "icao": ac.get("hex", ""),
                    "flight": ac.get("flight", "").strip(),
                    "lat": ac.get("lat", ""),
                    "lon": ac.get("lon", ""),
                    "altitude_ft": ac.get("alt_baro", ""),
                    "speed_kts": ac.get("gs", ""),
                    "track_deg": ac.get("track", ""),
                    "vertical_rate": ac.get("baro_rate", ""),
                    "squawk": ac.get("squawk", ""),
                    "rssi": ac.get("rssi", ""),
                    "messages": ac.get("messages", ""),
                    "seen_pos": ac.get("seen_pos", "")
                }
                rows.append(row)
            
            print(f"\r{timestamp} — {len(aircraft_list)} aircraft visible", end="")
            
            if len(rows) >= 500:
                df = pd.DataFrame(rows)
                df.to_csv(filename, mode='a',
                         header=not os.path.exists(filename),
                         index=False)
                rows = []
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filename, mode='a',
                     header=not os.path.exists(filename),
                     index=False)
        print(f"\nSession saved to {filename}")
        print(f"Total duration: {datetime.datetime.now() - session_start}")

if __name__ == "__main__":
    log_session()
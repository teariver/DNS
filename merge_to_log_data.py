# extract_features_full_1min.py
# ==========================================
# 🧠 LOG → FULL FEATURE TABLE (1-minute resample, wide format)
# ==========================================
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import entropy
import logging
from tqdm import tqdm

# ========== CONFIG ==========
RAW_LOG_FILE = r"C:\Users\Surface\Documents\DNS\_BANK_DNS_LOG_INFOBLOX_ALL_ENRICHED__202510231125.csv"
OUTPUT_DIR = r"C:\Users\Surface\Documents\DNS\processed_logs"
OUT_FILE = os.path.join(OUTPUT_DIR, "feature_timeseries.csv")
AGG_FREQ = "1s"
LOG_FILE = r"C:\Users\Surface\Documents\DNS\logs\pipeline.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
def log(s):
    print(s)
    logging.info(s)

# ========== HELPERS ==========
def calc_entropy(series):
    try:
        counts = series.value_counts()
        if len(counts) == 0:
            return 0.0
        return float(entropy(counts))
    except Exception:
        return 0.0

def top_value(series):
    try:
        if series.dropna().empty:
            return "Unknown"
        return series.value_counts().idxmax()
    except Exception:
        return "Unknown"

# safe access helper
def sget(series, name):
    return series.get(name) if hasattr(series, 'get') else series

# ========== LOAD RAW ==========
def load_raw(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    log(f"Loading CSV (low_memory=False): {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    if 'timestamp' not in df.columns:
        raise ValueError("Input file missing 'timestamp' column")
    # parse timestamp robustly
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    log(f"Raw rows after parsing timestamp: {len(df):,}")
    return df

# ========== BUILD FEATURES ==========
def build_full_features(df):
    log("Extracting global lists of categories and log_types...")
    cats = sorted(df['category'].dropna().unique().tolist()) if 'category' in df else []
    lts  = sorted(df['log_type'].dropna().unique().tolist()) if 'log_type' in df else []

    log(f"Found {len(cats)} categories, {len(lts)} log_types")

    # We'll aggregate per-minute only for minutes that HAVE logs, then reindex to full minute range
    records = []

    log("Aggregating windows per minute (only minutes with logs)...")
    # Iterate over resampled windows (this tolerates duplicate timestamps in index)
    for ts, window in tqdm(df.set_index('timestamp').resample(AGG_FREQ), desc="resample"):
        if window.shape[0] == 0:
            continue

        rec = {}
        rec['timestamp'] = ts

        # --- A) tổng quan
        rec['total_logs'] = int(len(window))
        rec['unique_src_ip'] = int(window['src_ip'].nunique()) if 'src_ip' in window else 0
        rec['unique_dst_ip'] = int(window['dst_ip'].nunique()) if 'dst_ip' in window else 0
        rec['unique_fqdn'] = int(window['fqdn'].nunique()) if 'fqdn' in window else 0
        rec['query_rate_dns'] = float((window['protocol'] == 'DNS').mean()) if 'protocol' in window else 0.0
        rec['drop_rate'] = float((window['action'] == 'DROP').mean()) if 'action' in window else 0.0
        rec['alert_rate'] = float((window['action'] == 'ALERT').mean()) if 'action' in window else 0.0

        # --- B) event / category
        rec['unique_event_count'] = int(window['event_name'].nunique()) if 'event_name' in window else 0
        rec['top_event_name'] = top_value(window['event_name']) if 'event_name' in window else "Unknown"
        # count by category & log_type (only present categories for this minute will be added)
        if 'category' in window:
            vc = window['category'].value_counts()
            for cat, cnt in vc.items():
                # sanitize column name
                col = f"count_by_category_{str(cat).replace(' ','_').replace('/','_')}"
                rec[col] = int(cnt)
        if 'log_type' in window:
            vc = window['log_type'].value_counts()
            for lt, cnt in vc.items():
                col = f"count_by_log_type_{str(lt).replace(' ','_').replace('/','_')}"
                rec[col] = int(cnt)

        # --- C) intensity / severity
        rec['avg_hit_count'] = float(window['hit_count'].astype(float).mean()) if 'hit_count' in window else 0.0
        rec['max_hit_count'] = float(window['hit_count'].astype(float).max()) if 'hit_count' in window else 0.0
        rec['avg_severity'] = float(window['severity'].astype(float).mean()) if 'severity' in window else 0.0
        rec['count_threshold_crossed'] = int((window.get('threshold_crossed_flag', pd.Series(dtype=int)) == 1).sum())
        rec['flood_alert_rate'] = float((window.get('flood_alert_flag', pd.Series(dtype=int)) == 1).mean())
        rec['avg_threshold_value'] = float(window.get('threshold_value', pd.Series(dtype=float)).astype(float).mean()) if 'threshold_value' in window else 0.0
        rec['avg_high_value'] = float(window.get('high_value', pd.Series(dtype=float)).astype(float).mean()) if 'high_value' in window else 0.0
        rec['avg_low_value'] = float(window.get('low_value', pd.Series(dtype=float)).astype(float).mean()) if 'low_value' in window else 0.0

        # --- D) IP / port
        rec['top_src_ip'] = top_value(window['src_ip']) if 'src_ip' in window else "Unknown"
        rec['top_dst_ip'] = top_value(window['dst_ip']) if 'dst_ip' in window else "Unknown"
        rec['ip_entropy'] = float(calc_entropy(window['src_ip'])) if 'src_ip' in window else 0.0
        rec['port_entropy'] = float(calc_entropy(window['dst_port'])) if 'dst_port' in window else 0.0

        # --- E) DNS / RPZ / fqdn
        rec['rpz_activity_count'] = int((window.get('log_type', pd.Series()) == 'RPZ').sum())
        rec['invalid_fqdn_ratio'] = float(window['fqdn'].isnull().mean()) if 'fqdn' in window else 0.0
        rec['blacklist_hits'] = int(window['fqdn'].astype(str).str.contains('antimalware|blacklist', case=False, na=False).sum()) if 'fqdn' in window else 0
        rec['unique_zones'] = int(window.get('zone', pd.Series()).nunique()) if 'zone' in window else 0

        # --- F) system
        rec['dhcp_event_rate'] = float((window.get('log_type', pd.Series()) == 'DHCP').mean())
        rec['serial_update_events'] = int(window.get('serial', pd.Series()).notnull().sum())
        rec['dns_latency_event'] = int((window.get('dns_update_latency', pd.Series()) == '0/0/0/0').sum())

        records.append(rec)

    log(f"Aggregated {len(records):,} minute-windows (non-empty). Building DataFrame...")
    agg = pd.DataFrame(records)
    # set timestamp index
    agg['timestamp'] = pd.to_datetime(agg['timestamp'])
    agg = agg.set_index('timestamp').sort_index()

    # Create full minute range and reindex the aggregated DataFrame (this is safe: agg index unique)
    full_index = pd.date_range(start=agg.index.min(), end=agg.index.max(), freq=AGG_FREQ)
    agg = agg.reindex(full_index)  # minutes without logs become NaN rows

    # Ensure we have consistent columns for all categories/log_types observed globally:
    # Add zeros for missing category/log_type columns (using global cats/lts)
    for cat in cats:
        col = f"count_by_category_{str(cat).replace(' ','_').replace('/','_')}"
        if col not in agg.columns:
            agg[col] = 0
    for lt in lts:
        col = f"count_by_log_type_{str(lt).replace(' ','_').replace('/','_')}"
        if col not in agg.columns:
            agg[col] = 0

    # Fill NaN: numeric -> 0, text -> 'Unknown'
    for c in agg.columns:
        if agg[c].dtype == 'O':
            agg[c] = agg[c].fillna('Unknown')
        else:
            agg[c] = agg[c].fillna(0)

    # Reset index to have timestamp column
    agg = agg.reset_index().rename(columns={'index':'timestamp'})
    log(f"Feature DataFrame shape after reindex: {agg.shape[0]:,} rows × {agg.shape[1]:,} cols")
    return agg

# ========== MAIN ==========
def main():
    try:
        df = load_raw(RAW_LOG_FILE)
        feat = build_full_features(df)
        feat.to_csv(OUT_FILE, index=False)
        log(f"Saved feature file: {OUT_FILE}")
        print("DONE ✅")
    except Exception as e:
        log(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()

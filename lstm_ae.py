#!/usr/bin/env python3
# train_lstm_autoencoder_fast_fixed.py
# ✅ Final optimized version — fixed DatetimeIndex bug + fast runtime

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ================= CONFIG =================
FEATURE_CSV = r"C:\Users\Surface\Documents\DNS\processed_logs\feature_timeseries.csv"
RAW_LOG_FILE = r"C:\Users\Surface\Documents\DNS\raw_log.csv"
OUTPUT_DIR = r"C:\Users\Surface\Documents\DNS\dl_output"

# === SPEED/QUALITY BALANCE ===
FAST_MODE = True          # ⚡️ Bật khi test nhanh, tắt khi train full
WINDOW_SIZE = 30
STRIDE = 5
BATCH_SIZE = 256
EPOCHS = 40
LATENT_DIM = 16
RANDOM_SEED = 42
THRESHOLD_STD_MULT = 3
MAX_SNIPPET_LEN = 150
PATIENCE = 3

# Safety setup
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ================= PERFORMANCE SETUP =================
try:
    cpus = os.cpu_count() or 1
    tf.config.threading.set_intra_op_parallelism_threads(cpus)
    tf.config.threading.set_inter_op_parallelism_threads(cpus)
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print("✅ GPU detected — using GPU acceleration.")
else:
    print("⚙️ No GPU detected — running multi-threaded on CPU.")

# ================= HELPERS =================
def safe_read_csv(path, parse_dates=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False, on_bad_lines="skip", parse_dates=parse_dates)

def make_sequences(arr, window=60, stride=1):
    seqs, idxs = [], []
    T = arr.shape[0]
    for start in range(0, T - window + 1, stride):
        end = start + window
        seqs.append(arr[start:end])
        idxs.append(end - 1)
    return np.stack(seqs), np.array(idxs)

def build_lstm_autoencoder(window_size, n_features, latent_dim=16):
    """Simplified, faster LSTM Autoencoder"""
    inp = layers.Input(shape=(window_size, n_features))
    x = layers.LSTM(64, return_sequences=False)(inp)
    z = layers.Dense(latent_dim, activation="relu")(x)
    x = layers.RepeatVector(window_size)(z)
    x = layers.LSTM(64, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n_features))(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

def compute_sequence_mse(orig, recon):
    return np.mean(np.square(orig - recon), axis=(1, 2))

def compute_feature_mse(orig, recon):
    return np.mean(np.square(orig - recon), axis=(0, 1))

def safe_timestamp_col(df):
    candidates = [c for c in df.columns if any(x in c.lower() for x in ["time", "timestamp", "date"])]
    return candidates[0] if candidates else None

# ================= TRAINING PIPELINE =================
def train_and_export():
    print("🚀 Loading features...")
    df_feat = safe_read_csv(FEATURE_CSV, parse_dates=["timestamp"])
    if "timestamp" not in df_feat.columns:
        raise RuntimeError("❌ Feature CSV must contain 'timestamp' column.")
    df_feat = df_feat.sort_values("timestamp").reset_index(drop=True)

    if FAST_MODE:
        df_feat = df_feat.head(10000)
        print("⚡ FAST_MODE active: using first 10,000 rows for quick training test")

    timestamps = df_feat["timestamp"].copy()
    X_df = df_feat.drop(columns=["timestamp"])

    # Convert numeric
    for c in X_df.columns:
        if X_df[c].dtype == object:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise RuntimeError("❌ No numeric columns found.")
    dropped_cols = [c for c in X_df.columns if c not in numeric_cols]
    with open(os.path.join(OUTPUT_DIR, "dropped_non_numeric_columns.json"), "w", encoding="utf-8") as f:
        json.dump(dropped_cols, f, indent=2, ensure_ascii=False)

    X = X_df[numeric_cols].fillna(0.0).astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    seqs, seq_idx = make_sequences(X_scaled, window=WINDOW_SIZE, stride=STRIDE)
    print(f"📈 Sequences created: {seqs.shape}")

    n_seq = seqs.shape[0]
    n_train = int(n_seq * 0.8)
    X_train, X_val = seqs[:n_train], seqs[n_train:]

    # Dataset pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, X_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_lstm_autoencoder(WINDOW_SIZE, X.shape[1], LATENT_DIM)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[es], verbose=2)

    model.save(os.path.join(OUTPUT_DIR, "lstm_ae_model.h5"))
    print("✅ Model saved.")

    recon_train = model.predict(X_train, batch_size=BATCH_SIZE)
    recon_all = model.predict(seqs, batch_size=BATCH_SIZE)
    mse_train = compute_sequence_mse(X_train, recon_train)
    mse_all = compute_sequence_mse(seqs, recon_all)

    mean_train, std_train = mse_train.mean(), mse_train.std()
    threshold = mean_train + THRESHOLD_STD_MULT * std_train
    print(f"📊 Threshold: {threshold:.6f}")

    # ✅ FIXED: convert timestamps safely to Series to avoid DatetimeIndex error
    seq_timestamps = pd.Series(pd.to_datetime(timestamps.iloc[seq_idx].values)).reset_index(drop=True)

    anomaly_df = pd.DataFrame({
        "timestamp": seq_timestamps,
        "reconstruction_error": mse_all,
        "is_anomaly": (mse_all > threshold)
    })
    anomaly_csv = os.path.join(OUTPUT_DIR, "anomaly_result.csv")
    anomaly_df.to_csv(anomaly_csv, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
    print(f"✅ anomaly_result saved -> {anomaly_csv}")

    # feature importance
    recon_val = model.predict(X_val, batch_size=BATCH_SIZE)
    feat_mse = compute_feature_mse(X_val, recon_val)
    feat_importance = {numeric_cols[i]: float(feat_mse[i]) for i in range(len(numeric_cols))}
    feat_importance = dict(sorted(feat_importance.items(), key=lambda kv: kv[1], reverse=True))
    with open(os.path.join(OUTPUT_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(feat_importance, f, indent=2)

    # metrics
    mse_val = compute_sequence_mse(X_val, recon_val)
    metrics = {
        "train_mse_mean": float(mean_train),
        "train_mse_std": float(std_train),
        "val_mse_mean": float(mse_val.mean()),
        "val_mse_std": float(mse_val.std()),
        "threshold": float(threshold)
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

    # plot anomaly timeline
    plt.figure(figsize=(14, 5))
    sns.lineplot(x="timestamp", y="reconstruction_error", data=anomaly_df)
    plt.axhline(threshold, color="orange", linestyle="--", label="Threshold")
    plt.scatter(anomaly_df.loc[anomaly_df["is_anomaly"], "timestamp"],
                anomaly_df.loc[anomaly_df["is_anomaly"], "reconstruction_error"],
                color="red", s=20, label="Anomaly")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_plot.png"), dpi=150)
    plt.close()

    return anomaly_df

# ================= MERGE & SUMMARY =================
def merge_and_summarize(anomaly_csv):
    print("\n🔍 Merging anomaly results with raw log file...")
    df_raw = safe_read_csv(RAW_LOG_FILE)
    if df_raw.empty:
        raise RuntimeError("❌ Raw log file empty or unreadable.")

    time_col = safe_timestamp_col(df_raw)
    if time_col is None:
        raise RuntimeError("❌ No timestamp column found in raw log.")
    df_raw["time"] = pd.to_datetime(df_raw[time_col], errors="coerce")

    df_result = pd.read_csv(anomaly_csv, parse_dates=["timestamp"])
    df_raw = df_raw.sort_values("time").dropna(subset=["time"]).reset_index(drop=True)
    df_result = df_result.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)

    merged = pd.merge_asof(
        df_raw,
        df_result.rename(columns={"timestamp": "time"}),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=2)
    )

    merged["is_anomaly"] = merged["is_anomaly"].fillna(False)
    merged["reconstruction_error"] = merged["reconstruction_error"].fillna(0.0)
    merged.to_csv(os.path.join(OUTPUT_DIR, "raw_with_anomaly_flag.csv"), index=False)
    print("✅ raw_with_anomaly_flag.csv saved.")

    # summary
    anomalies = merged[merged["is_anomaly"] == True].copy()
    if anomalies.empty:
        print("⚠️ No anomalies detected.")
        return

    text_candidates = ["message", "msg", "log", "info", "description", "detail", "query"]
    text_cols = [c for c in merged.columns if c.lower() in text_candidates]
    if not text_cols:
        text_cols = [c for c in merged.columns if merged[c].dtype == object]
    anomalies["combined_text"] = anomalies[text_cols].astype(str).agg(" ".join, axis=1).str.lower() if text_cols else ""

    def classify(text):
        t = str(text).lower()
        if "thread_protect" in t: return "Thread Protect", "DROP", "Critical"
        if "refused" in t: return "Connection Refused", "Check ACLs", "Warning"
        if "nxdomain" in t: return "NXDOMAIN", "Ignore", "Info"
        if "rate limit" in t: return "Rate Limit", "Throttle", "Critical"
        if "timeout" in t: return "Timeout", "Retry", "Info"
        if "servfail" in t: return "SERVFAIL", "Check upstream", "Warning"
        if "denied" in t: return "Access Denied", "Review policy", "Warning"
        if "suspicious" in t: return "Suspicious Query", "Manual check", "Critical"
        return "Unknown", "Investigate manually", "Warning"

    anomalies[["anomaly_type", "recommended_action", "severity"]] = anomalies["combined_text"].apply(lambda t: pd.Series(classify(t)))

    def snippet(row):
        for c in text_cols:
            if c in row and isinstance(row[c], str) and row[c].strip():
                return row[c][:MAX_SNIPPET_LEN]
        return row.get("combined_text", "")[:MAX_SNIPPET_LEN]

    anomalies["related_message"] = anomalies.apply(snippet, axis=1)
    anomalies["timestamp_str"] = anomalies["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    summary = anomalies[["timestamp_str", "anomaly_type", "recommended_action", "severity", "related_message", "reconstruction_error"]]
    summary = summary.rename(columns={"timestamp_str": "timestamp"})
    summary_path = os.path.join(OUTPUT_DIR, "anomaly_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"✅ anomaly_summary.csv saved -> {summary_path}")

# ================= MAIN =================
if __name__ == "__main__":
    result = train_and_export()
    merge_and_summarize(os.path.join(OUTPUT_DIR, "anomaly_result.csv"))
    print("\n🎯 All done! Outputs in:", OUTPUT_DIR)

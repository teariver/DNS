# =============================================================
# 🧠 FINAL CNN–LSTM AUTOENCODER — TIMESTAMP-SAFE + LOW-FP ANOMALY DETECTION
# =============================================================
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, RepeatVector, UpSampling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# 🔇 TensorFlow logs
tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ⚙️ Thread optimization
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# ---------------- CONFIG ----------------
DATA_FILE = r"C:\Users\Surface\Documents\DNS\processed_logs\feature_timeseries.csv"
RAW_LOG_FILE = r"C:\Users\Surface\Documents\DNS\raw_log.csv"
OUTPUT_DIR = r"C:\Users\Surface\Documents\DNS\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_STEP = 20
TEST_SPLIT = 0.2
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 1️⃣ LOAD & CLEAN
# ============================================================
print("📥 Loading feature timeseries...")
df = pd.read_csv(DATA_FILE, low_memory=False)

if "timestamp" not in df.columns:
    raise ValueError("❌ Missing 'timestamp' column in feature file!")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in df.columns:
    if c not in numeric_cols and c != "timestamp":
        try:
            df[c] = pd.to_numeric(df[c])
            numeric_cols.append(c)
        except:
            pass

features = df[numeric_cols].to_numpy(dtype=np.float32)
median_val = np.nanmedian(features)
features = np.nan_to_num(features, nan=median_val, posinf=median_val, neginf=median_val)

# ============================================================
# 2️⃣ SCALE
# ============================================================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features).astype(np.float32)

if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
    raise ValueError("❌ Scaled data contains NaN/Inf!")

print(f"📊 Features scaled: {len(numeric_cols)} | Range = [{scaled_data.min():.3f}, {scaled_data.max():.3f}]")

# ============================================================
# 3️⃣ CREATE SEQUENCES
# ============================================================
def create_sequences_fast(data, time_steps=TIME_STEP):
    n_samples = len(data) - time_steps
    stride = data.strides[0]
    return np.lib.stride_tricks.as_strided(
        data,
        shape=(n_samples, time_steps, data.shape[1]),
        strides=(stride, stride, data.strides[1])
    ).copy()

X = create_sequences_fast(scaled_data, TIME_STEP)
split = int((1 - TEST_SPLIT) * len(X))
X_train, X_test = X[:split], X[split:]

train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_test, X_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"✅ Data ready: train={X_train.shape}, test={X_test.shape}")

# ============================================================
# 4️⃣ MODEL
# ============================================================
model = Sequential([
    Conv1D(32, 3, activation="relu", padding="same", input_shape=(TIME_STEP, X.shape[2])),
    MaxPooling1D(2, padding="same"),
    LSTM(64, activation="tanh", return_sequences=False),
    RepeatVector(TIME_STEP // 2),
    LSTM(64, activation="tanh", return_sequences=True),
    UpSampling1D(2),
    Conv1D(X.shape[2], 3, activation="sigmoid", padding="same")
])

optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())

checkpoint_path = os.path.join(OUTPUT_DIR, "cnn_lstm_autoencoder_best.h5")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True)
]

model.summary()

# ============================================================
# 5️⃣ TRAIN
# ============================================================
print("🚀 Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks
)

model.save(os.path.join(OUTPUT_DIR, "cnn_lstm_autoencoder_final.h5"))
print("✅ Model saved successfully.")

# ============================================================
# 6️⃣ RECONSTRUCTION ERROR — with smoothing & quantile-based threshold
# ============================================================
print("🧮 Calculating reconstruction errors...")
X_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
errors = np.mean(np.mean(np.square(X_pred - X), axis=2), axis=1)

# 🔹 Apply rolling smoothing to reduce noise
smooth_window = 5
errors_smooth = pd.Series(errors).rolling(window=smooth_window, min_periods=1).mean().to_numpy()

# 🔹 Adaptive threshold (reduce false positives)
threshold = np.quantile(errors_smooth, 0.995)  # 99.5th percentile = only top 0.5% flagged
print(f"🚨 Adaptive threshold = {threshold:.6f} (99.5th percentile)")

timestamps = df["timestamp"][TIME_STEP:].reset_index(drop=True)
result_df = pd.DataFrame({
    "timestamp": timestamps,
    "reconstruction_error": errors_smooth,
    "is_anomaly": errors_smooth > threshold
})
result_df.to_csv(os.path.join(OUTPUT_DIR, "anomaly_result.csv"), index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
print("✅ Saved anomaly results.")

# ============================================================
# 7️⃣ METRICS & PLOT
# ============================================================
mse = float(np.mean(np.square(errors)))
rmse = math.sqrt(mse)
metrics = {"MSE": mse, "RMSE": rmse, "threshold": float(threshold)}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
print(f"📊 Metrics saved. RMSE={rmse:.6f}")

plt.figure(figsize=(15, 5))
plt.plot(result_df["timestamp"], result_df["reconstruction_error"], label="Reconstruction Error", lw=1)
plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold (adaptive)")
plt.scatter(result_df.loc[result_df["is_anomaly"], "timestamp"],
            result_df.loc[result_df["is_anomaly"], "reconstruction_error"], s=20, color="red", label="Anomaly")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_plot.png"))
plt.close()
print("✅ Plot saved.")

# ============================================================
# 8️⃣ MERGE WITH RAW LOG (timestamp-safe)
# ============================================================
print("🔍 Merging anomaly results with raw log file...")

try:
    df_raw = pd.read_csv(RAW_LOG_FILE, low_memory=False, on_bad_lines='skip')
    df_result = pd.read_csv(os.path.join(OUTPUT_DIR, "anomaly_result.csv"), low_memory=False)

    time_col_candidates = [c for c in df_raw.columns if any(x in c.lower() for x in ["time", "timestamp", "date"])]
    if not time_col_candidates:
        raise ValueError("❌ Không tìm thấy cột thời gian trong raw log.")
    raw_time_col = time_col_candidates[0]

    df_raw["time"] = pd.to_datetime(df_raw[raw_time_col], errors="coerce")
    df_result["timestamp"] = pd.to_datetime(df_result["timestamp"], errors="coerce")
    df_raw = df_raw.sort_values("time").dropna(subset=["time"]).reset_index(drop=True)
    df_result = df_result.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)

    merged = pd.merge_asof(
        df_raw,
        df_result.rename(columns={"timestamp": "time"}),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=2)
    )

    merged["reconstruction_error"] = merged["reconstruction_error"].fillna(0)
    merged["is_anomaly"] = merged["is_anomaly"].fillna(False)

    output_file = os.path.join(OUTPUT_DIR, "raw_with_anomaly_flag.csv")
    merged.to_csv(output_file, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
    print(f"✅ Merged file saved: {output_file}")
    print(f"📊 Total anomalies flagged: {int(merged['is_anomaly'].sum())}")

    # ============================================================
    # 9️⃣ DETAILED ANOMALY SUMMARY CSV
    # ============================================================
    print("🧾 Generating detailed anomaly summary CSV...")
    anomalies = merged[merged["is_anomaly"] == True].copy()

    if anomalies.empty:
        print("⚠️ No anomalies detected — skipping detailed summary.")
    else:
        text_cols = [c for c in merged.columns if c.lower() in ["message", "msg", "log", "log_message", "info", "description", "detail", "query"]]
        if not text_cols:
            text_cols = [c for c in merged.columns if merged[c].dtype == object]

        anomalies["combined_text"] = anomalies[text_cols].astype(str).agg(" ".join, axis=1).str.lower() if text_cols else ""

        def get_snippet(row, cols=text_cols, max_len=150):
            for c in cols:
                if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
                    return str(row[c]).strip()[:max_len]
            return row.get("combined_text", "")[:max_len]

        def classify_and_recommend(text):
            if "thread_protect" in text: return "Thread Protect Trigger", "DROP - Drop suspicious thread", "Critical"
            if "refused" in text: return "Connection Refused", "ALERT - Check ACLs / policies", "Warning"
            if "nxdomain" in text: return "Non-Existent Domain", "IGNORE - Likely client typo", "Info"
            if "malformed" in text: return "Malformed Query", "BLOCK - Drop malformed request", "Warning"
            if "servfail" in text: return "Server Failure", "RETRY - Check upstream servers", "Warning"
            if "timeout" in text: return "Query Timeout", "MONITOR - check backend latency", "Info"
            if "rate limit" in text: return "Rate Limit / Flood", "DROP / ALERT - Possible flood", "Critical"
            if "denied" in text: return "Access Denied", "ALERT - Check policy", "Warning"
            if "suspicious" in text: return "Suspicious Query", "ALERT - Manual review", "Critical"
            return "Unknown abnormal pattern", "INVESTIGATE manually", "Warning"

        classified = anomalies["combined_text"].apply(classify_and_recommend)
        anomalies["anomaly_type"] = classified.apply(lambda x: x[0])
        anomalies["recommended_action"] = classified.apply(lambda x: x[1])
        anomalies["severity"] = classified.apply(lambda x: x[2])
        anomalies["related_message"] = anomalies.apply(lambda r: get_snippet(r), axis=1)
        anomalies["timestamp_readable"] = anomalies["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        summary_cols = ["timestamp_readable", "anomaly_type", "recommended_action", "severity", "related_message", "reconstruction_error"]
        summary_df = anomalies[summary_cols].rename(columns={"timestamp_readable": "timestamp"})
        summary_path = os.path.join(OUTPUT_DIR, "anomaly_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S.%f")

        print(f"✅ Detailed anomaly summary saved: {summary_path}")
        print(summary_df.head(10))

except Exception as e:
    print(f"❌ Merge or summary error: {e}")

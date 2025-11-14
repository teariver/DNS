# -*- coding: utf-8 -*-
"""
DeepAnt (improved) — CNN Autoencoder for time-series anomaly detection
Sửa bởi: ChatGPT (phiên bản cải tiến)
Chạy trực tiếp: chỉnh các đường dẫn ở phần CONFIG nếu cần
"""
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv1D, Flatten, Dense, Dropout,
                                     Reshape, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ---------------- CONFIG ----------------
DATA_FILE = r"C:\Users\Surface\Documents\DNS\processed_logs\feature_timeseries.csv"
RAW_LOG_FILE = r"C:\Users\Surface\Documents\DNS\raw_log.csv"
RULE_RESULT_FILE = r"C:\Users\Surface\Documents\DNS\dl_output\dns_infoblox_rule_result.csv"  # optional
OUTPUT_DIR = r"C:\Users\Surface\Documents\DNS\deepant_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model / data params
TIME_STEP = 20
TEST_SPLIT = 0.2
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SEED = 42
THRESHOLD_METHOD = "percentile"   # "percentile" or "mean_std"
PERCENTILE = 99.7                 # nếu dùng percentile
THRESHOLD_STD_MULT = 3.0          # nếu dùng mean+mult*std
ERROR_AGG = "max"                 # "mean" hoặc "max" (per-sequence aggregation)
USE_ROBUST_SCALER = False         # True -> RobustScaler, False -> MinMaxScaler
FILTER_TRAIN_ANOMALIES = True     # nếu True: bỏ sequences có lỗi cao trước khi train (semi-supervised)
FILTER_PERCENTILE = 99.0          # nếu filter, loại top X% lỗi khỏi training

# reproducible
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

# ensure numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in df.columns:
    if c not in numeric_cols and c != "timestamp":
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if c not in numeric_cols and df[c].dtype != object:
                numeric_cols.append(c)
        except Exception:
            pass

numeric_cols = [c for c in numeric_cols if c != "timestamp"]
if len(numeric_cols) == 0:
    raise ValueError("❌ No numeric features detected to train on.")

features = df[numeric_cols].to_numpy(dtype=np.float32)
median_val = np.nanmedian(features)
features = np.nan_to_num(features, nan=median_val, posinf=median_val, neginf=median_val)

print(f"📊 Numeric features detected: {len(numeric_cols)} -> {numeric_cols[:10]}{'...' if len(numeric_cols)>10 else ''}")

# ============================================================
# 2️⃣ SCALE (configurable)
# ============================================================
if USE_ROBUST_SCALER:
    scaler = RobustScaler()
    print("🔧 Using RobustScaler")
else:
    scaler = MinMaxScaler()
    print("🔧 Using MinMaxScaler")

scaled_data = scaler.fit_transform(features).astype(np.float32)
if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
    raise ValueError("❌ Scaled data still contains NaN or Inf!")

print(f"📈 Scaled range = [{scaled_data.min():.6f}, {scaled_data.max():.6f}]")

# ============================================================
# 3️⃣ CREATE SEQUENCES (fast)
# ============================================================
def create_sequences_fast(data, time_steps=TIME_STEP):
    n_samples = len(data) - time_steps
    if n_samples <= 0:
        raise ValueError("Not enough rows to create sequences with the given TIME_STEP")
    stride0 = data.strides[0]
    stride1 = data.strides[1]
    return np.lib.stride_tricks.as_strided(
        data,
        shape=(n_samples, time_steps, data.shape[1]),
        strides=(stride0, stride0, stride1)
    ).copy()

X = create_sequences_fast(scaled_data, TIME_STEP)
timestamps = df['timestamp'].reset_index(drop=True)[TIME_STEP:].reset_index(drop=True)
if len(timestamps) != len(X):
    minlen = min(len(timestamps), len(X))
    timestamps = timestamps[:minlen]
    X = X[:minlen]

print(f"✅ Sequences created: {X.shape} (n_seq, time_step, n_features)")

# ============================================================
# 4️⃣ SIMPLE PRE-TRAIN RUN to compute errors and optionally filter bad sequences
# ============================================================
# We'll do a quick small model to estimate errors and optionally filter training set
def build_quick_model(n_features, time_step=TIME_STEP):
    m = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(time_step, n_features)),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        Dropout(0.2),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        Dense(time_step * n_features, activation='relu'),
        Reshape((time_step, n_features)),
        Conv1D(n_features, kernel_size=3, activation='sigmoid', padding='same')
    ])
    m.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return m

n_features = X.shape[2]

# Split for quick-run (small fraction) to estimate baseline errors
quick_split = int(0.02 * len(X)) if len(X) > 500 else int(0.1 * len(X))
if quick_split < 10:
    quick_split = min(10, len(X)//2)
qs_train = X[:quick_split]
qs_val = X[quick_split: quick_split + quick_split//2] if quick_split * 1.5 < len(X) else X[quick_split: quick_split + max(1, quick_split//4)]

print(f"ℹ️ Quick-run using {len(qs_train)} train seq and {len(qs_val)} val seq to estimate errors...")

quick_model = build_quick_model(n_features)
quick_hist = quick_model.fit(qs_train, qs_train,
                            validation_data=(qs_val, qs_val),
                            epochs=5,
                            batch_size=min(64, len(qs_train)),
                            verbose=0)

X_pred_quick = quick_model.predict(X, batch_size=256, verbose=0)
if ERROR_AGG == "mean":
    errors_quick = np.mean(np.mean(np.square(X_pred_quick - X), axis=2), axis=1)
else:  # max
    errors_quick = np.max(np.mean(np.square(X_pred_quick - X), axis=2), axis=1)

# decide filter
if FILTER_TRAIN_ANOMALIES:
    thr_filter = np.percentile(errors_quick, FILTER_PERCENTILE)
    keep_mask = errors_quick <= thr_filter
    X_filtered = X[keep_mask]
    print(f"🔍 Filtering training sequences: removed top {(100 - FILTER_PERCENTILE):.2f}% by quick-error. Kept {len(X_filtered)}/{len(X)} sequences.")
    if len(X_filtered) < 50:
        print("⚠️ After filtering, too few sequences left -> disabling filter")
        X_filtered = X.copy()
else:
    X_filtered = X.copy()

# ============================================================
# 5️⃣ TRAIN FINAL MODEL (deeper + callbacks)
# ============================================================
# split train/test
split = int((1 - TEST_SPLIT) * len(X_filtered))
X_train_all, X_test_all = X_filtered[:split], X_filtered[split:]

# if filtered removed some sequences from start, we still want evaluation on original tail
# so create a test set from the original X (unfiltered) last part
split_orig = int((1 - TEST_SPLIT) * len(X))
X_test_orig = X[split_orig:]

train_ds = tf.data.Dataset.from_tensor_slices((X_train_all, X_train_all)).shuffle(1024, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_test_all, X_test_all)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_final_model(n_features, time_step=TIME_STEP):
    model = Sequential([
        Conv1D(128, kernel_size=5, activation='relu', padding='same', input_shape=(time_step, n_features)),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        Dropout(0.3),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        Dense(time_step * n_features, activation='relu'),
        Reshape((time_step, n_features)),
        Conv1D(n_features, kernel_size=3, activation='sigmoid', padding='same')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

model = build_final_model(n_features)
model.summary()

checkpoint_path = os.path.join(OUTPUT_DIR, 'deepant_best.h5')
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
]

print("🚀 Training final DeepAnt model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks
)
final_model_path = os.path.join(OUTPUT_DIR, 'deepant_final.h5')
model.save(final_model_path)
print("✅ Model saved:", final_model_path)

# ============================================================
# 6️⃣ RECONSTRUCTION ERROR & THRESHOLD (final on full X)
# ============================================================
print("🧮 Calculating reconstruction errors on all sequences (original full X)...")
X_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)

if ERROR_AGG == "mean":
    errors = np.mean(np.mean(np.square(X_pred - X), axis=2), axis=1)
else:
    errors = np.max(np.mean(np.square(X_pred - X), axis=2), axis=1)

mean_err = float(np.mean(errors))
std_err = float(np.std(errors))

if THRESHOLD_METHOD == "percentile":
    threshold = float(np.percentile(errors, PERCENTILE))
    print(f"🚨 Using percentile {PERCENTILE} -> threshold={threshold:.6f}")
else:
    threshold = mean_err + THRESHOLD_STD_MULT * std_err
    print(f"🚨 Using mean+{THRESHOLD_STD_MULT}*std -> mean={mean_err:.6f}, std={std_err:.6f}, threshold={threshold:.6f}")

# create result df
timestamps_full = df['timestamp'].reset_index(drop=True)[TIME_STEP:].reset_index(drop=True)
minlen = min(len(timestamps_full), len(errors))
timestamps_full = timestamps_full[:minlen]
errors = errors[:minlen]

result_df = pd.DataFrame({
    'timestamp': timestamps_full,
    'reconstruction_error': errors,
    'is_anomaly': (errors > threshold)
})
out_result = os.path.join(OUTPUT_DIR, 'anomaly_result.csv')
result_df.to_csv(out_result, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
print("✅ Saved anomaly results:", out_result)

# ============================================================
# 7️⃣ METRICS & PLOT
# ============================================================
mse = float(np.mean(np.square(errors)))
rmse = math.sqrt(mse)
metrics = {'MSE': mse, 'RMSE': rmse, 'threshold': float(threshold), 'mean_err': mean_err, 'std_err': std_err,
           'n_sequences': len(errors), 'n_anomalies': int(result_df['is_anomaly'].sum())}
with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"📊 Metrics saved. RMSE={rmse:.6f} | anomalies={metrics['n_anomalies']} / {metrics['n_sequences']}")

# plot error histogram and time series
plt.figure(figsize=(10,4))
plt.hist(errors, bins=200)
plt.axvline(threshold, color='r', linestyle='--', label='threshold')
plt.title('Histogram of reconstruction errors')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'error_histogram.png'))
plt.close()

plt.figure(figsize=(15, 4))
plt.plot(timestamps_full, errors, lw=1, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
anoms = result_df[result_df['is_anomaly']]
plt.scatter(anoms['timestamp'], anoms['reconstruction_error'], s=18, color='red', label='Anomaly')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'anomaly_plot.png'))
plt.close()
print("✅ Plots saved: error_histogram.png, anomaly_plot.png")

# ============================================================
# 8️⃣ MERGE RAW + RULE + SUMMARY (unchanged logic, improved safety)
# ============================================================
print("🔍 Merging anomaly results with raw log file (and optional rule labels)...")
try:
    df_raw = pd.read_csv(RAW_LOG_FILE, low_memory=False, on_bad_lines='skip')
    df_result = result_df.copy()

    time_col_candidates = [c for c in df_raw.columns if 'time' in c.lower() or 'timestamp' in c.lower() or 'date' in c.lower()]
    if len(time_col_candidates) == 0:
        raise ValueError('❌ Không tìm thấy cột thời gian trong raw log.')
    raw_time_col = time_col_candidates[0]
    print(f"ℹ️ Using raw time column: {raw_time_col}")

    df_raw['time'] = pd.to_datetime(df_raw[raw_time_col], errors='coerce')
    df_result['timestamp'] = pd.to_datetime(df_result['timestamp'], errors='coerce')

    df_raw = df_raw.sort_values('time').dropna(subset=['time']).reset_index(drop=True)
    df_result = df_result.sort_values('timestamp').dropna(subset=['timestamp']).reset_index(drop=True)

    merged = pd.merge_asof(
        df_raw,
        df_result.rename(columns={'timestamp': 'time'}),
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=2)
    )

    merged['reconstruction_error'] = merged['reconstruction_error'].fillna(0)
    merged['is_anomaly'] = merged['is_anomaly'].fillna(False)

    if os.path.exists(RULE_RESULT_FILE):
        try:
            df_rule = pd.read_csv(RULE_RESULT_FILE, low_memory=False)
            tcols = [c for c in df_rule.columns if 'time' in c.lower() or 'timestamp' in c.lower() or 'date' in c.lower()]
            if len(tcols) == 0:
                print("⚠️ Rule file found but no time-like column to merge on. Skipping rule_label merge.")
            else:
                rule_time_col = tcols[0]
                df_rule[rule_time_col] = pd.to_datetime(df_rule[rule_time_col], errors='coerce')
                df_rule = df_rule.sort_values(rule_time_col).dropna(subset=[rule_time_col]).reset_index(drop=True)
                if 'rule_label' not in df_rule.columns:
                    possible = [c for c in df_rule.columns if 'label' in c.lower() or 'rule' in c.lower()]
                    if possible:
                        df_rule = df_rule.rename(columns={possible[0]: 'rule_label'})
                    else:
                        print("⚠️ No rule_label column found in rule file. Skipping rule_label merge.")
                if 'rule_label' in df_rule.columns:
                    df_rule_small = df_rule[[rule_time_col, 'rule_label']].rename(columns={rule_time_col: 'time'})
                    merged = pd.merge_asof(
                        merged.sort_values('time'),
                        df_rule_small.sort_values('time'),
                        on='time',
                        direction='nearest',
                        tolerance=pd.Timedelta(seconds=2)
                    )
                    if 'rule_label' not in merged.columns:
                        merged['rule_label'] = pd.NA
                    print("✅ Merged rule_label from rule file.")
        except Exception as e_rule:
            print(f"⚠️ Error merging rule file: {e_rule}. Continuing without rule_label.")

    merged['time'] = pd.to_datetime(merged['time'], errors='coerce')
    out_file = os.path.join(OUTPUT_DIR, 'raw_with_anomaly_flag.csv')
    merged.to_csv(out_file, index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
    print(f'✅ Merged file saved: {out_file}')

    # Generate summary
    anomalies = merged[merged["is_anomaly"] == True].copy()
    if anomalies.empty:
        print("⚠️ No anomalies detected — skipping detailed summary.")
    else:
        text_cols = [c for c in merged.columns if c.lower() in ["message", "msg", "log", "log_message", "info", "description", "detail", "query"]]
        if not text_cols:
            for cand in ["message", "log", "msg", "info", "description", "detail", "query"]:
                if cand in merged.columns:
                    text_cols.append(cand)

        if text_cols:
            anomalies["combined_text"] = anomalies[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
        else:
            str_cols = [c for c in anomalies.columns if anomalies[c].dtype == object]
            if str_cols:
                anomalies["combined_text"] = anomalies[str_cols].astype(str).agg(" ".join, axis=1).str.lower()
            else:
                anomalies["combined_text"] = ""

        def get_snippet(row, cols=text_cols, max_len=150):
            for c in cols:
                if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
                    return str(row[c]).strip()[:max_len]
            txt = row.get("combined_text", "")
            return txt[:max_len]

        def classify_and_recommend(text):
            if not isinstance(text, str):
                text = str(text)
            if "thread_protect" in text:
                return "Thread Protect Trigger", "DROP - Drop suspicious thread", "Critical"
            if "refused" in text or "query refused" in text:
                return "Connection Refused", "ALERT - Check ACLs / policies", "Warning"
            if "nxdomain" in text or "non-existent domain" in text:
                return "Non-Existent Domain", "IGNORE - Likely client typo", "Info"
            if "malformed" in text or "formerr" in text:
                return "Malformed Query", "BLOCK - Drop malformed request", "Warning"
            if "servfail" in text:
                return "Server Failure (SERVFAIL)", "RETRY - Check upstream DNS servers", "Warning"
            if "timeout" in text:
                return "Query Timeout", "MONITOR / RETRY - possible backend latency", "Info"
            if "rate limit" in text:
                return "Rate Limit / Flood", "DROP / ALERT - Possible flood, consider throttling", "Critical"
            if "denied" in text:
                return "Access Denied", "ALERT - Check policy", "Warning"
            if "suspicious" in text:
                return "Suspicious Query", "ALERT - Manual review", "Critical"
            return "Unknown abnormal pattern", "ALERT - Investigate manually", "Warning"

        classified = anomalies["combined_text"].apply(classify_and_recommend)
        anomalies["anomaly_type"] = classified.apply(lambda x: x[0])
        anomalies["recommended_action"] = classified.apply(lambda x: x[1])
        anomalies["severity"] = classified.apply(lambda x: x[2])
        anomalies["related_message"] = anomalies.apply(lambda r: get_snippet(r, text_cols if text_cols else []), axis=1)
        anomalies["timestamp_readable"] = anomalies["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        summary_cols = ["timestamp_readable", "anomaly_type", "recommended_action", "severity", "related_message", "reconstruction_error"]
        summary_df = anomalies[summary_cols].rename(columns={"timestamp_readable": "timestamp"})
        summary_path = os.path.join(OUTPUT_DIR, "anomaly_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S.%f")
        print(f"✅ Detailed anomaly summary saved: {summary_path}")
        print(summary_df.head(10))

except Exception as e:
    print(f'❌ Merge or summary error: {e}')

# ============================================================
# 9️⃣ QUICK SUMMARY
# ============================================================
try:
    merged_sample = pd.read_csv(os.path.join(OUTPUT_DIR, 'raw_with_anomaly_flag.csv'), low_memory=False)
    total = len(merged_sample)
    anomalies = merged_sample['is_anomaly'].astype(bool).sum()
    print(f"ℹ️ Summary: total rows = {total}, anomalies = {anomalies}")
    if 'rule_label' in merged_sample.columns:
        print(f"ℹ️ Rows with rule_label = {merged_sample['rule_label'].notna().sum()}")
except Exception:
    pass

print("🎉 All done. Outputs in:", OUTPUT_DIR)

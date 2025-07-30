import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from pathlib import Path
import os

# --- LOAD DATA FROM JSON -------------------------------------------------
DATA_PATH = "data/runtimes.json"
with open(DATA_PATH, "r") as f:
    run_data = json.load(f)

df = pd.DataFrame(run_data)
df["date"] = pd.to_datetime(df["date"])

def time_str_to_min(tstr):
    m, s = map(int, tstr.split(":"))
    return m + s/60

df["time"] = df["time"].apply(time_str_to_min)
df["pace"] = df["time"] / df["distance"]
df["est_1.5_time"] = df["pace"] * 1.5

# --- PARAMETERS ----------------------------------------------------------
HALF_LIFE = 60  # days
SIMULATIONS = 10000
TARGET_DATE = pd.to_datetime("2025-08-15")
SIGMA_D = 0.4  # Distance Gaussian width (in miles)

# --- FEATURE ENGINEERING -------------------------------------------------
df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
latest_day = df["days_since_start"].max()

# Recency weight (exponential)
df["w_recency"] = np.exp(-(latest_day - df["days_since_start"]) / HALF_LIFE)

# Distance similarity weight (Gaussian bell curve centered at 1.5 miles)
df["w_distance"] = np.exp(-((df["distance"] - 1.5) ** 2) / (2 * SIGMA_D ** 2))

# Combined weight (product)
df["weight_total"] = df["w_recency"] * df["w_distance"]

# --- OUTPUT WEIGHTS TO JSON ----------------------------------------------
os.makedirs("data", exist_ok=True)
weights_out = []
for i, row in df.iterrows():
    weights_out.append({
        "date": str(row["date"].date()),
        "distance": float(row["distance"]),
        "time": f"{int(row['time'])}:{int(round((row['time']%1)*60)):02d}",
        "w_recency": float(row["w_recency"]),
        "w_distance": float(row["w_distance"]),
        "weight_total": float(row["weight_total"])
    })
# --- OUTPUT WEIGHTS TO JSON (with date) -------------------------------
today_str = datetime.now().strftime("%Y-%m-%d")
weights_dir = Path("data/weights")
weights_dir.mkdir(parents=True, exist_ok=True)
weights_file = weights_dir / f"{today_str}_weights.json"

with open(weights_file, "w") as f:
    json.dump(weights_out, f, indent=2)
# --- WEIGHTED LINEAR REGRESSION ------------------------------------------
X = df["days_since_start"].values.reshape(-1,1)
y = df["est_1.5_time"].values
w = df["weight_total"].values
reg = LinearRegression()
reg.fit(X, y, sample_weight=w)
b0, b1 = reg.intercept_, reg.coef_[0]

y_pred = reg.predict(X)
residuals = y - y_pred
sigma_hat = np.sqrt(np.average(residuals**2, weights=w))

# --- MONTE CARLO FORECASTING ---------------------------------------------
x_star = (TARGET_DATE - df["date"].min()).days
future_days = np.arange(0, x_star+1)
future_dates = [df["date"].min() + timedelta(days=int(d)) for d in future_days]
trend = b0 + b1 * future_days

np.random.seed(1)
samples = np.array([
    trend + np.random.normal(0, sigma_hat, trend.shape)
    for _ in range(1000)
])
mean_pred = samples.mean(axis=0)
lower_pred = np.percentile(samples, 5, axis=0)
upper_pred = np.percentile(samples, 95, axis=0)

target_trend = b0 + b1 * x_star
target_samples = target_trend + np.random.normal(0, sigma_hat, SIMULATIONS)
q5, q50, q95 = np.percentile(target_samples, [5, 50, 95])
prob_sub9 = np.mean(target_samples < 9.0)

# --- VISUALIZATION -------------------------------------------------------
plt.figure(figsize=(15,9))
plt.gca().set_facecolor('#fafafa')
plt.grid(color='#e1e1e1', linewidth=1.3, alpha=0.4, zorder=0)

plt.scatter(df["date"], df["est_1.5_time"], s=70, color="#324563", alpha=0.70,
            label="Normalized 1.5-Mile Results", edgecolor='white', linewidth=1.2, zorder=5)
plt.plot(future_dates, mean_pred, color="#0d0d0d", linewidth=3, label="Weighted Mean Forecast", zorder=4)
plt.fill_between(future_dates, lower_pred, upper_pred, color='#a8b9d1', alpha=0.32,
                 label="90% Confidence Band", zorder=3)
plt.axhline(9, color="#383838", linestyle="--", linewidth=2, label="9:00 Target", zorder=2)
plt.axvline(TARGET_DATE, color="#b2182b", linestyle=":", linewidth=2, zorder=2)
plt.scatter([TARGET_DATE], [q50], color="#b2182b", s=140, zorder=8, label="August 15, 2025 Median")
plt.text(TARGET_DATE, q50 + 0.1, f"Median: {q50:.2f} min", fontsize=14, color="#b2182b",
         ha="center", va='bottom', fontweight='bold', family='sans-serif')
plt.text(TARGET_DATE, 8.55, f"P(<9:00) = {prob_sub9:.1%}", fontsize=15, color="#0d0d0d",
         ha="center", va='bottom', fontweight='medium',
         bbox=dict(facecolor='#fafafa', edgecolor='#e1e1e1', boxstyle='round,pad=0.22', alpha=0.85))
plt.xlabel("Date", fontsize=17, weight='light', family='sans-serif', labelpad=12)
plt.ylabel("1.5-Mile Time (min)", fontsize=17, weight='light', family='sans-serif', labelpad=12)
plt.title("Probabilistic Forecast of Sub-9:00 1.5-Mile Run by August 15, 2025",
          fontsize=21, weight='medium', family='sans-serif', pad=17)
plt.legend(frameon=False, fontsize=13, loc='upper right')
plt.xticks(fontsize=13, family='sans-serif')
plt.yticks(fontsize=13, family='sans-serif')
plt.tight_layout()
plt.show()

# --- PRINT SUMMARY TABLE -------------------------------------------------
print("Forecast for August 15, 2025:")
print(f"  5th percentile (optimistic): {q5:.2f} min")
print(f"  Median forecast:             {q50:.2f} min")
print(f"  95th percentile (pessimistic): {q95:.2f} min")
print(f"  Probability of sub-9:00:     {prob_sub9:.1%}")
print("\nAll weights (recency, distance, combined) saved to data/weights.json")

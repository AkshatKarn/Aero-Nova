# diag_leak.py  (safe, updated)
import pandas as pd, numpy as np

# load processed files (same as train script)
aq = pd.read_csv("data/processed/aqi_master.csv")
we = pd.read_csv("data/processed/weather_master.csv")
tr = pd.read_csv("data/processed/traffic_master.csv")

def make_ts(df):
    # find a datetime-like column and create 'timestamp' if possible
    for col in df.columns:
        if any(k in col.lower() for k in ["timestamp","time","date"]):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df[col], errors="coerce")
            break
    try:
        df = df.set_index("timestamp").resample("1h").mean().reset_index()
    except Exception:
        pass
    return df

aq = make_ts(aq); we = make_ts(we); tr = make_ts(tr)

# merge as training does (1h tolerance)
merged = pd.merge_asof(aq.sort_values("timestamp"), we.sort_values("timestamp"),
                       on="timestamp", direction="nearest", tolerance=pd.Timedelta("1h"))
merged = pd.merge_asof(merged.sort_values("timestamp"), tr.sort_values("timestamp"),
                       on="timestamp", direction="nearest", tolerance=pd.Timedelta("1h"))

# choose target
if "AQI_real" in merged.columns and merged["AQI_real"].notna().sum() > 0:
    target = "AQI_real"
else:
    candidates = [c for c in merged.columns if "aqi" in c.lower()]
    target = candidates[0] if candidates else None

print("Merged shape:", merged.shape)
print("Target column chosen:", target)
if target is None:
    raise SystemExit("No AQI-like column found in merged data.")

y = pd.to_numeric(merged[target], errors="coerce")
X = merged.drop(columns=[target]).copy()

# coerce possible numeric columns (same strategy as train)
for c in X.columns:
    if X[c].dtype == object:
        X[c] = pd.to_numeric(X[c].astype(str).str.replace(",",""), errors="coerce")

print("\n-- y stats --")
print(y.describe())
print("y nunique:", y.nunique())
print("y top value counts (top 10):\n", y.value_counts().head(10).to_string())

# constant columns
const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
print("\nConstant columns (nunique<=1):", const_cols)

# helper to safely convert column to numeric vector for comparison
def col_to_numeric_for_compare(s):
    # datetimes -> epoch seconds
    if pd.api.types.is_datetime64_any_dtype(s):
        # convert datetime to seconds
        return s.astype('int64').astype(float) / 1e9
    # if object, try numeric coercion
    if s.dtype == object:
        return pd.to_numeric(s.astype(str).str.replace(",",""), errors='coerce')
    # if numeric already, return as float
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    # otherwise fallback: try numeric coercion
    return pd.to_numeric(s, errors='coerce')

# check identical / almost-identical
identical = []
almost_identical = []
for c in X.columns:
    sx = col_to_numeric_for_compare(X[c])
    mask = y.notna() & sx.notna()
    if mask.sum() < 5:
        continue
    ax = sx[mask].values
    ay = y[mask].astype(float).values
    try:
        if np.array_equal(ax, ay):
            identical.append(c)
        elif np.allclose(ax, ay, rtol=1e-6, atol=1e-6):
            almost_identical.append(c)
    except Exception:
        pass

print("\nColumns exactly identical to target:", identical)
print("Columns nearly identical to target (allclose):", almost_identical)

# correlation check (only numeric columns)
# convert any datetime columns in X to numeric for corr check too
X_corr = X.copy()
for c in X_corr.columns:
    if pd.api.types.is_datetime64_any_dtype(X_corr[c]):
        X_corr[c] = X_corr[c].astype('int64').astype(float) / 1e9
    if X_corr[c].dtype == object:
        X_corr[c] = pd.to_numeric(X_corr[c].astype(str).str.replace(",",""), errors='coerce')

num = X_corr.select_dtypes(include=[np.number])
corrs = {}
for c in num.columns:
    mask = y.notna() & num[c].notna()
    if mask.sum() < 5:
        continue
    try:
        corr = np.corrcoef(num.loc[mask, c].astype(float), y.loc[mask].astype(float))[0,1]
        corrs[c] = corr
    except Exception:
        pass

if corrs:
    corr_series = pd.Series(corrs).abs().sort_values(ascending=False)
    print("\nTop 10 absolute correlations with target:\n", corr_series.head(10).to_string())
    print("\nColumns with abs(corr)>0.999:", corr_series[corr_series>0.999].index.tolist())
else:
    print("\nNo numeric columns to compute correlation.")

# show small sample
print("\nX sample (first 6 cols and 5 rows):")
print(X.iloc[:5,:6].to_string(index=False))
print("\ny sample (first 10):\n", y.head(10).to_string())

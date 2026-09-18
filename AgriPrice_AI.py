"""
AgriPrice AI – Indian Mandi Price Analytics & Forecasting System
================================================================
BharatCare + IBM Internship Project
Single-file Streamlit application (backend + frontend)
"""

# ── SECTION 0: IMPORTS & CONFIGURATION ──────────────────────────────────────
import os
import warnings
import sqlite3
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# Constants
DATASET_PATH = "Agriculture_price_dataset.csv"
RANDOM_STATE = 42
MIN_SERIES_LENGTH = 90          # minimum days needed to train a forecast model
FORECAST_COMMODITY = "Potato"
FORECAST_MARKET    = "Kalipur"
FORECAST_STATE     = "West Bengal"

# State name canonical mapping (raw → clean)
STATE_MAP = {
    " Punjab":             "Punjab",
    "Tamilnadu":           "Tamil Nadu",
    "Chattisgarh":         "Chhattisgarh",
    "Jammu & Kashmir":     "Jammu and Kashmir",
    "Orissa":              "Odisha",
    "Uttrakhand":          "Uttarakhand",
    "Gao":                 "Unknown",
}

st.set_page_config(
    page_title="AgriPrice AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card{background:#f7f8fa;border:1px solid #e5e7eb;border-radius:8px;
          padding:16px 20px;text-align:center;}
.kpi-title{font-size:12px;color:#57606a;font-weight:600;text-transform:uppercase;
           letter-spacing:.05em;margin-bottom:4px;}
.kpi-value{font-size:28px;font-weight:700;color:#1f2328;}
.kpi-sub{font-size:12px;color:#57606a;margin-top:2px;}
.insight-data{background:#eef6ff;border-left:4px solid #3b82d4;
              padding:10px 14px;border-radius:4px;margin:6px 0;font-size:14px;}
.insight-ai{background:#f3f0ff;border-left:4px solid #7c5cd8;
            padding:10px 14px;border-radius:4px;margin:6px 0;font-size:14px;}
.flag-warn{background:#fff8e1;border-left:4px solid #f59e0b;
           padding:8px 12px;border-radius:4px;font-size:13px;}
</style>
""", unsafe_allow_html=True)


# ── SECTION 1: DATA LOADING ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    """Load raw CSV. Returns raw DataFrame."""
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found: `{DATASET_PATH}`. Place the CSV in the same folder.")
        st.stop()
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    return df


# ── SECTION 2: DATA CLEANING ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Cleaning data…")
def clean_data(df: pd.DataFrame):
    """
    Full cleaning pipeline.
    Returns: (df_full_flagged, df_clean)
      df_full_flagged  – all rows with Data_Quality_Flag column
      df_clean         – rows suitable for analytics & ML (flags removed)
    """
    d = df.copy()

    # --- 2a: Date conversion ---
    d["Price Date"] = pd.to_datetime(d["Price Date"], dayfirst=False, errors="coerce")
    d = d.dropna(subset=["Price Date"])

    # --- 2b: Strip whitespace on all string columns ---
    str_cols = d.select_dtypes(include="object").columns
    for c in str_cols:
        d[c] = d[c].str.strip()

    # --- 2c: State name normalization ---
    d["STATE"] = d["STATE"].replace(STATE_MAP)

    # --- 2d: Derived time columns ---
    d["Year"]        = d["Price Date"].dt.year
    d["Month"]       = d["Price Date"].dt.month
    d["Month_Name"]  = d["Price Date"].dt.strftime("%b")
    d["Quarter"]     = d["Price Date"].dt.quarter
    d["Day"]         = d["Price Date"].dt.day
    d["Day_of_Week"] = d["Price Date"].dt.dayofweek
    d["Day_Name"]    = d["Price Date"].dt.strftime("%a")
    d["Week_of_Year"]= d["Price Date"].dt.isocalendar().week.astype(int)

    # --- 2e: Derived price metrics ---
    d["Price_Range"]        = d["Max_Price"] - d["Min_Price"]
    d["Price_Spread_Pct"]   = np.where(
        d["Modal_Price"] > 0,
        (d["Price_Range"] / d["Modal_Price"]) * 100, np.nan
    )
    d["Modal_vs_Min"] = d["Modal_Price"] - d["Min_Price"]
    d["Modal_vs_Max"] = d["Max_Price"]   - d["Modal_Price"]

    # --- 2f: Quality flags ---
    flags = pd.Series(["OK"] * len(d), index=d.index)
    flags = flags.where(d["Modal_Price"] != 0,               "ZERO_MODAL")
    flags = flags.where(d["Min_Price"] <= d["Max_Price"],     "PRICE_LOGIC_ERROR")
    flags = flags.where(d["Modal_Price"] >= d["Min_Price"],   "MODAL_BELOW_MIN")
    flags = flags.where(d["Modal_Price"] <= d["Max_Price"],   "MODAL_ABOVE_MAX")

    # IQR × 5 outlier fence per commodity
    for comm in d["Commodity"].unique():
        mask = d["Commodity"] == comm
        sub  = d.loc[mask, "Modal_Price"]
        q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
        iqr    = q3 - q1
        upper  = q3 + 5 * iqr
        lower  = max(0, q1 - 5 * iqr)
        outlier_mask = mask & ((d["Modal_Price"] > upper) | (d["Modal_Price"] < lower))
        flags = flags.where(~outlier_mask, "EXTREME_OUTLIER")

    d["Data_Quality_Flag"] = flags

    # df_full: all rows with flags (for analytics & transparency)
    df_full = d.copy()

    # df_clean: only OK rows (for ML and reliable analytics)
    df_clean = d[d["Data_Quality_Flag"] == "OK"].copy().reset_index(drop=True)

    return df_full, df_clean


def get_data_quality_report(df_full: pd.DataFrame) -> pd.DataFrame:
    """Returns a summary DataFrame of flag counts."""
    counts  = df_full["Data_Quality_Flag"].value_counts().reset_index()
    counts.columns = ["Flag", "Count"]
    counts["Percentage"] = (counts["Count"] / len(df_full) * 100).round(2)
    return counts


# ── SECTION 3: ANALYTICS FUNCTIONS ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def commodity_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-commodity price statistics."""
    grp = df.groupby("Commodity")["Modal_Price"].agg(
        Records="count",
        Mean_Price="mean",
        Median_Price="median",
        Std_Price="std",
        Min_Price="min",
        Max_Price="max",
    ).reset_index()
    grp["CV_Pct"] = (grp["Std_Price"] / grp["Mean_Price"] * 100).round(2)
    grp = grp.round(2)
    return grp.sort_values("Mean_Price", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def state_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-state average modal price and record count."""
    grp = df.groupby("STATE").agg(
        Records       = ("Modal_Price", "count"),
        Avg_Modal     = ("Modal_Price", "mean"),
        Std_Modal     = ("Modal_Price", "std"),
        Min_Modal     = ("Modal_Price", "min"),
        Max_Modal     = ("Modal_Price", "max"),
    ).reset_index()
    grp["CV_Pct"] = (grp["Std_Modal"] / grp["Avg_Modal"] * 100).round(2)
    return grp.round(2).sort_values("Avg_Modal", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def market_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-market summary statistics."""
    grp = df.groupby(["Market Name", "STATE"]).agg(
        Records      = ("Modal_Price", "count"),
        Avg_Modal    = ("Modal_Price", "mean"),
        Std_Modal    = ("Modal_Price", "std"),
        Avg_Range    = ("Price_Range", "mean"),
    ).reset_index()
    grp["CV_Pct"] = (grp["Std_Modal"] / grp["Avg_Modal"] * 100).round(2)
    return grp.round(2).sort_values("Avg_Modal", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly average modal price per commodity."""
    df2 = df.copy()
    df2["YearMonth"] = df2["Price Date"].dt.to_period("M").astype(str)
    grp = df2.groupby(["YearMonth", "Commodity"])["Modal_Price"].mean().reset_index()
    grp.columns = ["YearMonth", "Commodity", "Avg_Modal"]
    grp = grp.sort_values("YearMonth").reset_index(drop=True)
    return grp.round(2)


@st.cache_data(show_spinner=False)
def seasonal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly seasonal index per commodity (mean / annual_mean)."""
    annual = df.groupby("Commodity")["Modal_Price"].mean().rename("Annual_Mean")
    monthly = df.groupby(["Commodity", "Month"])["Modal_Price"].mean().rename("Monthly_Mean").reset_index()
    monthly = monthly.merge(annual, on="Commodity")
    monthly["Seasonal_Index"] = (monthly["Monthly_Mean"] / monthly["Annual_Mean"] * 100).round(2)
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["Month_Name"] = monthly["Month"].map(month_names)
    return monthly.sort_values(["Commodity","Month"]).reset_index(drop=True)


# ── SECTION 4: SQL ANALYTICS ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_sql_analytics(df: pd.DataFrame) -> dict:
    """
    Load df into in-memory SQLite, run 10 analytical queries.
    Returns dict of {query_name: result_dataframe}.
    """
    conn = sqlite3.connect(":memory:")
    export = df[["STATE","District Name","Market Name","Commodity","Variety",
                 "Grade","Min_Price","Max_Price","Modal_Price","Price Date",
                 "Year","Month","Quarter","Price_Range","Data_Quality_Flag"]].copy()
    export["Price Date"] = export["Price Date"].astype(str)
    export.to_sql("mandi_prices", conn, index=False, if_exists="replace")

    queries = {
        "Q1_Avg_By_Commodity": """
            SELECT Commodity,
                   ROUND(AVG(Modal_Price),2) AS Avg_Modal,
                   ROUND(MIN(Modal_Price),2) AS Min_Modal,
                   ROUND(MAX(Modal_Price),2) AS Max_Modal,
                   COUNT(*)                 AS Records
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
            GROUP BY Commodity
            ORDER BY Avg_Modal DESC
        """,
        "Q2_Avg_By_State": """
            SELECT STATE,
                   ROUND(AVG(Modal_Price),2) AS Avg_Modal,
                   COUNT(*)                  AS Records
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
            GROUP BY STATE
            ORDER BY Avg_Modal DESC
        """,
        "Q3_Top10_Markets_By_Price": """
            SELECT "Market Name", STATE,
                   ROUND(AVG(Modal_Price),2) AS Avg_Modal,
                   COUNT(*)                  AS Records
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
            GROUP BY "Market Name", STATE
            HAVING Records >= 30
            ORDER BY Avg_Modal DESC
            LIMIT 10
        """,
        "Q4_Most_Volatile_Commodities": """
            SELECT Commodity,
                   ROUND(AVG(Modal_Price),2)  AS Avg_Modal,
                   ROUND(
                     SQRT(AVG(Modal_Price*Modal_Price) - AVG(Modal_Price)*AVG(Modal_Price)),
                   2) AS Std_Modal
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
            GROUP BY Commodity
            ORDER BY Std_Modal DESC
        """,
        "Q5_Monthly_Avg_By_Commodity": """
            SELECT Commodity, Year, Month,
                   ROUND(AVG(Modal_Price),2) AS Avg_Modal,
                   COUNT(*)                  AS Records
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
            GROUP BY Commodity, Year, Month
            ORDER BY Commodity, Year, Month
        """,
        "Q6_State_Commodity_Matrix": """
            SELECT STATE, Commodity,
                   ROUND(AVG(Modal_Price),2) AS Avg_Modal
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
            GROUP BY STATE, Commodity
            ORDER BY STATE, Commodity
        """,
        "Q7_Highest_Price_Spread_Markets": """
            SELECT "Market Name", STATE,
                   ROUND(AVG(Price_Range),2) AS Avg_Spread,
                   COUNT(*)                  AS Records
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
            GROUP BY "Market Name", STATE
            HAVING Records >= 30
            ORDER BY Avg_Spread DESC
            LIMIT 10
        """,
        "Q8_Price_Extremes": """
            SELECT Commodity, "Market Name", STATE, Modal_Price, "Price Date"
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
              AND Modal_Price = (SELECT MAX(Modal_Price) FROM mandi_prices WHERE Data_Quality_Flag='OK')
            UNION ALL
            SELECT Commodity, "Market Name", STATE, Modal_Price, "Price Date"
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
              AND Modal_Price = (SELECT MIN(Modal_Price) FROM mandi_prices
                                 WHERE Data_Quality_Flag='OK' AND Modal_Price > 0)
        """,
        "Q9_Market_Activity": """
            SELECT "Market Name", STATE,
                   COUNT(*) AS Total_Records,
                   COUNT(DISTINCT "Price Date") AS Trading_Days
            FROM mandi_prices
            GROUP BY "Market Name", STATE
            ORDER BY Total_Records DESC
            LIMIT 15
        """,
        "Q10_YoY_Price_Change": """
            SELECT Commodity,
                   ROUND(AVG(CASE WHEN Year=2023 THEN Modal_Price END),2) AS Avg_2023,
                   ROUND(AVG(CASE WHEN Year=2024 THEN Modal_Price END),2) AS Avg_2024,
                   ROUND(AVG(CASE WHEN Year=2025 THEN Modal_Price END),2) AS Avg_2025
            FROM mandi_prices
            WHERE Data_Quality_Flag = 'OK'
              AND Commodity IN ('Potato','Onion')
            GROUP BY Commodity
        """,
    }

    results = {}
    for name, sql in queries.items():
        try:
            results[name] = pd.read_sql_query(sql, conn)
        except Exception as e:
            results[name] = pd.DataFrame({"Error": [str(e)]})
    conn.close()
    return results


# ── SECTION 5: FORECASTING PIPELINE ─────────────────────────────────────────

def prepare_forecast_series(df: pd.DataFrame, commodity: str, market: str) -> pd.DataFrame:
    """
    Filter df for commodity+market, aggregate per date (mean across varieties),
    sort chronologically, forward-fill gaps ≤ 3 days, return clean daily series.
    """
    sub = df[(df["Commodity"] == commodity) & (df["Market Name"] == market)].copy()
    if len(sub) < MIN_SERIES_LENGTH:
        return pd.DataFrame()

    # Aggregate per date: mean of Modal_Price across varieties/grades
    ts = (
        sub.groupby("Price Date")["Modal_Price"]
        .mean()
        .reset_index()
        .sort_values("Price Date")
        .rename(columns={"Price Date": "ds", "Modal_Price": "y"})
    )
    ts = ts.drop_duplicates(subset=["ds"]).reset_index(drop=True)

    # Reindex to daily frequency — forward-fill gaps ≤ 3 days
    full_range = pd.date_range(ts["ds"].min(), ts["ds"].max(), freq="D")
    ts = ts.set_index("ds").reindex(full_range).rename_axis("ds").reset_index()
    gap_mask = ts["y"].isna()
    gap_lengths = gap_mask.astype(int)
    # Only forward-fill where gap ≤ 3 consecutive days
    ts["y"] = ts["y"].ffill(limit=3)
    # Drop rows still NaN (larger gaps)
    ts = ts.dropna(subset=["y"]).reset_index(drop=True)
    return ts


def engineer_features(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features. All rolling operations use shift(1) FIRST
    to prevent look-ahead leakage.
    """
    t = ts.copy()
    p = t["y"]

    # Lags
    t["lag_1"]  = p.shift(1)
    t["lag_7"]  = p.shift(7)
    t["lag_14"] = p.shift(14)
    t["lag_30"] = p.shift(30)

    # Rolling stats (on shifted series — no leakage)
    t["roll_mean_7"]  = p.shift(1).rolling(7).mean()
    t["roll_mean_14"] = p.shift(1).rolling(14).mean()
    t["roll_mean_30"] = p.shift(1).rolling(30).mean()
    t["roll_std_7"]   = p.shift(1).rolling(7).std()
    t["roll_std_30"]  = p.shift(1).rolling(30).std()

    # Calendar features
    t["month"]       = t["ds"].dt.month
    t["quarter"]     = t["ds"].dt.quarter
    t["day_of_week"] = t["ds"].dt.dayofweek
    t["week_of_year"]= t["ds"].dt.isocalendar().week.astype(int)
    t["year"]        = t["ds"].dt.year

    # Drop rows with NaN from lag/rolling windows
    t = t.dropna().reset_index(drop=True)
    return t


FEATURE_COLS = [
    "lag_1","lag_7","lag_14","lag_30",
    "roll_mean_7","roll_mean_14","roll_mean_30",
    "roll_std_7","roll_std_30",
    "month","quarter","day_of_week","week_of_year","year"
]


def chronological_split(ts_feat: pd.DataFrame):
    """
    60% train / 17% validation / 23% test — strictly chronological.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test,
             dates_train, dates_val, dates_test)
    """
    n = len(ts_feat)
    i_val  = int(n * 0.60)
    i_test = int(n * 0.77)

    train = ts_feat.iloc[:i_val]
    val   = ts_feat.iloc[i_val:i_test]
    test  = ts_feat.iloc[i_test:]

    X_tr, y_tr = train[FEATURE_COLS].values, train["y"].values
    X_va, y_va = val[FEATURE_COLS].values,   val["y"].values
    X_te, y_te = test[FEATURE_COLS].values,  test["y"].values

    return (X_tr, y_tr, X_va, y_va, X_te, y_te,
            train["ds"], val["ds"], test["ds"])


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Compute MAE, RMSE, R², MAPE (only when y_true > 0)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    valid = y_true > 0
    mape = (np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])).mean() * 100 \
           if valid.sum() > 0 else np.nan
    return {"Model": model_name, "MAE": round(mae,2), "RMSE": round(rmse,2),
            "R2": round(r2,4), "MAPE": round(mape,2) if not np.isnan(mape) else None}


def rolling_mean_baseline(y_train: np.ndarray, X_test_lag1: np.ndarray,
                           window: int = 7) -> np.ndarray:
    """Predict using last-known 7-day rolling mean (uses lag_1 column)."""
    # lag_1 is the most recent known price — use its rolling mean as baseline
    return np.full(len(X_test_lag1), np.mean(y_train[-window:]))


@st.cache_resource(show_spinner="Training prediction models…")
def get_or_train_model(commodity: str, market: str, _df_clean: pd.DataFrame):
    """
    Trains all models for the given commodity+market combination.
    @st.cache_resource: persists model objects across rerenders.
    _df_clean prefixed with _ to avoid hashing the large DataFrame.
    Returns a results dict or None if insufficient data.
    """
    ts = prepare_forecast_series(_df_clean, commodity, market)
    if ts.empty or len(ts) < MIN_SERIES_LENGTH:
        return None

    ts_feat = engineer_features(ts)
    if len(ts_feat) < MIN_SERIES_LENGTH:
        return None

    (X_tr, y_tr, X_va, y_va, X_te, y_te,
     d_tr, d_va, d_te) = chronological_split(ts_feat)

    results = {}

    # --- Baseline: 7-day rolling mean ---
    base_pred_val  = np.full(len(y_va), np.mean(y_tr[-7:]))
    base_pred_test = np.full(len(y_te), np.mean(
        np.concatenate([y_tr, y_va])[-7:]
    ))
    results["Baseline"] = {
        "val_metrics":  calc_metrics(y_va, base_pred_val,  "Baseline"),
        "test_metrics": calc_metrics(y_te, base_pred_test, "Baseline"),
        "test_pred": base_pred_test,
        "model": None,
    }

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    lr_pred_val  = lr.predict(X_va)
    lr_pred_test = lr.predict(X_te)
    results["Linear Regression"] = {
        "val_metrics":  calc_metrics(y_va, lr_pred_val,  "Linear Regression"),
        "test_metrics": calc_metrics(y_te, lr_pred_test, "Linear Regression"),
        "test_pred": lr_pred_test,
        "model": lr,
    }

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                               random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred_val  = rf.predict(X_va)
    rf_pred_test = rf.predict(X_te)
    results["Random Forest"] = {
        "val_metrics":  calc_metrics(y_va, rf_pred_val,  "Random Forest"),
        "test_metrics": calc_metrics(y_te, rf_pred_test, "Random Forest"),
        "test_pred": rf_pred_test,
        "model": rf,
    }

    # --- XGBoost ---
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      verbose=False)
        xgb_pred_val  = xgb_model.predict(X_va)
        xgb_pred_test = xgb_model.predict(X_te)
        results["XGBoost"] = {
            "val_metrics":  calc_metrics(y_va, xgb_pred_val,  "XGBoost"),
            "test_metrics": calc_metrics(y_te, xgb_pred_test, "XGBoost"),
            "test_pred": xgb_pred_test,
            "model": xgb_model,
        }

    # Select best model by validation RMSE
    best_name = min(
        [k for k in results if k != "Baseline"],
        key=lambda k: results[k]["val_metrics"]["RMSE"]
    )

    return {
        "ts":         ts,
        "ts_feat":    ts_feat,
        "X_tr": X_tr, "y_tr": y_tr,
        "X_va": X_va, "y_va": y_va,
        "X_te": X_te, "y_te": y_te,
        "d_tr": d_tr, "d_va": d_va, "d_te": d_te,
        "results":    results,
        "best_model": best_name,
        "commodity":  commodity,
        "market":     market,
    }


def generate_future_forecast(model_data: dict, horizon: int) -> pd.DataFrame:
    """
    Walk-forward forecast for `horizon` days beyond the last known date.
    Uses the best model. Re-uses lag/rolling from the tail of ts_feat.
    """
    if model_data is None:
        return pd.DataFrame()

    best_name = model_data["best_model"]
    model     = model_data["results"][best_name]["model"]
    ts        = model_data["ts"].copy()

    if model is None:   # Baseline fallback
        last_7_mean = float(ts["y"].iloc[-7:].mean())
        dates = pd.date_range(ts["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.DataFrame({"ds": dates, "predicted": last_7_mean})

    # Iterative walk-forward prediction
    history = list(ts["y"].values)
    future_dates = pd.date_range(ts["ds"].iloc[-1] + pd.Timedelta(days=1),
                                 periods=horizon, freq="D")
    predictions = []

    for i, fdate in enumerate(future_dates):
        h = history  # all known values so far
        def lag(n): return h[-n] if len(h) >= n else np.nan
        def roll_mean(n):
            window = h[-n:] if len(h) >= n else h
            return np.mean(window) if window else np.nan
        def roll_std(n):
            window = h[-n:] if len(h) >= n else h
            return np.std(window) if len(window) > 1 else 0.0

        row = [
            lag(1), lag(7), lag(14), lag(30),
            roll_mean(7), roll_mean(14), roll_mean(30),
            roll_std(7), roll_std(30),
            fdate.month, fdate.quarter, fdate.dayofweek,
            int(fdate.isocalendar()[1]), fdate.year
        ]
        if any(np.isnan(v) for v in row):
            pred = float(np.mean(h[-7:]))
        else:
            pred = float(model.predict(np.array(row).reshape(1, -1))[0])
        predictions.append(pred)
        history.append(pred)

    return pd.DataFrame({"ds": future_dates, "predicted": predictions})


# ── SECTION 6: AI INSIGHTS ───────────────────────────────────────────────────

def generate_ai_insights(df_clean: pd.DataFrame, model_data: dict) -> list:
    """
    Rule-based insight generator. Every statement is derived from
    calculated values — no fabrication.
    Returns list of {category, type, text}.
    """
    insights = []

    # 1. Most volatile commodity
    ca = commodity_analytics(df_clean)
    most_vol = ca.sort_values("CV_Pct", ascending=False).iloc[0]
    insights.append({
        "category": "Volatility",
        "type": "DATA_DRIVEN",
        "text": (
            f"**{most_vol['Commodity']}** has the highest price volatility with a "
            f"Coefficient of Variation of **{most_vol['CV_Pct']:.1f}%**, "
            f"meaning prices fluctuate significantly around the "
            f"₹{most_vol['Mean_Price']:,.0f}/quintal annual average."
        )
    })

    # 2. Most stable commodity
    most_stab = ca.sort_values("CV_Pct").iloc[0]
    insights.append({
        "category": "Stability",
        "type": "DATA_DRIVEN",
        "text": (
            f"**{most_stab['Commodity']}** is the most price-stable commodity "
            f"with CV of **{most_stab['CV_Pct']:.1f}%** — "
            f"average price ₹{most_stab['Mean_Price']:,.0f}/quintal."
        )
    })

    # 3. Seasonal insight (Potato)
    try:
        sea = seasonal_analysis(df_clean)
        potato_sea = sea[sea["Commodity"] == "Potato"]
        if not potato_sea.empty:
            peak_month = potato_sea.loc[potato_sea["Seasonal_Index"].idxmax()]
            low_month  = potato_sea.loc[potato_sea["Seasonal_Index"].idxmin()]
            insights.append({
                "category": "Seasonal",
                "type": "AI_INTERPRETATION",
                "text": (
                    f"Potato prices historically peak in **{peak_month['Month_Name']}** "
                    f"(seasonal index {peak_month['Seasonal_Index']:.1f}) and reach their "
                    f"lowest in **{low_month['Month_Name']}** "
                    f"(index {low_month['Seasonal_Index']:.1f}). "
                    f"This suggests procurement in {low_month['Month_Name']} "
                    f"offers better value for buyers."
                )
            })
    except Exception:
        pass

    # 4. State price disparity
    sa = state_analytics(df_clean)
    if len(sa) >= 2:
        high_state = sa.iloc[0]
        low_state  = sa.iloc[-1]
        diff_pct = ((high_state["Avg_Modal"] - low_state["Avg_Modal"]) /
                    low_state["Avg_Modal"] * 100)
        insights.append({
            "category": "Market Disparity",
            "type": "DATA_DRIVEN",
            "text": (
                f"There is a **{diff_pct:.0f}% price difference** between the highest-price "
                f"state (**{high_state['STATE']}** — ₹{high_state['Avg_Modal']:,.0f}) "
                f"and lowest-price state (**{low_state['STATE']}** — "
                f"₹{low_state['Avg_Modal']:,.0f}). "
                f"This disparity indicates regional supply-demand imbalances."
            )
        })

    # 5. Forecast direction (if model data available)
    if model_data is not None:
        try:
            forecast_df = generate_future_forecast(model_data, 14)
            if not forecast_df.empty:
                ts = model_data["ts"]
                last_price = float(ts["y"].iloc[-1])
                avg_forecast = float(forecast_df["predicted"].mean())
                pct_change = (avg_forecast - last_price) / last_price * 100
                direction = "increase" if pct_change > 0 else "decrease"
                insights.append({
                    "category": "Forecast",
                    "type": "AI_INTERPRETATION",
                    "text": (
                        f"Based on historical patterns, the model estimates that "
                        f"**{model_data['commodity']}** prices at **{model_data['market']}** "
                        f"may **{direction}** by approximately **{abs(pct_change):.1f}%** "
                        f"over the next 14 days. "
                        f"Current price: ₹{last_price:,.0f} → "
                        f"Estimated avg: ₹{avg_forecast:,.0f}/quintal. "
                        f"*(Model estimate — not a guaranteed price)*"
                    )
                })
        except Exception:
            pass

    # 6. Data coverage note
    date_range = df_clean["Price Date"].max() - df_clean["Price Date"].min()
    insights.append({
        "category": "Data Coverage",
        "type": "DATA_DRIVEN",
        "text": (
            f"The dataset covers **{date_range.days} days** of mandi price data "
            f"across **{df_clean['STATE'].nunique()} states**, "
            f"**{df_clean['Market Name'].nunique()} markets**, and "
            f"**{df_clean['Commodity'].nunique()} commodities**, "
            f"with {len(df_clean):,} clean records available for analytics."
        )
    })

    return insights


# ── SECTION 7: CHART HELPERS ─────────────────────────────────────────────────

def kpi_card(title: str, value: str, sub: str = "") -> str:
    return f"""<div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""


def fig_commodity_bar(ca: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Min Price",    x=ca["Commodity"], y=ca["Min_Price"],   marker_color="#93c5fd"))
    fig.add_trace(go.Bar(name="Avg Modal",    x=ca["Commodity"], y=ca["Mean_Price"],  marker_color="#3b82d4"))
    fig.add_trace(go.Bar(name="Max Price",    x=ca["Commodity"], y=ca["Max_Price"],   marker_color="#1e3a5f"))
    fig.update_layout(barmode="group", title="Commodity Price Comparison (₹/quintal)",
                      xaxis_title="Commodity", yaxis_title="Price (₹/quintal)",
                      legend=dict(orientation="h"), height=400)
    return fig


def fig_monthly_trend(mt: pd.DataFrame, selected_commodities: list = None) -> go.Figure:
    if selected_commodities:
        mt = mt[mt["Commodity"].isin(selected_commodities)]
    fig = px.line(mt, x="YearMonth", y="Avg_Modal", color="Commodity",
                  title="Monthly Average Modal Price (₹/quintal)",
                  labels={"Avg_Modal": "Avg Modal Price (₹)", "YearMonth": "Month"})
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=420)
    return fig


def fig_seasonal(sea: pd.DataFrame, commodity: str) -> go.Figure:
    sub = sea[sea["Commodity"] == commodity]
    fig = px.bar(sub, x="Month_Name", y="Seasonal_Index",
                 title=f"Seasonal Price Index — {commodity}",
                 color="Seasonal_Index",
                 color_continuous_scale=["#93c5fd","#3b82d4","#1e3a5f"],
                 labels={"Seasonal_Index": "Index (100 = annual avg)"})
    fig.add_hline(y=100, line_dash="dash", line_color="red",
                  annotation_text="Annual Avg (100)")
    fig.update_layout(height=380)
    return fig


def fig_volatility(ca: pd.DataFrame) -> go.Figure:
    ca_sorted = ca.sort_values("CV_Pct", ascending=True)
    fig = px.bar(ca_sorted, x="CV_Pct", y="Commodity", orientation="h",
                 title="Price Volatility by Commodity (CV%)",
                 color="CV_Pct", color_continuous_scale="Reds",
                 labels={"CV_Pct": "Coefficient of Variation (%)"})
    fig.update_layout(height=320)
    return fig


def fig_state_bar(sa: pd.DataFrame, top_n: int = 15) -> go.Figure:
    sub = sa.head(top_n)
    fig = px.bar(sub, x="Avg_Modal", y="STATE", orientation="h",
                 color="Avg_Modal", color_continuous_scale="Blues",
                 title=f"Top {top_n} States by Avg Modal Price (₹/quintal)",
                 labels={"Avg_Modal": "Avg Modal Price (₹)", "STATE": "State"})
    fig.update_layout(height=480)
    return fig


def fig_actual_vs_pred(d_test, y_test, predictions_dict: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(d_test), y=y_test,
                             mode="lines", name="Actual", line=dict(color="#1f2328", width=2)))
    colors = {"Baseline":"#94a3b8","Linear Regression":"#f59e0b",
              "Random Forest":"#22c55e","XGBoost":"#3b82d4"}
    for mname, preds in predictions_dict.items():
        fig.add_trace(go.Scatter(x=list(d_test), y=preds, mode="lines",
                                 name=mname,
                                 line=dict(color=colors.get(mname,"#888"), dash="dot")))
    fig.update_layout(title="Actual vs Predicted — Test Set",
                      xaxis_title="Date", yaxis_title="Modal Price (₹/quintal)",
                      height=420, legend=dict(orientation="h"))
    return fig


def fig_forecast(ts_hist: pd.DataFrame, forecast_df: pd.DataFrame,
                 commodity: str, market: str) -> go.Figure:
    # show last 90 days of history
    hist = ts_hist.tail(90)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"],
                             mode="lines", name="Historical",
                             line=dict(color="#3b82d4", width=2)))
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["predicted"],
                             mode="lines+markers", name="Forecast (Estimate)",
                             line=dict(color="#f59e0b", dash="dash", width=2),
                             marker=dict(size=5)))
    fig.update_layout(
        title=f"Price Forecast — {commodity} @ {market}",
        xaxis_title="Date", yaxis_title="Modal Price (₹/quintal)",
        height=420,
        annotations=[dict(
            text="⚠️ Model estimates — not guaranteed prices",
            xref="paper", yref="paper", x=0.01, y=0.98,
            showarrow=False, font=dict(size=11, color="#57606a")
        )]
    )
    return fig


def fig_feature_importance(model, top_n: int = 14) -> go.Figure:
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.DataFrame({"Feature": FEATURE_COLS,
                        "Importance": model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=True).tail(top_n)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 title="Feature Importance",
                 color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=420)
    return fig


def fig_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=residuals, mode="markers",
                             marker=dict(color="#3b82d4", opacity=0.5, size=5),
                             name="Residual"))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title=f"Residuals — {model_name} (Test Set)",
                      xaxis_title="Sample Index",
                      yaxis_title="Residual (₹/quintal)",
                      height=360)
    return fig


# ── SECTION 8: STREAMLIT PAGES ───────────────────────────────────────────────

def page_home(df_full: pd.DataFrame, df_clean: pd.DataFrame):
    st.title("🌾 AgriPrice AI")
    st.subheader("Indian Mandi Price Analytics & Forecasting System")
    st.caption("Indian Mandi Price Analytics & Forecasting System")
    st.markdown("---")

    # KPI row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    total_clean = len(df_clean)
    n_states    = df_clean["STATE"].nunique()
    n_markets   = df_clean["Market Name"].nunique()
    n_comm      = df_clean["Commodity"].nunique()
    avg_modal   = df_clean["Modal_Price"].mean()
    avg_range   = df_clean["Price_Range"].mean()
    date_range  = (df_clean["Price Date"].max() - df_clean["Price Date"].min()).days

    c1.markdown(kpi_card("Clean Records",   f"{total_clean:,}",         f"of {len(df_full):,} total"), unsafe_allow_html=True)
    c2.markdown(kpi_card("States Covered",  str(n_states),              "after normalization"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Markets",         str(n_markets),             "unique mandis"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Commodities",     str(n_comm),                "tracked"), unsafe_allow_html=True)
    c5.markdown(kpi_card("Avg Modal Price", f"₹{avg_modal:,.0f}",       "per quintal"), unsafe_allow_html=True)
    c6.markdown(kpi_card("Avg Price Range", f"₹{avg_range:,.0f}",       f"over {date_range} days"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    ca  = commodity_analytics(df_clean)
    mt  = monthly_trend(df_clean)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_commodity_bar(ca), use_container_width=True)
    with col2:
        st.plotly_chart(fig_monthly_trend(mt, ["Potato","Onion"]), use_container_width=True)

    # Charts row 2
    sa = state_analytics(df_clean)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_state_bar(sa), use_container_width=True)
    with col4:
        st.plotly_chart(fig_volatility(ca), use_container_width=True)

    # Data quality summary
    st.markdown("---")
    st.subheader("📋 Data Quality Summary")
    dq = get_data_quality_report(df_full)
    st.dataframe(dq, use_container_width=True)
    st.caption("Flagged records are retained for analytics but excluded from ML training.")


def page_data_overview(df_full: pd.DataFrame, df_clean: pd.DataFrame):
    st.title("📊 Data Explorer")

    tab1, tab2, tab3 = st.tabs(["Dataset Info", "Sample Data", "Quality Flags"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset Dimensions**")
            st.metric("Total Rows (Raw)",   f"{len(df_full):,}")
            st.metric("Total Rows (Clean)", f"{len(df_clean):,}")
            st.metric("Columns",            str(df_full.shape[1]))
        with col2:
            st.markdown("**Date Coverage**")
            st.metric("From", str(df_clean["Price Date"].min().date()))
            st.metric("To",   str(df_clean["Price Date"].max().date()))
            st.metric("Calendar Days",
                      str((df_clean["Price Date"].max() - df_clean["Price Date"].min()).days + 1))

        st.markdown("**Column Summary**")
        dtypes_df = pd.DataFrame({
            "Column":   df_full.columns.tolist(),
            "Dtype":    [str(t) for t in df_full.dtypes.values],
            "Non-Null": df_full.notna().sum().values,
            "Unique":   [df_full[c].nunique() for c in df_full.columns],
        })
        st.dataframe(dtypes_df, use_container_width=True)

        st.markdown("**Price Statistics (Clean Data)**")
        st.dataframe(
            df_clean[["Min_Price","Max_Price","Modal_Price","Price_Range"]].describe().round(2),
            use_container_width=True
        )

    with tab2:
        st.markdown("**Raw Data Sample (first 1000 rows)**")
        st.dataframe(df_full.head(1000), use_container_width=True)
        buf = io.StringIO()
        df_clean.to_csv(buf, index=False)
        st.download_button("⬇️ Download Cleaned CSV",
                           data=buf.getvalue(),
                           file_name="AgriPrice_Cleaned.csv",
                           mime="text/csv")

    with tab3:
        dq = get_data_quality_report(df_full)
        st.dataframe(dq, use_container_width=True)
        st.markdown("**State Name Normalization Applied**")
        norm_df = pd.DataFrame(list(STATE_MAP.items()), columns=["Raw Value","Normalized To"])
        st.dataframe(norm_df, use_container_width=True)

        st.markdown("**Records Excluded from ML (flagged)**")
        flagged = df_full[df_full["Data_Quality_Flag"] != "OK"][
            ["STATE","Market Name","Commodity","Min_Price","Max_Price",
             "Modal_Price","Price Date","Data_Quality_Flag"]
        ].head(500)
        st.dataframe(flagged, use_container_width=True)


def page_commodity_analytics(df_clean: pd.DataFrame):
    st.title("🌾 Commodity Analytics")

    commodity = st.selectbox("Select Commodity", sorted(df_clean["Commodity"].unique()))
    sub = df_clean[df_clean["Commodity"] == commodity]

    # Stats row
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Records",       f"{len(sub):,}")
    col2.metric("Avg Modal",     f"₹{sub['Modal_Price'].mean():,.0f}")
    col3.metric("Max Modal",     f"₹{sub['Modal_Price'].max():,.0f}")
    col4.metric("Volatility CV", f"{sub['Modal_Price'].std()/sub['Modal_Price'].mean()*100:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["Price Trend", "Seasonal Pattern",
                                       "State Comparison", "Variety Breakdown"])
    with tab1:
        mt = monthly_trend(df_clean)
        st.plotly_chart(fig_monthly_trend(mt, [commodity]), use_container_width=True)
        # daily distribution
        fig_box = px.box(sub, x="Year", y="Modal_Price",
                         title=f"{commodity} — Modal Price Distribution by Year",
                         color="Year", labels={"Modal_Price":"Modal Price (₹)"})
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        sea = seasonal_analysis(df_clean)
        st.plotly_chart(fig_seasonal(sea, commodity), use_container_width=True)
        st.dataframe(
            sea[sea["Commodity"]==commodity][
                ["Month_Name","Monthly_Mean","Seasonal_Index"]
            ].rename(columns={"Monthly_Mean":"Avg Price (₹)","Seasonal_Index":"Seasonal Index"}),
            use_container_width=True
        )

    with tab3:
        state_comm = (
            sub.groupby("STATE")["Modal_Price"]
            .agg(Avg_Modal="mean", Records="count")
            .reset_index().sort_values("Avg_Modal", ascending=False)
        )
        fig_sc = px.bar(state_comm, x="STATE", y="Avg_Modal",
                        color="Avg_Modal", color_continuous_scale="Blues",
                        title=f"{commodity} — Avg Modal Price by State",
                        labels={"Avg_Modal":"Avg Modal Price (₹)","STATE":"State"})
        fig_sc.update_xaxes(tickangle=45)
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab4:
        var_counts = sub["Variety"].value_counts().reset_index()
        var_counts.columns = ["Variety","Count"]
        fig_var = px.pie(var_counts.head(10), names="Variety", values="Count",
                         title=f"{commodity} — Top 10 Varieties by Record Count")
        st.plotly_chart(fig_var, use_container_width=True)


def page_market_analytics(df_clean: pd.DataFrame):
    st.title("🏪 Market Analytics")

    # Cascading filters
    col1, col2 = st.columns(2)
    with col1:
        states = ["All"] + sorted(df_clean["STATE"].unique())
        sel_state = st.selectbox("Filter by State", states)
    sub = df_clean if sel_state == "All" else df_clean[df_clean["STATE"] == sel_state]
    with col2:
        commodities = ["All"] + sorted(sub["Commodity"].unique())
        sel_comm = st.selectbox("Filter by Commodity", commodities)
    if sel_comm != "All":
        sub = sub[sub["Commodity"] == sel_comm]

    ma = market_analytics(sub)

    tab1, tab2, tab3 = st.tabs(["Top Markets", "Price Spread", "Activity"])
    with tab1:
        top_n = st.slider("Show top N markets", 5, 30, 10)
        top_by_price = ma.head(top_n)
        fig_mp = px.bar(top_by_price, x="Avg_Modal", y="Market Name", orientation="h",
                        color="Avg_Modal", color_continuous_scale="Blues",
                        title=f"Top {top_n} Markets by Avg Modal Price",
                        labels={"Avg_Modal":"Avg Modal Price (₹)","Market Name":"Market"})
        st.plotly_chart(fig_mp, use_container_width=True)

    with tab2:
        spread_sorted = ma.sort_values("Avg_Range", ascending=False).head(top_n)
        fig_sp = px.bar(spread_sorted, x="Avg_Range", y="Market Name", orientation="h",
                        color="Avg_Range", color_continuous_scale="Oranges",
                        title=f"Top {top_n} Markets by Avg Price Spread (Max−Min)",
                        labels={"Avg_Range":"Avg Spread (₹)","Market Name":"Market"})
        st.plotly_chart(fig_sp, use_container_width=True)

    with tab3:
        active_sorted = ma.sort_values("Records", ascending=False).head(top_n)
        fig_act = px.bar(active_sorted, x="Records", y="Market Name", orientation="h",
                         color="Records", color_continuous_scale="Greens",
                         title=f"Top {top_n} Most Active Markets (by record count)",
                         labels={"Records":"Total Records","Market Name":"Market"})
        st.plotly_chart(fig_act, use_container_width=True)


def page_price_trends(df_clean: pd.DataFrame):
    st.title("📈 Price Trends")

    tab1, tab2, tab3 = st.tabs(["Monthly Trends", "Year-over-Year", "Price Distribution"])

    with tab1:
        comms = st.multiselect("Select Commodities",
                               sorted(df_clean["Commodity"].unique()),
                               default=["Potato","Onion"])
        mt = monthly_trend(df_clean)
        if comms:
            st.plotly_chart(fig_monthly_trend(mt, comms), use_container_width=True)
        else:
            st.info("Select at least one commodity.")

    with tab2:
        st.markdown("**Year-over-Year Comparison — Potato & Onion** *(only commodities with 3-year data)*")
        yoy_data = []
        for comm in ["Potato","Onion"]:
            for yr in [2023, 2024, 2025]:
                sub = df_clean[(df_clean["Commodity"]==comm) & (df_clean["Year"]==yr)]
                if len(sub) > 0:
                    yoy_data.append({
                        "Commodity": comm, "Year": str(yr),
                        "Avg_Modal": round(sub["Modal_Price"].mean(), 2),
                        "Records": len(sub)
                    })
        yoy_df = pd.DataFrame(yoy_data)
        if not yoy_df.empty:
            fig_yoy = px.bar(yoy_df, x="Year", y="Avg_Modal", color="Commodity",
                             barmode="group", title="Year-over-Year Avg Modal Price",
                             labels={"Avg_Modal":"Avg Modal Price (₹)"})
            st.plotly_chart(fig_yoy, use_container_width=True)
            st.dataframe(yoy_df, use_container_width=True)
        st.markdown("""
        > ⚠️ **Note:** Tomato data only available for 2023. Rice only for 2025. 
        > Wheat data drops off after mid-2024. Only Potato and Onion support full YoY comparison.
        """)

    with tab3:
        sel_comm = st.selectbox("Commodity", sorted(df_clean["Commodity"].unique()), key="dist_comm")
        sub = df_clean[df_clean["Commodity"] == sel_comm]
        fig_hist = px.histogram(sub, x="Modal_Price", nbins=80,
                                title=f"{sel_comm} — Modal Price Distribution",
                                labels={"Modal_Price":"Modal Price (₹/quintal)"},
                                color_discrete_sequence=["#3b82d4"])
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_violin = px.violin(sub, x="Year", y="Modal_Price",
                               title=f"{sel_comm} — Modal Price Violin by Year",
                               color="Year",
                               labels={"Modal_Price":"Modal Price (₹/quintal)"})
        st.plotly_chart(fig_violin, use_container_width=True)


def page_sql_analytics(df_clean: pd.DataFrame):
    st.title("🗄️ SQL Analytics")
    st.markdown("""
    The cleaned dataset is loaded into an **in-memory SQLite database** and queried using 
    standard SQL. Results are computed live — no pre-stored values.
    """)

    sql_results = run_sql_analytics(df_clean)

    query_labels = {
        "Q1_Avg_By_Commodity":         "Q1 — Average Modal Price by Commodity",
        "Q2_Avg_By_State":             "Q2 — Average Modal Price by State",
        "Q3_Top10_Markets_By_Price":   "Q3 — Top 10 Markets by Average Price",
        "Q4_Most_Volatile_Commodities":"Q4 — Most Volatile Commodities",
        "Q5_Monthly_Avg_By_Commodity": "Q5 — Monthly Average Price by Commodity",
        "Q6_State_Commodity_Matrix":   "Q6 — State × Commodity Price Matrix",
        "Q7_Highest_Price_Spread_Markets":"Q7 — Markets with Highest Price Spread",
        "Q8_Price_Extremes":           "Q8 — Highest & Lowest Modal Price Records",
        "Q9_Market_Activity":          "Q9 — Most Active Markets",
        "Q10_YoY_Price_Change":        "Q10 — Year-over-Year Price Change",
    }

    selected_q = st.selectbox("Select SQL Query to View", list(query_labels.values()))
    q_key = [k for k,v in query_labels.items() if v == selected_q][0]
    result_df = sql_results.get(q_key, pd.DataFrame())
    st.dataframe(result_df, use_container_width=True)

    # Show SQL text
    sql_texts = {
        "Q1_Avg_By_Commodity": "SELECT Commodity, ROUND(AVG(Modal_Price),2) AS Avg_Modal, ... FROM mandi_prices WHERE Data_Quality_Flag='OK' GROUP BY Commodity ORDER BY Avg_Modal DESC",
        "Q10_YoY_Price_Change": "SELECT Commodity, AVG(CASE WHEN Year=2023 THEN Modal_Price END) AS Avg_2023, AVG(CASE WHEN Year=2024 THEN Modal_Price END) AS Avg_2024, AVG(CASE WHEN Year=2025 THEN Modal_Price END) AS Avg_2025 FROM mandi_prices WHERE Commodity IN ('Potato','Onion') GROUP BY Commodity",
    }
    if q_key in sql_texts:
        with st.expander("View SQL"):
            st.code(sql_texts[q_key], language="sql")

    st.markdown("---")
    st.subheader("📊 SQL Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        q1 = sql_results["Q1_Avg_By_Commodity"]
        if not q1.empty:
            fig = px.bar(q1, x="Commodity", y="Avg_Modal",
                         color="Avg_Modal", color_continuous_scale="Blues",
                         title="Q1 — Avg Modal Price by Commodity")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        q10 = sql_results["Q10_YoY_Price_Change"]
        if not q10.empty and "Avg_2023" in q10.columns:
            q10_melt = q10.melt(id_vars="Commodity",
                                value_vars=["Avg_2023","Avg_2024","Avg_2025"],
                                var_name="Year", value_name="Avg_Price")
            q10_melt = q10_melt.dropna()
            fig_yoy = px.bar(q10_melt, x="Year", y="Avg_Price", color="Commodity",
                             barmode="group", title="Q10 — YoY Price Change")
            st.plotly_chart(fig_yoy, use_container_width=True)


def page_forecasting(df_clean: pd.DataFrame):
    st.title("🔮 Price Forecasting")
    st.markdown(
        "Select a commodity, market and forecast horizon to estimate future modal prices "
        "using machine learning models trained on historical mandi price patterns."
    )

    # ── Forecast Settings (in main page, not sidebar) ──
    st.markdown("### ⚙️ Forecast Settings")
    fs_col1, fs_col2, fs_col3 = st.columns(3)
    with fs_col1:
        commodities = sorted(df_clean["Commodity"].unique())
        sel_comm = st.selectbox(
            "Commodity",
            commodities,
            index=commodities.index(FORECAST_COMMODITY)
            if FORECAST_COMMODITY in commodities else 0,
            key="fc_comm",
        )
    # Cascading: market list updates when commodity changes
    comm_markets = sorted(
        df_clean[df_clean["Commodity"] == sel_comm]["Market Name"].unique()
    )
    default_mkt = FORECAST_MARKET if FORECAST_MARKET in comm_markets else comm_markets[0]
    with fs_col2:
        sel_market = st.selectbox(
            "Market",
            comm_markets,
            index=comm_markets.index(default_mkt),
            key="fc_mkt",
        )
    with fs_col3:
        horizon = st.select_slider("Forecast Horizon (days)", [7, 14, 30], value=14)
    st.markdown("---")

    # Check data sufficiency
    ts_check = prepare_forecast_series(df_clean, sel_comm, sel_market)
    if ts_check.empty or len(ts_check) < MIN_SERIES_LENGTH:
        st.warning(
            f"⚠️ Insufficient historical data for **{sel_comm} @ {sel_market}**. "
            f"Need at least {MIN_SERIES_LENGTH} daily observations — "
            f"found {len(ts_check)}. Please select a different market."
        )
        return

    # Train / load model
    with st.spinner(f"Training models for {sel_comm} @ {sel_market}…"):
        model_data = get_or_train_model(sel_comm, sel_market, df_clean)

    if model_data is None:
        st.error("Model training failed. Insufficient clean data after feature engineering.")
        return

    best_name = model_data["best_model"]
    forecast_df = generate_future_forecast(model_data, horizon)

    # KPI summary
    ts = model_data["ts"]
    last_price   = float(ts["y"].iloc[-1])
    avg_forecast = float(forecast_df["predicted"].mean())
    pct_change   = (avg_forecast - last_price) / last_price * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Known Price",   f"₹{last_price:,.0f}")
    col2.metric("Avg Forecast Price", f"₹{avg_forecast:,.0f}",
                delta=f"{pct_change:+.1f}%")
    col3.metric("Best Model",         best_name)
    col4.metric("Forecast Horizon",   f"{horizon} days")

    st.markdown("""
    <div class="flag-warn">
    ⚠️ <strong>Disclaimer:</strong> These are model-generated estimates based on historical price patterns.
    They are NOT guaranteed future prices. Actual mandi prices depend on weather, policy, supply chain, and
    demand factors not captured in this model.
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig_forecast(ts, forecast_df, sel_comm, sel_market),
                    use_container_width=True)

    # Forecast table
    with st.expander("📋 Forecast Table"):
        display_fc = forecast_df.copy()
        display_fc["ds"] = display_fc["ds"].dt.strftime("%Y-%m-%d")
        display_fc["predicted"] = display_fc["predicted"].round(2)
        display_fc.columns = ["Date","Predicted Modal Price (₹/quintal)"]
        st.dataframe(display_fc, use_container_width=True)

        buf = io.StringIO()
        display_fc.to_csv(buf, index=False)
        st.download_button("⬇️ Download Forecast CSV", buf.getvalue(),
                           file_name=f"forecast_{sel_comm}_{sel_market}.csv",
                           mime="text/csv")

    # Model metrics
    st.markdown("---")
    st.subheader(f"📉 {best_name} — Test Set Metrics")
    tm = model_data["results"][best_name]["test_metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("MAE",  f"₹{tm['MAE']:,.2f}")
    mc2.metric("RMSE", f"₹{tm['RMSE']:,.2f}")
    mc3.metric("R²",   str(tm['R2']))
    mc4.metric("MAPE", f"{tm['MAPE']:.2f}%" if tm['MAPE'] else "N/A")


def page_model_performance(df_clean: pd.DataFrame):
    st.title("📉 Model Performance")
    st.markdown(f"Model evaluation for: **{FORECAST_COMMODITY} @ {FORECAST_MARKET}**")

    with st.spinner("Loading model results…"):
        model_data = get_or_train_model(FORECAST_COMMODITY, FORECAST_MARKET, df_clean)

    if model_data is None:
        st.error("Model data unavailable.")
        return

    results  = model_data["results"]
    d_te     = model_data["d_te"]
    y_te     = model_data["y_te"]
    d_tr     = model_data["d_tr"]
    d_va     = model_data["d_va"]

    # Split summary
    st.subheader("🗓️ Train / Validation / Test Periods")
    col1, col2, col3 = st.columns(3)
    col1.metric("Training",   f"{d_tr.min().date()} → {d_tr.max().date()}",
                f"{len(d_tr)} days")
    col2.metric("Validation", f"{d_va.min().date()} → {d_va.max().date()}",
                f"{len(d_va)} days")
    col3.metric("Testing",    f"{d_te.min().date()} → {d_te.max().date()}",
                f"{len(d_te)} days")

    # Model comparison table
    st.subheader("📊 Model Comparison — Test Set Metrics")
    comp_rows = []
    for mname, mdict in results.items():
        row = mdict["test_metrics"].copy()
        row["Validation RMSE"] = mdict["val_metrics"]["RMSE"]
        row["Best Model?"]     = "✅" if mname == model_data["best_model"] else ""
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df.set_index("Model"), use_container_width=True)
    st.caption("Best model selected by lowest Validation RMSE.")

    # Actual vs Predicted
    st.subheader("📈 Actual vs Predicted — All Models")
    pred_dict = {mname: mdict["test_pred"]
                 for mname, mdict in results.items()}
    st.plotly_chart(fig_actual_vs_pred(d_te, y_te, pred_dict),
                    use_container_width=True)

    # Feature importance
    best_name   = model_data["best_model"]
    best_model  = results[best_name]["model"]
    fig_fi = fig_feature_importance(best_model)
    if fig_fi:
        st.subheader(f"🔍 Feature Importance — {best_name}")
        st.plotly_chart(fig_fi, use_container_width=True)

    # Residuals
    best_pred = results[best_name]["test_pred"]
    st.subheader(f"📊 Residuals — {best_name}")
    st.plotly_chart(fig_residuals(y_te, best_pred, best_name), use_container_width=True)

    # Metrics explanation
    with st.expander("ℹ️ Metric Explanations"):
        st.markdown("""
        | Metric | Description | Good Value |
        |--------|-------------|-----------|
        | **MAE** | Mean Absolute Error — average ₹ error per prediction | As low as possible (in ₹/quintal) |
        | **RMSE** | Root Mean Square Error — penalizes large errors more | Lower than MAE indicates few large errors |
        | **R²** | Coefficient of Determination — variance explained by model | Closer to 1.0 is better |
        | **MAPE** | Mean Absolute Percentage Error — % error per prediction | Lower is better; <20% generally acceptable |
        
        > **Limitation:** High R² does not guarantee good real-world forecasts. 
        > The model captures historical patterns but cannot predict policy shocks, 
        > weather events, or export bans.
        """)


def page_ai_insights(df_clean: pd.DataFrame):
    st.title("🤖 AI Insights")
    st.markdown("""
    Insights below are generated by a **rule-based AI layer** grounded entirely in 
    calculated analytical and model results. No fabrication. No external paid API required.
    """)

    with st.spinner("Generating insights…"):
        model_data = get_or_train_model(FORECAST_COMMODITY, FORECAST_MARKET, df_clean)
        insights   = generate_ai_insights(df_clean, model_data)

    for ins in insights:
        badge_color = "#3b82d4" if ins["type"] == "DATA_DRIVEN" else "#7c5cd8"
        badge_label = "📊 DATA-DRIVEN RESULT" if ins["type"] == "DATA_DRIVEN" else "🤖 AI-GENERATED INTERPRETATION"
        css_class   = "insight-data"         if ins["type"] == "DATA_DRIVEN" else "insight-ai"
        st.markdown(
            f'<div class="{css_class}"><span style="font-size:11px;font-weight:600;'
            f'color:{badge_color}">{badge_label} — {ins["category"].upper()}</span><br>'
            f'{ins["text"]}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("""
    ### Methodology
    - All **DATA-DRIVEN** statements trace directly to computed Pandas aggregations on the clean dataset
    - All **AI-INTERPRETED** statements use calculated results as input to rule-based templates
    - No external LLM API is called
    - Results are consistent and reproducible — running the app again produces the same insights
    
    ### Disclaimer
    AI-generated interpretations are pattern-based observations. They do not constitute 
    agricultural market advice.
    """)


def page_powerbi_guide():
    st.title("📊 Power BI Dashboard Guide")
    st.markdown("""
    ## Setup Instructions
    
    1. **Export the cleaned dataset** from the *Data Explorer* page using the **Download Cleaned CSV** button
    2. Open Power BI Desktop → **Get Data** → **Text/CSV** → select `AgriPrice_Cleaned.csv`
    3. In Power Query, verify `Price Date` is parsed as **Date** type
    4. Load the data
    
    ---
    
    ## Recommended Dashboard Pages
    
    ### PAGE 1 — Executive Overview
    **KPI Cards:** Total Records · States · Markets · Commodities · Avg Modal Price · Avg Price Range  
    **Visuals:** Commodity Price Bar · State Map/Bar · Monthly Trend Line · Commodity Donut · Top 10 Markets Bar
    
    ### PAGE 2 — Commodity Analysis
    **Slicers:** Commodity · Year · State  
    **Visuals:** Monthly Price Line · Seasonal Index Bar · State Comparison Bar · Min/Max/Modal Scatter
    
    ### PAGE 3 — Market Analytics
    **Slicers:** State · District · Commodity  
    **Visuals:** Market Price Bar · Price Spread Bar · Market Activity Bar
    
    ### PAGE 4 — Prediction View
    Import `forecast_Potato_Kalipur.csv` (exported from Forecasting page)  
    **Visuals:** Actual vs Predicted Line · Forecast Horizon Line · Error KPI Cards
    
    ---
    
    ## Recommended DAX Measures
    ```dax
    Avg_Modal_Price = AVERAGE(AgriPrice_Cleaned[Modal_Price])
    
    CV_Pct = DIVIDE(STDEV.P(AgriPrice_Cleaned[Modal_Price]),
                    AVERAGE(AgriPrice_Cleaned[Modal_Price])) * 100
    
    Price_Range_Avg = AVERAGE(AgriPrice_Cleaned[Price_Range])
    
    YoY_Change_2024_vs_2023 =
    VAR y24 = CALCULATE([Avg_Modal_Price], AgriPrice_Cleaned[Year] = 2024)
    VAR y23 = CALCULATE([Avg_Modal_Price], AgriPrice_Cleaned[Year] = 2023)
    RETURN DIVIDE(y24 - y23, y23) * 100
    ```
    """)


def page_about():
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## AgriPrice AI — Indian Mandi Price Analytics & Forecasting System
    
    **Internship:** BharatCare + IBM Data Analytics with AI

    ---
    
    ### Problem Statement
    Indian agricultural mandi prices vary significantly across commodities, states, districts, 
    and markets. This project analyzes historical mandi price patterns and builds a predictive 
    system to estimate future modal prices for selected agricultural commodities and markets.
    
    ### Dataset
    - **Source:** Agriculture_price_dataset.csv  
    - **Coverage:** 737,392 records · 10 columns · 2023-06-06 to 2025-06-11  
    - **Commodities:** Onion, Potato, Wheat, Tomato, Rice  
    - **Markets:** 1,598 unique mandis across 30 states (22 after normalization)
    
    ### Technologies Used
    | Layer | Technology |
    |-------|-----------|
    | Data Processing | Python · Pandas · NumPy |
    | Database | SQLite (in-memory) |
    | Machine Learning | Scikit-learn · XGBoost |
    | Visualization | Plotly Express / Graph Objects |
    | Frontend | Streamlit |
    | Business Analytics | Power BI (external) |
    
    ### Architecture
    Single-file design: `AgriPrice_AI.py` contains all backend logic, ML pipeline, 
    analytics, SQL queries, and the Streamlit frontend.
    
    ### Primary Forecasting Series
    - **Commodity:** Potato  
    - **Market:** Kalipur, West Bengal  
    - **Coverage:** ~735 unique dates (near-complete daily coverage)  
    - **Target:** Modal_Price (₹/quintal)
    
    ### Limitations
    - Only 5 commodities in the dataset
    - Rice and Tomato data limited to single years — not forecasted
    - Model cannot predict policy shocks (MSP changes, export bans)
    - Prices are in ₹/quintal (100 kg) — standard mandi unit
    
    ### Running the Application
    ```bash
    pip install -r requirements.txt
    streamlit run AgriPrice_AI.py
    ```
    """)


# ── SECTION 9: MAIN ENTRY POINT ──────────────────────────────────────────────

def main():
    st.sidebar.title("AgriPrice AI 🌾")
    st.sidebar.caption("Indian Mandi Price Analytics")
    st.sidebar.markdown("---")

    PAGES = {
        "🏠 Home / Dashboard":        "home",
        "📊 Data Explorer":            "data",
        "🌾 Commodity Analytics":      "commodity",
        "🏪 Market Analytics":         "market",
        "📈 Price Trends":             "trends",
        "🔮 Forecasting":              "forecast",
        "📉 Model Performance":        "model",
        "🗄️ SQL Analytics":            "sql",
        "🤖 AI Insights":              "insights",
        "📊 Power BI Guide":           "powerbi",
        "ℹ️ About Project":            "about",
    }

    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    page_id = PAGES[page]

    # Load and clean data (cached)
    df_raw   = load_data()
    df_full, df_clean = clean_data(df_raw)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Records loaded:** {len(df_clean):,} clean")
    st.sidebar.markdown(f"**Date range:** {df_clean['Price Date'].min().date()} → {df_clean['Price Date'].max().date()}")

    if page_id == "home":
        page_home(df_full, df_clean)
    elif page_id == "data":
        page_data_overview(df_full, df_clean)
    elif page_id == "commodity":
        page_commodity_analytics(df_clean)
    elif page_id == "market":
        page_market_analytics(df_clean)
    elif page_id == "trends":
        page_price_trends(df_clean)
    elif page_id == "forecast":
        page_forecasting(df_clean)
    elif page_id == "model":
        page_model_performance(df_clean)
    elif page_id == "sql":
        page_sql_analytics(df_clean)
    elif page_id == "insights":
        page_ai_insights(df_clean)
    elif page_id == "powerbi":
        page_powerbi_guide()
    elif page_id == "about":
        page_about()


if __name__ == "__main__":
    main()

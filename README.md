# AgriPrice AI — Indian Mandi Price Analytics & Forecasting System

> **BharatCare + IBM Data Analytics with AI Internship Project**  
> Developed with IBM Bob AI Assistant

---

## Project Overview

AgriPrice AI is a complete end-to-end **Data Analytics + Machine Learning** project built on real Indian agricultural mandi (wholesale market) price data covering **737,392 records** across **5 commodities**, **1,598 markets**, and **30 states** from **June 2023 to June 2025**.

The system provides:
- Interactive analytics dashboard
- SQL-based business intelligence
- Time-series price forecasting using ML
- AI-generated insights
- Power BI integration guide

---

## Problem Statement

Indian agricultural mandi prices vary significantly across commodities, states, districts, and markets. This creates pricing opacity that affects farmers, traders, and policy makers. The objective is to:

1. Analyze historical mandi price patterns across commodities and markets
2. Identify price trends, seasonality, and volatility
3. Build a predictive system that estimates future modal prices
4. Surface actionable insights for stakeholders

---

## Objectives

- Clean and validate 737K+ mandi price records
- Perform comprehensive EDA across commodity, state, market, and time dimensions
- Build and evaluate ML forecasting models (Baseline, Linear Regression, Random Forest, XGBoost)
- Deploy an interactive Streamlit dashboard
- Demonstrate SQL analytics using SQLite
- Generate AI-assisted, data-grounded insights
- Provide Power BI dashboard design and DAX measures

---

## Dataset

| Field | Value |
|-------|-------|
| Filename | `Agriculture_price_dataset.csv` |
| Total Records | 737,392 |
| Columns | 10 |
| Date Range | 2023-06-06 to 2025-06-11 |
| Commodities | Onion, Potato, Wheat, Tomato, Rice |
| States | 30 (22 after normalization) |
| Districts | 373 |
| Markets | 1,598 |
| Price Unit | ₹ per quintal (100 kg) |

**Columns:**
`STATE`, `District Name`, `Market Name`, `Commodity`, `Variety`, `Grade`, `Min_Price`, `Max_Price`, `Modal_Price`, `Price Date`

---

## Technologies Used

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Database | SQLite (in-memory, built-in) |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Plotly Express / Graph Objects |
| Frontend | Streamlit |
| Business Analytics | Power BI (external, optional) |
| AI Assistance | IBM Bob |

---

## System Architecture

```
Agriculture_price_dataset.csv
           │
    [Section 1] load_data()           ← @st.cache_data
           │
    [Section 2] clean_data()          ← @st.cache_data
           │
    ┌──────┴──────┐
    │             │
[Section 3]  [Section 5]
Analytics    Forecasting Pipeline
(Pandas)     prepare_forecast_series()
             engineer_features()
             get_or_train_model()      ← @st.cache_resource
             generate_future_forecast()
    │             │
    └──────┬──────┘
    [Section 4] SQL (SQLite in-memory)
    [Section 6] AI Insight Generator
    [Section 7] Chart Helpers (Plotly)
    [Section 8] Streamlit Pages (9 pages)
    [Section 9] main() entry point
```

Everything is contained in **one file**: `AgriPrice_AI.py`

---

## Data Cleaning Pipeline

| Step | Action |
|------|--------|
| Date parsing | `Price Date` (object) → `datetime64` with `errors='coerce'` |
| State normalization | 8 state name aliases mapped to canonical names |
| String normalization | `.str.strip()` on all categorical columns |
| Derived columns | Year, Month, Quarter, Day, Week_of_Year, Day_of_Week |
| Price metrics | Price_Range, Price_Spread_Pct, Modal_vs_Min, Modal_vs_Max |
| Quality flagging | Zero price, Min>Max, Modal outside Min/Max, IQR×5 outliers |
| ML exclusion | ~3,704 flagged rows excluded from training (retained for analytics) |

---

## Analytics Methodology

- **Commodity analytics:** mean/median/std/min/max/CV% per commodity
- **State analytics:** avg modal price per state, ranking
- **Market analytics:** top-N by price, activity, spread
- **Temporal analysis:** monthly trend, YoY comparison (Potato/Onion only)
- **Seasonal analysis:** monthly seasonal index (mean / annual mean × 100)
- **Price distribution:** histogram, violin, box plots by year

---

## SQL Analysis (SQLite)

10 analytical SQL queries executed against an in-memory SQLite database:

| Query | Business Question |
|-------|------------------|
| Q1 | Avg modal price by commodity |
| Q2 | Avg modal price by state |
| Q3 | Top 10 markets by average price |
| Q4 | Most volatile commodities |
| Q5 | Monthly average price by commodity |
| Q6 | State × Commodity price matrix |
| Q7 | Markets with highest price spread |
| Q8 | Highest and lowest price records |
| Q9 | Most active markets (record count) |
| Q10 | Year-over-year price change (Potato & Onion) |

---

## Power BI Dashboard

Export the cleaned dataset from the Data Explorer page (`Download Cleaned CSV`) and import into Power BI Desktop.

**4 Pages:**
1. Executive Overview — KPIs, commodity comparison, state map, monthly trend
2. Commodity Analysis — Seasonal index, state comparison, variety breakdown
3. Market Analytics — Price spread, activity, volatility
4. Prediction View — Import forecast CSV from Forecasting page

See the **Power BI Guide** page in the application for DAX measures.

---

## Prediction Methodology

**Primary forecasting series:** Potato → Kalipur market → West Bengal  
**Reason:** Near-complete daily coverage (735/737 dates), all 3 years represented, no extreme outliers.

**Aggregation:** Multiple varieties per date aggregated by `mean(Modal_Price)`

**Feature Engineering (14 features):**

| Feature | Type | Leakage Risk |
|---------|------|-------------|
| lag_1, lag_7, lag_14, lag_30 | Lag | ✅ Safe (explicit shift) |
| roll_mean_7/14/30 | Rolling | ✅ Safe (shift(1) before rolling) |
| roll_std_7/30 | Rolling | ✅ Safe |
| month, quarter, day_of_week, week_of_year, year | Calendar | ✅ Safe |

**Train/Validation/Test Split (Chronological):**

| Set | Period | ~Size |
|-----|--------|-------|
| Train | 2023-07 → 2024-09 | 60% |
| Validation | 2024-09 → 2024-12 | 17% |
| Test | 2025-01 → 2025-06 | 23% |

No random splitting. Strictly chronological to prevent data leakage.

**Models Compared:**

| Model | Selection Criterion |
|-------|-------------------|
| Rolling Mean Baseline | Reference — must be beaten |
| Linear Regression | Baseline ML |
| Random Forest | Selected if best validation RMSE |
| XGBoost | Selected if best validation RMSE |

Best model selected by **validation RMSE**. Reported metrics computed on **test set only**.

**Metrics:** MAE, RMSE, R², MAPE

---

## AI Integration

The AI Insights layer is **rule-based** — no paid external LLM required:
- Most volatile commodity (computed CV%)
- Most stable commodity
- Seasonal pattern (peak/trough month from seasonal index)
- State price disparity
- Forecast direction (14-day model estimate)
- Data coverage summary

All statements are clearly labeled as either `DATA-DRIVEN RESULT` or `AI-GENERATED INTERPRETATION`.

---

## Application Pages

| Page | Content |
|------|---------|
| 🏠 Home | 6 KPIs, commodity chart, monthly trend, state bar, volatility, DQ summary |
| 📊 Data Explorer | Dataset info, sample data, quality flags, download cleaned CSV |
| 🌾 Commodity Analytics | Per-commodity trend, seasonal index, state comparison, variety breakdown |
| 🏪 Market Analytics | Top markets by price/spread/activity, cascading filters |
| 📈 Price Trends | Monthly overlay, YoY comparison, price distribution |
| 🔮 Forecasting | Commodity+market selector, forecast chart, future price table, model KPIs |
| 📉 Model Performance | Comparison table, actual vs predicted, feature importance, residuals |
| 🗄️ SQL Analytics | 10 live SQL queries via in-memory SQLite |
| 🤖 AI Insights | Rule-based data-grounded insights |
| 📊 Power BI Guide | Setup instructions + DAX measures |
| ℹ️ About | Project info, architecture, limitations |

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `Agriculture_price_dataset.csv` in the same folder as `AgriPrice_AI.py`

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run AgriPrice_AI.py
```

The application will open in your browser at `http://localhost:8501`

---

## Project Files

```
IBM Project/
├── AgriPrice_AI.py                    ← Single-file application (backend + frontend)
├── Agriculture_price_dataset.csv      ← Input dataset
├── requirements.txt                   ← Python dependencies
├── README.md                          ← This file
└── AgriPrice_AI_ProjectReport.docx   ← Project report
```

---

## Limitations

- **Only 5 commodities** — Onion, Potato, Wheat, Tomato, Rice
- **Tomato**: data only available for 2023 — not forecastable
- **Rice**: data only available for 2025 — not forecastable  
- **Wheat**: drops off after mid-2024 — limited YoY analysis
- The ML model captures historical price patterns but **cannot predict**:
  - Government MSP announcements
  - Export bans / import duties
  - Weather shocks
  - Supply chain disruptions
- Prices are in **₹/quintal** (100 kg) — not per kg

---

## Future Enhancements

- Integrate weather data (rainfall, temperature) as additional features
- Add LSTM/Prophet time-series models for comparison
- Expand to more commodities and a larger date range
- Add commodity-specific MSP thresholds as reference lines
- Build SMS/WhatsApp alert system for price spike detection
- Deploy on cloud (Streamlit Cloud / AWS / Azure)

---

## Author

**Arbaz AP**  
BharatCare + IBM Data Analytics with AI Internship  
Developed using IBM Bob AI Assistant

---

*All analytical results, model metrics, and AI insights in this project are computed from the actual dataset. No values have been invented or hard-coded.*

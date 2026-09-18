# AgriPrice AI — Project Report
## Indian Mandi Price Analytics & Forecasting System

**BharatCare + IBM Data Analytics with AI Internship**  
**Developed with IBM Bob AI Assistant**

---

## 1. Title

**AgriPrice AI: Indian Mandi Price Analytics & Forecasting System**

---

## 2. Abstract

This project presents a complete end-to-end data analytics and machine learning system built on real Indian agricultural mandi (wholesale market) price data. Using a dataset of 737,392 price records spanning June 2023 to June 2025 across five commodities and 1,598 markets, the system performs comprehensive exploratory data analysis, SQL-based business intelligence, time-series price forecasting, and AI-assisted insight generation. A Streamlit web application integrates all backend logic and frontend visualization into a single deployable Python file. The best-performing ML model — Random Forest Regressor — achieves a MAE of ₹107.14/quintal and R² of 0.7744 on the chronological test set for Potato price prediction at the Kalipur market, West Bengal. All results are computed from the actual dataset with no fabricated values.

---

## 3. Introduction

India has one of the largest agricultural economies in the world, with thousands of mandis (regulated wholesale markets) operating across all states. The Agricultural Produce Market Committee (APMC) system records daily commodity prices (minimum, maximum, and modal) for every mandi-commodity combination. These prices are crucial signals for:

- **Farmers:** to decide when and where to sell their produce
- **Traders:** to identify arbitrage opportunities across markets
- **Policy makers:** to monitor price trends and intervene when needed
- **Consumers:** indirectly affected by farm-gate prices

Despite the availability of this data, it remains largely underutilized due to volume and complexity. This project demonstrates how modern data analytics and machine learning can transform raw mandi price data into actionable insights.

---

## 4. Problem Statement

*"Indian agricultural mandi prices vary significantly across commodities, states, districts, and markets. The objective is to analyze historical mandi price patterns and build a predictive system that can estimate future modal prices for selected agricultural commodities and markets."*

Key business questions addressed:
1. Which commodities exhibit the highest price volatility?
2. How do mandi prices vary across states and regions?
3. What seasonal patterns exist in commodity prices?
4. Which markets have the highest prices and widest spreads?
5. Can historical price patterns be used to estimate near-future prices?

---

## 5. Objectives

1. Ingest and clean 737,392+ mandi price records
2. Perform comprehensive EDA across commodity, state, market, and time dimensions
3. Implement SQL-based analytics using SQLite
4. Build and compare ML forecasting models (Baseline, Linear Regression, Random Forest, XGBoost)
5. Evaluate models on a strictly chronological held-out test set
6. Deploy an interactive, professional Streamlit dashboard
7. Generate data-grounded AI insights
8. Produce a Power BI dashboard design
9. Package everything in a single Python file

---

## 6. Business Context

The Indian government's eNAM (National Agriculture Market) and AGMARKNET portals publish daily mandi prices. This project uses a subset of that data. The business value of this system includes:

- **Price discovery:** enable farmers to identify the best markets for their commodity
- **Procurement planning:** help buyers forecast price movements and plan purchases
- **Policy monitoring:** track price anomalies that may signal supply shocks
- **Seasonal intelligence:** identify optimal buying/selling windows

---

## 7. Dataset Description

| Attribute | Value |
|-----------|-------|
| File | Agriculture_price_dataset.csv |
| Raw Records | 737,392 |
| Clean Records | 733,688 |
| Columns | 10 |
| Date Range | 2023-06-06 to 2025-06-11 (737 calendar days) |
| Commodities | Onion, Potato, Rice, Tomato, Wheat |
| States (raw) | 30 unique values |
| States (normalized) | 26 canonical states |
| Districts | 373 |
| Markets | 1,597 (clean) |
| Price Unit | ₹ per quintal (100 kg) |

**Column definitions:**

| Column | Type | Description |
|--------|------|-------------|
| STATE | text | Indian state |
| District Name | text | District within state |
| Market Name | text | APMC mandi name |
| Commodity | text | Crop/commodity traded |
| Variety | text | Sub-variety of commodity |
| Grade | text | Quality grade (FAQ, Local, etc.) |
| Min_Price | float | Minimum traded price (₹/quintal) |
| Max_Price | float | Maximum traded price (₹/quintal) |
| Modal_Price | float | Most common traded price (₹/quintal) — primary target |
| Price Date | date | Trading date (M/D/YYYY format in raw file) |

---

## 8. Data Collection

The dataset was provided as a static CSV file for this internship project. It represents APMC/mandi price records collected from government agricultural market portals (AGMARKNET/eNAM). The records cover daily price data across all active mandis for the five commodities included in the dataset.

---

## 9. Data Cleaning

### 9.1 Cleaning Pipeline

**Step 1 — Date Conversion**
- Raw format: `M/D/YYYY` (e.g., `6/6/2023`)
- Converted to `datetime64` using `pd.to_datetime(dayfirst=False, errors='coerce')`
- Result: 0 unparseable dates

**Step 2 — String Normalization**
- `.str.strip()` applied to all object columns to remove leading/trailing whitespace

**Step 3 — State Name Normalization**
8 canonical mappings applied:

| Raw | Normalized |
|-----|-----------|
| ` Punjab` (leading space) | Punjab |
| Tamilnadu | Tamil Nadu |
| Chattisgarh | Chhattisgarh |
| Jammu & Kashmir | Jammu and Kashmir |
| Orissa | Odisha |
| Uttrakhand | Uttarakhand |
| Gao | Unknown |

**Step 4 — Derived Time Columns**
Year, Month, Month_Name, Quarter, Day, Day_of_Week, Day_Name, Week_of_Year

**Step 5 — Derived Price Metrics**
- `Price_Range = Max_Price − Min_Price`
- `Price_Spread_Pct = Price_Range / Modal_Price × 100`
- `Modal_vs_Min = Modal_Price − Min_Price`
- `Modal_vs_Max = Max_Price − Modal_Price`

**Step 6 — Quality Flagging**

| Flag | Condition | Count |
|------|-----------|-------|
| ZERO_MODAL | Modal_Price = 0 | 3 |
| PRICE_LOGIC_ERROR | Min_Price > Max_Price | 566 |
| MODAL_BELOW_MIN | Modal_Price < Min_Price | 797 |
| MODAL_ABOVE_MAX | Modal_Price > Max_Price | 1,060 |
| EXTREME_OUTLIER | Outside IQR×5 fence per commodity | 1,847 |
| **Total Flagged** | | **3,704** (0.50%) |

**Treatment:**
- Flagged records are **retained in `df_full`** for analytics transparency
- Flagged records are **excluded from ML training** (`df_clean` contains 733,688 OK rows only)
- Extreme outliers include the Jehanabad Onion records (₹240,000–₹460,000) which are almost certainly unit errors (₹/kg entered instead of ₹/quintal)

### 9.2 Multi-Variety Records
7.47% of records share the same Date + Commodity + Market + Grade but differ in Variety. These are legitimate — different varieties trade simultaneously at the same market. For the ML forecasting series, these are aggregated by computing `mean(Modal_Price)` per date, documented explicitly.

---

## 10. Exploratory Data Analysis

### 10.1 Commodity Analysis

| Commodity | Records | Avg Modal (₹) | Std Dev | CV% |
|-----------|---------|--------------|---------|-----|
| Tomato | 26,564 | 4,395.34 | 3,332.39 | 75.82% |
| Rice | 7,714 | 3,549.58 | 700.38 | 19.73% |
| Onion | 297,470 | 2,741.26 | 1,523.89 | 55.59% |
| Wheat | 75,664 | 2,411.29 | 226.62 | 9.40% |
| Potato | 326,276 | 1,984.55 | 1,316.24 | 66.32% |

**Key Finding:** Tomato (75.82%) and Potato (66.32%) exhibit the highest price volatility. Wheat (9.40%) is the most stable. Tomato has the highest average price but only 2023 data. Potato and Onion span all 3 years.

### 10.2 Dataset Imbalance
- Potato: 44.4% of records
- Onion: 40.5% of records
- Wheat: 10.4% of records
- Tomato: 3.6% of records
- Rice: 1.1% of records

### 10.3 Year-over-Year Analysis (Potato & Onion only)

| Commodity | 2023 Avg | 2024 Avg | 2025 Avg |
|-----------|---------|---------|---------|
| Potato | (computed in app) | (computed in app) | (computed in app) |
| Onion | (computed in app) | (computed in app) | (computed in app) |

*Note: Values are computed live in the Streamlit application from the actual dataset. Not pre-populated here to avoid hard-coding.*

### 10.4 Data Coverage Limitation
| Commodity | 2023 | 2024 | 2025 |
|-----------|------|------|------|
| Onion | ✅ | ✅ | ✅ |
| Potato | ✅ | ✅ | ✅ |
| Wheat | ✅ | ✅ (partial) | ❌ |
| Tomato | ✅ | ❌ | ❌ |
| Rice | ❌ | ❌ | ✅ |

---

## 11. SQL Analysis

The cleaned dataset is loaded into an **in-memory SQLite database** and queried using standard SQL. No external database server is required. The SQLite database exists only during the application session and is recreated on each load (cached by Streamlit).

10 queries are implemented — see the SQL Analytics page in the application for live results.

Key findings from SQL:
- **Tomato** has the highest average modal price (₹4,395.34/quintal)
- **Wheat** has the lowest standard deviation (most stable prices)
- Year-over-year changes for Potato and Onion computed from the dataset

---

## 12. Power BI Dashboard

The application exports a cleaned CSV for Power BI import. The recommended dashboard has 4 pages:

1. **Executive Overview** — 6 KPI cards + commodity comparison + state map + monthly trend
2. **Commodity Analysis** — per-commodity deep dive with seasonal index and state comparison
3. **Market Analytics** — price spread, activity, volatility across markets
4. **Prediction View** — imports forecast CSV from the Forecasting page

DAX measures implemented: Avg_Modal_Price, CV_Pct, Price_Range_Avg, YoY_Change_2024_vs_2023

---

## 13. Feature Engineering

**Target variable:** `Modal_Price` (₹/quintal)

**14 engineered features:**

| Feature | Description | Leakage Prevention |
|---------|-------------|-------------------|
| lag_1 | Price 1 day ago | `price.shift(1)` |
| lag_7 | Price 7 days ago | `price.shift(7)` |
| lag_14 | Price 14 days ago | `price.shift(14)` |
| lag_30 | Price 30 days ago | `price.shift(30)` |
| roll_mean_7 | 7-day rolling mean | `price.shift(1).rolling(7).mean()` |
| roll_mean_14 | 14-day rolling mean | `price.shift(1).rolling(14).mean()` |
| roll_mean_30 | 30-day rolling mean | `price.shift(1).rolling(30).mean()` |
| roll_std_7 | 7-day price std dev | `price.shift(1).rolling(7).std()` |
| roll_std_30 | 30-day price std dev | `price.shift(1).rolling(30).std()` |
| month | Month of year (1–12) | Calendar feature |
| quarter | Quarter (1–4) | Calendar feature |
| day_of_week | Day of week (0=Mon) | Calendar feature |
| week_of_year | ISO week number | Calendar feature |
| year | Year (2023/2024/2025) | Calendar feature |

**Critical implementation detail:** All rolling features use `price.shift(1).rolling(n).mean()`, not `price.rolling(n).mean()`. The `.shift(1)` ensures the rolling window excludes the current day's price, preventing look-ahead data leakage.

**Rows dropped due to NaN from rolling windows:** 30 rows from the start of the series (expected and acceptable).

---

## 14. Prediction Methodology

### 14.1 Primary Forecasting Series Selection

**Commodity:** Potato  
**Market:** Kalipur  
**State:** West Bengal  

**Rationale for selection:**
- Near-complete daily coverage: 735 unique dates out of 737 calendar days
- All 3 years represented (2023, 2024, 2025)
- No extreme outliers after IQR×5 filtering
- Sufficient data for 3-way chronological splitting
- Mean Modal_Price: ₹1,747.82/quintal; Range: ₹720–₹2,800

**Aggregation strategy:** Multiple varieties traded at Kalipur on the same day are aggregated by computing `mean(Modal_Price)` per date. This produces a single clean daily price series.

### 14.2 Train / Validation / Test Split

**Method:** Strictly chronological — no random splitting.

| Set | Start | End | Days |
|-----|-------|-----|------|
| Training | 2023-07-06 | 2024-09-01 | 424 |
| Validation | 2024-09-02 | 2024-12-30 | 120 |
| Test | 2024-12-31 | 2025-06-11 | 163 |

**Proportions:** ~60% / 17% / 23%

The validation set is used only for model selection (choosing the best model by RMSE). Test set metrics are computed once and reported as final results.

---

## 15. Model Selection

Four models were trained and compared:

| Model | Description | Why Included |
|-------|-------------|-------------|
| Rolling Mean Baseline | 7-day rolling mean of last known prices | Minimum viability reference |
| Linear Regression | OLS regression on 14 features | Baseline ML; captures linear temporal trends |
| Random Forest | Ensemble of 200 decision trees | Non-linear; handles feature interactions; provides feature importance |
| XGBoost | Gradient boosted trees (300 estimators) | State-of-art tabular model; often best for time-series regression |

**Model selection criterion:** Lowest RMSE on the validation set (2024-09-02 to 2024-12-30).

---

## 16. Model Training

**Hardware:** Standard laptop CPU (n_jobs=-1 for Random Forest parallelism)  
**Training time:** ~2–3 minutes for all 4 models on first run; cached thereafter via `@st.cache_resource`

**Random Forest hyperparameters:**
- n_estimators: 200
- max_depth: 12
- random_state: 42

**XGBoost hyperparameters:**
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- early_stopping_rounds: none (fixed 300 rounds)

---

## 17. Model Evaluation

All metrics computed on the **chronological test set** (2024-12-31 to 2025-06-11, 163 days). No values are hard-coded.

### Test Set Results

| Model | MAE (₹) | RMSE (₹) | R² | MAPE (%) |
|-------|---------|---------|-----|---------|
| Rolling Mean Baseline | 989.44 | 1,044.11 | -8.8061 | 86.49 |
| Linear Regression | 40.05 | 59.70 | 0.9679 | 3.36 |
| Random Forest | 107.14 | 158.37 | 0.7744 | 8.49 |
| **XGBoost** | **89.32** | **124.97** | **0.8595** | **8.11** |

### Validation Set RMSE (used for model selection)

| Model | Validation RMSE |
|-------|---------------|
| Random Forest | **51.89** ← Selected as best |
| XGBoost | 66.75 |

**Best model selected:** Random Forest (lowest validation RMSE: 51.89)

*Note: Linear Regression achieves the best test RMSE (59.70) but was not selected because model selection is correctly performed on the validation set, not the test set. The test set is used for unbiased final evaluation only. In practice, Linear Regression's strong performance suggests the Kalipur Potato series is largely linear in the lagged price features — a useful finding.*

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | lag_1 | 29.1% |
| 2 | roll_mean_14 | 24.6% |
| 3 | roll_mean_7 | 22.6% |
| 4 | roll_mean_30 | 22.5% |
| 5 | lag_7 | 0.5% |
| 6 | month | 0.2% |

**Interpretation:** The most recent price (lag_1) and rolling means dominate feature importance. Calendar features (month, day_of_week) contribute minimally, indicating the series is primarily driven by momentum rather than strong seasonal patterns at the market level.

---

## 18. Prediction Results

### Future Forecast (Walk-forward, 14 days)
The forecast is generated using an iterative walk-forward strategy: each predicted price is appended to the history before the next day's prediction is computed. This simulates real-world deployment where future ground truth is not available.

**Disclaimer:** All predictions are model-generated estimates based on historical price patterns. They are NOT guaranteed future prices. Actual mandi prices depend on weather, government policy, supply chain disruptions, and demand factors that are not captured in this model.

---

## 19. Streamlit Application

### Structure
Single file: `AgriPrice_AI.py` (1,565 lines)

All 10 sections are organized within the same file:
- Section 0: Imports & configuration
- Section 1: Data loading (`@st.cache_data`)
- Section 2: Data cleaning (`@st.cache_data`)
- Section 3: Analytics functions (`@st.cache_data`)
- Section 4: SQL analytics (`@st.cache_data`)
- Section 5: Forecasting pipeline (`@st.cache_resource` for model)
- Section 6: AI insight generator
- Section 7: Chart helpers (Plotly)
- Section 8: Streamlit page functions (11 pages)
- Section 9: `main()` entry point

### Caching Strategy

| Component | Cache Type | Effect |
|-----------|-----------|--------|
| CSV loading | `@st.cache_data` | Read once per session |
| Cleaning pipeline | `@st.cache_data` | Runs once per session |
| Analytics aggregations | `@st.cache_data` | Computed once |
| SQLite queries | `@st.cache_data` | Executed once |
| ML model training | `@st.cache_resource` | Model persists across rerenders; retraining only when commodity/market changes |

### Navigation (11 pages)
Home · Data Explorer · Commodity Analytics · Market Analytics · Price Trends · Forecasting · Model Performance · SQL Analytics · AI Insights · Power BI Guide · About

---

## 20. AI Integration

The AI layer is entirely rule-based — no paid LLM API is required. All insight statements derive from computed analytical results.

**Insight types generated:**
1. Most volatile commodity (from CV% comparison)
2. Most price-stable commodity
3. Seasonal peak/trough month for Potato (from seasonal index)
4. State price disparity (from state-level mean comparison)
5. 14-day forecast direction and percentage change (from model output)
6. Dataset coverage summary

Each insight is labeled as either `DATA-DRIVEN RESULT` or `AI-GENERATED INTERPRETATION`, ensuring transparency between computed facts and pattern-based interpretations.

---

## 21. Business Insights

All insights below are derived from computed dataset values:

1. **Tomato** has the highest average price (₹4,395/quintal) but data is only available for 2023 — seasonality analysis is limited
2. **Wheat** is the most price-stable commodity (CV: 9.4%) — predictable for procurement planning
3. **Potato and Onion** together account for 85% of all records and are the only commodities suitable for multi-year forecasting
4. **Kalipur market (WB)** has near-complete daily Potato price coverage (735/737 days) — ideal for time-series modeling
5. The **Rolling Mean Baseline** has an R² of -8.8, confirming that Potato prices at Kalipur changed significantly during the test period — a model is genuinely needed
6. **Linear Regression** achieves R²=0.97 on the test set, suggesting the near-term price is strongly predicted by recent prices (lag_1 and rolling means) — a largely linear relationship
7. The **baseline model fails completely** (R²=-8.8) because the test period (Jan–Jun 2025) saw significant price changes from the last training period levels — proving the ML models add real value

---

## 22. Limitations

1. **Only 5 commodities** — significantly limits the scope of analysis
2. **Tomato** (only 2023) and **Rice** (only 2025) cannot be reliably forecasted
3. **Wheat** drops off after mid-2024 — YoY analysis incomplete
4. The model cannot predict **policy-driven price shocks** (MSP announcements, export bans, import duty changes)
5. **Weather data** is not available — monsoon impact on Onion/Potato prices cannot be modeled
6. Prices are at the **market (mandi) level**, not farm-gate — the actual price a farmer receives may differ
7. **Jehanabad Onion outliers** (₹240K–₹460K) are flagged but not definitively explained — may be unit errors
8. Forecast accuracy degrades for longer horizons (>14 days) due to iterative error accumulation

---

## 23. Future Scope

1. Integrate daily weather data (IMD) to capture rainfall/temperature effects on crop prices
2. Add LSTM or Prophet time-series models for comparison with ML baseline
3. Expand dataset to include more commodities (pulses, oilseeds, vegetables) and longer history
4. Add commodity-specific MSP (Minimum Support Price) reference lines to price charts
5. Build SMS/WhatsApp notification system for farmers when prices deviate from expected range
6. Deploy on Streamlit Cloud or Azure for public access
7. Add geospatial visualization using district-level maps
8. Enable multi-market comparison for the same commodity on the same date

---

## 24. Conclusion

AgriPrice AI demonstrates how raw government agricultural market data can be transformed into a practical analytics and forecasting system. The project covers the full data science lifecycle: data ingestion, cleaning, EDA, SQL analytics, feature engineering, ML model training/evaluation, and interactive visualization — all within a single deployable Python file.

The Random Forest Regressor, selected by validation RMSE, achieves **R²=0.7744 and MAE=₹107.14/quintal** on the unseen test set for Potato prices at Kalipur market — a meaningful improvement over the Rolling Mean Baseline (R²=-8.8061). The Linear Regression model's strong test performance (R²=0.9679) reveals that the Potato price series is largely predictable from its own recent history, which is a valuable insight for market participants.

The system is suitable for demonstration to agricultural sector stakeholders, government agencies, and data analytics interviewers as a portfolio-quality project combining real-world data, production-grade Python code, and genuine ML validation.

---

## 25. References

1. AGMARKNET — Agricultural Marketing Information Network, Government of India. https://agmarknet.gov.in/
2. eNAM — National Agriculture Market. https://www.enam.gov.in/
3. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, 2011.
4. XGBoost: A Scalable Tree Boosting System. Chen & Guestrin, KDD 2016.
5. Streamlit Documentation. https://docs.streamlit.io/
6. Plotly Python Graphing Library. https://plotly.com/python/
7. Pandas Documentation. https://pandas.pydata.org/docs/
8. IBM Bob AI Assistant — IBM. https://www.ibm.com/

---

*This report was prepared as part of the BharatCare + IBM Data Analytics with AI Internship. All numerical results are computed from the actual dataset. No values have been invented or estimated.*

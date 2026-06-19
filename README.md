# Nexus Demand Forecaster & Sales Analytics System

Nexus Demand Forecaster is a production-grade, interactive machine learning forecasting and regression web dashboard. It enables warehouse managers, supply chain specialists, and business analysts to ingest transactional historical records, run automated temporal feature engineering pipelines, fit predictive regression estimators, and run dynamic price elasticity simulations.

---

## 🏗️ System Architecture

```
            ┌──────────────────────────────────────┐
            │          Glassmorphic Web            │
            │          Dashboard (HTML/JS)         │
            └──────────────────┬───────────────────┘
                               │ (REST API / Forms)
                               ▼
            ┌──────────────────────────────────────┐
            │             Flask Server             │
            └──────────┬───────────────────────┬───┘
                       │                       │
                       ▼ (Inference / Train)   │ (Auth / Logs)
            ┌──────────────────────┐           │
            │    ML Forecaster     │           │
            │ (scikit-learn Pipeline)          ▼
            └──────────────────────┘  ┌──────────────────────┐
                                      │   CRUD Service Layer │
                                      └──────────┬───────────┘
                                                 ▼
                                      ┌──────────────────────┐
                                      │  SQLAlchemy 2.0 ORM  │
                                      └──────────┬───────────┘
                                                 ▼ (PRAGMA foreign_keys=ON)
                                      ┌──────────────────────┐
                                      │ SQLite Database File │
                                      │     (forecast.db)    │
                                      └──────────────────────┘
```

---

## ⚡ Key Machine Learning & Application Features

### 1. Robust Time-Series Feature Engineering
To predict tomorrow's sales quantities (`units_sold`), the system generates descriptive historical features from the raw transactional stream:
- **Autoregressive Lags**: Lags of 1 to 7 days (`sales_lag_1` to `sales_lag_7`) and 14 days to capture weekly seasonality cycles.
- **Historical Rolling Windows**: 7-day and 30-day moving averages (`rolling_mean_7`, `rolling_mean_30`) and rolling standard deviation to capture current demand levels and variance. All rolling statistics are strictly offset by 1 day to prevent leakage of the current target value into the features.
- **Cyclical Calendar Features**: Extraction of `day_of_week`, `month`, `day_of_year`, and `is_weekend` to represent weekly and yearly seasonal peaks.
- **Price Elasticity & Promotions**: Integration of average pricing (`unit_price`) and promotional triggers (`is_promo`) to capture pricing sensitivity.

### 2. Time-Based Validation Split (Zero-Leakage)
Standard random train/test splits violate chronological ordering and cause **data leakage** in time-series forecasting. Nexus OMS enforces a **strictly chronological split**:
- Training set is defined as the first $N\%$ of calendar days.
- Validation/Test set is defined as the final $(100 - N)\%$ of calendar days.
- Model performance metrics (RMSE, MAE, R²) are evaluated strictly on this validation set to verify real-world forecast accuracy.

### 3. Glassmorphic Single Page App (SPA)
The web interface features premium CSS glassmorphism, responsive grids, and rich interactive layouts:
- **Interactive Forecast Chart**: Integrated dual-axis Chart.js plotting historical sales values and projecting future 30-day demand.
- **Elasticity Simulator**: Drag-and-drop sliders allowing users to adjust pricing and toggle promos, generating real-time predictions of quantity sold.
- **Model Console**: Select machine learning algorithms (Ridge Regression vs. Random Forest Regressor), set training hyper-parameters, and retrain in a single click with validation visualizers.
- **CSV Data Ingester**: Validate and batch-load raw CSV reports into SQLite.

---

## 📁 Repository Structure
```
pg5/
├── data/
│   └── forecast.db            # SQLite database file containing transaction logs & users
├── models/
│   └── forecaster_pipeline.joblib  # Serialized scikit-learn model pipeline
├── src/
│   ├── database.py            # SQLite engine, sessions local factory, event listeners
│   ├── models.py              # User and SalesTransaction ORM mappings
│   ├── schemas.py             # Input validation schemas (Pydantic v2)
│   ├── crud.py                # Database queries and transaction helpers
│   ├── forecaster.py          # ML feature engineering, training, recursive forecasting
│   ├── app.py                 # Flask server routes and API controllers
│   └── static/                # Single-page frontend assets
│       ├── index.html         # Jinja2-enabled main HTML template
│       ├── styles.css         # Glassmorphic premium CSS styling
│       └── app.js             # API bindings, reactive elements, Chart.js drawings
├── verify_ml.py               # Automated pipeline validation suite (leakage checking)
├── requirements.txt           # Python packages list
└── README.md                  # Project manual (this file)
```

---

## 🚀 Operations and Installation Manual

### 1. Installation
Ensure Python 3.10+ is installed. Run:
```bash
# Install all required libraries
pip install -r requirements.txt
```

### 2. Run Automated Test Validation Suite
Run the test suite to verify that database triggers, authentication password hashing, feature engineering, and temporal split validation are all functioning correctly:
```bash
python verify_ml.py
```

### 3. Launch the Management Console CLI
Nexus Demand Forecaster includes an interactive rich CLI to manage operations:
```bash
python src/cli.py
```
**Options inside CLI:**
1. **Seed Historical Sales Data**: Generates 2 years of synthetic sales data with weekly seasonality (peaks on Friday/Saturday), holiday seasonal spikes (peaks in December), pricing promotions, price elasticity, and random noise.
2. **Train Forecasting Models**: Fits the forecasting model and prints R², RMSE, MAE, and feature importances.
3. **Create Admin Credentials**: Setup administrative credentials manually.
4. **Start Flask Web Server**: Runs the web app dashboard at `http://127.0.0.1:8000`.

### 4. Direct CLI Arguments
You can bypass the interactive menu by invoking operations directly:
```bash
# Seed the database
python src/cli.py seed

# Train the machine learning model
python src/cli.py train

# Create a custom user account
python src/cli.py admin

# Start the Flask web server
python src/cli.py server
```
Once the server is running, open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

*Default login credentials (auto-created on start if empty):*
- **Username:** `admin`
- **Password:** `admin`

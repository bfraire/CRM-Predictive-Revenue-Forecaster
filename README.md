# CRM Revenue Forecasting - Predictive Model & Dashboard 

A full end-to-end machine learning project that forecasts CRM revenue for 2025 using four years of synthetic opportunity data (2021–2024). 
Built in three phases: data preparation, modelling, and an interactive Streamlit dashboard.

## Goal 
The purpose of this project was to generate a revenue forecasting solution for CRM metrics. This project proposes using a predictive model
to analyze historical CRM record keeping and present insights in a streamlined dashboard. The key deliverable is a dashboard with the 
following KPIs: total forecasted revenue for next year, forecasted revenue per day and forecasted revenue per product. 

## Phases
### Phase 1 - Data Preparation 
[View Notebook](data_prepping.ipynb)
- Filters Won opportunities from raw CRM data
- Joins accounts (customer region), products (series), and sales teams
- Engineers temporal features: month, day of week, year, quarter end flag
- Aggregates to daily revenue by product × office location × deal type
- Outputs train_daily_revenue.csv and test_future_dates.csv

### Phase 2 - Modelling 
[View Notebook](data_model.ipynb)
- Categorical features were one-hot encoded and a time-based train/validation split was applied (2021–2023 train, 2024 validate). XGBoost and Random Forest were both 
tuned via hyperparameter grid search and evaluated on RMSE, MAE, and R². 

#### XGBoost 
XGBoost builds trees sequentially — each tree corrects the errors of the previous one 
by focusing on residuals. This makes it highly accurate on complex datasets but more 
sensitive to hyperparameter settings and prone to overfitting if not carefully tuned.

##### Random Forest
Random Forest builds trees independently in parallel and averages their predictions. 
This makes it more stable and less prone to overfitting, particularly on datasets where 
the signal is clean and consistent rather than highly complex.

| Model | Best Params | Validation RMSE |
|---|---|---|
| XGBoost | n_estimators=75, max_depth=7 | $8,065 |
| Random Forest | n_estimators=150, max_depth=7 | $6,985 |

Random Forest was selected as the winning model with an R² of 0.997, driven primarily by enterprise 
product features (Analytics Suite, CoreCRM Enterprise). Predictions generated for all 2025 dates and merged with historical actuals into a single combined forecast file.


## Phase 3 - Dashboard
- A Streamlit application visualises historical actuals (2021–2024) alongside the 2025 machine learning forecast. Users can filter by date range, product, office 
location, and deal type. Three chart views are available — Monthly Revenue Trend, Daily Revenue, and Top Products — alongside KPI cards for Total Revenue, Total Deals, and 
Average Deal Size.
<p align="center">
  <img src="images/Screenshot 2026-05-19 185450.png" width="100%">
</p>
<p align="center">
  <img src="images/Screenshot 2026-05-19 185923.png" width="100%">
</p>
<p align="center">
  <img src="images/Screenshot 2026-05-19 185959.png" width="100%">
</p>

## Assumptions 
- The top 50 most active product × location × deal type combinations were selected for 
  the 2025 forecast template. Segments with sparse historical data were excluded to avoid 
  unreliable predictions.
- Analysis and forecasting were performed exclusively on Won opportunities, as they 
  represent realised revenue. Open, Lost, and in-progress deals were excluded.
- No real CRM dataset was available for this project. 
  The synthetic generator was designed to mirror realistic B2B SaaS patterns as closely 
  as possible. Replacing the CSV files with actual Salesforce or HubSpot exports would 
  require minimal schema changes.

## Barriers
- **Synthetic data quality (v1)** — The initial dataset had close dates distributed 
  randomly with no seasonality, quarter-end clustering, or year-over-year growth. This 
  produced a flat, unrealistic revenue signal that resulted in a poor model fit (R² of 
  0.20). The data generator was rebuilt from scratch with realistic B2B patterns before 
  modelling could proceed.
- **Boundary spike** — The close date generator clamped out-of-range dates to a single 
  boundary date (Dec 31), creating an artificial revenue spike of ~$720k on one day. This 
  was resolved by spreading overflow dates across the last two weeks of December and 
  applying 99th percentile winsorization in the modelling pipeline.
- **Feature redundancy** — avg_deal_size (correlation 0.99 with total_revenue) and 
  quarter (correlation 0.97 with month) were identified via correlation heatmap and 
  dropped before modelling to prevent the model from learning spurious relationships.

-------
## Project Structure

Predictive_Model_Revenue_CRM/
├── data/
│   ├── opportunities.csv          # 14,343 synthetic CRM opportunities (2021–2024)
│   ├── accounts.csv               # 80 accounts with region and sector attributes
│   ├── leads.csv                  # 1,500 leads with source and conversion tracking
│   ├── products.csv               # 12 products across 5 series with list prices
│   └── sales_teams.csv            # 35 sales agents across 6 regional offices
├── data_prepping.ipynb            # Phase 1 — data aggregation and feature engineering
├── Modelling.ipynb                # Phase 2 — model training, tuning, and evaluation
├── CRM_Dashboard.py               # Phase 3 — interactive Streamlit dashboard
├── train_daily_revenue.csv        # Aggregated training dataset (Won deals, 2021–2024)
├── test_future_dates.csv          # Future date template for 2025 forecasting
└── final_combined_forecast.csv    # Historical actuals + 2025 predictions 

## Tech Stack

- Python 3.x — pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, streamlit, altair

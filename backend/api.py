from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib


# =========================================================
# APP CONFIG
# =========================================================

app = FastAPI(
    title="1-Day-Ahead Retail Sales Forecasting API",
    description="XGBoost-based retail sales forecasting API",
    version="1.0.0"
)


# Allow Streamlit to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "xgboost_sales_forecasting.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)

metadata = joblib.load(
    "model_metadata.pkl"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "train_processed.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.sort_values(
    ["store_nbr", "item_nbr", "date"]
).reset_index(drop=True)


# =========================================================
# REQUEST SCHEMA
# =========================================================

class PredictionRequest(BaseModel):

    store_nbr: int
    item_nbr: int
    forecast_date: str
    onpromotion: int


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():

    return {
        "message": "1-Day-Ahead Retail Sales Forecasting API",
        "status": "running",
        "model": "XGBoost"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_features(
    store_nbr,
    item_nbr,
    forecast_date,
    promotion
):

    forecast_date = pd.Timestamp(
        forecast_date
    )

    # ---------------------------------------------
    # Get historical data for selected product
    # ---------------------------------------------

    data = df[
        (df["store_nbr"] == store_nbr)
        &
        (df["item_nbr"] == item_nbr)
    ].copy()

    data = data.sort_values("date")

    if len(data) < 30:

        raise ValueError(
            "Not enough historical data for this "
            "store-product combination."
        )

    # ---------------------------------------------
    # Make sure forecast date is after history
    # ---------------------------------------------

    historical_data = data[
        data["date"] < forecast_date
    ].copy()

    if len(historical_data) < 30:

        raise ValueError(
            "The selected forecast date does not have "
            "at least 30 days of historical data."
        )

    # ---------------------------------------------
    # Last 30 observations
    # ---------------------------------------------

    sales = historical_data[
        "sales_target"
    ]

    promotion_history = historical_data[
        "onpromotion"
    ]

    row = {}

    # ---------------------------------------------
    # Store & item
    # ---------------------------------------------

    row["store_nbr"] = store_nbr
    row["item_nbr"] = item_nbr

    # ---------------------------------------------
    # Calendar features
    # ---------------------------------------------

    row["year"] = forecast_date.year
    row["month"] = forecast_date.month
    row["day"] = forecast_date.day
    row["day_of_week"] = forecast_date.dayofweek
    row["week_of_year"] = int(
        forecast_date.isocalendar().week
    )
    row["is_weekend"] = int(
        forecast_date.dayofweek >= 5
    )

    # ---------------------------------------------
    # Holiday
    # ---------------------------------------------

    holiday_data = df[
        df["date"] == forecast_date
    ]

    if len(holiday_data) > 0:

        row["is_holiday"] = int(
            holiday_data["is_holiday"].max()
        )

    else:

        # If future date is not present in the
        # processed dataset, assume non-holiday.
        row["is_holiday"] = 0

    # ---------------------------------------------
    # Promotion
    # ---------------------------------------------

    row["onpromotion"] = promotion

    # ---------------------------------------------
    # Sales lag features
    # ---------------------------------------------

    row["lag_1"] = sales.iloc[-1]
    row["lag_7"] = sales.iloc[-7]
    row["lag_14"] = sales.iloc[-14]
    row["lag_21"] = sales.iloc[-21]
    row["lag_28"] = sales.iloc[-28]
    row["lag_30"] = sales.iloc[-30]

    # ---------------------------------------------
    # Rolling features
    # ---------------------------------------------

    row["rolling_mean_7"] = (
        sales.iloc[-7:].mean()
    )

    row["rolling_mean_30"] = (
        sales.iloc[-30:].mean()
    )

    row["rolling_median_7"] = (
        sales.iloc[-7:].median()
    )

    row["rolling_std_7"] = (
        sales.iloc[-7:].std()
    )

    # ---------------------------------------------
    # Promotion history
    # ---------------------------------------------

    row["promotion_lag_1"] = (
        promotion_history.iloc[-1]
    )

    row["promotion_rolling_7"] = (
        promotion_history.iloc[-7:].sum()
    )

    # ---------------------------------------------
    # Transaction features
    # ---------------------------------------------

    if "transactions_lag_1" in historical_data.columns:

        row["transactions_lag_1"] = (
            historical_data[
                "transactions_lag_1"
            ].iloc[-1]
        )

    if "transactions_lag_7" in historical_data.columns:

        row["transactions_lag_7"] = (
            historical_data[
                "transactions_lag_7"
            ].iloc[-1]
        )

    if "transactions_rolling_7" in historical_data.columns:

        row["transactions_rolling_7"] = (
            historical_data[
                "transactions_rolling_7"
            ].iloc[-1]
        )

    # ---------------------------------------------
    # Other features
    # ---------------------------------------------

    for col in [
        "oil_price",
        "perishable",
        "cluster",
        "class"
    ]:

        if col in historical_data.columns:

            row[col] = historical_data[
                col
            ].iloc[-1]

    # ---------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------

    feature_df = pd.DataFrame([row])

    # ---------------------------------------------
    # Make feature order identical to training
    # ---------------------------------------------

    feature_df = feature_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Replace potential NaN
    feature_df = feature_df.fillna(0)

    return feature_df


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(
    request: PredictionRequest
):

    try:

        # -----------------------------------------
        # Validate promotion
        # -----------------------------------------

        if request.onpromotion not in [0, 1]:

            raise HTTPException(
                status_code=400,
                detail="onpromotion must be 0 or 1."
            )

        # -----------------------------------------
        # Build model features
        # -----------------------------------------

        X = build_features(
            store_nbr=request.store_nbr,
            item_nbr=request.item_nbr,
            forecast_date=request.forecast_date,
            promotion=request.onpromotion
        )

        # -----------------------------------------
        # Predict
        # -----------------------------------------

        prediction_log = model.predict(X)[0]

        # Model was trained using log1p(target)
        prediction = np.expm1(
            prediction_log
        )

        prediction = max(
            0,
            float(prediction)
        )

        # -----------------------------------------
        # Response
        # -----------------------------------------

        return {
            "store_nbr": request.store_nbr,
            "item_nbr": request.item_nbr,
            "forecast_date": request.forecast_date,
            "onpromotion": request.onpromotion,
            "predicted_sales": round(
                prediction,
                2
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
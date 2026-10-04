import streamlit as st
import pandas as pd
import requests


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# API CONFIG
# =========================================================

API_URL = "http://127.0.0.1:8000"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "../backend/train_processed.csv"
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


df = load_data()


# =========================================================
# HEADER
# =========================================================

st.title("📈 Retail Sales Forecasting")

st.write(
    "Predict product sales using an XGBoost "
    "machine learning model."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Forecast Settings"
)


# Store selection
stores = sorted(
    df["store_nbr"]
    .dropna()
    .unique()
)

selected_store = st.sidebar.selectbox(
    "Store",
    stores
)


# Product selection
store_data = df[
    df["store_nbr"] == selected_store
]

items = sorted(
    store_data["item_nbr"]
    .dropna()
    .unique()
)

selected_item = st.sidebar.selectbox(
    "Product",
    items
)


# Product data
item_data = store_data[
    store_data["item_nbr"] == selected_item
].copy()


if len(item_data) == 0:

    st.error(
        "No data available for this product."
    )

    st.stop()


# =========================================================
# FORECAST DATE
# =========================================================

latest_date = item_data[
    "date"
].max()

forecast_date = st.sidebar.date_input(
    "Forecast Date",
    value=latest_date + pd.Timedelta(days=1)
)

latest_date = df["date"].max().date()

st.write(f"Latest available data: **{latest_date}**")

forecast_date = latest_date + pd.Timedelta(days=1)

st.info(
    f"Forecast horizon: **1 day ahead**\n\n"
    f"Prediction date: **{forecast_date}**"
)
# =========================================================
# PROMOTION
# =========================================================

promotion = st.sidebar.selectbox(
    "On Promotion?",
    [0, 1],
    format_func=lambda x:
        "Yes" if x == 1 else "No"
)


# =========================================================
# PRODUCT SUMMARY
# =========================================================

st.subheader(
    "Product Overview"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Store",
        selected_store
    )


with col2:

    st.metric(
        "Product",
        selected_item
    )


with col3:

    st.metric(
        "Historical Sales",
        f"{len(item_data):,}"
    )


with col4:

    st.metric(
        "Last Date",
        latest_date.strftime(
            "%Y-%m-%d"
        )
    )


# =========================================================
# HISTORICAL CHART
# =========================================================

st.subheader(
    "Historical Sales"
)

chart_data = item_data[
    ["date", "sales_target"]
].tail(90)

chart_data = chart_data.set_index(
    "date"
)

st.line_chart(
    chart_data["sales_target"]
)


# =========================================================
# PREDICTION
# =========================================================

st.subheader(
    "Sales Forecast"
)


if st.button(
    "🔮 Predict Sales",
    type="primary"
):

    payload = {

        "store_nbr": int(
            selected_store
        ),

        "item_nbr": int(
            selected_item
        ),

        "forecast_date":
            str(forecast_date),

        "onpromotion": int(
            promotion
        )
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=60
        )

        # ---------------------------------------------
        # Successful response
        # ---------------------------------------------

        if response.status_code == 200:

            result = response.json()

            prediction = result[
                "predicted_sales"
            ]

            st.success(
                "Prediction completed!"
            )

            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.metric(
                    "Predicted Sales",
                    f"{prediction:.2f} units"
                )


            with result_col2:

                st.metric(
                    "Forecast Date",
                    str(forecast_date)
                )


            # -----------------------------------------
            # Prediction summary
            # -----------------------------------------

            st.info(
                f"The model predicts approximately "
                f"**{prediction:.2f} units** of product "
                f"**{selected_item}** will be sold at "
                f"store **{selected_store}** on "
                f"**{forecast_date}**."
            )


        else:

            try:

                error_detail = response.json()[
                    "detail"
                ]

            except:

                error_detail = response.text

            st.error(
                f"API Error: {error_detail}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the FastAPI backend. "
            "Make sure the backend is running."
        )


    except requests.exceptions.Timeout:

        st.error(
            "The API request timed out."
        )


    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader(
    "🤖 Model Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.write("**Model**")
    st.write("XGBoost")


with col2:

    st.write("**Prediction Target**")
    st.write("Daily Product Sales")


with col3:

    st.write("**Backend**")
    st.write("FastAPI")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Retail Sales Forecasting | "
    "XGBoost + FastAPI + Streamlit"
)
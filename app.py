import streamlit as st
import pandas as pd
import plotly.express as px

# Sample DataFrame (Replace with real data)
macrotable_df = pd.DataFrame({
    "shop_id": [1, 2, 3],
    "total_traffic": [10000, 15000, 20000],
    "prev_year_total_traffic": [9500, 14000, 19000],
    "ytd_total_traffic": [8000, 13000, 17000],
    "locals_new_effectiveness": [0.05, 0.06, 0.07],
    "prospect_generation": [0.03, 0.035, 0.04],
    "prospect_effectiveness": [0.4, 0.42, 0.45],
    "local_come_back": [0.2, 0.22, 0.25],
    "tourist_new_effectiveness": [0.08, 0.09, 0.1],
    "tourist_come_back": [0.15, 0.18, 0.2],
    "avg_amt_ticket": [50, 55, 60],
    "avg_num_ticket_per_customer": [1.5, 1.8, 2.0],
    "db_buyers_locals": [500, 700, 900],
    "db_buyers_tourist": [300, 500, 600],
    "prev_year_sales": [600000, 800000, 1000000],
    "ytd_sales": [450000, 600000, 750000],
    "budget_sales": [650000, 850000, 1100000],
})

# Function to calculate sales
def calculate_sales(df):
    return (
        (df["total_traffic"] * df["locals_new_effectiveness"]) +
        (df["total_traffic"] * df["prospect_generation"]) +
        (df["db_buyers_locals"] * df["prospect_effectiveness"]) +
        (df["local_come_back"] * df["db_buyers_locals"]) +
        (df["total_traffic"] * df["tourist_new_effectiveness"]) +
        (df["tourist_come_back"] * df["db_buyers_tourist"])
    ) * df["avg_amt_ticket"] * df["avg_num_ticket_per_customer"]

# Sidebar: Select Shop ID
shop_id = st.sidebar.selectbox("Select Shop ID", macrotable_df["shop_id"])
shop_data = macrotable_df[macrotable_df["shop_id"] == shop_id].copy()

st.sidebar.markdown("### What is your target category?")
team_type = st.sidebar.radio(
    "Select Team Type",
    ["Traffic", "Locals", "Tourist", "Avg Ticket"]
)

# **Quarterly Seasonality Adjustment**
st.sidebar.markdown("### Adjust Quarterly Traffic Distribution")
q1_multiplier = st.sidebar.slider("Q1 (%)", 0.0, 1.0, 0.25)
q2_multiplier = st.sidebar.slider("Q2 (%)", 0.0, 1.0, 0.25)
q3_multiplier = st.sidebar.slider("Q3 (%)", 0.0, 1.0, 0.25)
q4_multiplier = st.sidebar.slider("Q4 (%)", 0.0, 1.0, 0.25)

# Ensure the sum is 100%
total_multiplier = q1_multiplier + q2_multiplier + q3_multiplier + q4_multiplier
if total_multiplier != 1.0:
    st.sidebar.error("⚠️ The sum of Q1-Q4 must be exactly 100% (1.0)")

# Adjust Traffic using Quarterly Multiplier
shop_data["adjusted_traffic"] = shop_data["total_traffic"] * total_multiplier

# **Adjust KPIs Based on Selected Team Type**
st.sidebar.markdown("### Adjust KPIs")
if team_type == "Traffic":
    shop_data["total_traffic"] = st.sidebar.slider("Total Traffic", 5000, 30000, int(shop_data["total_traffic"]))

elif team_type == "Locals":
    shop_data["locals_new_effectiveness"] = st.sidebar.slider("Locals New Effectiveness", 0.01, 0.1, float(shop_data["locals_new_effectiveness"]))
    shop_data["prospect_generation"] = st.sidebar.slider("Prospect Generation", 0.01, 0.1, float(shop_data["prospect_generation"]))
    shop_data["prospect_effectiveness"] = st.sidebar.slider("Prospect Effectiveness", 0.1, 0.6, float(shop_data["prospect_effectiveness"]))
    shop_data["local_come_back"] = st.sidebar.slider("Local Comeback", 0.1, 0.4, float(shop_data["local_

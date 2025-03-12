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
    shop_data["local_come_back"] = st.sidebar.slider("Local Comeback", 0.1, 0.4, float(shop_data["local_come_back"]))

elif team_type == "Tourist":
    shop_data["tourist_new_effectiveness"] = st.sidebar.slider("Tourist New Effectiveness", 0.05, 0.15, float(shop_data["tourist_new_effectiveness"]))
    shop_data["tourist_come_back"] = st.sidebar.slider("Tourist Comeback", 0.1, 0.4, float(shop_data["tourist_come_back"]))

elif team_type == "Avg Ticket":
    shop_data["avg_amt_ticket"] = st.sidebar.slider("Avg Ticket Amount", 20, 100, int(shop_data["avg_amt_ticket"]))
    shop_data["avg_num_ticket_per_customer"] = st.sidebar.slider("Avg Tickets per Customer", 1.0, 3.0, float(shop_data["avg_num_ticket_per_customer"]))

# Calculate Projected Sales
shop_data["projected_sales"] = calculate_sales(shop_data)


# Ensure real_sales and projected_sales are computed
shop_data["real_sales"] = calculate_sales(shop_data)
shop_data["projected_sales"] = calculate_sales(shop_data)

# Debugging: Print available columns
st.write("Available Columns in shop_data:", shop_data.columns.tolist())

# Handle missing columns
columns_to_include = ["prev_year_sales", "ytd_sales", "budget_sales", "real_sales", "projected_sales"]
kpi_comparison = shop_data[[col for col in columns_to_include if col in shop_data.columns]]

# Rename for display
kpi_comparison = kpi_comparison.rename(columns={
    "prev_year_sales": "Previous Year Sales",
    "ytd_sales": "YTD Sales",
    "budget_sales": "Budget Sales",
    "real_sales": "Current Sales",
    "projected_sales": "Projected Sales",
})

st.dataframe(kpi_comparison)



# **Create Stacked Bar Chart for Sales Comparison**
sales_comparison_df = pd.DataFrame({
    "Category": ["Previous Year", "YTD", "Budget", "Current", "Projected"],
    "Sales": [
        shop_data["prev_year_sales"].values[0], shop_data["ytd_sales"].values[0],
        shop_data["budget_sales"].values[0], shop_data["real_sales"].values[0],
        shop_data["projected_sales"].values[0]
    ],
    "Type": ["Historical", "Historical", "Goal", "Actual", "Forecast"]
})

fig = px.bar(
    sales_comparison_df,
    x="Category",
    y="Sales",
    color="Type",
    text="Sales",
    title=f"Sales Overview for Shop {shop_id}",
    labels={"Sales": "Sales Value", "Category": "Sales Type"},
    barmode="group"
)

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
st.plotly_chart(fig)

# **Additional Charts for KPI Breakdown**
st.markdown("## KPI Breakdown by Category")

category_df = pd.DataFrame({
    "Category": ["New Locals", "Prospect Locals", "Existing Locals", "New Tourist", "Existing Tourist"],
    "Metric Value": [
        shop_data["locals_new_effectiveness"].values[0],
        shop_data["prospect_generation"].values[0],
        shop_data["local_come_back"].values[0],
        shop_data["tourist_new_effectiveness"].values[0],
        shop_data["tourist_come_back"].values[0]
    ]
})

fig2 = px.bar(
    category_df,
    x="Category",
    y="Metric Value",
    title="Comparison of Different Customer Categories",
    text="Metric Value"
)

fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
st.plotly_chart(fig2)



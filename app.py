import streamlit as st
import pandas as pd

# Load or create initial data
macrotable_df = pd.DataFrame({
    "shop_id": [1, 2, 3],  # Example shops
    "total_traffic": [10000, 15000, 20000],
    "locals_new": [500, 700, 900],
    "prospect_generation": [0.05, 0.06, 0.07],  # Prospect/Traffic Ratio
    "prospect_effectiveness": [0.3, 0.4, 0.35],  # Prospect/Last 4 Months DB
    "tourist_new": [300, 500, 800],
    "avg_amt_ticket": [50, 55, 60],  # Avg ticket amount
    "avg_num_ticket_per_customer": [1.2, 1.3, 1.1],  # Avg tickets per customer
    "real_sales": [500000, 750000, 1000000]  # Example actual sales
})

# Create a copy for modifications
final_kpis_df = macrotable_df.copy()

# Streamlit UI
st.title("Sales KPI Optimization Tool")
st.sidebar.header("Adjust KPIs to Reach the Sales Budget")

# Create adjustable sliders for key KPIs
for col in ["total_traffic", "locals_new", "tourist_new", "avg_amt_ticket", "avg_num_ticket_per_customer"]:
    final_kpis_df[col] = st.sidebar.slider(
        f"Adjust {col}",
        min_value=int(macrotable_df[col].min() * 0.5),
        max_value=int(macrotable_df[col].max() * 1.5),
        value=int(macrotable_df[col].mean())
    )

# Sales Projection Calculation
final_kpis_df["projected_sales"] = (
    (final_kpis_df["locals_new"] + final_kpis_df["tourist_new"])
    * final_kpis_df["avg_amt_ticket"] * final_kpis_df["avg_num_ticket_per_customer"]
)

# Display Initial KPIs, Adjusted KPIs, and Projected Sales
st.subheader("Initial vs. Adjusted KPIs")
st.write("Compare baseline KPIs with adjustments.")
st.dataframe(final_kpis_df)

# Show projected sales vs. real sales
st.subheader("Projected vs. Real Sales")
st.write("See how KPI adjustments impact sales.")
st.bar_chart(final_kpis_df[["real_sales", "projected_sales"]].set_index(final_kpis_df["shop_id"]))

# KPI Effectiveness Table
st.subheader("Effectiveness Metrics")
final_kpis_df["prospect_generation_effectiveness"] = final_kpis_df["prospect_generation"] * final_kpis_df["prospect_effectiveness"]
st.dataframe(final_kpis_df[["shop_id", "prospect_generation_effectiveness"]])

st.success("Adjust KPIs in the sidebar and track changes in real-time!")

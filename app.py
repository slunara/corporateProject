import streamlit as st
import pandas as pd
import plotly.express as px

# Sample data
macrotable_df = pd.DataFrame({
    "shop_id": [1, 2, 3],  
    "total_traffic": [10000, 15000, 20000],
    "locals_new": [500, 700, 900],
    "prospect_generation": [0.05, 0.06, 0.07],  # Prospect/Traffic Ratio
    "prospect_effectiveness": [0.3, 0.4, 0.35],  # Prospect/Last 4 Months DB
    "tourist_new": [300, 500, 600],
    "avg_amt_ticket": [50, 55, 60],
    "avg_num_ticket_per_customer": [1.5, 1.8, 2.0],
})

# Function to calculate sales
def calculate_sales(df):
    return (
        (df["locals_new"] + df["tourist_new"]) * df["avg_amt_ticket"] * df["avg_num_ticket_per_customer"]
    )

# Add Real Sales column
macrotable_df["real_sales"] = calculate_sales(macrotable_df)

# Sidebar: Select Shop ID
st.sidebar.header("Select Shop & Adjust KPIs")
shop_id = st.sidebar.selectbox("Select Shop ID", macrotable_df["shop_id"])

# Filter selected shop
shop_data = macrotable_df[macrotable_df["shop_id"] == shop_id].copy()

# Sidebar: Adjust KPIs
st.sidebar.subheader("Adjust KPIs for Projection")
shop_data["total_traffic"] = st.sidebar.slider("Total Traffic", 5000, 30000, int(shop_data["total_traffic"]))
shop_data["locals_new"] = st.sidebar.slider("New Local Customers", 100, 1500, int(shop_data["locals_new"]))
shop_data["tourist_new"] = st.sidebar.slider("New Tourist Customers", 100, 1500, int(shop_data["tourist_new"]))
shop_data["avg_amt_ticket"] = st.sidebar.slider("Avg Ticket Amount", 20, 100, int(shop_data["avg_amt_ticket"]))
shop_data["avg_num_ticket_per_customer"] = st.sidebar.slider("Avg Tickets per Customer", 1.0, 3.0, float(shop_data["avg_num_ticket_per_customer"]))

# Calculate Projected Sales
shop_data["projected_sales"] = calculate_sales(shop_data)

# Create Bar Chart Data
sales_comparison = pd.DataFrame({
    "Sales Type": ["Real Sales", "Projected Sales"],
    "Sales Value": [macrotable_df[macrotable_df["shop_id"] == shop_id]["real_sales"].values[0], shop_data["projected_sales"].values[0]],
})

# Plot Bar Chart
fig = px.bar(sales_comparison, x="Sales Type", y="Sales Value", text="Sales Value", color="Sales Type",
             color_discrete_map={"Real Sales": "lightblue", "Projected Sales": "blue"})

st.title("📊 Projected vs. Real Sales")
st.write(f"**Shop ID: {shop_id}**")
st.plotly_chart(fig, use_container_width=True)

# Display Data
st.subheader("KPI Data Overview")
st.write(shop_data)

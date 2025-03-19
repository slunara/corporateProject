import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Add the title
st.title("KPIs Optimizer")
st.markdown("### Adjust different KPIs to optimize your sales projections.")

def compute_customer_forecast(n_months, traffic, prob_prospect_generation,
                              prob_prospect_conversion, 
                              prob_direct_customer_conversion, retention_prob, prob_existing_clients_conversion,
                              existing_customers):

    prospects = traffic * prob_prospect_generation  

    rows, cols = np.meshgrid(np.arange(n_months), np.arange(n_months), indexing="ij")

    M_conversion_matrix_ops = np.where(cols <= rows, prob_prospect_conversion[rows-cols], 0)

    new_customers_from_prospects = M_conversion_matrix_ops @ prospects

    new_customers_direct = traffic * prob_direct_customer_conversion

    M_retention_matrix_ops = np.where(cols <= rows, retention_prob[rows-cols], 0)

    retained_customers_from_prospects = M_retention_matrix_ops @ new_customers_from_prospects

    retained_customers_direct = M_retention_matrix_ops @ new_customers_direct

    existing_customers_vector = np.full(n_months, existing_customers * prob_existing_clients_conversion)

    total_existing_customers = existing_customers_vector + retained_customers_from_prospects + retained_customers_direct

    forecast_df = pd.DataFrame({
    "prospects": prospects,
    "new_customers_from_prospects": new_customers_from_prospects,  # Sum across rows
    "new_customers_direct": new_customers_direct,
    "total_existing_customers": total_existing_customers
    }, index=[f"Month {i+1}" for i in range(n_months)])

    return forecast_df

def compute_revenue_forecast(local_customer_forecast_df, tourist_customer_forecast_df, avg_ticket_df):

    total_revenue_df = pd.DataFrame(index=local_customer_forecast_df.index)

    total_revenue_df['revenue_local_new_from_prospects'] = local_customer_forecast_df['new_customers_from_prospects']*avg_ticket_df['local_avg_ticket_new_from_prospects'].iloc[0]  
    total_revenue_df['revenue_local_new_from_direct'] = local_customer_forecast_df['new_customers_direct']*avg_ticket_df['local_avg_ticket_new_direct'].iloc[0]  
    total_revenue_df['revenue_local_existing'] = local_customer_forecast_df['total_existing_customers']*avg_ticket_df['local_avg_ticket_existing'].iloc[0]  
    total_revenue_df['revenue_local_total'] = total_revenue_df['revenue_local_new_from_prospects']+total_revenue_df['revenue_local_new_from_direct']+total_revenue_df['revenue_local_existing'] 
    
    total_revenue_df['revenue_tourist_new_from_prospects'] = tourist_customer_forecast_df['new_customers_from_prospects']*avg_ticket_df['tourist_avg_ticket_new_from_prospects'].iloc[0]  
    total_revenue_df['revenue_tourist_new_from_direct'] = tourist_customer_forecast_df['new_customers_direct']*avg_ticket_df['tourist_avg_ticket_new_direct'].iloc[0]  
    total_revenue_df['revenue_tourist_existing'] = tourist_customer_forecast_df['total_existing_customers']*avg_ticket_df['tourist_avg_ticket_existing'].iloc[0]  
    total_revenue_df['revenue_tourist_total'] = total_revenue_df['revenue_tourist_new_from_prospects']+total_revenue_df['revenue_tourist_new_from_direct']+total_revenue_df['revenue_tourist_existing']
    
    total_revenue_df['revenue_total'] = total_revenue_df['revenue_local_total']+total_revenue_df['revenue_tourist_total']
    return total_revenue_df 
    
# Streamlit UI
st.title("Sales Forecast and Budget Comparison")

n_months = 6
traffic = np.array([1000, 1200, 1100, 1300, 1250, 1400])  # traffic
prob_prospect_generation = 0.30
prob_prospect_conversion = np.array([0.3, 0.2, 0.1, 0, 0, 0])  # 30% of traffic converts into prospects

local_prob_direct_customer_conversion = 0.1  # 10% of traffic converts directly into customers
tourist_prob_direct_customer_conversion = 0.05

local_retention_prob = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.1])  # Retention probability for new customers
tourist_retention_prob = np.array([0, 0.7, 0.5, 0.3, 0.2, 0.1])  # Retention probability for new customers

local_prob_existing_clients_conversion = 0.03
tourist_prob_existing_clients_conversion = 0.01

existing_local_customers = 5000  # Existing local customers before the forecast
existing_tourist_customers = 1000  # Existing tourist customers before the forecast

local_avg_ticket_new_from_prospects = 50
local_avg_ticket_new_direct = 60
local_avg_ticket_existing = 40

tourist_avg_ticket_new_from_prospects = 0
tourist_avg_ticket_new_direct = 80
tourist_avg_ticket_existing = 45

avg_ticket_df = pd.DataFrame({
    "local_avg_ticket_new_from_prospects": [local_avg_ticket_new_from_prospects],
    "local_avg_ticket_new_direct": [local_avg_ticket_new_direct],
    "local_avg_ticket_existing": [local_avg_ticket_existing],
    "tourist_avg_ticket_new_from_prospects": [tourist_avg_ticket_new_from_prospects],
    "tourist_avg_ticket_new_direct": [tourist_avg_ticket_new_direct],
    "tourist_avg_ticket_existing": [tourist_avg_ticket_existing]
})


# Compute forecasts for Local Customers
local_customer_forecast_df = compute_customer_forecast(
    n_months, traffic, prob_prospect_generation,prob_prospect_conversion, 
    local_prob_direct_customer_conversion, local_retention_prob, local_prob_existing_clients_conversion,
    existing_local_customers
)

# Compute forecasts for Tourist Customers
tourist_customer_forecast_df = compute_customer_forecast(
    n_months, traffic, prob_prospect_generation,  prob_prospect_conversion, 
    tourist_prob_direct_customer_conversion, tourist_retention_prob, tourist_prob_existing_clients_conversion,
    existing_tourist_customers 
)

total_revenue=compute_revenue_forecast(local_customer_forecast_df,tourist_customer_forecast_df,avg_ticket_df)
sales_budget = st.number_input("Sales budget", 0, 10000000, 1000000)

st.subheader("Revenue Forecast")
st.dataframe(total_revenue_df)

import streamlit as st
import numpy as np
import pandas as pd

def compute_customer_forecast(n_months, traffic, prob_prospect_generation,
                              prob_prospect_conversion, prob_direct_customer_conversion, 
                              retention_prob, prob_existing_clients_conversion,
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
        "new_customers_from_prospects": new_customers_from_prospects,
        "new_customers_direct": new_customers_direct,
        "total_existing_customers": total_existing_customers
    }, index=[f"Month {i+1}" for i in range(n_months)])
    return forecast_df

def compute_revenue_forecast(local_customer_forecast_df, tourist_customer_forecast_df, avg_ticket_df):
    total_revenue_df = pd.DataFrame(index=local_customer_forecast_df.index)
    total_revenue_df['revenue_local_new_from_prospects'] = local_customer_forecast_df['new_customers_from_prospects'] * avg_ticket_df['local_avg_ticket_new_from_prospects'].iloc[0]  
    total_revenue_df['revenue_local_new_from_direct'] = local_customer_forecast_df['new_customers_direct'] * avg_ticket_df['local_avg_ticket_new_direct'].iloc[0]  
    total_revenue_df['revenue_local_existing'] = local_customer_forecast_df['total_existing_customers'] * avg_ticket_df['local_avg_ticket_existing'].iloc[0]  
    total_revenue_df['revenue_local_total'] = total_revenue_df['revenue_local_new_from_prospects'] + total_revenue_df['revenue_local_new_from_direct'] + total_revenue_df['revenue_local_existing'] 
    total_revenue_df['revenue_tourist_new_from_prospects'] = tourist_customer_forecast_df['new_customers_from_prospects'] * avg_ticket_df['tourist_avg_ticket_new_from_prospects'].iloc[0]  
    total_revenue_df['revenue_tourist_new_from_direct'] = tourist_customer_forecast_df['new_customers_direct'] * avg_ticket_df['tourist_avg_ticket_new_direct'].iloc[0]  
    total_revenue_df['revenue_tourist_existing'] = tourist_customer_forecast_df['total_existing_customers'] * avg_ticket_df['tourist_avg_ticket_existing'].iloc[0]  
    total_revenue_df['revenue_tourist_total'] = total_revenue_df['revenue_tourist_new_from_prospects'] + total_revenue_df['revenue_tourist_new_from_direct'] + total_revenue_df['revenue_tourist_existing']
    total_revenue_df['revenue_total'] = total_revenue_df['revenue_local_total'] + total_revenue_df['revenue_tourist_total']
    return total_revenue_df 

def compute_new_forecast(sales_budget, sensitivity_df, baseline_total_revenue):
    increment_needed = (sales_budget / baseline_total_revenue) - 1
    increment_per_action = (0.01 * increment_needed / sensitivity_df) * 100
    increment_per_action = increment_per_action.rename(columns={"% Impact on Total Revenue": "Required Percentage Change"})
    return increment_per_action

st.title("Sales Forecast and Budget Comparison")

n_months = st.number_input("Number of months to forecast", min_value=1, max_value=12, value=6)
traffic = np.array(st.text_area("Enter traffic per month (comma-separated)", "1000, 1200, 1100, 1300, 1250, 1400").split(","), dtype=int)
existing_local_customers = 5000
existing_tourist_customers = 1000

sensitivity_results = {}
for var in variables.keys():
    modified_variables = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in variables.items()}
    modified_variables[var] = variables[var] * 1.01
    modified_variables["existing_customers"] = existing_local_customers
    local_forecast = compute_customer_forecast(n_months, traffic, **modified_variables)
    modified_variables["existing_customers"] = existing_tourist_customers
    tourist_forecast = compute_customer_forecast(n_months, traffic, **modified_variables)
    new_revenue_df = compute_revenue_forecast(local_forecast, tourist_forecast, avg_ticket_df)
    new_total_revenue = new_revenue_df['revenue_total'].sum()
    sensitivity_results[var] = ((new_total_revenue - new_revenue_df['revenue_total'].sum()) / new_revenue_df['revenue_total'].sum()) * 100

sensitivity_df = pd.DataFrame.from_dict(sensitivity_results, orient='index', columns=["% Impact on Total Revenue"])
st.subheader("Sensitivity Analysis")
st.dataframe(sensitivity_df)

if st.button("Simulate with Updated Variables"):
    updated_variables = {k: st.slider(f"% Change for {k}", 0, 20, 0, key=f"{k}_slider") / 100 for k in variables.keys()}
    updated_variables = {k: variables[k] * (1 + v) for k, v in updated_variables.items()}
    updated_variables["existing_customers"] = existing_local_customers
    local_forecast = compute_customer_forecast(n_months, traffic, **updated_variables)
    updated_variables["existing_customers"] = existing_tourist_customers
    tourist_forecast = compute_customer_forecast(n_months, traffic, **updated_variables)
    new_revenue_df = compute_revenue_forecast(local_forecast, tourist_forecast, avg_ticket_df)
    st.subheader("Updated Revenue Forecast")
    st.dataframe(new_revenue_df)

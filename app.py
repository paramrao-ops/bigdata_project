import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction & Analytics Dashboard")
st.markdown("Big Data Analytics Pipeline built with PySpark & Streamlit")

# Load Silver Cleaned Data
@st.cache_data
def load_data():
    return pd.read_parquet("data/silver/churn_cleaned.parquet")

df = load_data()

# KPI Metric Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("Churn Rate", f"{(df['ChurnBinary'].mean() * 100):.1f}%")
col3.metric("Avg Tenure (Months)", f"{df['tenure'].mean():.1f}")
col4.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")

st.divider()

# Visualizations Row
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Churn Rate by Contract Type")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Contract", y="ChurnBinary", ax=ax, palette="Blues_d")
    ax.set_ylabel("Churn Proportion")
    st.pyplot(fig)

with col_right:
    st.subheader("Monthly Charges Distribution by Churn")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax2, palette="Set2")
    st.pyplot(fig2)

st.subheader("Raw Data Preview (Silver Layer)")
st.dataframe(df.head(10))

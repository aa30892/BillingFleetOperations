import os
import streamlit as st
import pandas as pd

st.set_page_config(page_title="PO vs Billing Dashboard", layout="wide")
st.title("PO vs Billing Comparison")

conn = st.connection("snowflake", ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL"))


@st.cache_data
def load_po_data():
    return conn.query("""
        SELECT
            JOB_NOTIFICATION_ID,
            MATERIAL_ID,
            FOS_CONTRACT_ID,
            CUSTOMER_NAME,
            PO_POSTING_DATE,
            PO_QTY,
            NET_PRICE_EURO,
            MATERIAL_DESC,
            MATERIAL_GROUP_DESC
        FROM PROD.EU_INSIGHT.FOS_PURCHASE_ORDER_DATA
        WHERE PO_POSTING_YEAR = 2026
          AND PO_POSTING_MONTH BETWEEN 1 AND 5
          AND MATERIAL_GROUP_DESC IN ('service truck', 'service')
    """)


@st.cache_data
def load_billing_data():
    return conn.query("""
        SELECT
            JOB_ID,
            MATL_ID_TRIM,
            CONTRACT_ID,
            CUST_NAME,
            BILLING_DT,
            BILLED_QTY,
            BILLED_AMT_EURO,
            MATL_DESC,
            MATL_GRP_DESC
        FROM PROD.EU_BI_VWS.FOS_BILLING_EXTRACT
        WHERE BILLING_YEAR = 2026
          AND BILLING_MONTH BETWEEN 1 AND 5
          AND MATL_GRP_DESC IN ('Service Truck', 'Service')
    """)


with st.spinner("Loading data..."):
    po_df = load_po_data()
    billing_df = load_billing_data()

po_df["PO_POSTING_DATE"] = pd.to_datetime(po_df["PO_POSTING_DATE"])
billing_df["BILLING_DT"] = pd.to_datetime(billing_df["BILLING_DT"])

merged = po_df.merge(
    billing_df,
    left_on=["JOB_NOTIFICATION_ID", "MATERIAL_ID"],
    right_on=["JOB_ID", "MATL_ID_TRIM"],
    how="outer",
    suffixes=("_PO", "_BILL"),
)

merged["CONTRACT_ID_COMBINED"] = merged["FOS_CONTRACT_ID"].fillna(merged["CONTRACT_ID"])
merged["CUSTOMER_NAME_COMBINED"] = merged["CUSTOMER_NAME"].fillna(merged["CUST_NAME"])
merged["MATERIAL_ID_COMBINED"] = merged["MATERIAL_ID"].fillna(merged["MATL_ID_TRIM"])
merged["JOB_ID_COMBINED"] = merged["JOB_NOTIFICATION_ID"].fillna(merged["JOB_ID"])
merged["PO_POSTING_DATE_COMBINED"] = merged["PO_POSTING_DATE"].fillna(merged["BILLING_DT"])

with st.sidebar:
    st.header("Filters")

    if st.button("Clear cache & reload"):
        load_po_data.clear()
        load_billing_data.clear()
        st.rerun()

    contracts = sorted(merged["CONTRACT_ID_COMBINED"].dropna().unique().tolist())
    selected_contracts = st.multiselect("Contract ID", contracts)

    customers = sorted(merged["CUSTOMER_NAME_COMBINED"].dropna().unique().tolist())
    selected_customers = st.multiselect("Customer Name", customers)

    min_date = merged["PO_POSTING_DATE_COMBINED"].min()
    max_date = merged["PO_POSTING_DATE_COMBINED"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.date_input(
            "PO Posting Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
    else:
        date_range = None

filtered = merged.copy()
if selected_contracts:
    filtered = filtered[filtered["CONTRACT_ID_COMBINED"].isin(selected_contracts)]
if selected_customers:
    filtered = filtered[filtered["CUSTOMER_NAME_COMBINED"].isin(selected_customers)]
if date_range and len(date_range) == 2:
    filtered = filtered[
        (filtered["PO_POSTING_DATE_COMBINED"] >= pd.Timestamp(date_range[0]))
        & (filtered["PO_POSTING_DATE_COMBINED"] <= pd.Timestamp(date_range[1]))
    ]

filtered["PRICE_DIFF"] = filtered["NET_PRICE_EURO"].fillna(0) - filtered["BILLED_AMT_EURO"].fillna(0)
filtered["QTY_DIFF"] = filtered["PO_QTY"].fillna(0) - filtered["BILLED_QTY"].fillna(0)

with st.container(horizontal=True):
    st.metric("Total Records", f"{len(filtered):,}", border=True)
    st.metric(
        "Total Price Diff (PO - Billed) €",
        f"{filtered['PRICE_DIFF'].sum():,.2f}",
        border=True,
    )
    st.metric(
        "Total Qty Diff (PO - Billed)",
        f"{filtered['QTY_DIFF'].sum():,.1f}",
        border=True,
    )

st.subheader("Differences by Material ID")

summary = (
    filtered.groupby("MATERIAL_ID_COMBINED")
    .agg(
        PO_NET_PRICE_EURO=("NET_PRICE_EURO", "sum"),
        BILLED_AMT_EURO=("BILLED_AMT_EURO", "sum"),
        PO_QTY=("PO_QTY", "sum"),
        BILLED_QTY=("BILLED_QTY", "sum"),
    )
    .reset_index()
)
summary["PRICE_DIFF"] = summary["PO_NET_PRICE_EURO"].fillna(0) - summary["BILLED_AMT_EURO"].fillna(0)
summary["QTY_DIFF"] = summary["PO_QTY"].fillna(0) - summary["BILLED_QTY"].fillna(0)

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("**NET_PRICE_EURO vs BILLED_AMT_EURO**")
        chart_price = summary[["MATERIAL_ID_COMBINED", "PO_NET_PRICE_EURO", "BILLED_AMT_EURO"]].set_index(
            "MATERIAL_ID_COMBINED"
        )
        st.bar_chart(chart_price)

with col2:
    with st.container(border=True):
        st.markdown("**PO_QTY vs BILLED_QTY**")
        chart_qty = summary[["MATERIAL_ID_COMBINED", "PO_QTY", "BILLED_QTY"]].set_index(
            "MATERIAL_ID_COMBINED"
        )
        st.bar_chart(chart_qty)

st.subheader("Detail Table")
display_cols = [
    "JOB_ID_COMBINED",
    "MATERIAL_ID_COMBINED",
    "CONTRACT_ID_COMBINED",
    "CUSTOMER_NAME_COMBINED",
    "PO_POSTING_DATE_COMBINED",
    "NET_PRICE_EURO",
    "BILLED_AMT_EURO",
    "PRICE_DIFF",
    "PO_QTY",
    "BILLED_QTY",
    "QTY_DIFF",
]
st.dataframe(
    filtered[display_cols].rename(
        columns={
            "JOB_ID_COMBINED": "Job ID",
            "MATERIAL_ID_COMBINED": "Material ID",
            "CONTRACT_ID_COMBINED": "Contract ID",
            "CUSTOMER_NAME_COMBINED": "Customer",
            "PO_POSTING_DATE_COMBINED": "PO Posting Date",
            "NET_PRICE_EURO": "PO Net Price €",
            "BILLED_AMT_EURO": "Billed Amt €",
            "PRICE_DIFF": "Price Diff €",
            "PO_QTY": "PO Qty",
            "BILLED_QTY": "Billed Qty",
            "QTY_DIFF": "Qty Diff",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

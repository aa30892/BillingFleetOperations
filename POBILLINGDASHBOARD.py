import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="PO vs Billing Dashboard", layout="wide")
st.title("PO vs Billing Comparison")

with st.sidebar:
    st.header("Upload Data")
    po_file = st.file_uploader("Upload PO data (CSV)", type=["csv"])
    billing_file = st.file_uploader("Upload Billing data (CSV)", type=["csv"])

if po_file is None or billing_file is None:
    st.info("Please upload both PO and Billing CSV files to proceed.")
    st.stop()

po_df = pd.read_csv(po_file)
billing_df = pd.read_csv(billing_file)

po_df.columns = po_df.columns.str.upper()
billing_df.columns = billing_df.columns.str.upper()

if "PO_POSTING_DATE" in po_df.columns:
    po_df["PO_POSTING_DATE"] = pd.to_datetime(po_df["PO_POSTING_DATE"])
if "BILLING_DT" in billing_df.columns:
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
merged["PO_POSTING_DATE_COMBINED"] = merged["PO_POSTING_DATE"].fillna(
    merged["BILLING_DT"] if "BILLING_DT" in merged.columns else pd.NaT
)

with st.sidebar:
    st.header("Filters")

    contracts = sorted(merged["CONTRACT_ID_COMBINED"].dropna().unique().tolist())
    selected_contracts = st.multiselect("Contract ID", contracts)

    customers = sorted(merged["CUSTOMER_NAME_COMBINED"].dropna().unique().tolist())
    selected_customers = st.multiselect("Customer Name", customers)

    vendor_col = "VENDOR_NAME" if "VENDOR_NAME" in merged.columns else ("VEND_NAME" if "VEND_NAME" in merged.columns else None)
    if vendor_col:
        vendors = sorted(merged[vendor_col].dropna().unique().tolist())
        selected_vendors = st.multiselect("Vendor Name", vendors)
    else:
        selected_vendors = []

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
if selected_vendors and vendor_col:
    filtered = filtered[filtered[vendor_col].isin(selected_vendors)]
if date_range and len(date_range) == 2:
    filtered = filtered[
        (filtered["PO_POSTING_DATE_COMBINED"] >= pd.Timestamp(date_range[0]))
        & (filtered["PO_POSTING_DATE_COMBINED"] <= pd.Timestamp(date_range[1]))
    ]

filtered["PRICE_DIFF"] = filtered["NET_PRICE_EURO"].fillna(0) - filtered["BILLED_AMT_EURO"].fillna(0)
filtered["QTY_DIFF"] = filtered["PO_QTY"].fillna(0) - filtered["BILLED_QTY"].fillna(0)

tab_dashboard, tab_anomaly = st.tabs(["Dashboard", "AI Anomaly Analysis"])

with tab_dashboard:
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

with tab_anomaly:
    st.subheader("AI Anomaly Detection")
    st.markdown("Identifies outliers in price and quantity differences using statistical analysis (IQR method).")

    if filtered.empty:
        st.warning("No data available. Adjust filters to see anomaly analysis.")
    else:
        def detect_anomalies(df, column, label):
            data = df[column].dropna()
            if data.empty:
                return pd.DataFrame()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            anomalies = df[(df[column] < lower) | (df[column] > upper)].copy()
            anomalies["ANOMALY_TYPE"] = label
            anomalies["ANOMALY_VALUE"] = anomalies[column]
            anomalies["LOWER_BOUND"] = lower
            anomalies["UPPER_BOUND"] = upper
            return anomalies

        price_anomalies = detect_anomalies(filtered, "PRICE_DIFF", "Price Difference")
        qty_anomalies = detect_anomalies(filtered, "QTY_DIFF", "Quantity Difference")
        all_anomalies = pd.concat([price_anomalies, qty_anomalies], ignore_index=True)

        with st.container(horizontal=True):
            st.metric("Price Anomalies", len(price_anomalies), border=True)
            st.metric("Quantity Anomalies", len(qty_anomalies), border=True)
            st.metric(
                "Anomaly Rate",
                f"{(len(all_anomalies) / (2 * len(filtered)) * 100):.1f}%" if len(filtered) > 0 else "0%",
                border=True,
            )

        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                st.markdown("**Price Diff Distribution**")
                price_data = filtered["PRICE_DIFF"].dropna()
                if not price_data.empty:
                    hist_values = np.histogram(price_data, bins=30)[0]
                    st.bar_chart(hist_values)
                    if not price_anomalies.empty:
                        q1 = price_data.quantile(0.25)
                        q3 = price_data.quantile(0.75)
                        iqr = q3 - q1
                        st.caption(f"Bounds: [{q1 - 1.5*iqr:,.2f}, {q3 + 1.5*iqr:,.2f}]")

        with col_b:
            with st.container(border=True):
                st.markdown("**Qty Diff Distribution**")
                qty_data = filtered["QTY_DIFF"].dropna()
                if not qty_data.empty:
                    hist_values = np.histogram(qty_data, bins=30)[0]
                    st.bar_chart(hist_values)
                    if not qty_anomalies.empty:
                        q1 = qty_data.quantile(0.25)
                        q3 = qty_data.quantile(0.75)
                        iqr = q3 - q1
                        st.caption(f"Bounds: [{q1 - 1.5*iqr:,.2f}, {q3 + 1.5*iqr:,.2f}]")

        if not all_anomalies.empty:
            st.subheader("Flagged Anomalies")
            anomaly_display = all_anomalies[[
                "JOB_ID_COMBINED",
                "MATERIAL_ID_COMBINED",
                "CONTRACT_ID_COMBINED",
                "CUSTOMER_NAME_COMBINED",
                "ANOMALY_TYPE",
                "ANOMALY_VALUE",
                "LOWER_BOUND",
                "UPPER_BOUND",
            ]].rename(columns={
                "JOB_ID_COMBINED": "Job ID",
                "MATERIAL_ID_COMBINED": "Material ID",
                "CONTRACT_ID_COMBINED": "Contract ID",
                "CUSTOMER_NAME_COMBINED": "Customer",
                "ANOMALY_TYPE": "Type",
                "ANOMALY_VALUE": "Value",
                "LOWER_BOUND": "Lower Bound",
                "UPPER_BOUND": "Upper Bound",
            })
            st.dataframe(anomaly_display, hide_index=True, use_container_width=True)
        else:
            st.success("No anomalies detected in the current data.")

# PO vs Billing dashboard with customer grouping filter
# Co-authored with CoCo
import streamlit as st
import pandas as pd
import numpy as np
import io
from collections import defaultdict

st.set_page_config(page_title="PO vs Billing Dashboard", layout="wide")
st.title("PO vs Billing Comparison")

with st.sidebar:
    st.header("Upload Data")
    po_file = st.file_uploader("Upload PO data (CSV or Parquet)", type=["csv", "parquet"])
    billing_file = st.file_uploader("Upload Billing data (CSV or Parquet)", type=["csv", "parquet"])

if po_file is None or billing_file is None:
    st.info("Please upload both PO and Billing files to proceed.")
    st.stop()


def load_and_merge(po_file, billing_file):
    try:
        po_buf = io.BytesIO(po_file.getvalue())
        billing_buf = io.BytesIO(billing_file.getvalue())
        if po_file.name.endswith(".parquet"):
            po_df = pd.read_parquet(po_buf)
        else:
            po_df = pd.read_csv(po_buf)
        if billing_file.name.endswith(".parquet"):
            billing_df = pd.read_parquet(billing_buf)
        else:
            billing_df = pd.read_csv(billing_buf)
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()

    po_df.columns = po_df.columns.str.upper().str.strip()
    billing_df.columns = billing_df.columns.str.upper().str.strip()

    # Show debug info
    with st.sidebar.expander("Debug: Column Info", expanded=False):
        st.write("PO columns:", list(po_df.columns))
        st.write("PO rows:", len(po_df))
        st.write("Billing columns:", list(billing_df.columns))
        st.write("Billing rows:", len(billing_df))

    if "PO_POSTING_DATE" in po_df.columns:
        po_df["PO_POSTING_DATE"] = pd.to_datetime(po_df["PO_POSTING_DATE"], errors="coerce")
    if "BILLING_DT" in billing_df.columns:
        billing_df["BILLING_DT"] = pd.to_datetime(billing_df["BILLING_DT"], errors="coerce")

    # Determine join keys and cast to string to avoid type mismatches
    po_left_on = []
    billing_right_on = []
    if "JOB_NOTIFICATION_ID" in po_df.columns and "JOB_ID" in billing_df.columns:
        po_df["JOB_NOTIFICATION_ID"] = po_df["JOB_NOTIFICATION_ID"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        billing_df["JOB_ID"] = billing_df["JOB_ID"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        po_left_on.append("JOB_NOTIFICATION_ID")
        billing_right_on.append("JOB_ID")
    if "MATERIAL_ID" in po_df.columns and "MATL_ID_TRIM" in billing_df.columns:
        po_df["MATERIAL_ID"] = po_df["MATERIAL_ID"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        billing_df["MATL_ID_TRIM"] = billing_df["MATL_ID_TRIM"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        po_left_on.append("MATERIAL_ID")
        billing_right_on.append("MATL_ID_TRIM")
    if "VEHICLE_ID" in po_df.columns and "VEHICLE_ID" in billing_df.columns:
        po_df["VEHICLE_ID"] = po_df["VEHICLE_ID"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        billing_df["VEHICLE_ID"] = billing_df["VEHICLE_ID"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        po_left_on.append("VEHICLE_ID")
        billing_right_on.append("VEHICLE_ID")

    if not po_left_on:
        st.error(
            f"Cannot merge files. PO columns: {list(po_df.columns)}. "
            f"Billing columns: {list(billing_df.columns)}. "
            f"Expected: JOB_NOTIFICATION_ID/JOB_ID and MATERIAL_ID/MATL_ID_TRIM."
        )
        st.stop()

    try:
        merged = po_df.merge(
            billing_df,
            left_on=po_left_on,
            right_on=billing_right_on,
            how="outer",
            suffixes=("_PO", "_BILL"),
        )
    except Exception as e:
        st.error(f"Merge failed: {e}")
        st.stop()

    # Build combined columns safely
    def get_col(df, col):
        if col in df.columns:
            return df[col]
        return pd.Series(np.nan, index=df.index)

    if "FOS_CONTRACT_ID" in merged.columns or "CONTRACT_ID" in merged.columns:
        merged["CONTRACT_ID_COMBINED"] = (
            get_col(merged, "FOS_CONTRACT_ID")
            .combine_first(get_col(merged, "CONTRACT_ID"))
            .astype(str).str.replace(r'\.0$', '', regex=True)
        )
    else:
        merged["CONTRACT_ID_COMBINED"] = "N/A"

    merged["CUSTOMER_NAME_COMBINED"] = (
        get_col(merged, "CUSTOMER_NAME")
        .combine_first(get_col(merged, "CUST_NAME"))
        .fillna("Unknown")
        .astype(str)
    )
    merged["MATERIAL_ID_COMBINED"] = (
        get_col(merged, "MATERIAL_ID")
        .combine_first(get_col(merged, "MATL_ID_TRIM"))
        .astype(str).str.replace(r'\.0$', '', regex=True)
    )
    merged["JOB_ID_COMBINED"] = (
        get_col(merged, "JOB_NOTIFICATION_ID")
        .combine_first(get_col(merged, "JOB_ID"))
        .astype(str).str.replace(r'\.0$', '', regex=True)
    )

    # Combine VEHICLE_ID from both sources (suffixed after merge since both tables have it)
    if "VEHICLE_ID_PO" in merged.columns or "VEHICLE_ID_BILL" in merged.columns:
        merged["VEHICLE_ID"] = (
            get_col(merged, "VEHICLE_ID_PO")
            .combine_first(get_col(merged, "VEHICLE_ID_BILL"))
            .astype(str)
        )

    if "BILLING_DT" in merged.columns and "PO_POSTING_DATE" in merged.columns:
        merged["PO_POSTING_DATE_COMBINED"] = merged["PO_POSTING_DATE"].combine_first(merged["BILLING_DT"])
    elif "PO_POSTING_DATE" in merged.columns:
        merged["PO_POSTING_DATE_COMBINED"] = merged["PO_POSTING_DATE"]
    elif "BILLING_DT" in merged.columns:
        merged["PO_POSTING_DATE_COMBINED"] = merged["BILLING_DT"]

    return merged


merged = load_and_merge(po_file, billing_file)

vendor_col = "VENDOR_NAME" if "VENDOR_NAME" in merged.columns else ("VEND_NAME" if "VEND_NAME" in merged.columns else None)


def build_customer_groups(customer_names):
    """Group customers by common keyword prefix (first word of the name).
    If multiple customers share the same first word, they are grouped under that word."""
    prefix_map = defaultdict(list)
    for name in customer_names:
        parts = name.strip().split()
        if parts:
            prefix = parts[0].upper()
            prefix_map[prefix].append(name)
    groups = {}
    for prefix, members in prefix_map.items():
        if len(members) > 1:
            group_label = prefix.title()
            for m in members:
                groups[m] = group_label
        else:
            groups[members[0]] = members[0]
    return groups


all_customers = [x for x in merged["CUSTOMER_NAME_COMBINED"].unique().tolist() if x != "nan" and x.strip() != ""]
all_customers.sort()
customer_group_map = build_customer_groups(all_customers)
merged["CUSTOMER_GROUP"] = merged["CUSTOMER_NAME_COMBINED"].map(customer_group_map).fillna(merged["CUSTOMER_NAME_COMBINED"])

with st.sidebar:
    st.header("Filters")

    contracts = [x for x in merged["CONTRACT_ID_COMBINED"].unique().tolist() if x != "nan" and x.strip() != ""]
    contracts.sort()
    selected_contracts = st.multiselect("Contract ID", contracts, key="filter_contracts")

    customer_groups = sorted(merged["CUSTOMER_GROUP"].dropna().unique().tolist())
    customer_groups = [x for x in customer_groups if x != "nan" and x.strip() != ""]
    selected_customer_groups = st.multiselect(
        "Customer Group",
        customer_groups,
        help="Customers are grouped by shared first keyword (e.g. all 'Transdev ...' entries → 'Transdev')",
        key="filter_customer_groups",
    )

    customers = [x for x in merged["CUSTOMER_NAME_COMBINED"].unique().tolist() if x != "nan" and x.strip() != ""]
    customers.sort()
    selected_customers = st.multiselect("Customer Name", customers, key="filter_customers")

    if vendor_col:
        vendors = [str(x) for x in merged[vendor_col].dropna().unique().tolist() if str(x).strip() != ""]
        vendors.sort()
        selected_vendors = st.multiselect("Vendor Name", vendors, key="filter_vendors")
    else:
        selected_vendors = []

    job_ids = [str(x) for x in merged["JOB_ID_COMBINED"].unique().tolist() if str(x) != "nan" and str(x).strip() != ""]
    job_ids.sort()
    selected_job_ids = st.multiselect("Job ID", job_ids, key="filter_job_ids")

    min_date = merged["PO_POSTING_DATE_COMBINED"].min()
    max_date = merged["PO_POSTING_DATE_COMBINED"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.date_input(
            "PO Posting Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="filter_dates",
        )
    else:
        date_range = None

filtered = merged
if selected_contracts:
    filtered = filtered[filtered["CONTRACT_ID_COMBINED"].isin(selected_contracts)]
if selected_customer_groups:
    filtered = filtered[filtered["CUSTOMER_GROUP"].isin(selected_customer_groups)]
if selected_customers:
    filtered = filtered[filtered["CUSTOMER_NAME_COMBINED"].isin(selected_customers)]
if selected_vendors and vendor_col:
    filtered = filtered[filtered[vendor_col].astype(str).isin(selected_vendors)]
if selected_job_ids:
    filtered = filtered[filtered["JOB_ID_COMBINED"].astype(str).isin(selected_job_ids)]
if date_range and len(date_range) == 2:
    filtered = filtered[
        (filtered["PO_POSTING_DATE_COMBINED"] >= pd.Timestamp(date_range[0]))
        & (filtered["PO_POSTING_DATE_COMBINED"] <= pd.Timestamp(date_range[1]))
    ]

filtered["PRICE_DIFF"] = filtered["NET_PRICE_EURO"].fillna(0) - filtered["BILLED_AMT_EURO"].fillna(0)
filtered["QTY_DIFF"] = filtered["PO_QTY"].fillna(0) - filtered["BILLED_QTY"].fillna(0)

tab_dashboard, tab_repair, tab_anomaly = st.tabs(["Dashboard", "Contract Analysis", "AI Anomaly Analysis"])

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
            chart_price = (
                summary[["MATERIAL_ID_COMBINED", "PO_NET_PRICE_EURO", "BILLED_AMT_EURO", "PRICE_DIFF"]]
                .sort_values("PRICE_DIFF", ascending=False)
                .drop(columns=["PRICE_DIFF"])
                .set_index("MATERIAL_ID_COMBINED")
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

with tab_repair:
    st.subheader("Fleet Type Analysis — Materials per Job & Vehicle")
    st.markdown("Breaks down by `FLEET_TYPE` which materials are used per job and vehicle, with quantities and descriptions.")

    fleet_col = None
    for fc in ["FLEET_TYPE", "FLEET_TYPE_PO"]:
        if fc in filtered.columns:
            fleet_col = fc
            break

    vehicle_col_fleet = None
    for vc in ["VEHICLE_ID", "VEHICLE_ID_PO", "LICENCE_PLATE", "LICENCE_PLATE_ID"]:
        if vc in filtered.columns:
            vehicle_col_fleet = vc
            break

    licence_col_fleet = None
    if vehicle_col_fleet and "VEHICLE" in vehicle_col_fleet.upper():
        for lc in ["LICENCE_PLATE", "LICENCE_PLATE_ID"]:
            if lc in filtered.columns:
                licence_col_fleet = lc
                break

    mat_desc_fleet = None
    for mc in ["MATERIAL_DESC", "MATERIAL_DESC_PO", "MATL_DESC", "MATL_DESC_PO", "MATL_DESC_BILL"]:
        if mc in filtered.columns:
            mat_desc_fleet = mc
            break

    if fleet_col is None:
        st.warning("No FLEET_TYPE column found. Ensure your PO CSV includes FLEET_TYPE.")
    elif vehicle_col_fleet is None:
        st.warning("No VEHICLE column found. Ensure your CSV includes VEHICLE_ID or LICENCE_PLATE.")
    else:
        fleet_types = sorted([str(x) for x in filtered[fleet_col].dropna().unique().tolist() if str(x).strip() != ""])
        selected_fleet = st.multiselect("Filter by Fleet Type", fleet_types, key="fleet_filter")

        fleet_df = filtered.copy()
        if selected_fleet:
            fleet_df = fleet_df[fleet_df[fleet_col].astype(str).isin(selected_fleet)]

        group_cols_fleet = ["JOB_ID_COMBINED", vehicle_col_fleet]
        if licence_col_fleet:
            group_cols_fleet.append(licence_col_fleet)
        group_cols_fleet.append("MATERIAL_ID_COMBINED")
        if mat_desc_fleet:
            group_cols_fleet.append(mat_desc_fleet)
        group_cols_fleet.append(fleet_col)

        # Aggregate PO and Billing sides separately to avoid many-to-many inflation
        po_cols = [c for c in group_cols_fleet if c in fleet_df.columns]
        po_agg = fleet_df[fleet_df["PO_QTY"].notna()].drop_duplicates(
            subset=po_cols + ["NET_PRICE_EURO"] if "NET_PRICE_EURO" in fleet_df.columns else po_cols
        )
        bill_agg = fleet_df[fleet_df["BILLED_QTY"].notna()].drop_duplicates(
            subset=po_cols + ["BILLED_AMT_EURO"] if "BILLED_AMT_EURO" in fleet_df.columns else po_cols
        )

        po_grouped = po_agg.groupby(group_cols_fleet).agg(
            TIMES_USED_PO=("PO_QTY", "count"),
            **{"TOTAL_PO_EURO": ("NET_PRICE_EURO", "sum")} if "NET_PRICE_EURO" in po_agg.columns else {},
        ).reset_index()

        bill_grouped = bill_agg.groupby(group_cols_fleet).agg(
            TIMES_BILLED=("BILLED_QTY", "count"),
            **{"TOTAL_BILLED_EURO": ("BILLED_AMT_EURO", "sum")} if "BILLED_AMT_EURO" in bill_agg.columns else {},
        ).reset_index()

        fleet_analysis = po_grouped.merge(bill_grouped, on=group_cols_fleet, how="outer")
        fleet_analysis["TIMES_USED_PO"] = fleet_analysis["TIMES_USED_PO"].fillna(0).astype(int)
        fleet_analysis["TIMES_BILLED"] = fleet_analysis["TIMES_BILLED"].fillna(0).astype(int)
        if "TOTAL_PO_EURO" in fleet_analysis.columns and "TOTAL_BILLED_EURO" in fleet_analysis.columns:
            fleet_analysis["EURO_DIFF"] = fleet_analysis["TOTAL_PO_EURO"].fillna(0) - fleet_analysis["TOTAL_BILLED_EURO"].fillna(0)
        fleet_analysis = fleet_analysis.sort_values(["TIMES_USED_PO"], ascending=False).reset_index(drop=True)

        with st.container(horizontal=True):
            st.metric("Fleet Types", len(fleet_types), border=True)
            st.metric("Records", len(fleet_analysis), border=True)
            st.metric("Unique Vehicles", fleet_df[vehicle_col_fleet].nunique(), border=True)

        with st.container(border=True):
            st.markdown("**Material Usage by Fleet Type**")
            fleet_summary = (
                fleet_df.groupby([fleet_col, "MATERIAL_ID_COMBINED"])
                .agg(TOTAL_QTY=("PO_QTY", "sum"))
                .reset_index()
                .sort_values("TOTAL_QTY", ascending=False)
            )
            top_by_fleet = fleet_summary.groupby(fleet_col).head(10)
            if not top_by_fleet.empty:
                pivot = top_by_fleet.pivot_table(
                    index="MATERIAL_ID_COMBINED", columns=fleet_col, values="TOTAL_QTY", fill_value=0
                )
                st.bar_chart(pivot)

        rename_map_fleet = {
            "JOB_ID_COMBINED": "Job ID",
            vehicle_col_fleet: "Vehicle ID",
            "MATERIAL_ID_COMBINED": "Material ID",
            fleet_col: "Fleet Type",
            "TIMES_USED_PO": "Times Used PO",
            "TIMES_BILLED": "Times Billed",
            "TOTAL_PO_EURO": "PO € Total",
            "TOTAL_BILLED_EURO": "Billed € Total",
            "EURO_DIFF": "€ Difference",
        }
        if licence_col_fleet:
            rename_map_fleet[licence_col_fleet] = "Licence Plate"
        if mat_desc_fleet:
            rename_map_fleet[mat_desc_fleet] = "Material Description"

        st.dataframe(
            fleet_analysis.rename(columns=rename_map_fleet),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.subheader("Customer Group Analysis — PO vs Billing")
    st.markdown("Aggregates Times Used PO, Times Billed, PO € Total, and Billed € Total per **Customer Group**, highlighting discrepancies.")

    if "CUSTOMER_GROUP" not in filtered.columns:
        st.warning("No CUSTOMER_GROUP column found in the data.")
    else:
        cg_po = filtered[filtered["PO_QTY"].notna()].groupby("CUSTOMER_GROUP").agg(
            TIMES_USED_PO=("PO_QTY", "count"),
            **{"TOTAL_PO_EURO": ("NET_PRICE_EURO", "sum")} if "NET_PRICE_EURO" in filtered.columns else {},
        ).reset_index()

        cg_bill = filtered[filtered["BILLED_QTY"].notna()].groupby("CUSTOMER_GROUP").agg(
            TIMES_BILLED=("BILLED_QTY", "count"),
            **{"TOTAL_BILLED_EURO": ("BILLED_AMT_EURO", "sum")} if "BILLED_AMT_EURO" in filtered.columns else {},
        ).reset_index()

        cg_analysis = cg_po.merge(cg_bill, on="CUSTOMER_GROUP", how="outer")
        cg_analysis["TIMES_USED_PO"] = cg_analysis["TIMES_USED_PO"].fillna(0).astype(int)
        cg_analysis["TIMES_BILLED"] = cg_analysis["TIMES_BILLED"].fillna(0).astype(int)
        if "TOTAL_PO_EURO" in cg_analysis.columns and "TOTAL_BILLED_EURO" in cg_analysis.columns:
            cg_analysis["EURO_DIFF"] = cg_analysis["TOTAL_PO_EURO"].fillna(0) - cg_analysis["TOTAL_BILLED_EURO"].fillna(0)
        cg_analysis = cg_analysis.sort_values("TIMES_USED_PO", ascending=False).reset_index(drop=True)

        with st.container(horizontal=True):
            st.metric("Customer Groups", len(cg_analysis), border=True)
            if "EURO_DIFF" in cg_analysis.columns:
                total_diff = cg_analysis["EURO_DIFF"].sum()
                st.metric("Total € Difference", f"{total_diff:,.2f}", border=True)

        with st.container(border=True):
            st.markdown("**PO € vs Billed € by Customer Group**")
            if "TOTAL_PO_EURO" in cg_analysis.columns and "TOTAL_BILLED_EURO" in cg_analysis.columns:
                chart_cg = (
                    cg_analysis[["CUSTOMER_GROUP", "TOTAL_PO_EURO", "TOTAL_BILLED_EURO"]]
                    .sort_values("TOTAL_PO_EURO", ascending=False)
                    .set_index("CUSTOMER_GROUP")
                )
                st.bar_chart(chart_cg)

        cg_rename = {
            "CUSTOMER_GROUP": "Customer Group",
            "TIMES_USED_PO": "Times Used PO",
            "TIMES_BILLED": "Times Billed",
            "TOTAL_PO_EURO": "PO € Total",
            "TOTAL_BILLED_EURO": "Billed € Total",
            "EURO_DIFF": "€ Difference",
        }
        st.dataframe(
            cg_analysis.rename(columns=cg_rename),
            hide_index=True,
            use_container_width=True,
        )

with tab_anomaly:
    st.subheader("AI Anomaly Detection — Vendor Behaviour vs Material")
    st.markdown("Analyses vendor billing patterns per material description, flagging suspicious behaviour using IQR-based outlier detection.")

    if filtered.empty:
        st.warning("No data available. Adjust filters to see anomaly analysis.")
    else:
        vendor_col_anom = None
        for vc in ["VENDOR_NAME", "VEND_NAME", "VENDOR_NAME_PO", "VEND_NAME_BILL"]:
            if vc in filtered.columns:
                vendor_col_anom = vc
                break

        mat_desc_anom = None
        for mc in ["MATERIAL_DESC", "MATERIAL_DESC_PO", "MATL_DESC", "MATL_DESC_PO", "MATL_DESC_BILL"]:
            if mc in filtered.columns:
                mat_desc_anom = mc
                break

        filtered["PRICE_DIFF"] = filtered["NET_PRICE_EURO"].fillna(0) - filtered["BILLED_AMT_EURO"].fillna(0)
        filtered["QTY_DIFF"] = filtered["PO_QTY"].fillna(0) - filtered["BILLED_QTY"].fillna(0)

        def detect_anomalies(df, column):
            data = df[column].dropna()
            if data.empty or data.std() == 0:
                return pd.DataFrame()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return df[(df[column] < lower) | (df[column] > upper)]

        price_outliers = detect_anomalies(filtered, "PRICE_DIFF")
        qty_outliers = detect_anomalies(filtered, "QTY_DIFF")

        total_anomalies = len(price_outliers) + len(qty_outliers)
        with st.container(horizontal=True):
            st.metric("Price Outliers", f"{len(price_outliers):,}", border=True)
            st.metric("Qty Outliers", f"{len(qty_outliers):,}", border=True)
            st.metric("Total Anomalies", f"{total_anomalies:,}", border=True)
            st.metric(
                "Anomaly Rate",
                f"{total_anomalies / max(2 * len(filtered), 1) * 100:.1f}%",
                border=True,
            )

        st.divider()
        st.subheader("Vendor Anomaly Heatmap")
        st.markdown("Shows which vendors have the most anomalous transactions by material — higher counts indicate suspicious patterns.")

        if vendor_col_anom and mat_desc_anom:
            combined_outliers = pd.concat([price_outliers, qty_outliers]).drop_duplicates(subset=["JOB_ID_COMBINED", "MATERIAL_ID_COMBINED"])

            if not combined_outliers.empty:
                vendor_mat_counts = (
                    combined_outliers.groupby([vendor_col_anom, mat_desc_anom])
                    .size()
                    .reset_index(name="ANOMALY_COUNT")
                    .sort_values("ANOMALY_COUNT", ascending=False)
                )

                top_vendors = vendor_mat_counts[vendor_col_anom].value_counts().head(15).index.tolist()
                top_materials = vendor_mat_counts[mat_desc_anom].value_counts().head(10).index.tolist()

                heatmap_data = vendor_mat_counts[
                    (vendor_mat_counts[vendor_col_anom].isin(top_vendors)) &
                    (vendor_mat_counts[mat_desc_anom].isin(top_materials))
                ]

                if not heatmap_data.empty:
                    pivot_heat = heatmap_data.pivot_table(
                        index=vendor_col_anom, columns=mat_desc_anom, values="ANOMALY_COUNT", fill_value=0
                    )
                    st.dataframe(
                        pivot_heat,
                        use_container_width=True,
                    )
                else:
                    st.info("Not enough data for heatmap.")

                st.divider()
                st.subheader("Vendor Anomaly Heatmap — Job Type Patterns")
                st.markdown("Flags vendors with suspicious job type patterns: **Regular Breakdown**, **Regular Job**, **Turn on Rim**, and **Inspection on same vehicle > 2 times**.")

                job_type_col_anom = None
                for jtc in ["JOB_TYPE_CODE", "JOB_TYPE_CD", "JOB_TYPE_CODE_PO", "JOB_TYPE_CD_PO", "JOB_TYPE_CODE_BILL", "JOB_TYPE_CD_BILL"]:
                    if jtc in filtered.columns:
                        job_type_col_anom = jtc
                        break

                vehicle_col_anom = None
                for vhc in ["VEHICLE_ID", "VEHICLE_ID_PO", "LICENCE_PLATE", "LICENCE_PLATE_ID", "VEHICLE_REG_NBR"]:
                    if vhc in filtered.columns:
                        vehicle_col_anom = vhc
                        break

                if job_type_col_anom is None or vehicle_col_anom is None:
                    st.warning("Missing JOB_TYPE or VEHICLE column — cannot compute job type anomaly heatmap.")
                else:
                    job_types_lower = filtered[job_type_col_anom].astype(str).str.lower().str.strip()

                    reg_breakdown = filtered[job_types_lower.str.contains("regular.*breakdown|breakdown.*regular", na=False)]
                    reg_job = filtered[job_types_lower.str.contains("regular.*job|job.*regular", na=False)]
                    turn_on_rim = filtered[job_types_lower.str.contains("turn.*on.*rim", na=False)]

                    inspection_counts = (
                        filtered[job_types_lower.str.contains("inspection", na=False)]
                        .groupby([vendor_col_anom, vehicle_col_anom])
                        .size()
                        .reset_index(name="INSP_COUNT")
                    )
                    insp_exceed = inspection_counts[inspection_counts["INSP_COUNT"] > 2]
                    insp_vendors = set(insp_exceed[vendor_col_anom].unique()) if not insp_exceed.empty else set()

                    anomaly_rows = []
                    all_vendors = set()

                    if not reg_breakdown.empty:
                        for v in reg_breakdown[vendor_col_anom].unique():
                            cnt = int((reg_breakdown[vendor_col_anom] == v).sum())
                            anomaly_rows.append({"Vendor": v, "Anomaly": "Regular Breakdown", "Count": cnt})
                            all_vendors.add(v)

                    if not reg_job.empty:
                        for v in reg_job[vendor_col_anom].unique():
                            cnt = int((reg_job[vendor_col_anom] == v).sum())
                            anomaly_rows.append({"Vendor": v, "Anomaly": "Regular Job", "Count": cnt})
                            all_vendors.add(v)

                    if not turn_on_rim.empty:
                        for v in turn_on_rim[vendor_col_anom].unique():
                            cnt = int((turn_on_rim[vendor_col_anom] == v).sum())
                            anomaly_rows.append({"Vendor": v, "Anomaly": "Turn on Rim", "Count": cnt})
                            all_vendors.add(v)

                    if not insp_exceed.empty:
                        for v in insp_vendors:
                            cnt = int(insp_exceed[insp_exceed[vendor_col_anom] == v]["INSP_COUNT"].sum())
                            anomaly_rows.append({"Vendor": v, "Anomaly": "Inspection >2 Same Vehicle", "Count": cnt})
                            all_vendors.add(v)

                    if anomaly_rows:
                        anomaly_df = pd.DataFrame(anomaly_rows)
                        pivot_anomaly = anomaly_df.pivot_table(
                            index="Vendor", columns="Anomaly", values="Count", fill_value=0, aggfunc="sum"
                        ).sort_values(by=list(anomaly_df["Anomaly"].unique()), ascending=False)

                        st.dataframe(pivot_anomaly, use_container_width=True)

                        with st.container(horizontal=True):
                            st.metric("Vendors Flagged", len(all_vendors), border=True)
                            st.metric("Total Flags", sum(r["Count"] for r in anomaly_rows), border=True)
                    else:
                        st.success("No job type anomaly patterns detected in the filtered data.")

                st.divider()
                st.subheader("Top Offending Vendors")

                vendor_scores = (
                    combined_outliers.groupby(vendor_col_anom)
                    .agg(
                        ANOMALY_COUNT=("JOB_ID_COMBINED", "count"),
                        TOTAL_PRICE_DIFF=("PRICE_DIFF", "sum"),
                        AVG_PRICE_DIFF=("PRICE_DIFF", "mean"),
                        TOTAL_QTY_DIFF=("QTY_DIFF", "sum"),
                        UNIQUE_MATERIALS=("MATERIAL_ID_COMBINED", "nunique"),
                    )
                    .reset_index()
                    .sort_values("ANOMALY_COUNT", ascending=False)
                    .head(20)
                    .reset_index(drop=True)
                )
                vendor_scores.index = vendor_scores.index + 1
                vendor_scores.index.name = "Rank"

                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.markdown("**Anomaly Count by Vendor (Top 20)**")
                        st.bar_chart(vendor_scores.set_index(vendor_col_anom)["ANOMALY_COUNT"])
                with col2:
                    with st.container(border=True):
                        st.markdown("**Total Price Difference by Vendor (Top 20)**")
                        st.bar_chart(vendor_scores.set_index(vendor_col_anom)["TOTAL_PRICE_DIFF"])

                st.dataframe(
                    vendor_scores.rename(columns={
                        vendor_col_anom: "Vendor",
                        "ANOMALY_COUNT": "Anomalies",
                        "TOTAL_PRICE_DIFF": "Total Price Diff €",
                        "AVG_PRICE_DIFF": "Avg Price Diff €",
                        "TOTAL_QTY_DIFF": "Total Qty Diff",
                        "UNIQUE_MATERIALS": "Unique Materials",
                    }).round(2),
                    use_container_width=True,
                )

                st.divider()
                st.subheader("Material-Level Anomaly Breakdown")
                st.markdown("Which materials are most commonly involved in anomalous vendor behaviour?")

                mat_scores = (
                    combined_outliers.groupby([mat_desc_anom, "MATERIAL_ID_COMBINED"])
                    .agg(
                        ANOMALY_COUNT=("JOB_ID_COMBINED", "count"),
                        VENDORS_INVOLVED=(vendor_col_anom, "nunique"),
                        TOTAL_PRICE_DIFF=("PRICE_DIFF", "sum"),
                        TOTAL_QTY_DIFF=("QTY_DIFF", "sum"),
                    )
                    .reset_index()
                    .sort_values("ANOMALY_COUNT", ascending=False)
                    .head(25)
                    .reset_index(drop=True)
                )

                with st.container(border=True):
                    st.markdown("**Top 25 Materials by Anomaly Frequency**")
                    st.bar_chart(mat_scores.set_index(mat_desc_anom)["ANOMALY_COUNT"].head(15))

                st.dataframe(
                    mat_scores.rename(columns={
                        mat_desc_anom: "Material Description",
                        "MATERIAL_ID_COMBINED": "Material ID",
                        "ANOMALY_COUNT": "Anomalies",
                        "VENDORS_INVOLVED": "Vendors Involved",
                        "TOTAL_PRICE_DIFF": "Total Price Diff €",
                        "TOTAL_QTY_DIFF": "Total Qty Diff",
                    }).round(2),
                    hide_index=True,
                    use_container_width=True,
                )

                st.divider()
                st.subheader("Vendor × Material Detail — Anomalous Records")

                detail_cols = ["JOB_ID_COMBINED", vendor_col_anom, "MATERIAL_ID_COMBINED"]
                if mat_desc_anom:
                    detail_cols.append(mat_desc_anom)
                detail_cols.extend(["NET_PRICE_EURO", "BILLED_AMT_EURO", "PRICE_DIFF", "PO_QTY", "BILLED_QTY", "QTY_DIFF"])

                available_detail = [c for c in detail_cols if c in combined_outliers.columns]
                detail_df = combined_outliers[available_detail].sort_values("PRICE_DIFF", ascending=False, key=abs)

                rename_detail = {
                    "JOB_ID_COMBINED": "Job ID",
                    vendor_col_anom: "Vendor",
                    "MATERIAL_ID_COMBINED": "Material ID",
                    "NET_PRICE_EURO": "PO Net Price €",
                    "BILLED_AMT_EURO": "Billed Amt €",
                    "PRICE_DIFF": "Price Diff €",
                    "PO_QTY": "PO Qty",
                    "BILLED_QTY": "Billed Qty",
                    "QTY_DIFF": "Qty Diff",
                }
                if mat_desc_anom:
                    rename_detail[mat_desc_anom] = "Material Desc"

                st.dataframe(
                    detail_df.rename(columns=rename_detail).round(2),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.success("No anomalies detected in the current data.")
        elif vendor_col_anom is None:
            st.warning("No VENDOR column found. Ensure your CSV includes VENDOR_NAME or VEND_NAME.")
        elif mat_desc_anom is None:
            st.warning("No MATERIAL_DESC column found. Ensure your CSV includes MATERIAL_DESC or MATL_DESC.")

    st.divider()
    st.subheader("PO Net Price > Billed Amount — Vendor Severity Flags")
    st.markdown("Identifies vendors where PO value significantly exceeds what was actually billed per month. "
                "Large gaps may indicate phantom POs, inflated pricing, or services not rendered.")
    st.markdown("""
- **Groups by vendor and month**, calculates the underbilled gap
- **Assigns severity:** CRITICAL (≥80%), HIGH (≥60%), MEDIUM (≥40%), LOW (<40%)
- **Flags malpractice reasons** (phantom POs, inflated pricing, services not rendered)
- **Shows a bar chart** of top 20 vendors and a full detail table
""")

    vendor_col_anomaly = None
    for vc in ["VENDOR_NAME", "VEND_NAME", "VENDOR_NAME_PO", "VEND_NAME_BILL"]:
        if vc in filtered.columns:
            vendor_col_anomaly = vc
            break

    month_col = None
    for mc in ["PO_POSTING_MONTH", "PO_POSTING_MONTH_PO", "BILLING_MONTH"]:
        if mc in filtered.columns:
            month_col = mc
            break

    if vendor_col_anomaly is None:
        st.warning("No VENDOR column found. Ensure your CSV includes VENDOR_NAME or VEND_NAME.")
    elif filtered["NET_PRICE_EURO"].isna().all() and filtered["BILLED_AMT_EURO"].isna().all():
        st.info("No price data available for this analysis.")
    else:
        analysis_df = filtered[[vendor_col_anomaly, "NET_PRICE_EURO", "BILLED_AMT_EURO",
                                "PO_QTY", "BILLED_QTY", "JOB_ID_COMBINED"]].copy()
        if month_col and month_col in filtered.columns:
            analysis_df["MONTH"] = filtered[month_col]
        else:
            analysis_df["MONTH"] = filtered["PO_POSTING_DATE_COMBINED"].dt.month

        vendor_monthly = (
            analysis_df.groupby([vendor_col_anomaly, "MONTH"])
            .agg(
                TOTAL_PO_NET_PRICE=("NET_PRICE_EURO", "sum"),
                TOTAL_BILLED_AMT=("BILLED_AMT_EURO", "sum"),
                PO_JOBS=("JOB_ID_COMBINED", "count"),
            )
            .reset_index()
        )
        vendor_monthly["UNDERBILLED_AMT"] = (
            vendor_monthly["TOTAL_PO_NET_PRICE"].fillna(0) - vendor_monthly["TOTAL_BILLED_AMT"].fillna(0)
        )
        vendor_monthly["UNDERBILLED_PCT"] = (
            vendor_monthly["UNDERBILLED_AMT"] / vendor_monthly["TOTAL_PO_NET_PRICE"].replace(0, np.nan) * 100
        )

        flagged = vendor_monthly[vendor_monthly["UNDERBILLED_AMT"] > 0].copy()

        def assign_severity(pct):
            if pct >= 80:
                return "CRITICAL"
            elif pct >= 60:
                return "HIGH"
            elif pct >= 40:
                return "MEDIUM"
            else:
                return "LOW"

        def assign_flag(row):
            if row["UNDERBILLED_PCT"] >= 80:
                return "Extreme gap — vendor may be inflating PO prices or not rendering services"
            elif row["UNDERBILLED_PCT"] >= 60:
                return "High underbilling — possible phantom POs or delayed invoicing"
            elif row["UNDERBILLED_PCT"] >= 40:
                return "Significant gap — partial services or split billing across periods"
            else:
                return "Moderate gap — monitor for trends"

        if flagged.empty:
            st.success("No underbilling anomalies detected.")
        else:
            flagged["SEVERITY"] = flagged["UNDERBILLED_PCT"].apply(assign_severity)
            flagged["MALPRACTICE_FLAG"] = flagged.apply(assign_flag, axis=1)
            flagged = flagged.sort_values("UNDERBILLED_AMT", ascending=False).reset_index(drop=True)

            severity_counts = flagged["SEVERITY"].value_counts()
            with st.container(horizontal=True):
                st.metric("Critical", severity_counts.get("CRITICAL", 0), border=True)
                st.metric("High", severity_counts.get("HIGH", 0), border=True)
                st.metric("Medium", severity_counts.get("MEDIUM", 0), border=True)
                st.metric("Low", severity_counts.get("LOW", 0), border=True)

            with st.container(border=True):
                st.markdown("**Top 20 Vendors by Underbilled Amount**")
                top20 = flagged.head(20).set_index(vendor_col_anomaly)[["TOTAL_PO_NET_PRICE", "TOTAL_BILLED_AMT"]]
                st.bar_chart(top20)

            st.dataframe(
                flagged.rename(columns={
                    vendor_col_anomaly: "Vendor",
                    "MONTH": "Month",
                    "TOTAL_PO_NET_PRICE": "PO Net Price €",
                    "TOTAL_BILLED_AMT": "Billed Amt €",
                    "UNDERBILLED_AMT": "Underbilled €",
                    "UNDERBILLED_PCT": "Underbilled %",
                    "PO_JOBS": "PO Jobs",
                    "SEVERITY": "Severity",
                    "MALPRACTICE_FLAG": "Flag",
                }).round(2),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()
    st.subheader("Material Usage per Job & Vehicle — PO vs Billing Discrepancies")
    st.markdown("Shows per Job ID which Material IDs and how many times they were used on a vehicle, "
                "highlighting where PO quantity differs from Billing quantity.")
    st.markdown("""
- **PO Qty**: Total units ordered (sum of PO_QTY across all line items for that Job + Material + Vehicle)
- **Times Used**: Number of separate PO line entries for that combination
- **Billed Qty**: Total units actually invoiced
- **Qty Difference**: PO Qty minus Billed Qty — positive means more was ordered than billed
""")

    vehicle_col_mat = None
    for vc in ["VEHICLE_ID", "VEHICLE_ID_PO", "LICENCE_PLATE", "LICENCE_PLATE_ID"]:
        if vc in filtered.columns:
            vehicle_col_mat = vc
            break

    licence_col_mat = None
    if vehicle_col_mat and "VEHICLE" in vehicle_col_mat.upper():
        for lc in ["LICENCE_PLATE", "LICENCE_PLATE_ID"]:
            if lc in filtered.columns:
                licence_col_mat = lc
                break

    mat_desc_col = None
    for mc in ["MATERIAL_DESC", "MATERIAL_DESC_PO", "MATL_DESC", "MATL_DESC_PO", "MATL_DESC_BILL"]:
        if mc in filtered.columns:
            mat_desc_col = mc
            break

    required_cols = ["JOB_ID_COMBINED", "MATERIAL_ID_COMBINED", "PO_QTY", "BILLED_QTY"]
    if not all(c in filtered.columns for c in required_cols):
        st.warning("Missing required columns (JOB_ID, MATERIAL_ID, PO_QTY, or BILLED_QTY).")
    else:
        group_cols = ["JOB_ID_COMBINED", "MATERIAL_ID_COMBINED"]
        if mat_desc_col:
            group_cols.append(mat_desc_col)
        if vehicle_col_mat:
            group_cols.append(vehicle_col_mat)
        if licence_col_mat:
            group_cols.append(licence_col_mat)

        mat_usage = (
            filtered.groupby(group_cols)
            .agg(
                PO_QTY_TOTAL=("PO_QTY", "sum"),
                BILLED_QTY_TOTAL=("BILLED_QTY", "sum"),
                PO_LINES=("PO_QTY", "count"),
            )
            .reset_index()
        )
        mat_usage["QTY_DIFF"] = mat_usage["PO_QTY_TOTAL"].fillna(0) - mat_usage["BILLED_QTY_TOTAL"].fillna(0)
        mat_usage["HAS_DISCREPANCY"] = mat_usage["QTY_DIFF"] != 0

        discrepancies = mat_usage[mat_usage["HAS_DISCREPANCY"]].sort_values(
            "QTY_DIFF", ascending=False, key=abs
        ).reset_index(drop=True)

        with st.container(horizontal=True):
            st.metric("Total Job-Material Combos", len(mat_usage), border=True)
            st.metric("With Discrepancies", len(discrepancies), border=True)
            st.metric(
                "Discrepancy Rate",
                f"{len(discrepancies) / max(len(mat_usage), 1) * 100:.1f}%",
                border=True,
            )

        if discrepancies.empty:
            st.success("No quantity discrepancies found between PO and Billing.")
        else:
            rename_map = {
                "JOB_ID_COMBINED": "Job ID",
                "MATERIAL_ID_COMBINED": "Material ID",
                "PO_QTY_TOTAL": "PO Qty",
                "BILLED_QTY_TOTAL": "Billed Qty",
                "QTY_DIFF": "Qty Difference",
                "PO_LINES": "Times Used",
            }
            if mat_desc_col:
                rename_map[mat_desc_col] = "Material Description"
            if vehicle_col_mat:
                rename_map[vehicle_col_mat] = "Vehicle ID"
            if licence_col_mat:
                rename_map[licence_col_mat] = "Licence Plate"

            st.dataframe(
                discrepancies.drop(columns=["HAS_DISCREPANCY"]).rename(columns=rename_map),
                hide_index=True,
                use_container_width=True,
            )

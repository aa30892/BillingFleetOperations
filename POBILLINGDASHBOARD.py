# PO vs Billing dashboard — single pre-joined file input
# Co-authored with CoCo
import streamlit as st
import pandas as pd
import numpy as np
import io
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="PO vs Billing Dashboard", layout="wide")
st.title("PO vs Billing Comparison")

# ------------------------------------------------------------------
# SIDEBAR — single file source
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Data Source")
    source_type = st.radio(
        "Choose Data Source:",
        ["Upload File", "Load from Server"],
        index=0,
    )

    data_file = None

    if source_type == "Upload File":
        data_file = st.file_uploader(
            "Upload pre-joined PO/Billing data (CSV or Parquet)",
            type=["csv", "parquet"],
        )
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pattern_csv = os.path.join(current_dir, "*.csv")
        pattern_pq = os.path.join(current_dir, "*.parquet")
        server_files = sorted(
            glob.glob(pattern_csv) + glob.glob(pattern_pq),
            key=os.path.getmtime,
            reverse=True,
        )
        if server_files:
            display_map = {os.path.basename(f): f for f in server_files}
            selected_name = st.selectbox("Select File", list(display_map.keys()))
            data_file = open(display_map[selected_name], "rb")
        else:
            st.error(f"No CSV/Parquet files found in: {current_dir}")

if data_file is None:
    st.info("Please provide a pre-joined PO/Billing data file to proceed.")
    st.stop()

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
JOIN_KEYS = ["JOB_NOTIFICATION_ID", "MATERIAL_ID", "VEHICLE_ID", "PO_POSTING_DATE"]


@st.cache_data(show_spinner="Loading data...")
def load_data(file_obj):
    name = file_obj.name

    if name.endswith(".parquet"):
        if hasattr(file_obj, "getvalue"):
            df = pd.read_parquet(io.BytesIO(file_obj.getvalue()))
        else:
            file_obj.seek(0)
            df = pd.read_parquet(file_obj)
    else:
        if hasattr(file_obj, "getvalue"):
            df = pd.read_csv(io.BytesIO(file_obj.getvalue()), low_memory=False)
        else:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, low_memory=False)

    df.columns = df.columns.str.upper().str.strip()

    if "PO_POSTING_DATE" in df.columns:
        df["PO_POSTING_DATE"] = pd.to_datetime(df["PO_POSTING_DATE"], errors="coerce").dt.normalize()
    if "BILLING_DT" in df.columns:
        df["BILLING_DT"] = pd.to_datetime(df["BILLING_DT"], errors="coerce").dt.normalize()

    # --- combined columns (coalesce PO / billing variants) ---
    def _coalesce(df, a, b):
        if a in df.columns and b in df.columns:
            return df[a].combine_first(df[b])
        if a in df.columns:
            return df[a]
        if b in df.columns:
            return df[b]
        return pd.Series("N/A", index=df.index)

    df["CONTRACT_ID_COMBINED"] = (
        _coalesce(df, "FOS_CONTRACT_ID", "CONTRACT_ID")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    df["CUSTOMER_NAME_COMBINED"] = (
        _coalesce(df, "CUSTOMER_NAME", "CUST_NAME").fillna("Unknown").astype(str)
    )
    df["MATERIAL_ID_COMBINED"] = (
        _coalesce(df, "MATERIAL_ID", "MATL_ID_TRIM")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    df["JOB_ID_COMBINED"] = (
        _coalesce(df, "JOB_NOTIFICATION_ID", "JOB_ID")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    df["PO_POSTING_DATE_COMBINED"] = _coalesce(df, "PO_POSTING_DATE", "BILLING_DT")

    # --- compute TIMES_USED_PO / TIMES_BILLED per key group ---
    present_keys = [k for k in JOIN_KEYS if k in df.columns]
    if present_keys:
        if "NET_PRICE_EURO" in df.columns:
            po_counts = (
                df[df["NET_PRICE_EURO"].notna()]
                .groupby(present_keys, dropna=False)
                .size()
                .reset_index(name="TIMES_USED_PO")
            )
            df = df.merge(po_counts, on=present_keys, how="left")
            df["TIMES_USED_PO"] = df["TIMES_USED_PO"].fillna(0).astype(int)
        else:
            df["TIMES_USED_PO"] = 0

        if "BILLED_AMT_EURO" in df.columns:
            bill_counts = (
                df[df["BILLED_AMT_EURO"].notna()]
                .groupby(present_keys, dropna=False)
                .size()
                .reset_index(name="TIMES_BILLED")
            )
            df = df.merge(bill_counts, on=present_keys, how="left")
            df["TIMES_BILLED"] = df["TIMES_BILLED"].fillna(0).astype(int)
        else:
            df["TIMES_BILLED"] = 0
    else:
        df["TIMES_USED_PO"] = 0
        df["TIMES_BILLED"] = 0

    # lightweight dtype optimization
    category_cols = [
        "FLEET_TYPE", "CUSTOMER_NAME", "JOB_TYPE_CODE", "MATERIAL_DESC",
        "VENDOR_NAME", "CUST_NAME", "MATL_DESC", "JOB_TYPE_CD", "VEND_NAME",
        "CONTRACT_ID_COMBINED", "CUSTOMER_NAME_COMBINED", "MATERIAL_ID_COMBINED",
    ]
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


try:
    merged = load_data(data_file)
    if source_type == "Load from Server" and hasattr(data_file, "close"):
        data_file.close()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# ------------------------------------------------------------------
# DERIVED COLUMNS
# ------------------------------------------------------------------
vendor_col = next(
    (c for c in ["VENDOR_NAME", "VEND_NAME"] if c in merged.columns), None
)

merged["PRICE_DIFF"] = merged.get("NET_PRICE_EURO", pd.Series(0, index=merged.index)).fillna(0) - merged.get("BILLED_AMT_EURO", pd.Series(0, index=merged.index)).fillna(0)
merged["QTY_DIFF"] = merged.get("PO_QTY", pd.Series(0, index=merged.index)).fillna(0) - merged.get("BILLED_QTY", pd.Series(0, index=merged.index)).fillna(0)

if "FLEET_CUSTOMER_GROUP" in merged.columns:
    merged["CUSTOMER_GROUP"] = merged["FLEET_CUSTOMER_GROUP"].fillna("Unknown").astype(str)
else:
    merged["CUSTOMER_GROUP"] = merged["CUSTOMER_NAME_COMBINED"]

# ------------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    contracts = sorted(
        x for x in merged["CONTRACT_ID_COMBINED"].unique()
        if isinstance(x, str) and x not in ("nan", "") and x.strip()
    )
    selected_contracts = st.multiselect("Contract ID", contracts, key="filter_contracts")

    customer_groups = sorted(
        x for x in merged["CUSTOMER_GROUP"].dropna().unique()
        if isinstance(x, str) and x not in ("nan", "") and x.strip()
    )
    selected_customer_groups = st.multiselect(
        "Customer Group",
        customer_groups,
        help="Customers grouped by shared first keyword",
        key="filter_customer_groups",
    )

    customers = sorted(
        x for x in merged["CUSTOMER_NAME_COMBINED"].unique()
        if isinstance(x, str) and x not in ("nan", "") and x.strip()
    )
    selected_customers = st.multiselect("Customer Name", customers, key="filter_customers")

    if vendor_col:
        vendors = sorted(str(x) for x in merged[vendor_col].dropna().unique() if str(x).strip())
        selected_vendors = st.multiselect("Vendor Name", vendors, key="filter_vendors")
    else:
        selected_vendors = []

    job_ids = sorted(
        str(x) for x in merged["JOB_ID_COMBINED"].unique()
        if str(x) not in ("nan", "") and str(x).strip()
    )
    selected_job_ids = st.multiselect("Job ID", job_ids, key="filter_job_ids")

    material_ids = sorted(
        str(x) for x in merged["MATERIAL_ID_COMBINED"].unique()
        if str(x) not in ("nan", "") and str(x).strip()
    )
    selected_material_ids = st.multiselect("Material ID", material_ids, key="filter_material_ids")

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

# ------------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------------
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
if selected_material_ids:
    filtered = filtered[filtered["MATERIAL_ID_COMBINED"].astype(str).isin(selected_material_ids)]
if date_range and len(date_range) == 2:
    filtered = filtered[
        (filtered["PO_POSTING_DATE_COMBINED"] >= pd.Timestamp(date_range[0]))
        & (filtered["PO_POSTING_DATE_COMBINED"] <= pd.Timestamp(date_range[1]))
    ]

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_dashboard, tab_vendor, tab_repair, tab_anomaly = st.tabs(
    ["Dashboard", "Vendor", "Contract Analysis", "AI Anomaly Analysis"]
)

# ========================== DASHBOARD ==========================
with tab_dashboard:
    total_po = filtered["NET_PRICE_EURO"].fillna(0).sum()
    total_billed = filtered["BILLED_AMT_EURO"].fillna(0).sum()
    price_diff = total_po - total_billed
    diff_color = "green" if total_billed >= total_po else "red"

    with st.container(horizontal=True):
        st.metric("Total Records", f"{len(filtered):,}", border=True)
        st.metric("Total PO €", f"{total_po:,.2f}", border=True)
        st.metric("Total Billed €", f"{total_billed:,.2f}", border=True)
    with st.container(horizontal=True):
        st.markdown(
            f"**Total Price Diff (PO - Billed) €:** "
            f"<span style='color:{diff_color}; font-size:1.3em; font-weight:bold'>{price_diff:,.2f}</span>",
            unsafe_allow_html=True,
        )
        st.metric("Total Qty Diff (PO - Billed)", f"{filtered['QTY_DIFF'].sum():,.1f}", border=True)

    st.subheader("Differences by Material ID")
    summary = (
        filtered.groupby("MATERIAL_ID_COMBINED", dropna=False, observed=True)
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
            chart_qty = summary[["MATERIAL_ID_COMBINED", "PO_QTY", "BILLED_QTY"]].set_index("MATERIAL_ID_COMBINED")
            st.bar_chart(chart_qty)

    st.subheader("Times Used PO vs Times Billed")
    usage_summary = (
        filtered.groupby("MATERIAL_ID_COMBINED", dropna=False, observed=True)
        .agg(
            TIMES_USED_PO=("TIMES_USED_PO", "sum"),
            TIMES_BILLED=("TIMES_BILLED", "sum"),
        )
        .reset_index()
    )
    usage_summary["USAGE_DIFF"] = usage_summary["TIMES_USED_PO"] - usage_summary["TIMES_BILLED"]

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**TIMES_USED_PO vs TIMES_BILLED by Material**")
            chart_usage = (
                usage_summary[["MATERIAL_ID_COMBINED", "TIMES_USED_PO", "TIMES_BILLED"]]
                .sort_values("TIMES_USED_PO", ascending=False)
                .set_index("MATERIAL_ID_COMBINED")
            )
            st.bar_chart(chart_usage)
    with col2:
        with st.container(border=True):
            st.markdown("**Usage Difference (PO - Billed)**")
            chart_diff = (
                usage_summary[["MATERIAL_ID_COMBINED", "USAGE_DIFF"]]
                .sort_values("USAGE_DIFF", ascending=False)
                .set_index("MATERIAL_ID_COMBINED")
            )
            st.bar_chart(chart_diff)

    st.dataframe(
        usage_summary.rename(columns={
            "MATERIAL_ID_COMBINED": "Material ID",
            "TIMES_USED_PO": "Times Used PO",
            "TIMES_BILLED": "Times Billed",
            "USAGE_DIFF": "Difference",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Detail Table")
    detail_cols = [
        "JOB_ID_COMBINED", "MATERIAL_ID_COMBINED", "CONTRACT_ID_COMBINED",
        "CUSTOMER_NAME_COMBINED", "PO_POSTING_DATE_COMBINED",
        "NET_PRICE_EURO", "BILLED_AMT_EURO", "PRICE_DIFF",
        "PO_QTY", "BILLED_QTY", "QTY_DIFF",
        "TIMES_USED_PO", "TIMES_BILLED",
    ]
    detail_cols = [c for c in detail_cols if c in filtered.columns]
    st.dataframe(
        filtered[detail_cols].rename(columns={
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
            "TIMES_USED_PO": "Times Used PO",
            "TIMES_BILLED": "Times Billed",
        }),
        hide_index=True,
        use_container_width=True,
    )

# ========================== VENDOR ==========================
with tab_vendor:
    st.subheader("Vendor Analysis")

    vendor_display = next(
        (c for c in ["VENDOR_NAME", "VEND_NAME"] if c in filtered.columns), None
    )

    vendor_tab_df = filtered
    if vendor_display:
        vendor_list = sorted(str(x) for x in filtered[vendor_display].dropna().unique() if str(x).strip())
        selected_tab_vendors = st.multiselect("Filter by Vendor", vendor_list, key="vendor_tab_filter")
        if selected_tab_vendors:
            vendor_tab_df = filtered[filtered[vendor_display].astype(str).isin(selected_tab_vendors)]

    fleet_col_v = next(
        (c for c in ["FLEET_TYPE", "FLEET_TYPE_PO"] if c in filtered.columns), None
    )
    mat_desc_v = next(
        (c for c in ["MATERIAL_DESC", "MATL_DESC"] if c in filtered.columns), None
    )
    licence_col_v = next(
        (c for c in ["LICENCE_PLATE", "LICENCE_PLATE_ID"] if c in filtered.columns), None
    )

    group_cols_v = ["JOB_ID_COMBINED"]
    if licence_col_v:
        group_cols_v.append(licence_col_v)
    group_cols_v.append("MATERIAL_ID_COMBINED")
    if mat_desc_v:
        group_cols_v.append(mat_desc_v)
    if fleet_col_v:
        group_cols_v.append(fleet_col_v)
    if vendor_display:
        group_cols_v.append(vendor_display)

    vendor_agg = {"TOTAL_PO_EURO": ("NET_PRICE_EURO", "sum"), "TOTAL_BILLED_EURO": ("BILLED_AMT_EURO", "sum")}
    vendor_summary = (
        vendor_tab_df.groupby(group_cols_v, dropna=False, observed=True)
        .agg(**vendor_agg)
        .reset_index()
    )
    vendor_summary["EURO_DIFF"] = vendor_summary["TOTAL_PO_EURO"].fillna(0) - vendor_summary["TOTAL_BILLED_EURO"].fillna(0)
    vendor_summary = vendor_summary.sort_values("EURO_DIFF", ascending=False).reset_index(drop=True)

    rename_map = {
        "JOB_ID_COMBINED": "Job ID",
        "MATERIAL_ID_COMBINED": "Material ID",
        "TOTAL_PO_EURO": "Total PO €",
        "TOTAL_BILLED_EURO": "Total Billed €",
        "EURO_DIFF": "Euro Difference",
    }
    if licence_col_v:
        rename_map[licence_col_v] = "Licence Plate"
    if mat_desc_v:
        rename_map[mat_desc_v] = "Material Description"
    if fleet_col_v:
        rename_map[fleet_col_v] = "Fleet Type"
    if vendor_display:
        rename_map[vendor_display] = "Vendor"

    st.dataframe(
        vendor_summary.rename(columns=rename_map),
        hide_index=True,
        use_container_width=True,
    )

# ========================== CONTRACT ANALYSIS ==========================
with tab_repair:
    st.subheader("Fleet Type Analysis — Materials per Job & Vehicle")
    fleet_col = next(
        (fc for fc in ["FLEET_TYPE", "FLEET_TYPE_PO"] if fc in filtered.columns), None
    )
    vehicle_col_fleet = next(
        (vc for vc in ["VEHICLE_ID", "LICENCE_PLATE", "LICENCE_PLATE_ID"] if vc in filtered.columns), None
    )
    licence_col_fleet = next(
        (lc for lc in ["LICENCE_PLATE", "LICENCE_PLATE_ID"]
         if vehicle_col_fleet and "VEHICLE" in vehicle_col_fleet.upper() and lc in filtered.columns),
        None,
    )
    mat_desc_fleet = next(
        (mc for mc in ["MATERIAL_DESC", "MATL_DESC"] if mc in filtered.columns), None
    )

    if fleet_col is None:
        st.warning("No FLEET_TYPE column found.")
    elif vehicle_col_fleet is None:
        st.warning("No VEHICLE column found.")
    else:
        fleet_types = sorted(str(x) for x in filtered[fleet_col].dropna().unique() if str(x).strip())
        selected_fleet = st.multiselect("Filter by Fleet Type", fleet_types, key="fleet_filter")
        fleet_df = filtered if not selected_fleet else filtered[filtered[fleet_col].astype(str).isin(selected_fleet)]

        group_cols_fleet = ["JOB_ID_COMBINED", vehicle_col_fleet]
        if licence_col_fleet:
            group_cols_fleet.append(licence_col_fleet)
        group_cols_fleet.append("MATERIAL_ID_COMBINED")
        if mat_desc_fleet:
            group_cols_fleet.append(mat_desc_fleet)
        group_cols_fleet.append(fleet_col)

        fleet_agg_dict = {
            "TIMES_USED_PO": ("TIMES_USED_PO", "sum"),
            "TIMES_BILLED": ("TIMES_BILLED", "sum"),
        }
        if "NET_PRICE_EURO" in fleet_df.columns:
            fleet_agg_dict["TOTAL_PO_EURO"] = ("NET_PRICE_EURO", "sum")
        if "BILLED_AMT_EURO" in fleet_df.columns:
            fleet_agg_dict["TOTAL_BILLED_EURO"] = ("BILLED_AMT_EURO", "sum")

        fleet_analysis = (
            fleet_df.groupby(group_cols_fleet, dropna=False, observed=True)
            .agg(**fleet_agg_dict)
            .reset_index()
        )
        if "TOTAL_PO_EURO" in fleet_analysis.columns and "TOTAL_BILLED_EURO" in fleet_analysis.columns:
            fleet_analysis["EURO_DIFF"] = (
                fleet_analysis["TOTAL_PO_EURO"].fillna(0) - fleet_analysis["TOTAL_BILLED_EURO"].fillna(0)
            )
        fleet_analysis = fleet_analysis.sort_values("TIMES_USED_PO", ascending=False).reset_index(drop=True)

        st.dataframe(
            fleet_analysis.rename(columns={
                "JOB_ID_COMBINED": "Job ID",
                vehicle_col_fleet: "Vehicle ID",
                "MATERIAL_ID_COMBINED": "Material ID",
                fleet_col: "Fleet Type",
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.subheader("Customer Group Analysis — PO vs Billing")
    if "CUSTOMER_GROUP" in filtered.columns:
        cg_agg_dict = {
            "TIMES_USED_PO": ("TIMES_USED_PO", "sum"),
            "TIMES_BILLED": ("TIMES_BILLED", "sum"),
        }
        if "NET_PRICE_EURO" in filtered.columns:
            cg_agg_dict["TOTAL_PO_EURO"] = ("NET_PRICE_EURO", "sum")
        if "BILLED_AMT_EURO" in filtered.columns:
            cg_agg_dict["TOTAL_BILLED_EURO"] = ("BILLED_AMT_EURO", "sum")
        cg_analysis = (
            filtered.groupby("CUSTOMER_GROUP", dropna=False, observed=True)
            .agg(**cg_agg_dict)
            .reset_index()
        )
        st.dataframe(cg_analysis, hide_index=True, use_container_width=True)

# ========================== AI ANOMALY ANALYSIS ==========================
with tab_anomaly:
    st.subheader("AI Anomaly Analysis")

    st.info("""
### How AI Anomaly Detection Works

This analysis combines three complementary approaches:

#### 1. Rule-Based Checks
The dashboard identifies common reconciliation issues:

• PO exists but no billing found
• Billing exists but no PO found
• Price difference exceeds 20%
• Quantity difference exceeds 20%

Each triggered rule contributes to the Risk Score.

#### 2. Statistical Outlier Detection
Z-Scores are calculated for:

• Price Difference (€)
• Quantity Difference
• PO Amount (€)
• Billed Amount (€)
• PO Quantity
• Billed Quantity

Records more than 3 standard deviations away from the population average are flagged as outliers.

#### 3. AI Pattern Detection
An Isolation Forest machine-learning algorithm analyzes:

• Price Difference
• Quantity Difference
• PO Amount
• Billed Amount
• PO Quantity
• Billed Quantity
• Times Used in PO
• Times Billed

Unlike simple thresholds, the model identifies unusual combinations of values that differ from the majority of transactions.

### Risk Levels

🔴 HIGH:
Multiple anomaly indicators found.

🟡 MEDIUM:
Some unusual characteristics detected.

🟢 LOW:
No significant anomaly detected.

### AI Driver

The AI Driver identifies the strongest contributing anomaly dimension:

• PRICE_DIFF
• QTY_DIFF
• NET_PRICE_EURO
• BILLED_AMT_EURO
• TIMES_USED_PO
• TIMES_BILLED
""")

    anomaly_df = filtered.copy()

    # ==================== RULE BASED ====================
    anomaly_df["RULE_NO_BILLING"] = (
        anomaly_df["NET_PRICE_EURO"].fillna(0) > 0
    ) & (
        anomaly_df["BILLED_AMT_EURO"].fillna(0) == 0
    )
    anomaly_df["RULE_NO_PO"] = (
        anomaly_df["NET_PRICE_EURO"].fillna(0) == 0
    ) & (
        anomaly_df["BILLED_AMT_EURO"].fillna(0) > 0
    )

    anomaly_df["PRICE_DIFF_PCT"] = np.where(
        anomaly_df["NET_PRICE_EURO"].fillna(0) != 0,
        (anomaly_df["PRICE_DIFF"] / anomaly_df["NET_PRICE_EURO"].replace(0, np.nan)) * 100,
        0,
    )
    anomaly_df["QTY_DIFF_PCT"] = np.where(
        anomaly_df["PO_QTY"].fillna(0) != 0,
        (anomaly_df["QTY_DIFF"] / anomaly_df["PO_QTY"].replace(0, np.nan)) * 100,
        0,
    )

    anomaly_df["RULE_HIGH_PRICE_DIFF"] = anomaly_df["PRICE_DIFF_PCT"].abs() > 20
    anomaly_df["RULE_HIGH_QTY_DIFF"] = anomaly_df["QTY_DIFF_PCT"].abs() > 20

    anomaly_df["RULE_COUNT"] = anomaly_df[
        ["RULE_NO_BILLING", "RULE_NO_PO", "RULE_HIGH_PRICE_DIFF", "RULE_HIGH_QTY_DIFF"]
    ].sum(axis=1)

    st.subheader("Rule-Based Findings")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing Billing", int(anomaly_df["RULE_NO_BILLING"].sum()))
    c2.metric("Missing PO", int(anomaly_df["RULE_NO_PO"].sum()))
    c3.metric("High Price Diff", int(anomaly_df["RULE_HIGH_PRICE_DIFF"].sum()))
    c4.metric("High Qty Diff", int(anomaly_df["RULE_HIGH_QTY_DIFF"].sum()))

    # ==================== STATISTICAL OUTLIERS ====================
    stat_cols = [c for c in ["PRICE_DIFF", "QTY_DIFF", "NET_PRICE_EURO", "BILLED_AMT_EURO", "PO_QTY", "BILLED_QTY"] if c in anomaly_df.columns]
    z_data = anomaly_df[stat_cols].fillna(0)
    z_scores = (z_data - z_data.mean()) / z_data.std(ddof=0)
    z_scores = z_scores.fillna(0)
    anomaly_df["MAX_Z_SCORE"] = z_scores.abs().max(axis=1)
    anomaly_df["STAT_OUTLIER"] = anomaly_df["MAX_Z_SCORE"] > 3
    anomaly_df["TOP_STAT_DRIVER"] = z_scores.abs().idxmax(axis=1)

    st.subheader("Statistical Outliers")
    st.metric("Records with Z-Score > 3", int(anomaly_df["STAT_OUTLIER"].sum()))

    # ==================== AI / ML ====================
    st.subheader("AI Pattern Detection")

    ml_features = [
        c for c in [
            "PRICE_DIFF", "QTY_DIFF", "NET_PRICE_EURO", "BILLED_AMT_EURO",
            "PO_QTY", "BILLED_QTY", "TIMES_USED_PO", "TIMES_BILLED",
        ] if c in anomaly_df.columns
    ]

    if len(anomaly_df) >= 20 and len(ml_features) >= 2:
        X = anomaly_df[ml_features].fillna(0).replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(contamination=0.03, random_state=42, n_estimators=200)
        anomaly_df["AI_ANOMALY"] = model.fit_predict(X_scaled)
        anomaly_df["AI_SCORE"] = model.decision_function(X_scaled)
        anomaly_df["AI_FLAG"] = anomaly_df["AI_ANOMALY"] == -1

        contribution_df = pd.DataFrame(
            np.abs(X_scaled), columns=ml_features, index=anomaly_df.index,
        )
        anomaly_df["AI_DRIVER"] = contribution_df.idxmax(axis=1)
        st.metric("AI Detected Anomalies", int(anomaly_df["AI_FLAG"].sum()))
    else:
        anomaly_df["AI_FLAG"] = False
        anomaly_df["AI_SCORE"] = 0
        anomaly_df["AI_DRIVER"] = "N/A"
        st.warning("Not enough records for machine-learning analysis.")

    # ==================== RISK SCORE ====================
    anomaly_df["RISK_SCORE"] = (
        anomaly_df["RULE_COUNT"] * 3
        + anomaly_df["STAT_OUTLIER"].astype(int) * 2
        + anomaly_df["AI_FLAG"].astype(int) * 5
    )
    anomaly_df["RISK_LEVEL"] = np.select(
        [anomaly_df["RISK_SCORE"] >= 8, anomaly_df["RISK_SCORE"] >= 4],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )

    st.subheader("Risk Distribution")
    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk", int((anomaly_df["RISK_LEVEL"] == "HIGH").sum()))
    c2.metric("Medium Risk", int((anomaly_df["RISK_LEVEL"] == "MEDIUM").sum()))
    c3.metric("Low Risk", int((anomaly_df["RISK_LEVEL"] == "LOW").sum()))

    # ==================== EXPLANATION ====================
    def explain_row(row):
        reasons = []
        if row["RULE_NO_BILLING"]:
            reasons.append("PO exists but no billing")
        if row["RULE_NO_PO"]:
            reasons.append("Billing exists but no PO")
        if row["RULE_HIGH_PRICE_DIFF"]:
            reasons.append("Large price difference")
        if row["RULE_HIGH_QTY_DIFF"]:
            reasons.append("Large quantity difference")
        if row["STAT_OUTLIER"]:
            reasons.append(f"Statistical outlier ({row['TOP_STAT_DRIVER']})")
        if row["AI_FLAG"]:
            reasons.append(f"AI anomaly ({row['AI_DRIVER']})")
        return "; ".join(reasons)

    anomaly_df["ANOMALY_EXPLANATION"] = anomaly_df.apply(explain_row, axis=1)

    # ==================== AI DIMENSIONS ====================
    st.subheader("AI Anomaly Dimensions")
    ai_dim_summary = (
        anomaly_df[anomaly_df["AI_FLAG"]]
        .groupby("AI_DRIVER", observed=True)
        .size()
        .reset_index(name="Anomaly Count")
        .sort_values("Anomaly Count", ascending=False)
    )
    if len(ai_dim_summary) > 0:
        st.bar_chart(ai_dim_summary.set_index("AI_DRIVER"))
        st.dataframe(ai_dim_summary, hide_index=True, use_container_width=True)
    else:
        st.info("No AI anomalies detected.")

    # ==================== TOP ANOMALIES ====================
    st.subheader("Top Anomalies")
    display_cols = [
        c for c in [
            "JOB_ID_COMBINED", "CUSTOMER_NAME_COMBINED", "CONTRACT_ID_COMBINED",
            "MATERIAL_ID_COMBINED", "PRICE_DIFF", "QTY_DIFF", "RULE_COUNT",
            "MAX_Z_SCORE", "AI_SCORE", "AI_DRIVER", "RISK_LEVEL", "RISK_SCORE",
            "ANOMALY_EXPLANATION",
        ] if c in anomaly_df.columns
    ]
    top_anomalies = anomaly_df.sort_values(
        ["RISK_SCORE", "AI_SCORE"], ascending=[False, True]
    ).head(200)
    st.dataframe(top_anomalies[display_cols], hide_index=True, use_container_width=True)

    # ==================== EXPORT ====================
    csv = (
        anomaly_df.sort_values("RISK_SCORE", ascending=False)
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        "Download Anomaly Report",
        data=csv,
        file_name="PO_Billing_Anomaly_Report.csv",
        mime="text/csv",
    )

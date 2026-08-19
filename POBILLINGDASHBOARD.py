# PO vs Billing dashboard with customer grouping filter
# Co-authored with CoCo
import streamlit as st
import pandas as pd
import numpy as np
import io
import glob
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest



st.set_page_config(page_title="PO vs Billing Dashboard", layout="wide")
st.title("PO vs Billing Comparison")

with st.sidebar:
    st.header("Data Source Selection")
    source_type = st.radio(
        "Choose Data Source:",
        ["Upload Local Files", "Load from Server"],
        index=0,
        key="source_type"
    )
    
    po_file = None
    billing_file = None
    
    if source_type == "Upload Local Files":
        st.subheader("Upload Data")
        po_file = st.file_uploader("Upload PO data (CSV or Parquet)", type=["csv", "parquet"])
        billing_file = st.file_uploader("Upload Billing data (CSV or Parquet)", type=["csv", "parquet"])
    else:
        st.subheader("Server Files")
        # Find the absolute directory of the currently running script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build search patterns tied to that specific folder
        po_pattern = os.path.join(current_dir, "PO*.parquet")
        billing_pattern = os.path.join(current_dir, "Billing*.parquet")
        
        server_po_files = sorted(glob.glob(po_pattern), key=os.path.getmtime, reverse=True)
        server_billing_files = sorted(glob.glob(billing_pattern), key=os.path.getmtime, reverse=True)
        
        if server_po_files:
            # Display only the filename in the dropdown for better readability
            po_display_names = {os.path.basename(f): f for f in server_po_files}
            selected_po_name = st.selectbox("Select Server PO File", list(po_display_names.keys()))
            po_file = open(po_display_names[selected_po_name], "rb")
        else:
            st.error(f"No files matching 'PO*.parquet' found in folder: {current_dir}")
            
        if server_billing_files:
            billing_display_names = {os.path.basename(f): f for f in server_billing_files}
            selected_billing_name = st.selectbox("Select Server Billing File", list(billing_display_names.keys()))
            billing_file = open(billing_display_names[selected_billing_name], "rb")
        else:
            st.error(f"No files matching 'Billing*.parquet' found in folder: {current_dir}")

if po_file is None or billing_file is None:
    st.info("Please ensure both PO and Billing data are selected to proceed.")
    st.stop()


def _clean_columns(columns):
    """Normalize column names for matching: strip a UTF-8 BOM character (which
    some exports fuse onto the first column's name, e.g. 'FLEET_TYPE' becomes
    '\ufeffFLEET_TYPE'), then uppercase and strip surrounding whitespace.
    str.strip() alone does not remove a BOM since it isn't whitespace."""
    return (
        pd.Index(columns)
        .str.replace("\ufeff", "", regex=False)
        .str.upper()
        .str.strip()
    )


def _clean_keys(df, key_cols):
    for k in key_cols:
        df[k] = df[k].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    return df


# Descriptive text columns that repeat heavily and have bounded cardinality (at
# most a few thousand distinct values in real exports) — safe to store as
# category dtype. Deliberately excludes JOB_NOTIFICATION_ID, JOB_ID,
# MATERIAL_ID, MATL_ID_TRIM, VEHICLE_ID, LICENCE_PLATE, LICENCE_PLATE_ID, and
# VEHICLE_REG_NBR: those are near-unique per row and are also used as groupby
# keys throughout the app, where categorical dtype risks pandas materializing
# the full cartesian product of category levels if a groupby is ever missing
# observed=True.
SAFE_CATEGORY_COLS = [
    "FLEET_TYPE", "CUSTOMER_NAME", "JOB_TYPE_CODE", "MATERIAL_DESC",
    "VENDOR_NAME", "CUST_NAME", "MATL_DESC", "JOB_TYPE_CD", "VEND_NAME",
    "CONTRACT_ID", "FOS_CONTRACT_ID",
]


def _categorize(df):
    for col in SAFE_CATEGORY_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _aggregate_frame(df, key_cols, numeric_sum_cols, count_col):
    """Collapse df to one row per key_cols combination: sum numeric_sum_cols,
    keep the first value of every other (descriptive) column, and add a count
    of how many source rows were folded into each combo."""
    desc_cols = [c for c in df.columns if c not in key_cols and c not in numeric_sum_cols]
    agg_spec = {c: (c, "sum") for c in numeric_sum_cols}
    agg_spec.update({c: (c, "first") for c in desc_cols})
    counts = df.groupby(key_cols, dropna=False).size().reset_index(name=count_col)
    agg = df.groupby(key_cols, dropna=False).agg(**agg_spec).reset_index()
    return agg.merge(counts, on=key_cols, how="left")


def load_and_preaggregate(file_obj, keep_cols, key_cols, numeric_sum_cols, count_col,
                           date_col=None, chunksize=200_000):
    """Read a PO/Billing export and collapse it to one row per key_cols combo,
    without ever holding the full raw file in memory at once. CSVs are streamed
    in chunks (each chunk aggregated immediately, then discarded); Parquet files
    are read with column selection only, since columnar formats are already far
    more memory-efficient than CSV for the same data."""
    
    # Handle both Streamlit UploadedFile objects (which have an attribute '.name')
    # and standard Python file objects opened from the server (which have '.name')
    name = file_obj.name

    if name.endswith(".parquet"):
        # read_parquet handles bytes arrays or file descriptors gracefully
        if hasattr(file_obj, "getvalue"):
            buf = io.BytesIO(file_obj.getvalue())
            df = pd.read_parquet(buf)
            del buf
        else:
            file_obj.seek(0)
            df = pd.read_parquet(file_obj)
            
        df.columns = _clean_columns(df.columns)
        present_keys = [k for k in key_cols if k in df.columns]
        df = df[[c for c in keep_cols if c in df.columns] + [k for k in present_keys if k not in keep_cols]]
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = _clean_keys(df, present_keys)
        df = _categorize(df)
        numeric_present = [c for c in numeric_sum_cols if c in df.columns]
        agg = _aggregate_frame(df, present_keys, numeric_present, count_col)
        del df
        return agg, present_keys

    # CSV path: peek the header to resolve real column names/casing, then stream
    # in chunks so we never hold more than one chunk of raw rows in memory.
    header = pd.read_csv(io.BytesIO(file_obj.getvalue()), nrows=0)
    header_map = {c.replace("\ufeff", "").upper().strip(): c for c in header.columns}
    present_keys = [k for k in key_cols if k in header_map]
    usecols = [orig for upper, orig in header_map.items() if upper in keep_cols]
    for k in present_keys:
        if header_map[k] not in usecols:
            usecols.append(header_map[k])

    partial_aggs = []
    reader = pd.read_csv(io.BytesIO(file_obj.getvalue()), usecols=usecols, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        chunk.columns = _clean_columns(chunk.columns)
        if date_col and date_col in chunk.columns:
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        chunk = _clean_keys(chunk, present_keys)
        chunk = _categorize(chunk)
        numeric_present = [c for c in numeric_sum_cols if c in chunk.columns]
        partial_aggs.append(_aggregate_frame(chunk, present_keys, numeric_present, count_col))
        del chunk

    combined = pd.concat(partial_aggs, ignore_index=True)
    del partial_aggs
    numeric_present = [c for c in numeric_sum_cols if c in combined.columns]
    # Re-aggregate the concatenated per-chunk partials: sum the numeric totals AND
    # the per-chunk counts, keep first descriptive value across chunks.
    desc_cols = [c for c in combined.columns if c not in present_keys and c not in numeric_present and c != count_col]
    final_spec = {c: (c, "sum") for c in numeric_present}
    final_spec[count_col] = (count_col, "sum")
    final_spec.update({c: (c, "first") for c in desc_cols})
    agg = combined.groupby(present_keys, dropna=False).agg(**final_spec).reset_index()
    del combined
    return agg, present_keys


@st.cache_data(show_spinner="Loading and aggregating data...")
def load_and_merge(po_file, billing_file):
    # Peek both headers first to decide which columns actually form valid join
    # keys (same column must exist, under its expected name, on both sides).
    def _peek_cols(f):
        if f.name.endswith(".parquet"):
            if hasattr(f, "getvalue"):
                return set(_clean_columns(pd.read_parquet(io.BytesIO(f.getvalue())).columns))
            else:
                f.seek(0)
                return set(_clean_columns(pd.read_parquet(f).columns))
        return set(_clean_columns(pd.read_csv(io.BytesIO(f.getvalue()), nrows=0).columns))

    try:
        po_cols = _peek_cols(po_file)
        billing_cols = _peek_cols(billing_file)
    except Exception as e:
        raise ValueError(f"Error reading files: {e}")

    po_key_candidates, billing_key_candidates = [], []
    if "JOB_NOTIFICATION_ID" in po_cols and "JOB_ID" in billing_cols:
        po_key_candidates.append("JOB_NOTIFICATION_ID")
        billing_key_candidates.append("JOB_ID")
    if "MATERIAL_ID" in po_cols and "MATL_ID_TRIM" in billing_cols:
        po_key_candidates.append("MATERIAL_ID")
        billing_key_candidates.append("MATL_ID_TRIM")
    if "VEHICLE_ID" in po_cols and "VEHICLE_ID" in billing_cols:
        po_key_candidates.append("VEHICLE_ID")
        billing_key_candidates.append("VEHICLE_ID")

    if not po_key_candidates:
        raise ValueError(
            f"Cannot merge files. PO columns: {sorted(po_cols)}. "
            f"Billing columns: {sorted(billing_cols)}. "
            f"Expected: JOB_NOTIFICATION_ID/JOB_ID and MATERIAL_ID/MATL_ID_TRIM."
        )

    # Columns the app actually uses downstream.
    po_keep = [
        "FLEET_TYPE", "PO_POSTING_DATE", "PO_POSTING_MONTH", "FOS_CONTRACT_ID",
        "CUSTOMER_NAME", "JOB_NOTIFICATION_ID", "JOB_TYPE_CODE", "MATERIAL_ID",
        "MATERIAL_DESC", "VEHICLE_ID", "LICENCE_PLATE", "VENDOR_NAME",
        "PO_QTY", "NET_PRICE_EURO",
    ]
    billing_keep = [
        "BILLING_DT", "CONTRACT_ID", "CUST_NAME", "MATL_ID_TRIM", "MATL_DESC",
        "JOB_ID", "JOB_TYPE_CD", "VEND_NAME",
        "LICENCE_PLATE_ID", "VEHICLE_ID", "VEHICLE_REG_NBR",
        "BILLED_QTY", "BILLED_AMT_EURO",
    ]

    try:
        po_agg, po_left_on = load_and_preaggregate(
            po_file, po_keep, po_key_candidates, ["PO_QTY", "NET_PRICE_EURO"],
            "TIMES_USED_PO", date_col="PO_POSTING_DATE",
        )
        bill_agg, billing_right_on = load_and_preaggregate(
            billing_file, billing_keep, billing_key_candidates, ["BILLED_QTY", "BILLED_AMT_EURO"],
            "TIMES_BILLED", date_col="BILLING_DT",
        )
    except Exception as e:
        raise ValueError(f"Error reading/aggregating files: {e}")

    try:
        merged = po_agg.merge(
            bill_agg,
            left_on=po_left_on,
            right_on=billing_right_on,
            how="outer",
            suffixes=("_PO", "_BILL"),
        )
        merged["TIMES_USED_PO"] = merged["TIMES_USED_PO"].fillna(0).astype(int)
        merged["TIMES_BILLED"] = merged["TIMES_BILLED"].fillna(0).astype(int)
        del po_agg, bill_agg
    except Exception as e:
        raise ValueError(f"Merge failed: {e}")

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

    merged = merged.drop(
        columns=[c for c in ["JOB_NOTIFICATION_ID", "JOB_ID", "MATERIAL_ID", "MATL_ID_TRIM",
                              "PO_POSTING_DATE", "BILLING_DT"] if c in merged.columns]
    )

    category_candidates = [
        "FLEET_TYPE", "CUSTOMER_NAME", "JOB_TYPE_CODE", "MATERIAL_DESC",
        "VENDOR_NAME", "CUST_NAME", "MATL_DESC", "JOB_TYPE_CD", "VEND_NAME",
        "CONTRACT_ID_COMBINED", "CUSTOMER_NAME_COMBINED", "MATERIAL_ID_COMBINED",
    ]
    for col in category_candidates:
        if col in merged.columns:
            merged[col] = merged[col].astype("category")
    for col in merged.select_dtypes(include="float64").columns:
        merged[col] = pd.to_numeric(merged[col], downcast="float")
    for col in merged.select_dtypes(include="int64").columns:
        merged[col] = pd.to_numeric(merged[col], downcast="integer")

    return merged


try:
    merged = load_and_merge(po_file, billing_file)
    # Safely close server-side files if they were opened manually
    if source_type == "Load from Server":
        po_file.close()
        billing_file.close()
except ValueError as e:
    st.error(str(e))
    st.stop()

# Rest of the dashboard logic follows...
vendor_col = "VENDOR_NAME" if "VENDOR_NAME" in merged.columns else ("VEND_NAME" if "VEND_NAME" in merged.columns else None)

@st.cache_data
def build_customer_groups(customer_names):
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

all_customers = [x for x in merged["CUSTOMER_NAME_COMBINED"].unique().tolist() if isinstance(x, str) and x != "nan" and x.strip() != ""]
all_customers.sort()
customer_group_map = build_customer_groups(all_customers)
merged["CUSTOMER_GROUP"] = merged["CUSTOMER_NAME_COMBINED"].map(customer_group_map).fillna(merged["CUSTOMER_NAME_COMBINED"])

with st.sidebar:
    st.header("Filters")

    contracts = [x for x in merged["CONTRACT_ID_COMBINED"].unique().tolist() if isinstance(x, str) and x != "nan" and x.strip() != ""]
    contracts.sort()
    selected_contracts = st.multiselect("Contract ID", contracts, key="filter_contracts")

    customer_groups = sorted(merged["CUSTOMER_GROUP"].dropna().unique().tolist())
    customer_groups = [x for x in customer_groups if isinstance(x, str) and x != "nan" and x.strip() != ""]
    selected_customer_groups = st.multiselect(
        "Customer Group",
        customer_groups,
        help="Customers are grouped by shared first keyword (e.g. all 'Transdev ...' entries → 'Transdev')",
        key="filter_customer_groups",
    )

    customers = [x for x in merged["CUSTOMER_NAME_COMBINED"].unique().tolist() if isinstance(x, str) and x != "nan" and x.strip() != ""]
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
        st.metric("Total Price Diff (PO - Billed) €", f"{filtered['PRICE_DIFF'].sum():,.2f}", border=True)
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

    st.subheader("Detail Table")
    display_cols = ["JOB_ID_COMBINED", "MATERIAL_ID_COMBINED", "CONTRACT_ID_COMBINED", "CUSTOMER_NAME_COMBINED", "PO_POSTING_DATE_COMBINED", "NET_PRICE_EURO", "BILLED_AMT_EURO", "PRICE_DIFF", "PO_QTY", "BILLED_QTY", "QTY_DIFF"]
    st.dataframe(
        filtered[display_cols].rename(
            columns={
                "JOB_ID_COMBINED": "Job ID", "MATERIAL_ID_COMBINED": "Material ID", "CONTRACT_ID_COMBINED": "Contract ID", "CUSTOMER_NAME_COMBINED": "Customer", "PO_POSTING_DATE_COMBINED": "PO Posting Date", "NET_PRICE_EURO": "PO Net Price €", "BILLED_AMT_EURO": "Billed Amt €", "PRICE_DIFF": "Price Diff €", "PO_QTY": "PO Qty", "BILLED_QTY": "Billed Qty", "QTY_DIFF": "Qty Diff",
            }
        ),
        hide_index=True, use_container_width=True,
    )

with tab_repair:
    st.subheader("Fleet Type Analysis — Materials per Job & Vehicle")
    fleet_col = next((fc for fc in ["FLEET_TYPE", "FLEET_TYPE_PO"] if fc in filtered.columns), None)
    vehicle_col_fleet = next((vc for vc in ["VEHICLE_ID", "VEHICLE_ID_PO", "LICENCE_PLATE", "LICENCE_PLATE_ID"] if vc in filtered.columns), None)
    licence_col_fleet = next((lc for lc in ["LICENCE_PLATE", "LICENCE_PLATE_ID"] if vehicle_col_fleet and "VEHICLE" in vehicle_col_fleet.upper() and lc in filtered.columns), None)
    mat_desc_fleet = next((mc for mc in ["MATERIAL_DESC", "MATERIAL_DESC_PO", "MATL_DESC", "MATL_DESC_PO", "MATL_DESC_BILL"] if mc in filtered.columns), None)

    if fleet_col is None:
        st.warning("No FLEET_TYPE column found.")
    elif vehicle_col_fleet is None:
        st.warning("No VEHICLE column found.")
    else:
        fleet_types = sorted([str(x) for x in filtered[fleet_col].dropna().unique().tolist() if str(x).strip() != ""])
        selected_fleet = st.multiselect("Filter by Fleet Type", fleet_types, key="fleet_filter")
        fleet_df = filtered if not selected_fleet else filtered[filtered[fleet_col].astype(str).isin(selected_fleet)]

        group_cols_fleet = ["JOB_ID_COMBINED", vehicle_col_fleet]
        if licence_col_fleet: group_cols_fleet.append(licence_col_fleet)
        group_cols_fleet.append("MATERIAL_ID_COMBINED")
        if mat_desc_fleet: group_cols_fleet.append(mat_desc_fleet)
        group_cols_fleet.append(fleet_col)

        fleet_agg_dict = {"TIMES_USED_PO": ("TIMES_USED_PO", "sum"), "TIMES_BILLED": ("TIMES_BILLED", "sum")}
        if "NET_PRICE_EURO" in fleet_df.columns: fleet_agg_dict["TOTAL_PO_EURO"] = ("NET_PRICE_EURO", "sum")
        if "BILLED_AMT_EURO" in fleet_df.columns: fleet_agg_dict["TOTAL_BILLED_EURO"] = ("BILLED_AMT_EURO", "sum")

        fleet_analysis = fleet_df.groupby(group_cols_fleet, dropna=False, observed=True).agg(**fleet_agg_dict).reset_index()
        if "TOTAL_PO_EURO" in fleet_analysis.columns and "TOTAL_BILLED_EURO" in fleet_analysis.columns:
            fleet_analysis["EURO_DIFF"] = fleet_analysis["TOTAL_PO_EURO"].fillna(0) - fleet_analysis["TOTAL_BILLED_EURO"].fillna(0)
        fleet_analysis = fleet_analysis.sort_values(["TIMES_USED_PO"], ascending=False).reset_index(drop=True)

        st.dataframe(fleet_analysis.rename(columns={"JOB_ID_COMBINED": "Job ID", vehicle_col_fleet: "Vehicle ID", "MATERIAL_ID_COMBINED": "Material ID", fleet_col: "Fleet Type"}), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Customer Group Analysis — PO vs Billing")
    if "CUSTOMER_GROUP" in filtered.columns:
        cg_agg_dict = {"TIMES_USED_PO": ("TIMES_USED_PO", "sum"), "TIMES_BILLED": ("TIMES_BILLED", "sum")}
        if "NET_PRICE_EURO" in filtered.columns: cg_agg_dict["TOTAL_PO_EURO"] = ("NET_PRICE_EURO", "sum")
        if "BILLED_AMT_EURO" in filtered.columns: cg_agg_dict["TOTAL_BILLED_EURO"] = ("BILLED_AMT_EURO", "sum")
        cg_analysis = filtered.groupby("CUSTOMER_GROUP", dropna=False, observed=True).agg(**cg_agg_dict).reset_index()
        st.dataframe(cg_analysis, hide_index=True, use_container_width=True)
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

    # ==========================================================
    # RULE BASED ANOMALIES
    # ==========================================================

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
        (
            anomaly_df["PRICE_DIFF"]
            / anomaly_df["NET_PRICE_EURO"].replace(0, np.nan)
        ) * 100,
        0,
    )

    anomaly_df["QTY_DIFF_PCT"] = np.where(
        anomaly_df["PO_QTY"].fillna(0) != 0,
        (
            anomaly_df["QTY_DIFF"]
            / anomaly_df["PO_QTY"].replace(0, np.nan)
        ) * 100,
        0,
    )

    anomaly_df["RULE_HIGH_PRICE_DIFF"] = (
        anomaly_df["PRICE_DIFF_PCT"].abs() > 20
    )

    anomaly_df["RULE_HIGH_QTY_DIFF"] = (
        anomaly_df["QTY_DIFF_PCT"].abs() > 20
    )

    anomaly_df["RULE_COUNT"] = (
        anomaly_df[
            [
                "RULE_NO_BILLING",
                "RULE_NO_PO",
                "RULE_HIGH_PRICE_DIFF",
                "RULE_HIGH_QTY_DIFF",
            ]
        ]
        .sum(axis=1)
    )

    st.subheader("Rule-Based Findings")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Missing Billing",
        int(anomaly_df["RULE_NO_BILLING"].sum())
    )

    c2.metric(
        "Missing PO",
        int(anomaly_df["RULE_NO_PO"].sum())
    )

    c3.metric(
        "High Price Diff",
        int(anomaly_df["RULE_HIGH_PRICE_DIFF"].sum())
    )

    c4.metric(
        "High Qty Diff",
        int(anomaly_df["RULE_HIGH_QTY_DIFF"].sum())
    )

    # ==========================================================
    # STATISTICAL OUTLIERS
    # ==========================================================

    stat_cols = [
        "PRICE_DIFF",
        "QTY_DIFF",
        "NET_PRICE_EURO",
        "BILLED_AMT_EURO",
        "PO_QTY",
        "BILLED_QTY",
    ]

    stat_cols = [
        c for c in stat_cols
        if c in anomaly_df.columns
    ]

    z_data = anomaly_df[stat_cols].fillna(0)

    z_scores = (
        z_data - z_data.mean()
    ) / z_data.std(ddof=0)

    z_scores = z_scores.fillna(0)

    anomaly_df["MAX_Z_SCORE"] = (
        z_scores.abs().max(axis=1)
    )

    anomaly_df["STAT_OUTLIER"] = (
        anomaly_df["MAX_Z_SCORE"] > 3
    )

    anomaly_df["TOP_STAT_DRIVER"] = (
        z_scores.abs().idxmax(axis=1)
    )

    st.subheader("Statistical Outliers")

    st.metric(
        "Records with Z-Score > 3",
        int(anomaly_df["STAT_OUTLIER"].sum())
    )

    # ==========================================================
    # AI / MACHINE LEARNING
    # ==========================================================

    st.subheader("AI Pattern Detection")

    ml_features = [
        "PRICE_DIFF",
        "QTY_DIFF",
        "NET_PRICE_EURO",
        "BILLED_AMT_EURO",
        "PO_QTY",
        "BILLED_QTY",
        "TIMES_USED_PO",
        "TIMES_BILLED",
    ]

    ml_features = [
        c for c in ml_features
        if c in anomaly_df.columns
    ]

    if len(anomaly_df) >= 20 and len(ml_features) >= 2:

        X = (
            anomaly_df[ml_features]
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            contamination=0.03,
            random_state=42,
            n_estimators=200,
        )

        anomaly_df["AI_ANOMALY"] = (
            model.fit_predict(X_scaled)
        )

        anomaly_df["AI_SCORE"] = (
            model.decision_function(X_scaled)
        )

        anomaly_df["AI_FLAG"] = (
            anomaly_df["AI_ANOMALY"] == -1
        )

        contribution_df = pd.DataFrame(
            np.abs(X_scaled),
            columns=ml_features,
            index=anomaly_df.index,
        )

        anomaly_df["AI_DRIVER"] = (
            contribution_df.idxmax(axis=1)
        )

        st.metric(
            "AI Detected Anomalies",
            int(anomaly_df["AI_FLAG"].sum())
        )

    else:

        anomaly_df["AI_FLAG"] = False
        anomaly_df["AI_SCORE"] = 0
        anomaly_df["AI_DRIVER"] = "N/A"

        st.warning(
            "Not enough records for machine-learning analysis."
        )

    # ==========================================================
    # RISK SCORE
    # ==========================================================

    anomaly_df["RISK_SCORE"] = (
        anomaly_df["RULE_COUNT"] * 3
        + anomaly_df["STAT_OUTLIER"].astype(int) * 2
        + anomaly_df["AI_FLAG"].astype(int) * 5
    )

    anomaly_df["RISK_LEVEL"] = np.select(
        [
            anomaly_df["RISK_SCORE"] >= 8,
            anomaly_df["RISK_SCORE"] >= 4,
        ],
        [
            "HIGH",
            "MEDIUM",
        ],
        default="LOW",
    )

    st.subheader("Risk Distribution")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "High Risk",
        int((anomaly_df["RISK_LEVEL"] == "HIGH").sum())
    )

    c2.metric(
        "Medium Risk",
        int((anomaly_df["RISK_LEVEL"] == "MEDIUM").sum())
    )

    c3.metric(
        "Low Risk",
        int((anomaly_df["RISK_LEVEL"] == "LOW").sum())
    )

    # ==========================================================
    # HUMAN READABLE EXPLANATION
    # ==========================================================

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
            reasons.append(
                f"Statistical outlier ({row['TOP_STAT_DRIVER']})"
            )

        if row["AI_FLAG"]:
            reasons.append(
                f"AI anomaly ({row['AI_DRIVER']})"
            )

        return "; ".join(reasons)

    anomaly_df["ANOMALY_EXPLANATION"] = (
        anomaly_df.apply(explain_row, axis=1)
    )

    # ==========================================================
    # AI DIMENSIONS
    # ==========================================================

    st.subheader("AI Anomaly Dimensions")

    ai_dim_summary = (
        anomaly_df[anomaly_df["AI_FLAG"]]
        .groupby("AI_DRIVER", observed=True)
        .size()
        .reset_index(name="Anomaly Count")
        .sort_values("Anomaly Count", ascending=False)
    )

    if len(ai_dim_summary) > 0:

        st.bar_chart(
            ai_dim_summary.set_index("AI_DRIVER")
        )

        st.dataframe(
            ai_dim_summary,
            hide_index=True,
            use_container_width=True,
        )

    else:
        st.info("No AI anomalies detected.")

    # ==========================================================
    # TOP ANOMALIES
    # ==========================================================

    st.subheader("Top Anomalies")

    display_cols = [
        "JOB_ID_COMBINED",
        "CUSTOMER_NAME_COMBINED",
        "CONTRACT_ID_COMBINED",
        "MATERIAL_ID_COMBINED",
        "PRICE_DIFF",
        "QTY_DIFF",
        "RULE_COUNT",
        "MAX_Z_SCORE",
        "AI_SCORE",
        "AI_DRIVER",
        "RISK_LEVEL",
        "RISK_SCORE",
        "ANOMALY_EXPLANATION",
    ]

    display_cols = [
        c for c in display_cols
        if c in anomaly_df.columns
    ]

    top_anomalies = (
        anomaly_df
        .sort_values(
            ["RISK_SCORE", "AI_SCORE"],
            ascending=[False, True]
        )
        .head(200)
    )

    st.dataframe(
        top_anomalies[display_cols],
        hide_index=True,
        use_container_width=True,
    )

    # ==========================================================
    # EXPORT
    # ==========================================================

    csv = (
        anomaly_df
        .sort_values("RISK_SCORE", ascending=False)
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "Download Anomaly Report",
        data=csv,
        file_name="PO_Billing_Anomaly_Report.csv",
        mime="text/csv",
    )
    
    

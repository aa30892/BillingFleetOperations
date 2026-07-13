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
    name = file_obj.name

    if name.endswith(".parquet"):
        buf = io.BytesIO(file_obj.getvalue())
        df = pd.read_parquet(buf)
        del buf
        df.columns = df.columns.str.upper().str.strip()
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
    header_map = {c.upper().strip(): c for c in header.columns}
    present_keys = [k for k in key_cols if k in header_map]
    usecols = [orig for upper, orig in header_map.items() if upper in keep_cols]
    for k in present_keys:
        if header_map[k] not in usecols:
            usecols.append(header_map[k])

    partial_aggs = []
    reader = pd.read_csv(io.BytesIO(file_obj.getvalue()), usecols=usecols, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        chunk.columns = chunk.columns.str.upper().str.strip()
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
            return set(pd.read_parquet(io.BytesIO(f.getvalue())).columns.str.upper().str.strip())
        return set(pd.read_csv(io.BytesIO(f.getvalue()), nrows=0).columns.str.upper().str.strip())

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

    # Columns the app actually uses downstream. Deliberately excludes several
    # columns that were being loaded and aggregated but never referenced in any
    # tab, chart, filter, or display: JOB_DATE, the *_LOCAL currency fields,
    # FLEET_CUSTOMER_GROUP, CUSTOMER_COUNTRY_ID, FOS_CONTRACT_TYPE,
    # CUSTOMER_DEPOT, MATERIAL_GROUP_DESC/MATL_GRP_DESC, MATERIAL_TYPE_DESC/
    # MATL_TYPE_DESC, FOS_MATERIAL_TYPE, VENDOR_COUNTRY_CODE, CUST_CNTRY_CD, and
    # VENDOR_ID. None of these cost anything to drop functionally, and together
    # they were a meaningful share of peak memory during aggregation.
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

    # The raw join-key columns are now fully superseded by their *_COMBINED
    # versions and never referenced again downstream. They're also the biggest
    # memory cost in the whole table (high-cardinality object columns), so drop
    # them rather than carry duplicate copies of the same identifiers around.
    merged = merged.drop(
        columns=[c for c in ["JOB_NOTIFICATION_ID", "JOB_ID", "MATERIAL_ID", "MATL_ID_TRIM",
                              "PO_POSTING_DATE", "BILLING_DT"] if c in merged.columns]
    )

    # Memory optimization: repeated text columns (names, IDs, descriptions) compress
    # heavily as categoricals; numeric columns rarely need full float64/int64 range.
    category_candidates = [
        "FLEET_TYPE", "CUSTOMER_NAME", "JOB_TYPE_CODE", "MATERIAL_DESC",
        "VENDOR_NAME", "CUST_NAME", "MATL_DESC", "JOB_TYPE_CD", "VEND_NAME",
        "CONTRACT_ID_COMBINED", "CUSTOMER_NAME_COMBINED", "MATERIAL_ID_COMBINED",
    ]
    # Deliberately NOT categorized: JOB_ID_COMBINED, VEHICLE_ID, LICENCE_PLATE,
    # LICENCE_PLATE_ID, VEHICLE_REG_NBR. These are near-unique per row (hundreds of
    # thousands of distinct values), so categorical dtype buys little memory and is
    # actively dangerous as a groupby key: pandas' default observed=False on a
    # categorical groupby materializes the full cartesian product of category
    # levels, which for a 400k+ level column can attempt to allocate an
    # astronomically large result and crash far worse than the original bug.
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
except ValueError as e:
    st.error(str(e))
    st.stop()

vendor_col = "VENDOR_NAME" if "VENDOR_NAME" in merged.columns else ("VEND_NAME" if "VEND_NAME" in merged.columns else None)


@st.cache_data
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

        fleet_df = filtered
        if selected_fleet:
            fleet_df = fleet_df[fleet_df[fleet_col].astype(str).isin(selected_fleet)]

        group_cols_fleet = ["JOB_ID_COMBINED", vehicle_col_fleet]
        if licence_col_fleet:
            group_cols_fleet.append(licence_col_fleet)
        group_cols_fleet.append("MATERIAL_ID_COMBINED")
        if mat_desc_fleet:
            group_cols_fleet.append(mat_desc_fleet)
        group_cols_fleet.append(fleet_col)

        # Underlying PO/Billing lines were already pre-aggregated to one row per
        # (job, material, vehicle) combo in load_and_merge, with TIMES_USED_PO /
        # TIMES_BILLED / PO_QTY / NET_PRICE_EURO / BILLED_QTY / BILLED_AMT_EURO
        # already summed there — so this is a single cheap groupby-sum, not a
        # dataframe copy + multi-column drop_duplicates + two-way merge.
        fleet_agg_dict = {"TIMES_USED_PO": ("TIMES_USED_PO", "sum"), "TIMES_BILLED": ("TIMES_BILLED", "sum")}
        if "NET_PRICE_EURO" in fleet_df.columns:
            fleet_agg_dict["TOTAL_PO_EURO"] = ("NET_PRICE_EURO", "sum")
        if "BILLED_AMT_EURO" in fleet_df.columns:
            fleet_agg_dict["TOTAL_BILLED_EURO"] = ("BILLED_AMT_EURO", "sum")

        fleet_analysis = fleet_df.groupby(group_cols_fleet, dropna=False, observed=True).agg(**fleet_agg_dict).reset_index()
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
                fleet_df.groupby([fleet_col, "MATERIAL_ID_COMBINED"], dropna=False, observed=True)
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
        del fleet_analysis, fleet_df

    st.divider()
    st.subheader("Customer Group Analysis — PO vs Billing")
    st.markdown("Aggregates Times Used PO, Times Billed, PO € Total, and Billed € Total per **Customer Group**, highlighting discrepancies.")

    if "CUSTOMER_GROUP" not in filtered.columns:
        st.warning("No CUSTOMER_GROUP column found in the data.")
    else:
        cg_agg_dict = {"TIMES_USED_PO": ("TIMES_USED_PO", "sum"), "TIMES_BILLED": ("TIMES_BILLED", "sum")}
        if "NET_PRICE_EURO" in filtered.columns:
            cg_agg_dict["TOTAL_PO_EURO"] = ("NET_PRICE_EURO", "sum")
        if "BILLED_AMT_EURO" in filtered.columns:
            cg_agg_dict["TOTAL_BILLED_EURO"] = ("BILLED_AMT_EURO", "sum")

        cg_analysis = filtered.groupby("CUSTOMER_GROUP", dropna=False, observed=True).agg(**cg_agg_dict).reset_index()
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

        st.divider()
        st.subheader("Materials per Job & Vehicle — by Customer Group")
        st.markdown("Select a customer group to see detailed material usage per job and vehicle.")

        cg_list = sorted(cg_analysis["CUSTOMER_GROUP"].dropna().unique().tolist())
        selected_cg = st.selectbox("Select Customer Group", cg_list, key="cg_detail_filter")

        if selected_cg:
            cg_filtered = filtered[filtered["CUSTOMER_GROUP"] == selected_cg]

            if cg_filtered.empty:
                st.info("No data for the selected customer group.")
            else:
                vehicle_col_cg = None
                for vc in ["VEHICLE_ID", "VEHICLE_ID_PO", "LICENCE_PLATE", "LICENCE_PLATE_ID"]:
                    if vc in cg_filtered.columns:
                        vehicle_col_cg = vc
                        break

                licence_col_cg = None
                if vehicle_col_cg and "VEHICLE" in vehicle_col_cg.upper():
                    for lc in ["LICENCE_PLATE", "LICENCE_PLATE_ID"]:
                        if lc in cg_filtered.columns:
                            licence_col_cg = lc
                            break

                mat_desc_cg = None
                for mc in ["MATERIAL_DESC", "MATERIAL_DESC_PO", "MATL_DESC", "MATL_DESC_PO", "MATL_DESC_BILL"]:
                    if mc in cg_filtered.columns:
                        mat_desc_cg = mc
                        break

                group_cols_cg = ["JOB_ID_COMBINED", "MATERIAL_ID_COMBINED"]
                if vehicle_col_cg:
                    group_cols_cg.append(vehicle_col_cg)
                if licence_col_cg:
                    group_cols_cg.append(licence_col_cg)
                if mat_desc_cg:
                    group_cols_cg.append(mat_desc_cg)

                cg_detail_agg = {"TIMES_USED_PO": ("TIMES_USED_PO", "sum"), "TIMES_BILLED": ("TIMES_BILLED", "sum")}
                if "NET_PRICE_EURO" in cg_filtered.columns:
                    cg_detail_agg["TOTAL_PO_EURO"] = ("NET_PRICE_EURO", "sum")
                if "BILLED_AMT_EURO" in cg_filtered.columns:
                    cg_detail_agg["TOTAL_BILLED_EURO"] = ("BILLED_AMT_EURO", "sum")

                cg_detail = cg_filtered.groupby(group_cols_cg, dropna=False, observed=True).agg(**cg_detail_agg).reset_index()
                if "TOTAL_PO_EURO" in cg_detail.columns and "TOTAL_BILLED_EURO" in cg_detail.columns:
                    cg_detail["EURO_DIFF"] = cg_detail["TOTAL_PO_EURO"].fillna(0) - cg_detail["TOTAL_BILLED_EURO"].fillna(0)
                cg_detail = cg_detail.sort_values("TIMES_USED_PO", ascending=False).reset_index(drop=True)

                with st.container(horizontal=True):
                    st.metric("Jobs", cg_detail["JOB_ID_COMBINED"].nunique(), border=True)
                    st.metric("Materials", cg_detail["MATERIAL_ID_COMBINED"].nunique(), border=True)
                    if vehicle_col_cg:
                        st.metric("Vehicles", cg_detail[vehicle_col_cg].nunique(), border=True)
                    st.metric("Records", len(cg_detail), border=True)

                cg_detail_rename = {
                    "JOB_ID_COMBINED": "Job ID",
                    "MATERIAL_ID_COMBINED": "Material ID",
                    "TIMES_USED_PO": "Times Used PO",
                    "TIMES_BILLED": "Times Billed",
                    "TOTAL_PO_EURO": "PO € Total",
                    "TOTAL_BILLED_EURO": "Billed € Total",
                    "EURO_DIFF": "€ Difference",
                }
                if vehicle_col_cg:
                    cg_detail_rename[vehicle_col_cg] = "Vehicle ID"
                if licence_col_cg:
                    cg_detail_rename[licence_col_cg] = "Licence Plate"
                if mat_desc_cg:
                    cg_detail_rename[mat_desc_cg] = "Material Description"

                st.dataframe(
                    cg_detail.rename(columns=cg_detail_rename),
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

        # Outlier frames only ever need this handful of columns downstream (vendor
        # heatmap, top-offender tables, detail view) — slicing to them before
        # running IQR detection keeps price_outliers/qty_outliers/combined_outliers
        # a fraction of the size of the full 40+ column merged table.
        anomaly_cols = ["JOB_ID_COMBINED", "MATERIAL_ID_COMBINED", "PRICE_DIFF", "QTY_DIFF",
                         "NET_PRICE_EURO", "BILLED_AMT_EURO", "PO_QTY", "BILLED_QTY"]
        if vendor_col_anom:
            anomaly_cols.append(vendor_col_anom)
        if mat_desc_anom:
            anomaly_cols.append(mat_desc_anom)
        anomaly_slice = filtered[[c for c in anomaly_cols if c in filtered.columns]]

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

        price_outliers = detect_anomalies(anomaly_slice, "PRICE_DIFF")
        qty_outliers = detect_anomalies(anomaly_slice, "QTY_DIFF")

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
            del price_outliers, qty_outliers, anomaly_slice

            if not combined_outliers.empty:
                vendor_mat_counts = (
                    combined_outliers.groupby([vendor_col_anom, mat_desc_anom], dropna=False, observed=True)
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

                    anomaly_parts = []

                    if not reg_breakdown.empty:
                        rb = reg_breakdown.groupby(vendor_col_anom, dropna=False, observed=True).size().reset_index(name="Count")
                        rb["Anomaly"] = "Regular Breakdown"
                        rb.rename(columns={vendor_col_anom: "Vendor"}, inplace=True)
                        anomaly_parts.append(rb)

                    if not reg_job.empty:
                        rj = reg_job.groupby(vendor_col_anom, dropna=False, observed=True).size().reset_index(name="Count")
                        rj["Anomaly"] = "Regular Job"
                        rj.rename(columns={vendor_col_anom: "Vendor"}, inplace=True)
                        anomaly_parts.append(rj)

                    if not turn_on_rim.empty:
                        tr = turn_on_rim.groupby(vendor_col_anom, dropna=False, observed=True).size().reset_index(name="Count")
                        tr["Anomaly"] = "Turn on Rim"
                        tr.rename(columns={vendor_col_anom: "Vendor"}, inplace=True)
                        anomaly_parts.append(tr)

                    if not insp_exceed.empty:
                        ie = insp_exceed.groupby(vendor_col_anom, dropna=False, observed=True)["INSP_COUNT"].sum().reset_index(name="Count")
                        ie["Anomaly"] = "Inspection >2 Same Vehicle"
                        ie.rename(columns={vendor_col_anom: "Vendor"}, inplace=True)
                        anomaly_parts.append(ie)

                    if anomaly_parts:
                        anomaly_df = pd.concat(anomaly_parts, ignore_index=True)
                        all_vendors = set(anomaly_df["Vendor"].unique())
                        pivot_anomaly = anomaly_df.pivot_table(
                            index="Vendor", columns="Anomaly", values="Count", fill_value=0, aggfunc="sum"
                        ).sort_values(by=list(anomaly_df["Anomaly"].unique()), ascending=False)

                        st.dataframe(pivot_anomaly, use_container_width=True)

                        with st.container(horizontal=True):
                            st.metric("Vendors Flagged", len(all_vendors), border=True)
                            st.metric("Total Flags", int(anomaly_df["Count"].sum()), border=True)
                    else:
                        st.success("No job type anomaly patterns detected in the filtered data.")

                st.divider()
                st.subheader("Top Offending Vendors")

                vendor_scores = (
                    combined_outliers.groupby(vendor_col_anom, dropna=False, observed=True)
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
                    combined_outliers.groupby([mat_desc_anom, "MATERIAL_ID_COMBINED"], dropna=False, observed=True)
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
                detail_df = combined_outliers[available_detail].sort_values("PRICE_DIFF", ascending=False, key=abs).head(500)

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
            analysis_df.groupby([vendor_col_anomaly, "MONTH"], dropna=False, observed=True)
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
            filtered.groupby(group_cols, dropna=False, observed=True)
            .agg(
                PO_QTY_TOTAL=("PO_QTY", "sum"),
                BILLED_QTY_TOTAL=("BILLED_QTY", "sum"),
                PO_LINES=("TIMES_USED_PO", "sum"),
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
                discrepancies.head(1000).drop(columns=["HAS_DISCREPANCY"]).rename(columns=rename_map),
                hide_index=True,
                use_container_width=True,
            )

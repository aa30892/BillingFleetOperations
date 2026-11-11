# PO vs Billing dashboard — quarterly pre-joined file input
# Co-authored with CoCo
#
# Input: one or more quarterly pre-joined PO/Billing files (CSV or Parquet), e.g.
# POjoinBIlling_Q12026.csv. Each row is one PO line (JOB_NOTIFICATION_ID +
# MATERIAL_ID + VEHICLE_ID + PO_POSTING_DATE) with its billing already rolled up
# (BILLING_LINES, BILLED_QTY, BILLED_AMT_EURO, FIRST/LAST_BILLING_DT). Billing
# columns are empty when the PO line was never billed.
#
# Core question: for every Job ID × Material ID, how many times was it used in the
# PO versus how many times was it billed?
#   * Units mode (default): sum(PO_QTY)  vs sum(BILLED_QTY)
#   * Lines mode:           PO lines     vs sum(BILLING_LINES)
# Parquet: single files, several part files, or a folder of part files (e.g. a
# Snowflake COPY INTO unload) are supported. Requires pyarrow (pip install pyarrow).
# Older un-aggregated files (one row per billing line, BILLING_DT, no
# BILLING_LINES) are still supported: each billed row then counts as one line.
import decimal
import glob
import io
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PO vs Billing Dashboard", layout="wide")
st.title("PO vs Billing Comparison — Quarterly")

MAX_TABLE_ROWS = 5_000  # rows rendered in on-screen tables (full data via CSV export)
PO_KEYS = ["JOB_NOTIFICATION_ID", "MATERIAL_ID", "VEHICLE_ID", "PO_POSTING_DATE"]
JM_KEYS = ["JOB_ID_COMBINED", "MATERIAL_ID_COMBINED"]
STATUS_ORDER = ["Not billed", "Under-billed", "Over-billed", "Billed without PO", "Matched"]


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def _coalesce(df, cols):
    """First non-null value across the given columns (those that exist)."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return None
    out = df[present[0]]
    for c in present[1:]:
        out = out.combine_first(df[c])
    return out


def _id_str(s):
    """IDs as clean strings: 316132437.0 -> '316132437', NaN -> ''."""
    if s is None:
        return None
    if not pd.api.types.is_numeric_dtype(s):
        # parquet may hold numeric IDs as Decimal / dictionary-encoded values
        num = pd.to_numeric(s.astype("string").str.strip(), errors="coerce")
        if num.notna().sum() == s.notna().sum():
            s = num
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce").round().astype("Int64").astype("string")
    else:
        out = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return out.fillna("").astype(str)


def _text(s, default=""):
    if s is None:
        return default
    return s.astype("string").str.strip().fillna(default).replace("", default).astype(str)


def _normalize_types(df):
    """Make parquet-specific dtypes look like what the CSV reader produces."""
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.ArrowDtype):  # only when a pyarrow dtype backend is in use
            kind = s.dtype.kind
            if kind == "M":
                if getattr(s.dtype.pyarrow_dtype, "tz", None):
                    s = s.dt.tz_localize(None)
                s = s.astype("datetime64[ns]")
            elif kind in "biuf":
                s = s.astype("float64")
            else:
                s = s.astype("string")
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_localize(None)  # keep wall-clock time, drop timezone
        elif s.dtype == object:
            first = s.dropna().head(1)
            if len(first) and isinstance(first.iloc[0], decimal.Decimal):
                s = pd.to_numeric(s, errors="coerce").astype("float64")
            elif len(first) and isinstance(first.iloc[0], bytes):
                s = s.str.decode("utf-8", errors="replace")
        df[c] = s
    return df


def read_source(name, path, upload):
    is_parquet = name.lower().endswith(".parquet") or (path is not None and os.path.isdir(path))
    if is_parquet:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            raise RuntimeError("Reading Parquet files needs the pyarrow package: pip install pyarrow")
        buf = io.BytesIO(upload.getvalue()) if upload is not None else path
        return _normalize_types(pd.read_parquet(buf, engine="pyarrow"))
    buf = io.BytesIO(upload.getvalue()) if upload is not None else path
    return pd.read_csv(buf, low_memory=False)


def prepare_frame(df, source_name):
    df.columns = df.columns.str.upper().str.strip()

    for c in ["PO_POSTING_DATE", "BILLING_DT", "FIRST_BILLING_DT", "LAST_BILLING_DT"]:
        if c in df.columns:
            d = pd.to_datetime(df[c], errors="coerce")
            if isinstance(d.dtype, pd.DatetimeTZDtype):
                d = d.dt.tz_localize(None)
            df[c] = d.astype("datetime64[ns]").dt.normalize()

    had_billing_lines = "BILLING_LINES" in df.columns
    for c in ["PO_QTY", "NET_PRICE_EURO", "BILLED_QTY", "BILLED_AMT_EURO", "BILLING_LINES"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

    # --- combined (PO side first, billing side as fallback) ---
    out = pd.DataFrame(index=df.index)
    out["JOB_ID_COMBINED"] = _id_str(_coalesce(df, ["JOB_NOTIFICATION_ID", "JOB_ID"]))
    out["MATERIAL_ID_COMBINED"] = _id_str(_coalesce(df, ["MATERIAL_ID", "MATL_ID_TRIM"]))
    out["CONTRACT_ID_COMBINED"] = _id_str(_coalesce(df, ["FOS_CONTRACT_ID", "CONTRACT_ID"]))
    out["CUSTOMER_NAME_COMBINED"] = _text(_coalesce(df, ["CUSTOMER_NAME", "CUST_NAME"]), "Unknown")
    out["CUSTOMER_GROUP"] = (
        _text(df["FLEET_CUSTOMER_GROUP"], "Unknown")
        if "FLEET_CUSTOMER_GROUP" in df.columns
        else out["CUSTOMER_NAME_COMBINED"]
    )
    out["VENDOR_C"] = _text(_coalesce(df, ["VENDOR_NAME", "VEND_NAME"]), "Unknown")
    out["MATERIAL_DESC_C"] = _text(_coalesce(df, ["MATERIAL_DESC", "MATL_DESC"]))
    out["FLEET_TYPE_C"] = _text(_coalesce(df, ["FLEET_TYPE", "FLEET_TYPE_PO"]), "Unknown")
    out["JOB_TYPE_C"] = _text(_coalesce(df, ["JOB_TYPE_CODE", "JOB_TYPE_CD"]))
    out["LICENCE_PLATE_C"] = _text(_coalesce(df, ["LICENCE_PLATE", "LICENCE_PLATE_ID"]))
    veh = _id_str(_coalesce(df, ["VEHICLE_ID"]))
    out["VEHICLE_C"] = veh if veh is not None else out["LICENCE_PLATE_C"]

    po_date = _coalesce(df, ["PO_POSTING_DATE"])
    bill_first = _coalesce(df, ["FIRST_BILLING_DT", "BILLING_DT"])
    bill_last = _coalesce(df, ["LAST_BILLING_DT", "BILLING_DT"])
    nat = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    out["PO_POSTING_DATE_COMBINED"] = (po_date if po_date is not None else nat).combine_first(
        bill_first if bill_first is not None else nat
    )
    out["FIRST_BILLING_DATE"] = bill_first if bill_first is not None else nat
    out["LAST_BILLING_DATE"] = bill_last if bill_last is not None else nat
    q = out["PO_POSTING_DATE_COMBINED"].dt.to_period("Q").astype(str)
    out["QUARTER"] = q.str.replace(r"^(\d{4})Q(\d)$", r"\1-Q\2", regex=True).replace("NaT", "Unknown")
    out["PO_MONTH"] = out["PO_POSTING_DATE_COMBINED"].dt.strftime("%Y-%m").fillna("Unknown")

    # --- PO / billing presence and counts ---
    po_cols = [c for c in ["JOB_NOTIFICATION_ID", "NET_PRICE_EURO", "PO_QTY"] if c in df.columns]
    bill_cols = [c for c in ["JOB_ID", "BILLED_AMT_EURO", "BILLED_QTY", "BILLING_LINES"] if c in df.columns]
    has_po = df[po_cols].notna().any(axis=1)
    has_bill = df[bill_cols].notna().any(axis=1)
    out["PO_LINES"] = has_po.astype("int32")
    if had_billing_lines:
        out["BILL_LINES"] = np.where(has_bill, df["BILLING_LINES"].fillna(1), 0)
    else:
        out["BILL_LINES"] = has_bill.astype("int32")

    for c in ["PO_QTY", "NET_PRICE_EURO", "BILLED_QTY", "BILLED_AMT_EURO"]:
        out[c] = df[c]
    for c in PO_KEYS:  # keep raw keys for de-duplication across files
        if c in df.columns:
            out["_K_" + c] = df[c]

    out["SOURCE_FILE"] = source_name
    return out


@st.cache_data(show_spinner="Loading quarterly file(s)...", max_entries=3)
def load_sources(_sources, cache_key):
    """_sources: list of (name, path_or_None, uploaded_file_or_None). Not hashed;
    cache_key identifies the files (name/size/mtime)."""
    frames = []
    for name, path, upload in _sources:
        frames.append(prepare_frame(read_source(name, path, upload), name))

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    dropped = 0
    key_cols = [c for c in df.columns if c.startswith("_K_")]
    if len(frames) > 1 and key_cols:
        before = len(df)
        df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
        dropped = before - len(df)
    df = df.drop(columns=key_cols)

    cat_cols = [
        "JOB_ID_COMBINED", "MATERIAL_ID_COMBINED", "CONTRACT_ID_COMBINED",
        "CUSTOMER_NAME_COMBINED", "CUSTOMER_GROUP", "VENDOR_C", "MATERIAL_DESC_C",
        "FLEET_TYPE_C", "JOB_TYPE_C", "LICENCE_PLATE_C", "VEHICLE_C", "QUARTER",
        "PO_MONTH", "SOURCE_FILE",
    ]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df, dropped


@st.cache_data(show_spinner=False, max_entries=3)
def filter_options(_df, cache_key):
    def opts(col):
        return sorted(str(x) for x in _df[col].cat.categories if str(x).strip())

    return {
        c: opts(c)
        for c in [
            "QUARTER", "CONTRACT_ID_COMBINED", "CUSTOMER_GROUP", "CUSTOMER_NAME_COMBINED",
            "VENDOR_C", "MATERIAL_ID_COMBINED", "FLEET_TYPE_C",
        ]
    }


@st.cache_data(show_spinner="Aggregating Job × Material usage...", max_entries=8)
def job_material_base(_df, cache_key):
    """One row per Job ID × Material ID with PO and billing totals."""
    g = _df.groupby(JM_KEYS, observed=True, sort=False)
    agg = g.agg(
        PO_LINES=("PO_LINES", "sum"),
        BILL_LINES=("BILL_LINES", "sum"),
        PO_QTY=("PO_QTY", "sum"),
        BILLED_QTY=("BILLED_QTY", "sum"),
        PO_EURO=("NET_PRICE_EURO", "sum"),
        BILLED_EURO=("BILLED_AMT_EURO", "sum"),
        FIRST_PO_DATE=("PO_POSTING_DATE_COMBINED", "min"),
        LAST_BILLING_DATE=("LAST_BILLING_DATE", "max"),
    ).reset_index()
    attr_cols = [
        "MATERIAL_DESC_C", "VENDOR_C", "CONTRACT_ID_COMBINED", "CUSTOMER_NAME_COMBINED",
        "CUSTOMER_GROUP", "FLEET_TYPE_C", "JOB_TYPE_C", "LICENCE_PLATE_C", "VEHICLE_C", "QUARTER",
    ]
    attrs = _df[JM_KEYS + attr_cols].drop_duplicates(subset=JM_KEYS)
    agg = agg.merge(attrs, on=JM_KEYS, how="left")
    agg["EURO_DIFF"] = agg["PO_EURO"] - agg["BILLED_EURO"]  # PO − Billed
    return agg


def add_usage(jm, used_col, billed_col):
    jm = jm.copy()
    jm["TIMES_USED_PO"] = jm[used_col]
    jm["TIMES_BILLED"] = jm[billed_col]
    diff = (jm["TIMES_USED_PO"] - jm["TIMES_BILLED"]).round(6)
    jm["USAGE_DIFF"] = diff
    used, billed = jm["TIMES_USED_PO"], jm["TIMES_BILLED"]
    jm["STATUS"] = pd.Categorical(
        np.select(
            [(used > 0) & (billed == 0), (used == 0) & (billed > 0), diff > 0, diff < 0],
            ["Not billed", "Billed without PO", "Under-billed", "Over-billed"],
            default="Matched",
        ),
        categories=STATUS_ORDER,
    )
    jm["_IS_NOT_BILLED"] = (jm["STATUS"] == "Not billed").astype(int)
    jm["_IS_UNDER_BILLED"] = (jm["STATUS"] == "Under-billed").astype(int)
    jm["_IS_OVER_BILLED"] = (jm["STATUS"] == "Over-billed").astype(int)
    return jm


def summarize(jm, by):
    out = (
        jm.groupby(by, observed=True, dropna=False)
        .agg(
            JOB_MATERIALS=("STATUS", "size"),
            TIMES_USED_PO=("TIMES_USED_PO", "sum"),
            TIMES_BILLED=("TIMES_BILLED", "sum"),
            NOT_BILLED=("_IS_NOT_BILLED", "sum"),
            UNDER_BILLED=("_IS_UNDER_BILLED", "sum"),
            OVER_BILLED=("_IS_OVER_BILLED", "sum"),
            PO_EURO=("PO_EURO", "sum"),
            BILLED_EURO=("BILLED_EURO", "sum"),
        )
        .reset_index()
    )
    out["USAGE_DIFF"] = (out["TIMES_USED_PO"] - out["TIMES_BILLED"]).round(6)
    out["BILLED_PCT"] = np.where(
        out["TIMES_USED_PO"] > 0, (out["TIMES_BILLED"] / out["TIMES_USED_PO"] * 100).round(1), np.nan
    )
    out["EURO_DIFF"] = out["PO_EURO"] - out["BILLED_EURO"]
    return out


NICE = {
    "JOB_ID_COMBINED": "Job ID",
    "MATERIAL_ID_COMBINED": "Material ID",
    "MATERIAL_DESC_C": "Material Description",
    "CONTRACT_ID_COMBINED": "Contract ID",
    "CUSTOMER_NAME_COMBINED": "Customer",
    "CUSTOMER_GROUP": "Customer Group",
    "VENDOR_C": "Vendor",
    "FLEET_TYPE_C": "Fleet Type",
    "JOB_TYPE_C": "Job Type",
    "LICENCE_PLATE_C": "Licence Plate",
    "VEHICLE_C": "Vehicle ID",
    "QUARTER": "Quarter",
    "PO_MONTH": "PO Month",
    "STATUS": "Status",
    "TIMES_USED_PO": "Times Used PO",
    "TIMES_BILLED": "Times Billed",
    "USAGE_DIFF": "Difference (PO − Billed)",
    "BILLED_PCT": "Billed %",
    "JOB_MATERIALS": "Job×Material combos",
    "NOT_BILLED": "Not billed",
    "UNDER_BILLED": "Under-billed",
    "OVER_BILLED": "Over-billed",
    "PO_LINES": "PO Lines",
    "BILL_LINES": "Billing Lines",
    "PO_QTY": "PO Qty",
    "BILLED_QTY": "Billed Qty",
    "PO_EURO": "PO €",
    "BILLED_EURO": "Billed €",
    "EURO_DIFF": "€ Diff (PO − Billed)",
    "NET_PRICE_EURO": "PO Net Price €",
    "BILLED_AMT_EURO": "Billed Amt €",
    "FIRST_PO_DATE": "PO Posting Date",
    "PO_POSTING_DATE_COMBINED": "PO Posting Date",
    "FIRST_BILLING_DATE": "First Billing Date",
    "LAST_BILLING_DATE": "Last Billing Date",
    "SOURCE_FILE": "Source File",
}


def show_table(df, sort_by=None, key=None):
    if sort_by is not None and sort_by in df.columns:
        df = df.reindex(df[sort_by].abs().sort_values(ascending=False).index)
    if len(df) > MAX_TABLE_ROWS:
        st.caption(f"Showing top {MAX_TABLE_ROWS:,} of {len(df):,} rows — use the CSV export for all rows.")
    st.dataframe(df.head(MAX_TABLE_ROWS).rename(columns=NICE), hide_index=True, use_container_width=True, key=key)


def csv_export(df, label, file_name, key):
    """Building a CSV of hundreds of thousands of rows is slow, so only do it on request."""
    if st.checkbox(f"Prepare CSV: {label} ({len(df):,} rows)", key=f"prep_{key}"):
        data = df.rename(columns=NICE).to_csv(index=False).encode("utf-8")
        st.download_button(f"Download {label}", data=data, file_name=file_name, mime="text/csv", key=f"dl_{key}")


# ------------------------------------------------------------------
# SIDEBAR — data source (one or more quarterly files)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Data Source")
    source_type = st.radio("Choose Data Source:", ["Upload File(s)", "Load from Server"], index=0)

    sources, key_parts = [], []
    if source_type == "Upload File(s)":
        uploads = st.file_uploader(
            "Upload quarterly pre-joined PO/Billing file(s) (CSV or Parquet)",
            type=["csv", "parquet"],
            accept_multiple_files=True,
            help="Several quarters — or all part files of one Parquet export — can be loaded "
            "together. Parquet is much smaller and faster than CSV. Files over 200 MB need "
            "server.maxUploadSize raised in .streamlit/config.toml.",
        )
        for f in uploads or []:
            sources.append((f.name, None, f))
            key_parts.append((f.name, f.size, getattr(f, "file_id", None)))
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        server_files = []
        for d in [current_dir, os.path.join(current_dir, "data")]:
            server_files += glob.glob(os.path.join(d, "*.csv")) + glob.glob(os.path.join(d, "*.parquet"))
            # folders of parquet part files (e.g. data/Q1_2026/part_0_0_0.snappy.parquet)
            server_files += [
                sub for sub in glob.glob(os.path.join(d, "*"))
                if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.parquet"))
            ]
        server_files = sorted(set(server_files), key=os.path.getmtime, reverse=True)
        if server_files:
            display_map = {os.path.relpath(f, current_dir): f for f in server_files}
            selected_names = st.multiselect(
                "Select quarterly file(s)", list(display_map.keys()), default=list(display_map.keys())[:1]
            )
            for n in selected_names:
                p = display_map[n]
                if os.path.isdir(p):
                    parts = glob.glob(os.path.join(p, "*.parquet"))
                    sig = (len(parts), sum(os.path.getsize(f) for f in parts), max(os.path.getmtime(f) for f in parts))
                    sources.append((os.path.basename(p) + "/", p, None))
                    key_parts.append((p,) + sig)
                else:
                    sources.append((os.path.basename(p), p, None))
                    key_parts.append((p, os.path.getsize(p), os.path.getmtime(p)))
        else:
            st.error(f"No CSV/Parquet files found in: {current_dir} (or its data/ folder)")

if not sources:
    st.info("Please provide one or more quarterly pre-joined PO/Billing files to proceed.")
    st.stop()

data_key = tuple(sorted(key_parts, key=str))
try:
    merged, n_dupes = load_sources(sources, data_key)
except Exception as e:
    st.error(f"Error loading file(s): {e}")
    st.stop()
if n_dupes:
    st.sidebar.caption(f"{n_dupes:,} duplicate PO lines across files removed (newest kept).")

options = filter_options(merged, data_key)

# ------------------------------------------------------------------
# SIDEBAR — usage measure + filters
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Usage Measure")
    measure = st.radio(
        "Count 'times used / billed' by",
        ["Units (PO qty vs billed qty)", "Lines (PO lines vs billing lines)"],
        index=0,
        help="In the quarterly files one billing line is generally one billed unit, so a PO line "
        "with qty 4 is billed on 4 lines. Units compares like with like; Lines counts records.",
    )
    USED_COL, BILLED_COL = ("PO_QTY", "BILLED_QTY") if measure.startswith("Units") else ("PO_LINES", "BILL_LINES")

    st.header("Filters")
    sel_quarters = st.multiselect("Quarter", options["QUARTER"], key="filter_quarters")
    sel_contracts = st.multiselect("Contract ID", options["CONTRACT_ID_COMBINED"], key="filter_contracts")
    sel_groups = st.multiselect("Customer Group", options["CUSTOMER_GROUP"], key="filter_customer_groups")
    sel_customers = st.multiselect("Customer Name", options["CUSTOMER_NAME_COMBINED"], key="filter_customers")
    sel_vendors = st.multiselect("Vendor Name", options["VENDOR_C"], key="filter_vendors")
    sel_materials = st.multiselect("Material ID", options["MATERIAL_ID_COMBINED"], key="filter_material_ids")
    job_text = st.text_area(
        "Job ID(s)",
        key="filter_job_ids",
        placeholder="Paste one or more Job IDs, separated by commas, spaces or new lines",
        help="Free text instead of a dropdown — a quarter contains hundreds of thousands of Job IDs.",
    )
    sel_jobs = sorted({re.sub(r"\.0$", "", t) for t in re.split(r"[\s,;]+", job_text or "") if t})

    min_date = merged["PO_POSTING_DATE_COMBINED"].min()
    max_date = merged["PO_POSTING_DATE_COMBINED"].max()
    date_range = None
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.date_input(
            "PO Posting Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="filter_dates",
        )

# ------------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------------
mask = pd.Series(True, index=merged.index)
for col, sel in [
    ("QUARTER", sel_quarters),
    ("CONTRACT_ID_COMBINED", sel_contracts),
    ("CUSTOMER_GROUP", sel_groups),
    ("CUSTOMER_NAME_COMBINED", sel_customers),
    ("VENDOR_C", sel_vendors),
    ("MATERIAL_ID_COMBINED", sel_materials),
    ("JOB_ID_COMBINED", sel_jobs),
]:
    if sel:
        mask &= merged[col].isin(sel)

date_key = None
if date_range and len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    if d0 > min_date or d1 < max_date:  # only filter when narrowed (keeps undated rows otherwise)
        mask &= merged["PO_POSTING_DATE_COMBINED"].between(d0, d1)
        date_key = (str(d0.date()), str(d1.date()))

filtered = merged[mask] if not mask.all() else merged
filter_key = (
    data_key, tuple(sel_quarters), tuple(sel_contracts), tuple(sel_groups), tuple(sel_customers),
    tuple(sel_vendors), tuple(sel_materials), tuple(sel_jobs), date_key,
)

if filtered.empty:
    st.warning("No records match the current filters.")
    st.stop()

jm = add_usage(job_material_base(filtered, filter_key), USED_COL, BILLED_COL)
unit_word = "units" if USED_COL == "PO_QTY" else "lines"

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_dashboard, tab_jm, tab_vendor, tab_contract, tab_anomaly = st.tabs(
    ["Dashboard", "Job × Material Usage", "Vendor", "Contract Analysis", "AI Anomaly Analysis"]
)

# ========================== DASHBOARD ==========================
with tab_dashboard:
    quarters_txt = ", ".join(str(q) for q in filtered["QUARTER"].unique())
    st.caption(f"Quarter(s) in view: {quarters_txt} · usage counted in **{unit_word}**")

    total_po = filtered["NET_PRICE_EURO"].sum()
    total_billed = filtered["BILLED_AMT_EURO"].sum()
    price_diff = total_po - total_billed
    used_total = jm["TIMES_USED_PO"].sum()
    billed_total = jm["TIMES_BILLED"].sum()
    status_counts = jm["STATUS"].value_counts().reindex(STATUS_ORDER, fill_value=0)

    with st.container(horizontal=True):
        st.metric("PO Lines", f"{len(filtered):,}", border=True)
        st.metric("Job × Material combos", f"{len(jm):,}", border=True)
        st.metric("Total PO €", f"{total_po:,.2f}", border=True)
        st.metric("Total Billed €", f"{total_billed:,.2f}", border=True)
        st.metric("€ Diff (PO − Billed)", f"{price_diff:,.2f}", border=True)
    with st.container(horizontal=True):
        st.metric(f"Times Used PO ({unit_word})", f"{used_total:,.0f}", border=True)
        st.metric(f"Times Billed ({unit_word})", f"{billed_total:,.0f}", border=True)
        st.metric("Difference (PO − Billed)", f"{used_total - billed_total:,.0f}", border=True)
        st.metric(
            "Combos fully billed",
            f"{(status_counts['Matched'] / max(len(jm), 1)) * 100:,.1f}%",
            border=True,
        )

    st.subheader("Job × Material reconciliation status")
    status_summary = summarize(jm, "STATUS")
    status_summary["STATUS"] = pd.Categorical(status_summary["STATUS"], categories=STATUS_ORDER)
    status_summary = status_summary.sort_values("STATUS")
    c1, c2 = st.columns([1, 2])
    with c1:
        with st.container(border=True):
            st.bar_chart(status_summary.set_index("STATUS")["JOB_MATERIALS"], horizontal=True)
    with c2:
        st.dataframe(
            status_summary[["STATUS", "JOB_MATERIALS", "TIMES_USED_PO", "TIMES_BILLED", "USAGE_DIFF", "PO_EURO", "BILLED_EURO", "EURO_DIFF"]]
            .rename(columns=NICE),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("By PO month")
    monthly = (
        filtered.assign(_U=filtered[USED_COL].fillna(0), _B=filtered[BILLED_COL].fillna(0))
        .groupby("PO_MONTH", observed=True)
        .agg(TIMES_USED_PO=("_U", "sum"), TIMES_BILLED=("_B", "sum"))
    )
    st.bar_chart(monthly.rename(columns=NICE), stack=False)

    st.subheader("Materials with the largest usage difference")
    mat_summary = summarize(jm, ["MATERIAL_ID_COMBINED", "MATERIAL_DESC_C"])
    top_mat = mat_summary.reindex(mat_summary["USAGE_DIFF"].abs().sort_values(ascending=False).index).head(20)
    with st.container(border=True):
        st.markdown(f"**Top 20 materials — Times Used PO vs Times Billed ({unit_word})**")
        chart = top_mat.assign(
            LABEL=top_mat["MATERIAL_ID_COMBINED"].astype(str) + " " + top_mat["MATERIAL_DESC_C"].astype(str).str[:30]
        ).set_index("LABEL")[["TIMES_USED_PO", "TIMES_BILLED"]]
        st.bar_chart(chart.rename(columns=NICE), stack=False)
    show_table(
        mat_summary[[
            "MATERIAL_ID_COMBINED", "MATERIAL_DESC_C", "JOB_MATERIALS", "TIMES_USED_PO", "TIMES_BILLED",
            "USAGE_DIFF", "BILLED_PCT", "NOT_BILLED", "UNDER_BILLED", "OVER_BILLED", "PO_EURO", "BILLED_EURO", "EURO_DIFF",
        ]],
        sort_by="USAGE_DIFF",
        key="mat_table",
    )

# ========================== JOB × MATERIAL ==========================
with tab_jm:
    st.subheader("Job ID × Material ID — Times Used in PO vs Times Billed")
    st.caption(
        f"One row per Job ID and Material ID. Times Used PO = "
        f"{'sum of PO_QTY' if USED_COL == 'PO_QTY' else 'number of PO lines'}; Times Billed = "
        f"{'sum of BILLED_QTY' if USED_COL == 'PO_QTY' else 'sum of BILLING_LINES'}."
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sel_status = st.multiselect(
            "Status", STATUS_ORDER, default=[s for s in STATUS_ORDER if s != "Matched"], key="jm_status"
        )
    with c2:
        min_abs = st.number_input("Min |difference|", min_value=0.0, value=0.0, step=1.0, key="jm_min_abs")
    with c3:
        desc_search = st.text_input("Material description contains", key="jm_desc")

    jm_view = jm
    if sel_status:
        jm_view = jm_view[jm_view["STATUS"].isin(sel_status)]
    if min_abs > 0:
        jm_view = jm_view[jm_view["USAGE_DIFF"].abs() >= min_abs]
    if desc_search:
        jm_view = jm_view[jm_view["MATERIAL_DESC_C"].astype(str).str.contains(desc_search, case=False, regex=False)]

    with st.container(horizontal=True):
        st.metric("Combos shown", f"{len(jm_view):,}", border=True)
        st.metric(f"Times Used PO ({unit_word})", f"{jm_view['TIMES_USED_PO'].sum():,.0f}", border=True)
        st.metric(f"Times Billed ({unit_word})", f"{jm_view['TIMES_BILLED'].sum():,.0f}", border=True)
        st.metric("€ Diff (PO − Billed)", f"{jm_view['EURO_DIFF'].sum():,.2f}", border=True)

    jm_cols = [
        "JOB_ID_COMBINED", "MATERIAL_ID_COMBINED", "MATERIAL_DESC_C", "STATUS",
        "TIMES_USED_PO", "TIMES_BILLED", "USAGE_DIFF",
        "PO_LINES", "BILL_LINES", "PO_QTY", "BILLED_QTY",
        "PO_EURO", "BILLED_EURO", "EURO_DIFF",
        "FIRST_PO_DATE", "LAST_BILLING_DATE", "QUARTER",
        "VENDOR_C", "CUSTOMER_NAME_COMBINED", "CONTRACT_ID_COMBINED", "FLEET_TYPE_C", "LICENCE_PLATE_C",
    ]
    show_table(jm_view[jm_cols], sort_by="USAGE_DIFF", key="jm_table")
    csv_export(jm_view[jm_cols], "Job × Material usage", "Job_Material_Usage.csv", "jm")

    st.divider()
    st.subheader("Drill-down: PO lines behind a Job ID")
    drill = st.text_input("Job ID", key="jm_drill", placeholder="e.g. 316132437").strip()
    if drill:
        drill = re.sub(r"\.0$", "", drill)
        lines = filtered[filtered["JOB_ID_COMBINED"] == drill]
        if lines.empty:
            st.info("No PO lines for that Job ID within the current filters.")
        else:
            st.dataframe(
                lines[[
                    "JOB_ID_COMBINED", "MATERIAL_ID_COMBINED", "MATERIAL_DESC_C", "VEHICLE_C", "LICENCE_PLATE_C",
                    "PO_POSTING_DATE_COMBINED", "PO_QTY", "NET_PRICE_EURO", "BILL_LINES", "BILLED_QTY",
                    "BILLED_AMT_EURO", "FIRST_BILLING_DATE", "LAST_BILLING_DATE", "VENDOR_C", "SOURCE_FILE",
                ]].rename(columns=NICE),
                hide_index=True,
                use_container_width=True,
            )

# ========================== VENDOR ==========================
with tab_vendor:
    st.subheader("Vendor Analysis")
    vendor_summary = summarize(jm, "VENDOR_C")
    st.markdown(f"**Usage by vendor ({unit_word}) — sorted by largest difference**")
    show_table(
        vendor_summary[[
            "VENDOR_C", "JOB_MATERIALS", "TIMES_USED_PO", "TIMES_BILLED", "USAGE_DIFF", "BILLED_PCT",
            "NOT_BILLED", "UNDER_BILLED", "OVER_BILLED", "PO_EURO", "BILLED_EURO", "EURO_DIFF",
        ]],
        sort_by="USAGE_DIFF",
        key="vendor_table",
    )

    st.divider()
    sel_tab_vendors = st.multiselect(
        "Show Job × Material detail for vendor(s)",
        sorted(vendor_summary["VENDOR_C"].astype(str)),
        key="vendor_tab_filter",
    )
    if sel_tab_vendors:
        vd = jm[jm["VENDOR_C"].astype(str).isin(sel_tab_vendors)]
        show_table(
            vd[[
                "JOB_ID_COMBINED", "LICENCE_PLATE_C", "MATERIAL_ID_COMBINED", "MATERIAL_DESC_C", "FLEET_TYPE_C",
                "VENDOR_C", "STATUS", "TIMES_USED_PO", "TIMES_BILLED", "USAGE_DIFF", "PO_EURO", "BILLED_EURO",
            ]],
            sort_by="USAGE_DIFF",
            key="vendor_detail",
        )

# ========================== CONTRACT ANALYSIS ==========================
with tab_contract:
    st.subheader("Fleet Type Analysis")
    sel_fleet = st.multiselect("Filter by Fleet Type", options["FLEET_TYPE_C"], key="fleet_filter")
    jm_fleet = jm[jm["FLEET_TYPE_C"].astype(str).isin(sel_fleet)] if sel_fleet else jm

    fleet_summary = summarize(jm_fleet, "FLEET_TYPE_C")
    st.dataframe(
        fleet_summary.sort_values("TIMES_USED_PO", ascending=False).rename(columns=NICE),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Contract Analysis — PO vs Billing")
    contract_summary = summarize(jm_fleet, ["CONTRACT_ID_COMBINED", "CUSTOMER_NAME_COMBINED", "FLEET_TYPE_C"])
    show_table(contract_summary, sort_by="USAGE_DIFF", key="contract_table")

    st.divider()
    st.subheader("Customer Group Analysis — PO vs Billing")
    cg_summary = summarize(jm_fleet, "CUSTOMER_GROUP")
    show_table(cg_summary, sort_by="USAGE_DIFF", key="cg_table")

# ========================== AI ANOMALY ANALYSIS ==========================
@st.cache_data(show_spinner="Running anomaly detection...", max_entries=4)
def run_anomaly(_jm, cache_key):
    a = _jm.copy()
    used, billed = a["TIMES_USED_PO"], a["TIMES_BILLED"]
    both = (used > 0) & (billed > 0)

    # ---- rule based (Job × Material level) ----
    a["RULE_NO_BILLING"] = a["STATUS"] == "Not billed"
    a["RULE_NO_PO"] = a["STATUS"] == "Billed without PO"
    usage_pct = np.where(used > 0, a["USAGE_DIFF"] / used.where(used > 0) * 100, 0)
    euro_pct = np.where(a["PO_EURO"] != 0, a["EURO_DIFF"] / a["PO_EURO"].where(a["PO_EURO"] != 0) * 100, 0)
    a["USAGE_DIFF_PCT"] = np.round(usage_pct, 1)
    a["EURO_DIFF_PCT"] = np.round(euro_pct, 1)
    # % rules only when both PO and billing exist — otherwise they just duplicate the missing-side rules
    a["RULE_HIGH_USAGE_DIFF"] = both & (np.abs(usage_pct) > 20)
    a["RULE_HIGH_EURO_DIFF"] = both & (np.abs(euro_pct) > 20)
    rule_cols = ["RULE_NO_BILLING", "RULE_NO_PO", "RULE_HIGH_USAGE_DIFF", "RULE_HIGH_EURO_DIFF"]
    a["RULE_COUNT"] = a[rule_cols].sum(axis=1)

    # ---- statistical outliers ----
    stat_cols = ["USAGE_DIFF", "EURO_DIFF", "PO_EURO", "BILLED_EURO", "TIMES_USED_PO", "TIMES_BILLED"]
    z_data = a[stat_cols].fillna(0).replace([np.inf, -np.inf], 0)
    z = ((z_data - z_data.mean()) / z_data.std(ddof=0).replace(0, np.nan)).fillna(0)
    a["MAX_Z_SCORE"] = z.abs().max(axis=1).round(2)
    a["STAT_OUTLIER"] = a["MAX_Z_SCORE"] > 3
    a["TOP_STAT_DRIVER"] = z.abs().idxmax(axis=1)

    # ---- isolation forest ----
    ml_ok = len(a) >= 20
    if ml_ok:
        X = z_data.to_numpy()
        X_scaled = StandardScaler().fit_transform(X)
        model = IsolationForest(contamination=0.03, random_state=42, n_estimators=200, n_jobs=-1)
        fit_rows = X_scaled
        if len(X_scaled) > 200_000:  # IF subsamples per tree anyway; keeps fit time bounded
            rng = np.random.default_rng(42)
            fit_rows = X_scaled[rng.choice(len(X_scaled), 200_000, replace=False)]
        model.fit(fit_rows)
        a["AI_SCORE"] = model.decision_function(X_scaled).round(4)
        a["AI_FLAG"] = model.predict(X_scaled) == -1
        a["AI_DRIVER"] = np.array(stat_cols)[np.abs(X_scaled).argmax(axis=1)]
    else:
        a["AI_SCORE"] = 0.0
        a["AI_FLAG"] = False
        a["AI_DRIVER"] = "N/A"

    # ---- risk score ----
    a["RISK_SCORE"] = a["RULE_COUNT"] * 3 + a["STAT_OUTLIER"].astype(int) * 2 + a["AI_FLAG"].astype(int) * 5
    a["RISK_LEVEL"] = np.select([a["RISK_SCORE"] >= 8, a["RISK_SCORE"] >= 4], ["HIGH", "MEDIUM"], default="LOW")

    # ---- explanation (vectorised) ----
    expl = pd.Series("", index=a.index, dtype=object)
    for flag, text in [
        (a["RULE_NO_BILLING"], "PO exists but not billed"),
        (a["RULE_NO_PO"], "Billed without PO"),
        (a["RULE_HIGH_USAGE_DIFF"], "Usage difference > 20%"),
        (a["RULE_HIGH_EURO_DIFF"], "€ difference > 20%"),
        (a["STAT_OUTLIER"], "Statistical outlier (" + a["TOP_STAT_DRIVER"].astype(str) + ")"),
        (a["AI_FLAG"], "AI anomaly (" + a["AI_DRIVER"].astype(str) + ")"),
    ]:
        expl = expl.where(~flag, expl + "; " + text)
    a["ANOMALY_EXPLANATION"] = expl.str.lstrip("; ")
    return a, ml_ok


with tab_anomaly:
    st.subheader("AI Anomaly Analysis")
    with st.expander("How AI Anomaly Detection Works", expanded=False):
        st.markdown(f"""
Analysis runs on **Job ID × Material ID** combinations (usage counted in {unit_word}).

**1. Rule-based checks**
- PO exists but nothing billed
- Billed without a PO
- Usage difference (PO − Billed) above 20% of PO usage — only when both sides exist
- € difference above 20% of PO € — only when both sides exist

Each triggered rule adds 3 to the Risk Score.

**2. Statistical outliers** — Z-scores of usage difference, € difference, PO €, Billed €, Times Used PO
and Times Billed. Combos more than 3 standard deviations from the mean are flagged (+2).

**3. AI pattern detection** — an Isolation Forest on the same six features flags the ~3% most unusual
combinations of values (+5).

**Risk levels:** 🔴 HIGH ≥ 8 · 🟡 MEDIUM ≥ 4 · 🟢 LOW < 4

**AI Driver** is the feature with the most extreme standardised value for that combo.
""")

    anomaly_df, ml_ok = run_anomaly(jm, (filter_key, USED_COL))

    st.subheader("Rule-Based Findings")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PO not billed", f"{int(anomaly_df['RULE_NO_BILLING'].sum()):,}")
    c2.metric("Billed without PO", f"{int(anomaly_df['RULE_NO_PO'].sum()):,}")
    c3.metric("Usage diff > 20%", f"{int(anomaly_df['RULE_HIGH_USAGE_DIFF'].sum()):,}")
    c4.metric("€ diff > 20%", f"{int(anomaly_df['RULE_HIGH_EURO_DIFF'].sum()):,}")

    c1, c2 = st.columns(2)
    c1.metric("Statistical outliers (Z > 3)", f"{int(anomaly_df['STAT_OUTLIER'].sum()):,}")
    if ml_ok:
        c2.metric("AI Detected Anomalies", f"{int(anomaly_df['AI_FLAG'].sum()):,}")
    else:
        c2.warning("Not enough records for machine-learning analysis.")

    st.subheader("Risk Distribution")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High Risk", f"{int((anomaly_df['RISK_LEVEL'] == 'HIGH').sum()):,}")
    c2.metric("🟡 Medium Risk", f"{int((anomaly_df['RISK_LEVEL'] == 'MEDIUM').sum()):,}")
    c3.metric("🟢 Low Risk", f"{int((anomaly_df['RISK_LEVEL'] == 'LOW').sum()):,}")

    st.subheader("AI Anomaly Dimensions")
    ai_dim = (
        anomaly_df[anomaly_df["AI_FLAG"]]
        .groupby("AI_DRIVER")
        .size()
        .reset_index(name="Anomaly Count")
        .sort_values("Anomaly Count", ascending=False)
    )
    if len(ai_dim):
        c1, c2 = st.columns([2, 1])
        c1.bar_chart(ai_dim.set_index("AI_DRIVER"))
        c2.dataframe(ai_dim, hide_index=True, use_container_width=True)
    else:
        st.info("No AI anomalies detected.")

    st.subheader("Top Anomalies")
    display_cols = [
        "JOB_ID_COMBINED", "MATERIAL_ID_COMBINED", "MATERIAL_DESC_C", "CUSTOMER_NAME_COMBINED",
        "CONTRACT_ID_COMBINED", "VENDOR_C", "STATUS", "TIMES_USED_PO", "TIMES_BILLED", "USAGE_DIFF",
        "PO_EURO", "BILLED_EURO", "EURO_DIFF", "RULE_COUNT", "MAX_Z_SCORE", "AI_SCORE", "AI_DRIVER",
        "RISK_LEVEL", "RISK_SCORE", "ANOMALY_EXPLANATION",
    ]
    ranked = anomaly_df.sort_values(["RISK_SCORE", "AI_SCORE"], ascending=[False, True])
    st.dataframe(ranked[display_cols].head(200).rename(columns=NICE), hide_index=True, use_container_width=True)

    flagged = ranked[ranked["RISK_LEVEL"] != "LOW"]
    csv_export(flagged, "Anomaly Report (High + Medium risk)", "PO_Billing_Anomaly_Report.csv", "anomaly")

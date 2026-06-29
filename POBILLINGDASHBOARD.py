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


@st.cache_data
def load_and_merge(po_bytes, billing_bytes):
    po_df = pd.read_csv(po_bytes)
    billing_df = pd.read_csv(billing_bytes)
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
    merged["CONTRACT_ID_COMBINED"] = merged["FOS_CONTRACT_ID"].combine_first(merged["CONTRACT_ID"]).astype(str)
    merged["CUSTOMER_NAME_COMBINED"] = merged["CUSTOMER_NAME"].combine_first(merged["CUST_NAME"]).astype(str)
    merged["MATERIAL_ID_COMBINED"] = merged["MATERIAL_ID"].combine_first(merged["MATL_ID_TRIM"]).astype(str)
    merged["JOB_ID_COMBINED"] = merged["JOB_NOTIFICATION_ID"].combine_first(merged["JOB_ID"]).astype(str)
    if "BILLING_DT" in merged.columns:
        merged["PO_POSTING_DATE_COMBINED"] = merged["PO_POSTING_DATE"].combine_first(merged["BILLING_DT"])
    else:
        merged["PO_POSTING_DATE_COMBINED"] = merged["PO_POSTING_DATE"]
    return merged


merged = load_and_merge(po_file, billing_file)

vendor_col = "VENDOR_NAME" if "VENDOR_NAME" in merged.columns else ("VEND_NAME" if "VEND_NAME" in merged.columns else None)

with st.sidebar:
    st.header("Filters")

    contracts = [x for x in merged["CONTRACT_ID_COMBINED"].unique().tolist() if x != "nan" and x.strip() != ""]
    contracts.sort()
    selected_contracts = st.multiselect("Contract ID", contracts, key="filter_contracts")

    customers = [x for x in merged["CUSTOMER_NAME_COMBINED"].unique().tolist() if x != "nan" and x.strip() != ""]
    customers.sort()
    selected_customers = st.multiselect("Customer Name", customers, key="filter_customers")

    if vendor_col:
        vendors = [str(x) for x in merged[vendor_col].dropna().unique().tolist() if str(x).strip() != ""]
        vendors.sort()
        selected_vendors = st.multiselect("Vendor Name", vendors, key="filter_vendors")
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
            key="filter_dates",
        )
    else:
        date_range = None

filtered = merged
if selected_contracts:
    filtered = filtered[filtered["CONTRACT_ID_COMBINED"].isin(selected_contracts)]
if selected_customers:
    filtered = filtered[filtered["CUSTOMER_NAME_COMBINED"].isin(selected_customers)]
if selected_vendors and vendor_col:
    filtered = filtered[filtered[vendor_col].astype(str).isin(selected_vendors)]
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

with tab_repair:
    st.subheader("Materials Used on Repair Jobs (by Vehicle)")
    st.markdown("When `JOB_TYPE_CODE = 'repair'`, what other materials are used? Ranked from highest to lowest quantity.")

    job_type_col = None
    for col_name in ["JOB_TYPE_CODE", "JOB_TYPE_CD", "JOB_TYPE_CODE_PO", "JOB_TYPE_CD_PO", "JOB_TYPE_CODE_BILL", "JOB_TYPE_CD_BILL"]:
        if col_name in filtered.columns:
            job_type_col = col_name
            break

    if job_type_col is None:
        st.warning("No JOB_TYPE column found in the data. Ensure your CSV includes JOB_TYPE_CODE or JOB_TYPE_CD.")
    else:
        repair_mask = filtered[job_type_col].astype(str).str.lower().str.contains("repair", na=False)
        repair_df = filtered[repair_mask]

        if repair_df.empty:
            st.info("No repair jobs found in the filtered data.")
        else:
            st.metric("Total Repair Records", f"{len(repair_df):,}", border=True)

            repair_ranking = (
                repair_df.groupby("MATERIAL_ID_COMBINED")
                .agg(
                    TOTAL_PO_QTY=("PO_QTY", "sum"),
                    TOTAL_BILLED_QTY=("BILLED_QTY", "sum"),
                    JOB_COUNT=("JOB_ID_COMBINED", "count"),
                )
                .reset_index()
            )
            repair_ranking["TOTAL_QTY"] = repair_ranking["TOTAL_PO_QTY"].fillna(0) + repair_ranking["TOTAL_BILLED_QTY"].fillna(0)
            repair_ranking = repair_ranking.sort_values("TOTAL_QTY", ascending=False).reset_index(drop=True)
            repair_ranking.index = repair_ranking.index + 1
            repair_ranking.index.name = "Rank"

            with st.container(border=True):
                st.markdown("**Top Materials on Repair Jobs**")
                st.bar_chart(
                    repair_ranking.head(20).set_index("MATERIAL_ID_COMBINED")[["TOTAL_PO_QTY", "TOTAL_BILLED_QTY"]]
                )

            st.dataframe(
                repair_ranking.rename(columns={
                    "MATERIAL_ID_COMBINED": "Material ID",
                    "TOTAL_PO_QTY": "Total PO Qty",
                    "TOTAL_BILLED_QTY": "Total Billed Qty",
                    "TOTAL_QTY": "Combined Qty",
                    "JOB_COUNT": "Job Count",
                }),
                use_container_width=True,
            )

    st.divider()
    st.subheader("Vendors with Service & Repair on Same Vehicle & Same Date")
    st.markdown("Identifies vendors that logged both `service` and `repair` job types on the same vehicle on the same PO posting date.")

    vehicle_col = None
    for vc in ["VEHICLE_ID", "VEHICLE_ID_PO", "LICENCE_PLATE", "LICENCE_PLATE_ID", "VEHICLE_REG_NUMBER", "VEHICLE_REG_NBR"]:
        if vc in filtered.columns:
            vehicle_col = vc
            break

    vendor_display_col = None
    for vnc in ["VENDOR_NAME", "VEND_NAME", "VENDOR_NAME_PO", "VEND_NAME_BILL"]:
        if vnc in filtered.columns:
            vendor_display_col = vnc
            break

    if job_type_col is None or vehicle_col is None or vendor_display_col is None:
        st.warning("Missing required columns (JOB_TYPE, VEHICLE, or VENDOR). Ensure your CSVs include these fields.")
    else:
        svc_mask = filtered[job_type_col].astype(str).str.lower().str.contains("service", na=False)
        rep_mask = filtered[job_type_col].astype(str).str.lower().str.contains("repair", na=False)

        service_records = filtered[svc_mask][[vendor_display_col, vehicle_col, "PO_POSTING_DATE_COMBINED", "JOB_ID_COMBINED", "MATERIAL_ID_COMBINED"]].copy()
        repair_records = filtered[rep_mask][[vendor_display_col, vehicle_col, "PO_POSTING_DATE_COMBINED", "JOB_ID_COMBINED", "MATERIAL_ID_COMBINED"]].copy()

        if service_records.empty or repair_records.empty:
            st.info("No overlap found — need both service and repair records in the filtered data.")
        else:
            overlap = service_records.merge(
                repair_records,
                on=[vendor_display_col, vehicle_col, "PO_POSTING_DATE_COMBINED"],
                how="inner",
                suffixes=("_SERVICE", "_REPAIR"),
            )

            if overlap.empty:
                st.info("No vendors found with both service and repair on the same vehicle and same date.")
            else:
                vendor_summary = (
                    overlap.groupby(vendor_display_col)
                    .agg(
                        OCCURRENCES=(vehicle_col, "count"),
                        VEHICLES=(vehicle_col, "nunique"),
                    )
                    .reset_index()
                    .sort_values("OCCURRENCES", ascending=False)
                    .reset_index(drop=True)
                )
                vendor_summary.index = vendor_summary.index + 1
                vendor_summary.index.name = "Rank"

                st.metric("Vendors Flagged", len(vendor_summary), border=True)

                st.dataframe(
                    vendor_summary.rename(columns={
                        vendor_display_col: "Vendor Name",
                        "OCCURRENCES": "Occurrences",
                        "VEHICLES": "Unique Vehicles",
                    }),
                    use_container_width=True,
                )

                with st.expander("Detail: Overlapping Records"):
                    st.dataframe(
                        overlap.rename(columns={
                            vendor_display_col: "Vendor",
                            vehicle_col: "Vehicle",
                            "PO_POSTING_DATE_COMBINED": "Date",
                            "JOB_ID_COMBINED_SERVICE": "Service Job ID",
                            "MATERIAL_ID_COMBINED_SERVICE": "Service Material",
                            "JOB_ID_COMBINED_REPAIR": "Repair Job ID",
                            "MATERIAL_ID_COMBINED_REPAIR": "Repair Material",
                        }),
                        hide_index=True,
                        use_container_width=True,
                    )

    st.divider()
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

        group_cols_fleet = ["JOB_ID_COMBINED", vehicle_col_fleet, "MATERIAL_ID_COMBINED"]
        if mat_desc_fleet:
            group_cols_fleet.append(mat_desc_fleet)
        group_cols_fleet.append(fleet_col)

        fleet_analysis = (
            fleet_df.groupby(group_cols_fleet)
            .agg(
                PO_QTY=("PO_QTY", "sum"),
                BILLED_QTY=("BILLED_QTY", "sum"),
                OCCURRENCES=("JOB_ID_COMBINED", "count"),
            )
            .reset_index()
        )
        fleet_analysis["QTY_DIFF"] = fleet_analysis["PO_QTY"].fillna(0) - fleet_analysis["BILLED_QTY"].fillna(0)
        fleet_analysis = fleet_analysis.sort_values(["PO_QTY"], ascending=False).reset_index(drop=True)

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
            vehicle_col_fleet: "Vehicle",
            "MATERIAL_ID_COMBINED": "Material ID",
            fleet_col: "Fleet Type",
            "PO_QTY": "PO Qty",
            "BILLED_QTY": "Billed Qty",
            "QTY_DIFF": "Qty Diff",
            "OCCURRENCES": "Times Used",
        }
        if mat_desc_fleet:
            rename_map_fleet[mat_desc_fleet] = "Material Description"

        st.dataframe(
            fleet_analysis.rename(columns=rename_map_fleet),
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
                        pivot_heat.style.background_gradient(cmap="Reds", axis=None),
                        use_container_width=True,
                    )
                else:
                    st.info("Not enough data for heatmap.")

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
                rename_map[vehicle_col_mat] = "Vehicle"

            st.dataframe(
                discrepancies.drop(columns=["HAS_DISCREPANCY"]).rename(columns=rename_map),
                hide_index=True,
                use_container_width=True,
            )

"""Data ingestion: load ILI Excel data, normalize columns, validate quality."""

import datetime
import pandas as pd
import numpy as np
from config import (
    COLUMN_MAP, EVENT_TYPE_MAP, ANOMALY_TYPES, REFERENCE_TYPES,
    ID_OD_MAP_2007, RUN_YEARS,
)


def load_all_runs(filepath: str) -> dict[int, pd.DataFrame]:
    """Load and normalize all ILI run sheets from the Excel file."""
    xls = pd.ExcelFile(filepath, engine="openpyxl")
    runs = {}
    for year in RUN_YEARS:
        sheet = str(year)
        if sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            df = _normalize(df, year)
            runs[year] = df
    return runs


def load_summary(filepath: str) -> pd.DataFrame:
    """Load the Summary sheet with run metadata."""
    return pd.read_excel(filepath, sheet_name="Summary", engine="openpyxl")


def _normalize(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Apply column renaming, type coercion, and value normalization."""
    # Rename columns
    col_map = COLUMN_MAP.get(year, {})
    df = df.rename(columns=col_map)

    # Normalize event types
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].map(
            lambda x: EVENT_TYPE_MAP.get(str(x).strip(), str(x).strip())
            if pd.notna(x) else x
        )

    # Boolean flags
    df["is_anomaly"] = df["event_type"].isin(ANOMALY_TYPES)
    df["is_girth_weld"] = df["event_type"].isin(REFERENCE_TYPES)

    # Normalize clock position (pandas reads as datetime.time → decimal hours 0-12)
    if "clock_position" in df.columns:
        df["clock_hours"] = df["clock_position"].apply(_clock_to_hours)

    # Normalize ID/OD
    if year == 2007 and "id_od" in df.columns:
        df["id_od"] = df["id_od"].map(
            lambda x: ID_OD_MAP_2007.get(str(x).strip(), "Unknown")
            if pd.notna(x) else "Unknown"
        )
    elif "id_od" in df.columns:
        df["id_od"] = df["id_od"].fillna("Unknown").astype(str).str.strip()

    # Ensure numeric columns
    for col in ["depth_pct", "length_in", "width_in", "log_distance_ft",
                 "joint_number", "joint_length_ft", "wall_thickness_in",
                 "dist_to_us_weld_ft", "elevation_ft"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add run year
    df["run_year"] = year

    # Add original row index for traceability
    df["source_row_idx"] = df.index

    return df


def _clock_to_hours(val) -> float:
    """Convert clock position to decimal hours (0-12 scale).

    Handles datetime.time objects (from openpyxl) and numeric values.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, datetime.time):
        total_hours = val.hour + val.minute / 60.0 + val.second / 3600.0
        return total_hours % 12.0 if total_hours > 0 else np.nan
    if isinstance(val, (int, float)):
        # Excel time fraction (0-1 = 0-24h)
        hours_24 = float(val) * 24.0
        return hours_24 % 12.0
    if isinstance(val, str):
        try:
            parts = val.split(":")
            h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            return (h + m / 60.0) % 12.0
        except (ValueError, IndexError):
            return np.nan
    return np.nan


def get_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Extract anomaly rows with valid depth."""
    mask = df["is_anomaly"] & df["depth_pct"].notna() & df["log_distance_ft"].notna()
    return df[mask].copy()


def get_girth_welds(df: pd.DataFrame) -> pd.DataFrame:
    """Extract girth weld rows with valid joint number and distance."""
    mask = df["is_girth_weld"] & df["joint_number"].notna() & df["log_distance_ft"].notna()
    gw = df[mask].copy()
    gw["joint_number"] = gw["joint_number"].astype(int)
    return gw.sort_values("log_distance_ft")


def data_quality_report(runs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Generate a data quality summary for each run."""
    records = []
    for year, df in runs.items():
        anomalies = get_anomalies(df)
        gw = get_girth_welds(df)
        record = {
            "Run Year": year,
            "Total Rows": len(df),
            "Anomaly Count": len(anomalies),
            "Girth Weld Count": len(gw),
            "Depth % Missing": anomalies["depth_pct"].isna().sum() if "depth_pct" in df.columns else "N/A",
            "Length Missing": anomalies["length_in"].isna().sum() if "length_in" in df.columns else "N/A",
            "Width Missing": anomalies["width_in"].isna().sum() if "width_in" in df.columns else "N/A",
            "Clock Missing": anomalies["clock_hours"].isna().sum() if "clock_hours" in df.columns else "N/A",
            "Distance Range": f"{df['log_distance_ft'].min():.1f} – {df['log_distance_ft'].max():.1f}",
            "Joint Range": f"{gw['joint_number'].min()} – {gw['joint_number'].max()}",
        }
        records.append(record)
    return pd.DataFrame(records)


def column_completeness(runs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Compute column-level completeness (% non-null) for anomaly rows per run."""
    key_cols = ["depth_pct", "length_in", "width_in", "clock_hours",
                "log_distance_ft", "joint_number", "wall_thickness_in",
                "id_od", "dist_to_us_weld_ft"]
    rows = []
    for year, df in runs.items():
        anom = get_anomalies(df)
        for col in key_cols:
            if col in anom.columns:
                pct = (anom[col].notna().sum() / len(anom) * 100) if len(anom) > 0 else 0
                rows.append({"Run Year": year, "Column": col, "Completeness %": round(pct, 1)})
            else:
                rows.append({"Run Year": year, "Column": col, "Completeness %": 0.0})
    return pd.DataFrame(rows)

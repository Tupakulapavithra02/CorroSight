"""Alignment: girth weld matching and piecewise linear distance correction."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from data_ingestion import get_girth_welds


def match_girth_welds(runs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Match girth welds across runs by joint number.

    Returns a DataFrame with columns: joint_number, dist_{year} for each run,
    plus delta columns between adjacent runs.
    """
    gw_by_year = {}
    for year, df in runs.items():
        gw = get_girth_welds(df)
        # Take first occurrence per joint (avoid duplicates)
        gw_dedup = gw.drop_duplicates(subset="joint_number", keep="first")
        gw_by_year[year] = gw_dedup.set_index("joint_number")["log_distance_ft"]

    # Find common joint numbers across all runs
    joint_sets = [set(s.index) for s in gw_by_year.values()]
    common_joints = sorted(set.intersection(*joint_sets))

    records = []
    for jn in common_joints:
        record = {"joint_number": jn}
        for year in sorted(gw_by_year.keys()):
            record[f"dist_{year}"] = gw_by_year[year].loc[jn]
        records.append(record)

    result = pd.DataFrame(records)

    # Compute deltas between consecutive runs
    years = sorted(gw_by_year.keys())
    for i in range(1, len(years)):
        y_prev, y_curr = years[i - 1], years[i]
        result[f"delta_{y_prev}_{y_curr}"] = (
            result[f"dist_{y_curr}"] - result[f"dist_{y_prev}"]
        )

    return result


def build_distance_corrector(
    gw_alignment: pd.DataFrame,
    source_year: int,
    reference_year: int = 2022,
) -> interp1d:
    """Build piecewise linear interpolation function to correct distances.

    Maps distances from source_year's coordinate frame to reference_year's frame.
    """
    src_col = f"dist_{source_year}"
    ref_col = f"dist_{reference_year}"

    src_dists = gw_alignment[src_col].values
    ref_dists = gw_alignment[ref_col].values

    # interp1d with fill_value="extrapolate" for anomalies outside girth weld range
    corrector = interp1d(
        src_dists, ref_dists,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return corrector


def apply_distance_correction(
    runs: dict[int, pd.DataFrame],
    gw_alignment: pd.DataFrame,
    reference_year: int = 2022,
) -> dict[int, pd.DataFrame]:
    """Apply distance correction to all runs, adding 'corrected_distance' column.

    The reference year keeps its original distances. Other years get corrected
    to the reference frame via piecewise linear interpolation on girth welds.
    """
    corrected_runs = {}
    for year, df in runs.items():
        df = df.copy()
        if year == reference_year:
            df["corrected_distance"] = df["log_distance_ft"]
        else:
            corrector = build_distance_corrector(gw_alignment, year, reference_year)
            valid = df["log_distance_ft"].notna()
            df.loc[valid, "corrected_distance"] = corrector(
                df.loc[valid, "log_distance_ft"].values
            )
            df.loc[~valid, "corrected_distance"] = np.nan
        corrected_runs[year] = df

    return corrected_runs


def compute_alignment_stats(gw_alignment: pd.DataFrame) -> dict:
    """Compute alignment statistics for the dashboard."""
    years = sorted([int(c.split("_")[1]) for c in gw_alignment.columns if c.startswith("dist_")])
    stats = {
        "common_joints": len(gw_alignment),
        "joint_range": (int(gw_alignment["joint_number"].min()),
                        int(gw_alignment["joint_number"].max())),
    }
    for i in range(1, len(years)):
        y_prev, y_curr = years[i - 1], years[i]
        delta_col = f"delta_{y_prev}_{y_curr}"
        if delta_col in gw_alignment.columns:
            deltas = gw_alignment[delta_col]
            stats[f"drift_{y_prev}_{y_curr}"] = {
                "mean": round(deltas.mean(), 3),
                "std": round(deltas.std(), 3),
                "min": round(deltas.min(), 3),
                "max": round(deltas.max(), 3),
                "abs_mean": round(deltas.abs().mean(), 3),
            }
    return stats

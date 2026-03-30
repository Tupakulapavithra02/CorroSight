"""Integrity analytics: segment risk heatmap, ASME B31G interaction,
automated dig list, and population growth analytics."""

import numpy as np
import pandas as pd

from config import WALL_LOSS_REPAIR_THRESHOLD, MAX_PLAUSIBLE_GROWTH_RATE


# ── 1. Segment Risk Heatmap ─────────────────────────────────────────────────

def segment_risk_analysis(
    matches_df: pd.DataFrame,
    corrected_runs: dict,
    segment_length_ft: float = 1000.0,
) -> list[dict]:
    """Divide pipeline into segments and compute composite risk score per segment.

    Composite score (0-100) = anomaly_density (25) + max_depth (35) +
                              avg_growth_rate (25) + critical_count (15)
    """
    # Get pipeline length from latest run
    latest_year = max(corrected_runs.keys())
    latest_run = corrected_runs[latest_year]
    max_dist = latest_run["corrected_distance"].max()
    n_segments = int(np.ceil(max_dist / segment_length_ft))

    # Build anomaly list with growth data from the most recent pair
    if matches_df.empty:
        # Fall back to raw anomalies without growth data
        anom = latest_run[latest_run["is_anomaly"]].copy()
        anom = anom.rename(columns={
            "corrected_distance": "distance",
            "depth_pct": "depth",
        })
        anom["growth_rate"] = np.nan
        anom["risk_category"] = "Unknown"
    else:
        anom = matches_df[["later_distance", "later_depth_pct",
                           "depth_growth_rate", "risk_category"]].copy()
        anom.columns = ["distance", "depth", "growth_rate", "risk_category"]

    segments = []
    for i in range(n_segments):
        start = i * segment_length_ft
        end = start + segment_length_ft
        mid = start + segment_length_ft / 2

        seg_anom = anom[(anom["distance"] >= start) & (anom["distance"] < end)]
        count = len(seg_anom)

        if count == 0:
            segments.append({
                "segment": i + 1,
                "start_ft": round(start, 1),
                "end_ft": round(end, 1),
                "midpoint_ft": round(mid, 1),
                "anomaly_count": 0,
                "max_depth_pct": 0,
                "avg_growth_rate": 0,
                "critical_count": 0,
                "risk_score": 0,
            })
            continue

        max_depth = seg_anom["depth"].max() if not seg_anom["depth"].isna().all() else 0
        avg_rate = seg_anom["growth_rate"].dropna()
        avg_rate = avg_rate[avg_rate >= 0].mean() if len(avg_rate[avg_rate >= 0]) > 0 else 0
        critical_count = (seg_anom["risk_category"] == "Critical").sum()

        # Density score (0-25): normalize by expected density (~5 per segment is high)
        density_score = min(25, count * 25 / 5)

        # Max depth score (0-35): 80% = full score
        depth_score = min(35, (max_depth / WALL_LOSS_REPAIR_THRESHOLD) * 35) if not np.isnan(max_depth) else 0

        # Growth rate score (0-25): 3%/yr = full score
        rate_score = min(25, (avg_rate / 3.0) * 25) if not np.isnan(avg_rate) else 0

        # Critical count score (0-15): 3 criticals = full score
        crit_score = min(15, critical_count * 15 / 3)

        risk_score = round(density_score + depth_score + rate_score + crit_score, 1)

        segments.append({
            "segment": i + 1,
            "start_ft": round(start, 1),
            "end_ft": round(end, 1),
            "midpoint_ft": round(mid, 1),
            "anomaly_count": int(count),
            "max_depth_pct": round(float(max_depth), 1) if not np.isnan(max_depth) else 0,
            "avg_growth_rate": round(float(avg_rate), 3) if not np.isnan(avg_rate) else 0,
            "critical_count": int(critical_count),
            "risk_score": risk_score,
        })

    return segments


# ── 2. ASME B31G Anomaly Interaction Assessment ─────────────────────────────

def interaction_assessment(matches_df: pd.DataFrame) -> list[dict]:
    """Detect anomalies that may interact per ASME B31G / RSTRENG rules.

    Per ASME B31G, two anomalies interact if their axial spacing is less than
    6 × wall_thickness. When anomalies interact, their combined effective
    length is greater than either alone, reducing the pipe's burst pressure.
    """
    if matches_df.empty:
        return []

    # Work with the latest-run anomaly data
    required = ["later_distance", "later_depth_pct", "later_wall_thickness",
                 "later_joint", "later_clock", "later_length_in",
                 "depth_growth_rate", "risk_score"]
    available = [c for c in required if c in matches_df.columns]
    df = matches_df[available].dropna(subset=["later_distance", "later_depth_pct"]).copy()
    df = df.sort_values("later_distance").reset_index(drop=True)

    if len(df) < 2:
        return []

    interactions = []
    used = set()

    for i in range(len(df)):
        if i in used:
            continue
        row_i = df.iloc[i]
        wt = row_i.get("later_wall_thickness", 0.3)
        if pd.isna(wt) or wt <= 0:
            wt = 0.3  # Default wall thickness

        # Interaction threshold: 6 × wall thickness (in inches), convert to feet
        threshold_ft = (6 * wt) / 12.0

        cluster = [i]
        # Find all anomalies within the threshold distance (chain forward)
        j = i + 1
        last_dist = row_i["later_distance"]
        while j < len(df):
            next_dist = df.iloc[j]["later_distance"]
            spacing = next_dist - last_dist

            # Account for anomaly length
            length_i = row_i.get("later_length_in", 0)
            if pd.isna(length_i):
                length_i = 0
            clear_spacing = spacing - (length_i / 12.0)

            if clear_spacing <= threshold_ft:
                cluster.append(j)
                used.add(j)
                last_dist = next_dist
                j += 1
            else:
                break

        if len(cluster) >= 2:
            used.add(i)
            members = df.iloc[cluster]

            # Combined effective length
            total_span_ft = members["later_distance"].max() - members["later_distance"].min()
            member_lengths = members.get("later_length_in", pd.Series([0] * len(members)))
            total_length_in = total_span_ft * 12 + member_lengths.fillna(0).max()

            max_depth = members["later_depth_pct"].max()
            avg_depth = members["later_depth_pct"].mean()
            max_growth = members.get("depth_growth_rate", pd.Series()).max()
            max_risk = members.get("risk_score", pd.Series()).max()

            # Interaction severity based on combined depth and count
            if max_depth >= 60 or len(cluster) >= 4:
                severity = "HIGH"
            elif max_depth >= 40 or len(cluster) >= 3:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            interactions.append({
                "cluster_id": len(interactions) + 1,
                "anomaly_count": len(cluster),
                "start_distance_ft": round(float(members["later_distance"].min()), 2),
                "end_distance_ft": round(float(members["later_distance"].max()), 2),
                "span_ft": round(float(total_span_ft), 2),
                "effective_length_in": round(float(total_length_in), 1),
                "max_depth_pct": round(float(max_depth), 1),
                "avg_depth_pct": round(float(avg_depth), 1),
                "max_growth_rate": round(float(max_growth), 3) if not pd.isna(max_growth) else None,
                "max_risk_score": round(float(max_risk), 1) if not pd.isna(max_risk) else None,
                "joint": int(members["later_joint"].iloc[0]) if "later_joint" in members.columns and not pd.isna(members["later_joint"].iloc[0]) else None,
                "wall_thickness_in": round(float(wt), 3),
                "interaction_threshold_in": round(float(6 * wt), 2),
                "severity": severity,
            })

    return interactions


# ── 3. Automated Dig List / Repair Prioritization ───────────────────────────

def generate_dig_list(matches_df: pd.DataFrame) -> list[dict]:
    """Generate prioritized repair schedule with IMMEDIATE / SCHEDULED / MONITOR.

    Urgency score = current_depth_component (40) + growth_rate_component (30) +
                    remaining_life_component (30)

    Categories:
      IMMEDIATE: urgency >= 75 or depth >= 70% or remaining_life < 3 years
      SCHEDULED: urgency >= 50 or depth >= 50% or remaining_life < 7 years
      MONITOR:   everything else with growth or depth concern
    """
    if matches_df.empty:
        return []

    required = ["later_distance", "later_depth_pct", "later_joint", "later_clock",
                 "depth_growth_rate", "remaining_life_years", "risk_score",
                 "risk_category", "later_event_type", "later_id_od",
                 "later_wall_thickness", "confidence_label"]
    available = [c for c in required if c in matches_df.columns]
    df = matches_df[available].copy()
    df = df.dropna(subset=["later_depth_pct"])

    dig_items = []
    for _, row in df.iterrows():
        depth = row.get("later_depth_pct", 0)
        rate = row.get("depth_growth_rate", 0)
        rem_life = row.get("remaining_life_years", np.nan)

        if pd.isna(depth):
            depth = 0
        if pd.isna(rate) or rate < 0:
            rate = 0
        if pd.isna(rem_life):
            rem_life = 999

        # Skip low-concern anomalies (< 20% depth and no growth)
        if depth < 20 and rate <= 0.5:
            continue

        # Depth component (0-40): 80% = full
        depth_score = min(40, (depth / WALL_LOSS_REPAIR_THRESHOLD) * 40)

        # Growth rate component (0-30): 5%/yr = full
        rate_score = min(30, (rate / MAX_PLAUSIBLE_GROWTH_RATE) * 30)

        # Remaining life component (0-30): 0 yrs = full, 15 yrs = zero
        if rem_life <= 0:
            life_score = 30
        elif rem_life >= 15:
            life_score = 0
        else:
            life_score = 30 * (1 - rem_life / 15)

        urgency = round(depth_score + rate_score + life_score, 1)

        # Categorize
        if urgency >= 75 or depth >= 70 or rem_life < 3:
            category = "IMMEDIATE"
            priority = 1
        elif urgency >= 50 or depth >= 50 or rem_life < 7:
            category = "SCHEDULED"
            priority = 2
        else:
            category = "MONITOR"
            priority = 3

        dig_items.append({
            "joint": int(row.get("later_joint", 0)) if not pd.isna(row.get("later_joint")) else None,
            "distance_ft": round(float(row.get("later_distance", 0)), 2) if not pd.isna(row.get("later_distance")) else None,
            "clock": round(float(row.get("later_clock", 0)), 1) if not pd.isna(row.get("later_clock", np.nan)) else None,
            "depth_pct": round(float(depth), 1),
            "growth_rate": round(float(rate), 3),
            "remaining_life_years": round(float(rem_life), 1) if rem_life < 999 else None,
            "event_type": row.get("later_event_type", ""),
            "id_od": row.get("later_id_od", ""),
            "wall_thickness_in": round(float(row.get("later_wall_thickness", 0)), 3) if not pd.isna(row.get("later_wall_thickness")) else None,
            "urgency_score": urgency,
            "category": category,
            "priority": priority,
            "risk_category": row.get("risk_category", ""),
            "confidence": row.get("confidence_label", ""),
        })

    # Sort by priority then urgency (descending)
    dig_items.sort(key=lambda x: (x["priority"], -x["urgency_score"]))
    return dig_items


# ── 4. Population Growth Analytics ──────────────────────────────────────────

def population_analytics(matches_df: pd.DataFrame) -> dict:
    """Analyze growth patterns by clock quadrant, ID/OD, and depth band.

    Clock quadrants:
      Top (10-2 o'clock), Right (2-4), Bottom (4-8), Left (8-10)

    Reveals systemic corrosion patterns:
      - Bottom-of-pipe = water settling (internal), soil-side (external)
      - Top-of-pipe = gas phase corrosion (internal), coating failure (external)
    """
    if matches_df.empty:
        return {"by_quadrant": [], "by_id_od": [], "by_depth_band": [],
                "quadrant_id_od": []}

    df = matches_df.copy()
    rates = df["depth_growth_rate"].copy()

    # Only use valid positive growth rates
    valid_mask = rates.notna() & (rates >= 0)
    df_valid = df[valid_mask].copy()

    if df_valid.empty:
        return {"by_quadrant": [], "by_id_od": [], "by_depth_band": [],
                "quadrant_id_od": []}

    # Assign clock quadrant
    def _clock_quadrant(clock):
        if pd.isna(clock):
            return "Unknown"
        clock = clock % 12
        if clock >= 10 or clock < 2:
            return "Top (10-2)"
        elif clock >= 2 and clock < 4:
            return "Right (2-4)"
        elif clock >= 4 and clock < 8:
            return "Bottom (4-8)"
        else:
            return "Left (8-10)"

    df_valid["quadrant"] = df_valid["later_clock"].apply(_clock_quadrant)

    # Assign depth band
    def _depth_band(depth):
        if pd.isna(depth):
            return "Unknown"
        if depth < 20:
            return "0-20%"
        elif depth < 40:
            return "20-40%"
        elif depth < 60:
            return "40-60%"
        else:
            return "60%+"

    df_valid["depth_band"] = df_valid["later_depth_pct"].apply(_depth_band)

    # ── By Quadrant ──
    by_quadrant = []
    for quad, group in df_valid.groupby("quadrant"):
        rates_g = group["depth_growth_rate"]
        by_quadrant.append({
            "quadrant": quad,
            "count": int(len(group)),
            "mean_growth_rate": round(float(rates_g.mean()), 3),
            "median_growth_rate": round(float(rates_g.median()), 3),
            "max_growth_rate": round(float(rates_g.max()), 3),
            "pct_high_growth": round(float((rates_g > 3.0).mean() * 100), 1),
            "avg_depth": round(float(group["later_depth_pct"].mean()), 1),
        })

    # ── By ID/OD ──
    by_id_od = []
    id_od_col = "later_id_od" if "later_id_od" in df_valid.columns else None
    if id_od_col:
        for label, group in df_valid.groupby(id_od_col):
            if pd.isna(label) or label == "":
                label = "Unknown"
            rates_g = group["depth_growth_rate"]
            by_id_od.append({
                "type": str(label),
                "count": int(len(group)),
                "mean_growth_rate": round(float(rates_g.mean()), 3),
                "median_growth_rate": round(float(rates_g.median()), 3),
                "max_growth_rate": round(float(rates_g.max()), 3),
                "avg_depth": round(float(group["later_depth_pct"].mean()), 1),
            })

    # ── By Depth Band ──
    by_depth_band = []
    for band, group in df_valid.groupby("depth_band"):
        rates_g = group["depth_growth_rate"]
        by_depth_band.append({
            "band": band,
            "count": int(len(group)),
            "mean_growth_rate": round(float(rates_g.mean()), 3),
            "median_growth_rate": round(float(rates_g.median()), 3),
        })

    # ── Cross-tab: Quadrant × ID/OD ──
    quadrant_id_od = []
    if id_od_col:
        for (quad, idod), group in df_valid.groupby(["quadrant", id_od_col]):
            if pd.isna(idod) or idod == "":
                idod = "Unknown"
            rates_g = group["depth_growth_rate"]
            quadrant_id_od.append({
                "quadrant": quad,
                "id_od": str(idod),
                "count": int(len(group)),
                "mean_growth_rate": round(float(rates_g.mean()), 3),
                "avg_depth": round(float(group["later_depth_pct"].mean()), 1),
            })

    return {
        "by_quadrant": by_quadrant,
        "by_id_od": by_id_od,
        "by_depth_band": by_depth_band,
        "quadrant_id_od": quadrant_id_od,
    }


# ── Combined Dashboard Endpoint ─────────────────────────────────────────────

def compute_integrity_dashboard(cache: dict) -> dict:
    """Compute all integrity analytics from the cached analysis results."""
    results = cache.get("results", {})
    corrected_runs = cache.get("corrected_runs", {})

    # Get the most recent pairwise matches (prefer 2015-2022)
    pairwise = results.get("pairwise", {})
    best_key = None
    for key in [(2015, 2022), (2007, 2022), (2007, 2015)]:
        if key in pairwise:
            best_key = key
            break

    matches = pd.DataFrame()
    if best_key and "matches" in pairwise[best_key]:
        matches = pairwise[best_key]["matches"]

    segments = segment_risk_analysis(matches, corrected_runs)
    interactions = interaction_assessment(matches)
    dig_list = generate_dig_list(matches)
    population = population_analytics(matches)

    # Summary stats
    immediate_count = sum(1 for d in dig_list if d["category"] == "IMMEDIATE")
    scheduled_count = sum(1 for d in dig_list if d["category"] == "SCHEDULED")
    monitor_count = sum(1 for d in dig_list if d["category"] == "MONITOR")

    high_risk_segments = sum(1 for s in segments if s["risk_score"] >= 60)

    return {
        "summary": {
            "total_dig_items": len(dig_list),
            "immediate_count": immediate_count,
            "scheduled_count": scheduled_count,
            "monitor_count": monitor_count,
            "interaction_clusters": len(interactions),
            "high_risk_segments": high_risk_segments,
            "total_segments": len(segments),
            "match_pair": f"{best_key[0]}-{best_key[1]}" if best_key else "N/A",
        },
        "segments": segments,
        "interactions": interactions,
        "dig_list": dig_list,
        "population": population,
    }

"""Anomaly matching: candidate search, similarity scoring, Hungarian assignment."""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment

from config import (
    DISTANCE_TOLERANCE_FT, CLOCK_TOLERANCE_HOURS,
    WEIGHT_DISTANCE, WEIGHT_CLOCK, WEIGHT_DEPTH, WEIGHT_DIMENSIONS, WEIGHT_TYPE,
    HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE,
    COMPATIBLE_TYPES, MAX_PLAUSIBLE_GROWTH_RATE,
)
from data_ingestion import get_anomalies


# ── Clock utilities ───────────────────────────────────────────────────────────

def clock_distance(h1: float, h2: float) -> float:
    """Circular distance between two clock positions (0-12 scale). Result in [0, 6]."""
    if np.isnan(h1) or np.isnan(h2):
        return 6.0  # Maximum distance when unknown
    diff = abs(h1 - h2) % 12.0
    return min(diff, 12.0 - diff)


def clock_to_trig(hours: float) -> tuple[float, float]:
    """Convert clock hours to (cos, sin) for KD-tree embedding."""
    if np.isnan(hours):
        return (0.0, 0.0)
    theta = hours * 2 * np.pi / 12.0
    return (np.cos(theta), np.sin(theta))


# ── Similarity scoring ────────────────────────────────────────────────────────

def compute_similarity(a_later: pd.Series, a_earlier: pd.Series,
                       years_between: float) -> float:
    """Compute weighted multi-attribute similarity score (0-1)."""
    # Distance score
    dist_diff = abs(a_later["corrected_distance"] - a_earlier["corrected_distance"])
    s_dist = max(0, 1.0 - dist_diff / DISTANCE_TOLERANCE_FT)

    # Clock score
    clk_diff = clock_distance(a_later.get("clock_hours", np.nan),
                               a_earlier.get("clock_hours", np.nan))
    s_clock = max(0, 1.0 - clk_diff / (CLOCK_TOLERANCE_HOURS * 6.0))  # Normalize to 6h max

    # Depth score (asymmetric: penalize shrinkage more)
    d_later = a_later.get("depth_pct", np.nan)
    d_earlier = a_earlier.get("depth_pct", np.nan)
    if np.isnan(d_later) or np.isnan(d_earlier):
        s_depth = 0.5
    else:
        depth_diff = d_later - d_earlier
        if depth_diff >= 0:
            s_depth = max(0, 1.0 - depth_diff / 30.0)
        else:
            s_depth = max(0, 1.0 - abs(depth_diff) / 10.0)

    # Dimensional similarity
    len_diff = _safe_diff(a_later.get("length_in"), a_earlier.get("length_in"))
    wid_diff = _safe_diff(a_later.get("width_in"), a_earlier.get("width_in"))
    s_dim = max(0, 1.0 - (len_diff + wid_diff) / 6.0)

    # Type compatibility score
    type_later = a_later.get("event_type", "")
    type_earlier = a_earlier.get("event_type", "")
    s_type = 1.0 if type_later == type_earlier else 0.7

    score = (WEIGHT_DISTANCE * s_dist +
             WEIGHT_CLOCK * s_clock +
             WEIGHT_DEPTH * s_depth +
             WEIGHT_DIMENSIONS * s_dim +
             WEIGHT_TYPE * s_type)
    return score


def _safe_diff(a, b) -> float:
    """Absolute difference with NaN handling."""
    if pd.isna(a) or pd.isna(b):
        return 1.5  # Moderate penalty for missing data
    return abs(float(a) - float(b))


# ── Type compatibility ────────────────────────────────────────────────────────

def types_compatible(type_a: str, type_b: str) -> bool:
    """Check if two event types are compatible for matching."""
    if type_a == type_b:
        return True
    compat_a = COMPATIBLE_TYPES.get(type_a, {type_a})
    return type_b in compat_a


# ── Main matching pipeline ────────────────────────────────────────────────────

def match_anomalies(
    run_later: pd.DataFrame,
    run_earlier: pd.DataFrame,
    years_between: float,
) -> dict:
    """Match anomalies between two runs using Hungarian algorithm.

    Returns dict with keys: 'matches', 'new_anomalies', 'missing_anomalies', 'stats'.
    """
    anom_later = get_anomalies(run_later).reset_index(drop=True)
    anom_earlier = get_anomalies(run_earlier).reset_index(drop=True)

    n_later = len(anom_later)
    n_earlier = len(anom_earlier)

    if n_later == 0 or n_earlier == 0:
        return {
            "matches": pd.DataFrame(),
            "new_anomalies": anom_later,
            "missing_anomalies": anom_earlier,
            "stats": {"total_matches": 0},
        }

    # Build KD-tree for earlier run anomalies
    earlier_points = _build_search_points(anom_earlier)
    tree = KDTree(earlier_points)

    # For each later anomaly, find candidates and compute scores
    # Build cost matrix (sparse: only fill candidate pairs)
    LARGE_COST = 1e6
    cost_matrix = np.full((n_later, n_earlier), LARGE_COST)
    candidate_counts = np.zeros(n_later, dtype=int)

    # Search radius: distance tolerance in the embedded space
    # Scale factors: distance in ft, clock in trig coords (range -1 to 1)
    search_radius = max(DISTANCE_TOLERANCE_FT, 2.0)  # Conservative radius

    for i in range(n_later):
        later_point = _build_search_point_single(anom_later.iloc[i])
        # Query KD-tree for nearby candidates
        candidate_indices = tree.query_ball_point(later_point, r=search_radius)

        for j in candidate_indices:
            a_l = anom_later.iloc[i]
            a_e = anom_earlier.iloc[j]

            # Verify distance tolerance
            dist_diff = abs(a_l["corrected_distance"] - a_e["corrected_distance"])
            if dist_diff > DISTANCE_TOLERANCE_FT:
                continue

            # Verify clock tolerance
            clk_diff = clock_distance(
                a_l.get("clock_hours", np.nan),
                a_e.get("clock_hours", np.nan)
            )
            if clk_diff > CLOCK_TOLERANCE_HOURS:
                continue

            # Verify type compatibility
            if not types_compatible(a_l.get("event_type", ""), a_e.get("event_type", "")):
                continue

            # Compute similarity
            sim = compute_similarity(a_l, a_e, years_between)
            cost_matrix[i, j] = 1.0 - sim
            candidate_counts[i] += 1

    # Solve Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build match results
    matches = []
    matched_later = set()
    matched_earlier = set()

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] >= (1.0 - LOW_CONFIDENCE):
            continue  # Skip poor matches

        sim = 1.0 - cost_matrix[i, j]
        a_l = anom_later.iloc[i]
        a_e = anom_earlier.iloc[j]

        # Compute confidence
        confidence = _compute_confidence(
            sim, candidate_counts[i], a_l, a_e, years_between
        )
        conf_label = _classify_confidence(confidence)

        match_record = _build_match_record(a_l, a_e, sim, confidence, conf_label, years_between)
        matches.append(match_record)
        matched_later.add(i)
        matched_earlier.add(j)

    matches_df = pd.DataFrame(matches) if matches else pd.DataFrame()

    # Unmatched anomalies
    new_idx = [i for i in range(n_later) if i not in matched_later]
    missing_idx = [i for i in range(n_earlier) if i not in matched_earlier]
    new_anomalies = anom_later.iloc[new_idx].copy() if new_idx else pd.DataFrame()
    missing_anomalies = anom_earlier.iloc[missing_idx].copy() if missing_idx else pd.DataFrame()

    # Stats
    stats = _compute_match_stats(matches_df, new_anomalies, missing_anomalies)

    return {
        "matches": matches_df,
        "new_anomalies": new_anomalies,
        "missing_anomalies": missing_anomalies,
        "stats": stats,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_search_points(df: pd.DataFrame) -> np.ndarray:
    """Build search points array: (corrected_distance, cos_clock, sin_clock)."""
    points = np.zeros((len(df), 3))
    for i, (_, row) in enumerate(df.iterrows()):
        cos_c, sin_c = clock_to_trig(row.get("clock_hours", np.nan))
        points[i] = [row["corrected_distance"], cos_c, sin_c]
    return points


def _build_search_point_single(row: pd.Series) -> np.ndarray:
    """Build a single search point."""
    cos_c, sin_c = clock_to_trig(row.get("clock_hours", np.nan))
    return np.array([row["corrected_distance"], cos_c, sin_c])


def _compute_confidence(
    similarity: float,
    n_candidates: int,
    a_later: pd.Series,
    a_earlier: pd.Series,
    years_between: float,
) -> float:
    """Compute match confidence incorporating multiple factors."""
    # Factor 1: Raw similarity (40%)
    f_sim = similarity

    # Factor 2: Uniqueness — single candidate = high confidence (25%)
    if n_candidates <= 1:
        f_unique = 1.0
    elif n_candidates == 2:
        f_unique = 0.7
    else:
        f_unique = max(0.3, 1.0 - n_candidates * 0.1)

    # Factor 3: Growth plausibility (20%)
    d_l = a_later.get("depth_pct", np.nan)
    d_e = a_earlier.get("depth_pct", np.nan)
    if not np.isnan(d_l) and not np.isnan(d_e) and years_between > 0:
        growth_rate = (d_l - d_e) / years_between
        if 0 <= growth_rate <= MAX_PLAUSIBLE_GROWTH_RATE:
            f_plaus = 1.0
        elif growth_rate < 0:
            f_plaus = max(0, 0.5 + growth_rate / 10.0)
        else:
            f_plaus = max(0.2, 1.0 - (growth_rate - MAX_PLAUSIBLE_GROWTH_RATE) / 10.0)
    else:
        f_plaus = 0.5

    # Factor 4: Same joint number (15%)
    jn_l = a_later.get("joint_number", np.nan)
    jn_e = a_earlier.get("joint_number", np.nan)
    if not np.isnan(jn_l) and not np.isnan(jn_e):
        f_joint = 1.0 if int(jn_l) == int(jn_e) else 0.6
    else:
        f_joint = 0.5

    confidence = 0.40 * f_sim + 0.25 * f_unique + 0.20 * f_plaus + 0.15 * f_joint
    return round(confidence, 4)


def _classify_confidence(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE:
        return "HIGH"
    elif confidence >= MEDIUM_CONFIDENCE:
        return "MEDIUM"
    else:
        return "LOW"


def _build_match_record(
    a_later: pd.Series,
    a_earlier: pd.Series,
    similarity: float,
    confidence: float,
    conf_label: str,
    years_between: float,
) -> dict:
    """Build a single match record with all relevant fields."""
    # Growth calculations
    depth_growth = _safe_sub(a_later.get("depth_pct"), a_earlier.get("depth_pct"))
    length_growth = _safe_sub(a_later.get("length_in"), a_earlier.get("length_in"))
    width_growth = _safe_sub(a_later.get("width_in"), a_earlier.get("width_in"))

    depth_rate = depth_growth / years_between if not np.isnan(depth_growth) else np.nan
    length_rate = length_growth / years_between if not np.isnan(length_growth) else np.nan
    width_rate = width_growth / years_between if not np.isnan(width_growth) else np.nan

    return {
        "similarity": round(similarity, 4),
        "confidence": confidence,
        "confidence_label": conf_label,
        # Earlier run
        "earlier_year": a_earlier.get("run_year"),
        "earlier_joint": a_earlier.get("joint_number"),
        "earlier_distance": round(a_earlier.get("corrected_distance", np.nan), 2),
        "earlier_orig_distance": round(a_earlier.get("log_distance_ft", np.nan), 2),
        "earlier_clock": round(a_earlier.get("clock_hours", np.nan), 2),
        "earlier_depth_pct": a_earlier.get("depth_pct"),
        "earlier_length_in": a_earlier.get("length_in"),
        "earlier_width_in": a_earlier.get("width_in"),
        "earlier_event_type": a_earlier.get("event_type"),
        "earlier_id_od": a_earlier.get("id_od"),
        "earlier_comments": a_earlier.get("comments"),
        "earlier_wall_thickness": a_earlier.get("wall_thickness_in"),
        "earlier_row_idx": a_earlier.get("source_row_idx"),
        # Later run
        "later_year": a_later.get("run_year"),
        "later_joint": a_later.get("joint_number"),
        "later_distance": round(a_later.get("corrected_distance", np.nan), 2),
        "later_orig_distance": round(a_later.get("log_distance_ft", np.nan), 2),
        "later_clock": round(a_later.get("clock_hours", np.nan), 2),
        "later_depth_pct": a_later.get("depth_pct"),
        "later_length_in": a_later.get("length_in"),
        "later_width_in": a_later.get("width_in"),
        "later_event_type": a_later.get("event_type"),
        "later_id_od": a_later.get("id_od"),
        "later_comments": a_later.get("comments"),
        "later_wall_thickness": a_later.get("wall_thickness_in"),
        "later_row_idx": a_later.get("source_row_idx"),
        # Growth
        "years_between": years_between,
        "depth_growth_pct": round(depth_growth, 2) if not np.isnan(depth_growth) else np.nan,
        "depth_growth_rate": round(depth_rate, 3) if not np.isnan(depth_rate) else np.nan,
        "length_growth_in": round(length_growth, 2) if not np.isnan(length_growth) else np.nan,
        "length_growth_rate": round(length_rate, 3) if not np.isnan(length_rate) else np.nan,
        "width_growth_in": round(width_growth, 2) if not np.isnan(width_growth) else np.nan,
        "width_growth_rate": round(width_rate, 3) if not np.isnan(width_rate) else np.nan,
    }


def _safe_sub(a, b) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)


def _compute_match_stats(matches_df, new_df, missing_df) -> dict:
    """Compute summary statistics for matching results."""
    stats = {
        "total_matches": len(matches_df),
        "new_anomalies": len(new_df),
        "missing_anomalies": len(missing_df),
    }
    if len(matches_df) > 0:
        stats["high_confidence"] = (matches_df["confidence_label"] == "HIGH").sum()
        stats["medium_confidence"] = (matches_df["confidence_label"] == "MEDIUM").sum()
        stats["low_confidence"] = (matches_df["confidence_label"] == "LOW").sum()
        stats["avg_similarity"] = round(matches_df["similarity"].mean(), 3)
        stats["avg_confidence"] = round(matches_df["confidence"].mean(), 3)
        if "depth_growth_rate" in matches_df.columns:
            valid_rates = matches_df["depth_growth_rate"].dropna()
            stats["avg_depth_growth_rate"] = round(valid_rates.mean(), 3) if len(valid_rates) > 0 else np.nan
            stats["negative_growth_count"] = (valid_rates < 0).sum()
            stats["high_growth_count"] = (valid_rates > MAX_PLAUSIBLE_GROWTH_RATE).sum()
    return stats

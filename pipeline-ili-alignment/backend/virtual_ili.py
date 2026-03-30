"""Virtual ILI: Predict future inspection findings from historical growth trends."""

import numpy as np
import pandas as pd

from config import WALL_LOSS_REPAIR_THRESHOLD, MAX_PLAUSIBLE_GROWTH_RATE


def predict_future_inspection(cache: dict, target_year: int) -> dict:
    """Simulate what a future ILI run would find at target_year.

    Uses growth rates from matched anomalies to extrapolate depth forward.
    Returns predicted anomaly states, risk distribution, and summary stats.
    """
    results = cache.get("results", {})
    pairwise = results.get("pairwise", {})
    chain = results.get("chain", {})
    triple = chain.get("triple_matches", pd.DataFrame())

    # Use 2015-2022 matches as the primary source (most recent growth data)
    key = (2015, 2022)
    if key not in pairwise:
        key = list(pairwise.keys())[-1] if pairwise else None
    if key is None:
        return {"error": "No match data available"}

    matches = pairwise[key].get("matches", pd.DataFrame())
    if matches.empty:
        return {"error": "No matches to extrapolate"}

    base_year = 2022
    years_forward = target_year - base_year
    if years_forward <= 0:
        return {"error": "Target year must be after 2022"}

    predictions = []
    for _, row in matches.iterrows():
        current_depth = row.get("later_depth_pct", np.nan)
        growth_rate = row.get("depth_growth_rate", np.nan)

        if pd.isna(current_depth) or pd.isna(growth_rate):
            continue

        # Use triple-match refined rate if available
        refined_rate = growth_rate
        is_triple = False
        if not triple.empty:
            j = row.get("later_joint")
            tm = triple[triple["joint_2022"] == j]
            if not tm.empty:
                lr = tm.iloc[0].get("linear_rate", np.nan)
                if not pd.isna(lr):
                    refined_rate = lr
                    is_triple = True

        # Skip anomalies with negative growth rates (measurement artifacts)
        if refined_rate < 0:
            continue

        predicted_depth = current_depth + (refined_rate * years_forward)
        predicted_depth = max(0, min(100, predicted_depth))

        remaining_capacity = WALL_LOSS_REPAIR_THRESHOLD - predicted_depth
        if refined_rate > 0:
            years_to_threshold = remaining_capacity / refined_rate if remaining_capacity > 0 else 0
        else:
            years_to_threshold = None

        # Risk at predicted time
        if predicted_depth >= 70:
            risk = "Critical"
        elif predicted_depth >= 50:
            risk = "High"
        elif predicted_depth >= 30:
            risk = "Medium"
        else:
            risk = "Low"

        predictions.append({
            "joint": _safe(row.get("later_joint")),
            "distance_ft": _safe(row.get("later_distance")),
            "clock": _safe(row.get("later_clock")),
            "current_depth_2022": round(current_depth, 1),
            "growth_rate": round(refined_rate, 3),
            "predicted_depth": round(predicted_depth, 1),
            "predicted_risk": risk,
            "years_to_80pct": round(years_to_threshold, 1) if years_to_threshold is not None else None,
            "event_type": row.get("later_event_type", row.get("event_type", "Metal Loss")),
            "confidence": row.get("confidence_label", "Unknown"),
            "is_triple_tracked": is_triple,
        })

    if not predictions:
        return {"error": "No predictions could be generated"}

    pred_df = pd.DataFrame(predictions)

    # Threshold crossings
    thresholds = {}
    for thresh in [50, 60, 70, 80]:
        currently_below = pred_df["current_depth_2022"] < thresh
        predicted_above = pred_df["predicted_depth"] >= thresh
        newly_crossing = currently_below & predicted_above
        thresholds[f"crossing_{thresh}pct"] = int(newly_crossing.sum())

    # Risk distribution
    risk_dist = pred_df["predicted_risk"].value_counts().to_dict()

    # Top concerns (sorted by predicted depth)
    top_concerns = pred_df.nlargest(20, "predicted_depth").to_dict(orient="records")

    # Summary stats
    summary = {
        "target_year": target_year,
        "years_forward": years_forward,
        "total_predicted": len(pred_df),
        "mean_predicted_depth": round(pred_df["predicted_depth"].mean(), 1),
        "max_predicted_depth": round(pred_df["predicted_depth"].max(), 1),
        "critical_count": int((pred_df["predicted_risk"] == "Critical").sum()),
        "high_count": int((pred_df["predicted_risk"] == "High").sum()),
        "medium_count": int((pred_df["predicted_risk"] == "Medium").sum()),
        "low_count": int((pred_df["predicted_risk"] == "Low").sum()),
        "needing_repair_by_target": int((pred_df["predicted_depth"] >= WALL_LOSS_REPAIR_THRESHOLD).sum()),
    }

    # Depth distribution for histogram
    depth_bins = {
        "0-20%": int((pred_df["predicted_depth"] < 20).sum()),
        "20-40%": int(((pred_df["predicted_depth"] >= 20) & (pred_df["predicted_depth"] < 40)).sum()),
        "40-60%": int(((pred_df["predicted_depth"] >= 40) & (pred_df["predicted_depth"] < 60)).sum()),
        "60-80%": int(((pred_df["predicted_depth"] >= 60) & (pred_df["predicted_depth"] < 80)).sum()),
        "80-100%": int((pred_df["predicted_depth"] >= 80).sum()),
    }

    return {
        "summary": summary,
        "risk_distribution": risk_dist,
        "threshold_crossings": thresholds,
        "depth_distribution": depth_bins,
        "top_concerns": top_concerns,
        "all_predictions": pred_df.to_dict(orient="records"),
    }


def _safe(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else round(f, 2)
    return v

"""Configuration: column mappings, event normalization, tolerances, and weights."""

# ── Run metadata ──────────────────────────────────────────────────────────────
RUN_YEARS = [2007, 2015, 2022]
YEARS_BETWEEN = {(2007, 2015): 8, (2015, 2022): 7, (2007, 2022): 15}

# ── Column name mappings (raw → canonical) ────────────────────────────────────
COLUMN_MAP = {
    2007: {
        "J. no.": "joint_number",
        "J. len [ft]": "joint_length_ft",
        "t [in]": "wall_thickness_in",
        "to u/s w. [ft]": "dist_to_us_weld_ft",
        "log dist. [ft]": "log_distance_ft",
        "Height [ft]": "elevation_ft",
        "event": "event_type",
        "depth [%]": "depth_pct",
        "ID Reduction [%]": "id_reduction_pct",
        "length [in]": "length_in",
        "width [in]": "width_in",
        "P2 Burst / MOP": "burst_mop_ratio",
        "o'clock": "clock_position",
        "internal": "id_od",
        "comment": "comments",
    },
    2015: {
        "J. no.": "joint_number",
        "J. len [ft]": "joint_length_ft",
        "Wt [in]": "wall_thickness_in",
        "to u/s w. [ft]": "dist_to_us_weld_ft",
        "to d/s w. [ft]": "dist_to_ds_weld_ft",
        "Log Dist. [ft]": "log_distance_ft",
        "Event Description": "event_type",
        "ID/OD": "id_od",
        "Depth [%]": "depth_pct",
        "Depth [in]": "depth_in",
        "OD Reduction [%]": "od_reduction_pct",
        "OD Reduction [in]": "od_reduction_in",
        "Length [in]": "length_in",
        "Width [in]": "width_in",
        "O'clock": "clock_position",
        "Comments": "comments",
        "Anomalies per Joint": "anomalies_per_joint",
        "Tool Velocity [ft/s]": "tool_velocity",
        "Elevation [ft]": "elevation_ft",
        "MOP [PSI]": "mop_psi",
        "SMYS [PSI]": "smys_psi",
        "Pdesign [PSI]": "pdesign_psi",
        "B31G Psafe [PSI]": "b31g_psafe_psi",
        "B31G Pburst [PSI]": "b31g_pburst_psi",
        "Mod B31G Psafe [PSI]": "mod_b31g_psafe_psi",
        "Mod B31G Pburst [PSI]": "mod_b31g_pburst_psi",
        "Effective Area Psafe [PSI]": "eff_area_psafe_psi",
        "Effective Area Pburst [PSI]": "eff_area_pburst_psi",
        "ERF": "erf",
        "RPR": "rpr",
    },
    2022: {
        "Joint Number": "joint_number",
        "Joint Length [ft]": "joint_length_ft",
        "WT [in]": "wall_thickness_in",
        "Distance to U/S GW \n[ft]": "dist_to_us_weld_ft",
        "Distance to D/S GW \n[ft]": "dist_to_ds_weld_ft",
        "ILI Wheel Count \n[ft.]": "log_distance_ft",
        "Event Description": "event_type",
        "ID/OD": "id_od",
        "Metal Loss Depth \n[%]": "depth_pct",
        "Metal Loss Depth \n[in]": "depth_in",
        "Metal Loss Depth + Tolerance\n[%]": "depth_plus_tol_pct",
        "Dimension Classification": "dimension_class",
        "Dent Depth\n [%]": "dent_depth_pct",
        "Dent Depth\n [in]": "dent_depth_in",
        "Length [in]": "length_in",
        "Width [in]": "width_in",
        "O'clock\n[hh:mm]": "clock_position",
        "Comments": "comments",
        "Anomalies per Joint": "anomalies_per_joint",
        "Elevation [ft]": "elevation_ft",
        "Seam Position\n[hh:mm]": "seam_position",
        "Distance To Seam Weld \n[in]": "dist_to_seam_in",
        "Tool": "tool",
        "Evaluation Pressure [PSI]": "eval_pressure_psi",
        "SMYS [PSI]": "smys_psi",
        "Pipe Type": "pipe_type",
        "Pipe Diameter (O.D.) \n[in.]": "pipe_od_in",
        "Pdesign [PSI]": "pdesign_psi",
        "Mod B31G Psafe \n[PSI]": "mod_b31g_psafe_psi",
        "Mod B31G Pburst [PSI]": "mod_b31g_pburst_psi",
        "Effective Area Psafe [PSI]": "eff_area_psafe_psi",
        "Effective Area Pburst [PSI]": "eff_area_pburst_psi",
        "ERF": "erf",
        "RPR": "rpr",
    },
}

# ── Event type normalization ──────────────────────────────────────────────────
EVENT_TYPE_MAP = {
    # Metal loss
    "metal loss": "Metal Loss",
    "Metal Loss": "Metal Loss",
    # Cluster (treated as metal loss for matching)
    "Cluster": "Cluster",
    "cluster": "Cluster",
    # Manufacturing anomaly
    "metal loss-manufacturing anomaly": "Metal Loss Manufacturing",
    "metal loss manufacturing": "Metal Loss Manufacturing",
    "Metal Loss Manufacturing Anomaly": "Metal Loss Manufacturing",
    "Seam Weld Manufacturing Anomaly": "Seam Weld Manufacturing",
    "Seam Weld Anomaly - B": "Seam Weld Anomaly",
    "Seam Weld Dent": "Seam Weld Dent",
    # Girth weld
    "Girth Weld": "Girth Weld",
    "GirthWeld": "Girth Weld",
    "Girth Weld Anomaly": "Girth Weld Anomaly",
    # Dent
    "Dent": "Dent",
    # Structural features
    "Bend": "Bend",
    "Field Bend": "Bend",
    "Tap": "Tap",
    "Valve": "Valve",
    "Tee": "Tee",
    "Stopple Tee": "Tee",
    "Flange": "Flange",
    "Attachment": "Attachment",
    # Reference markers
    "Above Ground Marker": "AGM",
    "AGM": "AGM",
    "Magnet": "Magnet",
    "Support": "Support",
    "Cathodic Protection Point": "CP Point",
    # Area markers
    "Area Start Launcher": "Launcher Start",
    "Area End Launcher": "Launcher End",
    "Area End Launch Trap": "Launcher End",
    "Area Start Receiver": "Receiver Start",
    "Area End Receiver": "Receiver End",
    "Area Start Receive Trap": "Receiver Start",
    "Area Start Installation": "Installation Start",
    "Area End Installation": "Installation End",
    "Area Start Sleeve": "Sleeve Start",
    "Area End Sleeve": "Sleeve End",
    "Start Sleeve": "Sleeve Start",
    "End Sleeve": "Sleeve End",
    "Area Start Composite Wrap": "Composite Wrap Start",
    "Area End Composite Wrap": "Composite Wrap End",
    "Start Composite Wrap": "Composite Wrap Start",
    "End Composite Wrap": "Composite Wrap End",
    "Area Start Casing": "Casing Start",
    "Area End Casing": "Casing End",
    "Start Casing": "Casing Start",
    "End Casing": "Casing End",
    "Area Start Tee": "Tee Start",
    "Area End Tee": "Tee End",
    "Start Repair Marker": "Repair Start",
    "End Repair Marker": "Repair End",
    "Start Recoat": "Recoat Start",
    "End Recoat": "Recoat End",
}

# Feature types that are anomalies (for matching)
ANOMALY_TYPES = {
    "Metal Loss", "Cluster", "Metal Loss Manufacturing",
    "Dent", "Seam Weld Manufacturing", "Seam Weld Anomaly",
    "Seam Weld Dent", "Girth Weld Anomaly",
}

# Feature types that are reference points (for alignment)
REFERENCE_TYPES = {"Girth Weld"}

# Feature type compatibility for matching
COMPATIBLE_TYPES = {
    "Metal Loss": {"Metal Loss", "Cluster"},
    "Cluster": {"Metal Loss", "Cluster"},
    "Metal Loss Manufacturing": {"Metal Loss Manufacturing", "Seam Weld Manufacturing"},
    "Seam Weld Manufacturing": {"Metal Loss Manufacturing", "Seam Weld Manufacturing"},
    "Dent": {"Dent", "Seam Weld Dent"},
    "Seam Weld Dent": {"Dent", "Seam Weld Dent"},
    "Seam Weld Anomaly": {"Seam Weld Anomaly"},
    "Girth Weld Anomaly": {"Girth Weld Anomaly"},
}

# ── ID/OD normalization for 2007 ─────────────────────────────────────────────
ID_OD_MAP_2007 = {
    "YES": "Internal",
    "NO": "External",
    "N/A": "Unknown",
}

# ── Matching tolerances ───────────────────────────────────────────────────────
DISTANCE_TOLERANCE_FT = 3.0
CLOCK_TOLERANCE_HOURS = 1.0
DEPTH_TOLERANCE_PCT = 15.0
LENGTH_TOLERANCE_IN = 3.0
WIDTH_TOLERANCE_IN = 3.0

# ── Similarity weights ───────────────────────────────────────────────────────
WEIGHT_DISTANCE = 0.35
WEIGHT_CLOCK = 0.25
WEIGHT_DEPTH = 0.20
WEIGHT_DIMENSIONS = 0.10
WEIGHT_TYPE = 0.10

# ── Confidence thresholds ────────────────────────────────────────────────────
HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.60
LOW_CONFIDENCE = 0.40

# ── Growth thresholds ────────────────────────────────────────────────────────
MAX_PLAUSIBLE_GROWTH_RATE = 5.0  # %/yr
WALL_LOSS_REPAIR_THRESHOLD = 80.0  # % depth triggers repair consideration

# ── Canonical columns expected after normalization ───────────────────────────
CANONICAL_COLS = [
    "joint_number", "joint_length_ft", "wall_thickness_in",
    "dist_to_us_weld_ft", "log_distance_ft", "event_type",
    "depth_pct", "length_in", "width_in", "clock_position",
    "id_od", "comments", "elevation_ft", "run_year",
    "is_anomaly", "is_girth_weld",
]

"""
data_preparation.py
-------------------
Loaders for the two predictive maintenance datasets:
  - CMAPSS (NASA turbofan engine run-to-failure)
  - UCI Naval Propulsion Plants (gas turbine degradation)

Both loaders return DataFrames with min-max normalized sensor readings and a computed / attached RUL column.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# CMAPSS constants
# ---------------------------------------------------------------------------

_CMAPSS_RAW_COLS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors that are constant (zero variance) across the full multi-subset dataset.
# Dropping them avoids numerical issues and removes uninformative features.
# s1, s5, s10, s16, s18, s19 are universally constant in CMAPSS literature.
_CONST_SENSORS = {"s1", "s5", "s10", "s16", "s18", "s19"}

CMAPSS_SENSOR_COLS = [
    f"s{i}" for i in range(1, 22) if f"s{i}" not in _CONST_SENSORS
]  # s2–s21 minus the six dropped = 15 sensors

# All model-ready feature columns (operational settings + informative sensors)
CMAPSS_FEATURE_COLS = [
    "op_setting_1", "op_setting_2", "op_setting_3",
] + CMAPSS_SENSOR_COLS


# ---------------------------------------------------------------------------
# Naval constants
# ---------------------------------------------------------------------------

NAVAL_FEATURE_COLS = [
    "lever_position",
    "ship_speed",
    "gt_shaft_torque",
    "gt_rate_of_revolutions",
    "gas_generator_rate_of_revolutions",
    "starboard_propeller_torque",
    "port_propeller_torque",
    "hp_turbine_exit_temperature",
    "gt_compressor_inlet_air_temperature",
    "gt_compressor_outlet_air_temperature",
    "hp_turbine_exit_pressure",
    "gt_compressor_inlet_air_pressure",
    "gt_compressor_outlet_air_pressure",
    "gt_exhaust_gas_pressure",
    "turbine_injection_control",
    "fuel_flow",
]

NAVAL_TARGET_COLS = ["kMc", "kMt"]


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_cmapss(subset: str = "FD001",
                raw_dir: str = "Data/CMaps",
                rul_clip: int = 125) -> tuple:
    """Load a NASA CMAPSS subset and return ``(train_df, test_df)``.

    Processing steps
    ----------------
    1. Parse raw space-delimited files.
    2. Drop the six constant sensors (s1, s5, s10, s16, s18, s19).
    3. Min-max normalize remaining sensor columns using train statistics.
    4. Compute piece-wise linear RUL for the train set (clipped at *rul_clip*).
    5. Attach ground-truth RUL to the **last observed cycle** of each test
       engine from the companion ``RUL_<subset>.txt`` file.

    Parameters
    ----------
    subset : str
        One of ``"FD001"``, ``"FD002"``, ``"FD003"``, ``"FD004"``.
    raw_dir : str or Path
        Directory containing ``train_<subset>.txt``, ``test_<subset>.txt``,
        and ``RUL_<subset>.txt``.
    rul_clip : int
        Maximum RUL value (piece-wise linear ceiling).

    Returns
    -------
    train_df : pd.DataFrame
        Shape ``(n_train_cycles, 21)`` with columns
        unit_id, cycle, op_setting_1/2/3, 15 sensor cols, rul.
    test_df : pd.DataFrame
        Same columns; ``rul`` is non-NaN only at each engine's last cycle.
    """
    raw_dir = Path(raw_dir)

    # 1. Parse files
    train_raw = pd.read_csv(
        raw_dir / f"train_{subset}.txt",
        sep=r"\s+", header=None, names=_CMAPSS_RAW_COLS, engine="python",
    )
    test_raw = pd.read_csv(
        raw_dir / f"test_{subset}.txt",
        sep=r"\s+", header=None, names=_CMAPSS_RAW_COLS, engine="python",
    )
    rul_file = pd.read_csv(
        raw_dir / f"RUL_{subset}.txt",
        header=None, names=["rul"], engine="python",
    )

    # 2. Drop constant sensors
    drop_cols = [c for c in _CONST_SENSORS if c in train_raw.columns]
    train_raw.drop(columns=drop_cols, inplace=True)
    test_raw.drop(columns=drop_cols, inplace=True)

    # 3. Min-max normalize sensor readings using train statistics only
    scaler = MinMaxScaler()
    sensor_cols_present = [c for c in CMAPSS_SENSOR_COLS if c in train_raw.columns]
    train_raw[sensor_cols_present] = scaler.fit_transform(
        train_raw[sensor_cols_present]
    )
    test_raw[sensor_cols_present] = scaler.transform(
        test_raw[sensor_cols_present]
    )

    # 4. Compute RUL for training set (vectorized)
    max_cycles = train_raw.groupby("unit_id")["cycle"].max()
    train_raw["rul"] = (
        max_cycles[train_raw["unit_id"]].values - train_raw["cycle"]
    ).clip(upper=rul_clip)

    # 5. Attach ground-truth RUL to last cycle of each test engine
    rul_file["unit_id"] = range(1, len(rul_file) + 1)
    last_obs = (
        test_raw.groupby("unit_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "last_cycle"})
        .merge(rul_file, on="unit_id")
    )
    test_raw = test_raw.merge(last_obs, on="unit_id", how="left")
    # Expose RUL only at the final observed cycle; all earlier cycles are NaN
    mask = test_raw["cycle"] == test_raw["last_cycle"]
    test_raw["rul"] = np.where(mask, test_raw["rul"], np.nan)
    test_raw.drop(columns=["last_cycle"], inplace=True)

    return train_raw, test_raw


def load_naval(raw_dir: str = "Data/UCI CBM Dataset") -> tuple:
    """Load the UCI Naval Propulsion Plants dataset.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing ``data.txt``.

    Returns
    -------
    X : pd.DataFrame
        Shape ``(11934, 16)`` — operational and sensor features.
    y : pd.DataFrame
        Shape ``(11934, 2)`` — degradation targets ``kMc`` and ``kMt``.
    """
    raw_dir = Path(raw_dir)
    all_cols = NAVAL_FEATURE_COLS + NAVAL_TARGET_COLS

    df = pd.read_csv(
        raw_dir / "data.txt",
        sep=r"\s+", header=None, names=all_cols, engine="python",
    )

    return df[NAVAL_FEATURE_COLS].copy(), df[NAVAL_TARGET_COLS].copy()

"""
MODULE 08 - CAPITAL ALLOCATION

Purpose
-------
Convert the decision-rule output from Module 07 into a systematic
capital-allocation framework.

Input
-----
Data/Processed/nifty50_decision_rules.csv

Output
------
Data/Processed/nifty50_capital_allocation.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"

INPUT_FILE = PROCESSED_DIR / "nifty50_decision_rules.csv"
OUTPUT_FILE = PROCESSED_DIR / "nifty50_capital_allocation.csv"


# ============================================================
# DISPLAY
# ============================================================

print("=" * 60)
print("MODULE 08 - CAPITAL ALLOCATION")
print("=" * 60)

print()
print(f"Project Root : {PROJECT_ROOT}")
print(f"Input File   : {INPUT_FILE}")
print(f"Output File  : {OUTPUT_FILE}")
print()


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Module 07 output was not found:\n{INPUT_FILE}"
    )

print("Loading decision rules data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded    : {len(df)}")
print(f"Columns loaded : {len(df.columns)}")
print()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Close",
    "Market_Trend",
    "Detected_Volatility_Regime",
    "Market_Regime",
    "Decision_Confidence",
    "Decision_Confidence_Score",
    "Risk_State",
    "Positioning_Environment",
    "Drawdown_Risk",
    "Trend_Score",
    "Volatility_Composite",
    "Volatility_Stress_Score",
    "Rule_Score",
    "Rule_Strength",
    "System_Action",
    "Final_Rule_Label",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print()
    print("ERROR: Required columns missing from Module 07 output:")

    for column in missing_columns:
        print(f" - {column}")

    raise ValueError(
        "Module 08 cannot continue because Module 07 "
        "does not contain the required columns."
    )


# ============================================================
# PREPARE INPUTS
# ============================================================

print("Preparing capital-allocation inputs...")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_columns = [
    "Close",
    "Decision_Confidence_Score",
    "Trend_Score",
    "Volatility_Composite",
    "Volatility_Stress_Score",
    "Rule_Score",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# 1. BASE ALLOCATION SCORE
# ============================================================

print("Building base allocation score...")

df["Base_Allocation_Score"] = (
    0.40 * df["Trend_Score"].clip(0, 100)
    + 0.30 * df["Rule_Score"].clip(0, 100)
    + 0.30 * df["Decision_Confidence_Score"].clip(0, 100)
)


# ============================================================
# 2. VOLATILITY ADJUSTMENT
# ============================================================

print("Applying volatility adjustment...")

volatility_adjustment = np.select(
    [
        df["Detected_Volatility_Regime"].eq("Low"),
        df["Detected_Volatility_Regime"].eq("Moderate"),
        df["Detected_Volatility_Regime"].eq("Elevated"),
        df["Detected_Volatility_Regime"].eq("High"),
        df["Detected_Volatility_Regime"].eq("Crisis"),
    ],
    [
        10.0,
        0.0,
        -10.0,
        -20.0,
        -35.0,
    ],
    default=0.0,
)

df["Volatility_Allocation_Adjustment"] = volatility_adjustment


# ============================================================
# 3. RISK ADJUSTMENT
# ============================================================

print("Applying risk-state adjustment...")

risk_adjustment = np.select(
    [
        df["Risk_State"].eq("Normal"),
        df["Risk_State"].eq("Caution"),
        df["Risk_State"].eq("Elevated"),
        df["Risk_State"].eq("High Risk"),
        df["Risk_State"].eq("Crisis"),
    ],
    [
        5.0,
        0.0,
        -10.0,
        -20.0,
        -35.0,
    ],
    default=0.0,
)

df["Risk_Allocation_Adjustment"] = risk_adjustment


# ============================================================
# 4. DRAWDOWN ADJUSTMENT
# ============================================================

print("Applying drawdown adjustment...")

drawdown_adjustment = np.select(
    [
        df["Drawdown_Risk"].eq("Low"),
        df["Drawdown_Risk"].eq("Moderate"),
        df["Drawdown_Risk"].eq("High"),
        df["Drawdown_Risk"].eq("Severe"),
    ],
    [
        5.0,
        0.0,
        -10.0,
        -20.0,
    ],
    default=0.0,
)

df["Drawdown_Allocation_Adjustment"] = drawdown_adjustment


# ============================================================
# 5. SYSTEM ACTION ADJUSTMENT
# ============================================================

print("Applying system-action adjustment...")

system_action_adjustment = np.select(
    [
        df["System_Action"].eq("Increase Exposure"),
        df["System_Action"].eq("Maintain Bullish Exposure"),
        df["System_Action"].eq("Maintain Balanced Exposure"),
        df["System_Action"].eq("Review"),
        df["System_Action"].eq("Reduce Exposure"),
        df["System_Action"].eq("Defensive Positioning"),
    ],
    [
        10.0,
        5.0,
        0.0,
        -5.0,
        -15.0,
        -25.0,
    ],
    default=0.0,
)

df["System_Action_Adjustment"] = system_action_adjustment


# ============================================================
# 6. FINAL ALLOCATION SCORE
# ============================================================

print("Building final allocation score...")

df["Capital_Allocation_Score"] = (
    df["Base_Allocation_Score"]
    + df["Volatility_Allocation_Adjustment"]
    + df["Risk_Allocation_Adjustment"]
    + df["Drawdown_Allocation_Adjustment"]
    + df["System_Action_Adjustment"]
)

df["Capital_Allocation_Score"] = (
    df["Capital_Allocation_Score"]
    .clip(0, 100)
)


# ============================================================
# 7. TARGET EQUITY EXPOSURE
# ============================================================

print("Determining target equity exposure...")

df["Target_Equity_Exposure_pct"] = np.select(
    [
        df["Capital_Allocation_Score"] >= 85,
        df["Capital_Allocation_Score"] >= 75,
        df["Capital_Allocation_Score"] >= 65,
        df["Capital_Allocation_Score"] >= 55,
        df["Capital_Allocation_Score"] >= 45,
        df["Capital_Allocation_Score"] >= 30,
    ],
    [
        100.0,
        90.0,
        80.0,
        70.0,
        55.0,
        35.0,
    ],
    default=15.0,
)


# ============================================================
# 8. CASH / DEFENSIVE ALLOCATION
# ============================================================

print("Determining defensive allocation...")

df["Target_Cash_Exposure_pct"] = (
    100.0 - df["Target_Equity_Exposure_pct"]
)


# ============================================================
# 9. EQUITY ALLOCATION BAND
# ============================================================

print("Classifying equity allocation band...")

df["Equity_Allocation_Band"] = np.select(
    [
        df["Target_Equity_Exposure_pct"] >= 90,
        df["Target_Equity_Exposure_pct"] >= 75,
        df["Target_Equity_Exposure_pct"] >= 55,
        df["Target_Equity_Exposure_pct"] >= 35,
    ],
    [
        "Very High",
        "High",
        "Moderate",
        "Low",
    ],
    default="Very Low",
)


# ============================================================
# 10. CAPITAL POSTURE
# ============================================================

print("Determining capital posture...")

df["Capital_Posture"] = np.select(
    [
        df["Target_Equity_Exposure_pct"] >= 90,
        df["Target_Equity_Exposure_pct"] >= 75,
        df["Target_Equity_Exposure_pct"] >= 55,
        df["Target_Equity_Exposure_pct"] >= 35,
    ],
    [
        "Aggressive",
        "Growth",
        "Balanced",
        "Defensive",
    ],
    default="Capital Preservation",
)


# ============================================================
# 11. NEW CAPITAL GUIDANCE
# ============================================================

print("Determining new-capital deployment guidance...")

df["New_Capital_Deployment"] = np.select(
    [
        df["Capital_Allocation_Score"] >= 85,
        df["Capital_Allocation_Score"] >= 75,
        df["Capital_Allocation_Score"] >= 65,
        df["Capital_Allocation_Score"] >= 55,
        df["Capital_Allocation_Score"] >= 45,
        df["Capital_Allocation_Score"] >= 30,
    ],
    [
        "Deploy Aggressively",
        "Deploy Normally",
        "Deploy Gradually",
        "Deploy Conservatively",
        "Hold Partial Cash",
        "Preserve Capital",
    ],
    default="Maximum Capital Preservation",
)


# ============================================================
# 12. RISK BUDGET
# ============================================================

print("Determining risk budget...")

df["Risk_Budget_pct"] = np.select(
    [
        df["Capital_Allocation_Score"] >= 85,
        df["Capital_Allocation_Score"] >= 75,
        df["Capital_Allocation_Score"] >= 65,
        df["Capital_Allocation_Score"] >= 55,
        df["Capital_Allocation_Score"] >= 45,
        df["Capital_Allocation_Score"] >= 30,
    ],
    [
        100.0,
        90.0,
        80.0,
        70.0,
        55.0,
        35.0,
    ],
    default=20.0,
)


# ============================================================
# 13. POSITION SIZING GUIDANCE
# ============================================================

print("Determining position-sizing guidance...")

df["Position_Size_Guidance"] = np.select(
    [
        df["Capital_Allocation_Score"] >= 85,
        df["Capital_Allocation_Score"] >= 75,
        df["Capital_Allocation_Score"] >= 65,
        df["Capital_Allocation_Score"] >= 55,
        df["Capital_Allocation_Score"] >= 45,
    ],
    [
        "Full Position",
        "Large Position",
        "Standard Position",
        "Reduced Position",
        "Small Position",
    ],
    default="Minimal Position",
)


# ============================================================
# 14. REBALANCING GUIDANCE
# ============================================================

print("Determining rebalancing guidance...")

df["Rebalancing_Guidance"] = np.select(
    [
        df["Capital_Allocation_Score"] >= 85,
        df["Capital_Allocation_Score"] >= 75,
        df["Capital_Allocation_Score"] >= 65,
        df["Capital_Allocation_Score"] >= 55,
        df["Capital_Allocation_Score"] >= 45,
        df["Capital_Allocation_Score"] >= 30,
    ],
    [
        "Increase Risk Allocation",
        "Maintain High Equity Weight",
        "Maintain Growth Allocation",
        "Maintain Balanced Allocation",
        "Reduce Risk Gradually",
        "Increase Defensive Allocation",
    ],
    default="Move Toward Capital Preservation",
)


# ============================================================
# 15. ALLOCATION SIGNAL
# ============================================================

print("Building allocation signal...")

df["Allocation_Signal"] = np.select(
    [
        df["Capital_Allocation_Score"] >= 85,
        df["Capital_Allocation_Score"] >= 75,
        df["Capital_Allocation_Score"] >= 65,
        df["Capital_Allocation_Score"] >= 55,
        df["Capital_Allocation_Score"] >= 45,
        df["Capital_Allocation_Score"] >= 30,
    ],
    [
        "Strong Increase",
        "Increase",
        "Moderate Increase",
        "Maintain",
        "Moderate Reduction",
        "Reduce",
    ],
    default="Strong Reduction",
)


# ============================================================
# 16. ALLOCATION CONFIDENCE
# ============================================================

print("Determining allocation confidence...")

df["Allocation_Confidence"] = np.select(
    [
        df["Decision_Confidence_Score"] >= 75,
        df["Decision_Confidence_Score"] >= 60,
        df["Decision_Confidence_Score"] >= 45,
    ],
    [
        "High",
        "Medium",
        "Low",
    ],
    default="Very Low",
)


# ============================================================
# 17. CAPITAL ALLOCATION RATIONALE
# ============================================================

print("Generating capital-allocation rationale...")


def build_rationale(row):
    score = row["Capital_Allocation_Score"]
    trend = row["Market_Trend"]
    volatility = row["Detected_Volatility_Regime"]
    risk = row["Risk_State"]
    action = row["System_Action"]

    if score >= 85:
        allocation = "strongly increase equity exposure"
    elif score >= 75:
        allocation = "maintain a high equity allocation"
    elif score >= 65:
        allocation = "maintain a growth-oriented allocation"
    elif score >= 55:
        allocation = "maintain a balanced allocation"
    elif score >= 45:
        allocation = "reduce risk gradually"
    elif score >= 30:
        allocation = "increase defensive allocation"
    else:
        allocation = "prioritize capital preservation"

    return (
        f"Market trend is {trend}; volatility regime is {volatility}; "
        f"risk state is {risk}; system action is {action}. "
        f"Capital allocation framework therefore recommends "
        f"{allocation}."
    )


df["Capital_Allocation_Rationale"] = df.apply(
    build_rationale,
    axis=1,
)


# ============================================================
# 18. CLEAN NUMERIC VALUES
# ============================================================

print()
print("Cleaning infinite allocation values...")

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True,
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL CAPITAL ALLOCATION VALIDATION")
print("=" * 60)

print()

print(f"Rows              : {len(df)}")
print(f"Columns           : {len(df.columns)}")

print(
    f"Start Date        : "
    f"{df['Date'].min().strftime('%Y-%m-%d')}"
)

print(
    f"End Date          : "
    f"{df['Date'].max().strftime('%Y-%m-%d')}"
)


allocation_columns = [
    "Base_Allocation_Score",
    "Volatility_Allocation_Adjustment",
    "Risk_Allocation_Adjustment",
    "Drawdown_Allocation_Adjustment",
    "System_Action_Adjustment",
    "Capital_Allocation_Score",
    "Target_Equity_Exposure_pct",
    "Target_Cash_Exposure_pct",
    "Equity_Allocation_Band",
    "Capital_Posture",
    "New_Capital_Deployment",
    "Risk_Budget_pct",
    "Position_Size_Guidance",
    "Rebalancing_Guidance",
    "Allocation_Signal",
    "Allocation_Confidence",
    "Capital_Allocation_Rationale",
]

print()
print(f"Allocation Features : {len(allocation_columns)}")
print()

print("Allocation columns:")

for column in allocation_columns:
    print(f" - {column}")


# ============================================================
# LATEST SNAPSHOT
# ============================================================

latest = df.iloc[-1]

print()
print("=" * 60)
print("LATEST CAPITAL ALLOCATION SNAPSHOT")
print("=" * 60)

print()

print(
    f"Date                       : "
    f"{latest['Date'].strftime('%Y-%m-%d')}"
)

print(
    f"Close                      : "
    f"{latest['Close']:.2f}"
)

print(
    f"Market Trend               : "
    f"{latest['Market_Trend']}"
)

print(
    f"Volatility Regime          : "
    f"{latest['Detected_Volatility_Regime']}"
)

print(
    f"Market Regime              : "
    f"{latest['Market_Regime']}"
)

print(
    f"Capital Allocation Score   : "
    f"{latest['Capital_Allocation_Score']:.2f}"
)

print(
    f"Target Equity Exposure     : "
    f"{latest['Target_Equity_Exposure_pct']:.2f}%"
)

print(
    f"Target Cash Exposure       : "
    f"{latest['Target_Cash_Exposure_pct']:.2f}%"
)

print(
    f"Equity Allocation Band     : "
    f"{latest['Equity_Allocation_Band']}"
)

print(
    f"Capital Posture             : "
    f"{latest['Capital_Posture']}"
)

print(
    f"New Capital Deployment     : "
    f"{latest['New_Capital_Deployment']}"
)

print(
    f"Risk Budget                : "
    f"{latest['Risk_Budget_pct']:.2f}%"
)

print(
    f"Position Size Guidance     : "
    f"{latest['Position_Size_Guidance']}"
)

print(
    f"Allocation Signal          : "
    f"{latest['Allocation_Signal']}"
)

print(
    f"Allocation Confidence      : "
    f"{latest['Allocation_Confidence']}"
)


# ============================================================
# DISTRIBUTIONS
# ============================================================

print()
print("Capital posture distribution:")

print(
    df["Capital_Posture"]
    .value_counts()
)


print()
print("Allocation signal distribution:")

print(
    df["Allocation_Signal"]
    .value_counts()
)


# ============================================================
# SAVE
# ============================================================

print()
print("Saving capital allocation dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 60)
print("MODULE 08 COMPLETED SUCCESSFULLY")
print("=" * 60)

print()

print(f"Output rows    : {len(df)}")
print(f"Output columns : {len(df.columns)}")

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("=" * 60)
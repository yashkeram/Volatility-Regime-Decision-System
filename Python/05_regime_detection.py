import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

PROCESSED_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Processed"
)

INPUT_FILE = os.path.join(
    PROCESSED_FOLDER,
    "nifty50_volatility_model.csv"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_FOLDER,
    "nifty50_regime_detection.csv"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("MODULE 05 - REGIME DETECTION")
print("=" * 60)
print()

print(f"Project Root : {PROJECT_ROOT}")
print(f"Input File   : {INPUT_FILE}")
print(f"Output File  : {OUTPUT_FILE}")
print()


# ============================================================
# VALIDATE INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Volatility model file not found:\n{INPUT_FILE}"
    )


# ============================================================
# LOAD VOLATILITY MODEL
# ============================================================

print("Loading volatility model...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded    : {len(df):,}")
print(f"Columns loaded : {len(df.columns)}")
print()


# ============================================================
# DATE PREPARATION
# ============================================================

print("Preparing dates...")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Close",
    "Volatility_20D",
    "Volatility_60D",
    "Volatility_120D",
    "Volatility_252D",
    "Volatility_20D_Percentile",
    "Volatility_60D_Percentile",
    "Volatility_120D_Percentile",
    "Volatility_20D_ZScore",
    "Volatility_Acceleration",
    "Volatility_Momentum_20D",
    "Volatility_of_Volatility_20D",
    "Return_Dispersion_20D",
    "Return_Dispersion_60D",
    "Upside_Volatility_20D",
    "Downside_Volatility_20D",
    "Downside_Volatility_Ratio",
    "Range_Volatility_20D",
    "Range_Volatility_60D",
    "Volatility_Composite",
    "Volatility_Stress_Score",
    "Volatility_Direction",
    "Volatility_Environment",
    "Volatility_Pressure",
    "Return_20D",
    "Return_60D",
    "MA_20D",
    "MA_50D",
    "MA_200D",
    "Price_vs_MA20_pct",
    "Price_vs_MA50_pct",
    "Price_vs_MA200_pct",
    "Drawdown_20D_pct",
    "Drawdown_60D_pct",
    "Drawdown_252D_pct"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing from "
        "nifty50_volatility_model.csv:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print("Converting regime inputs to numeric...")

numeric_columns = [
    column
    for column in required_columns
    if column != "Volatility_Direction"
    and column != "Volatility_Environment"
    and column != "Volatility_Pressure"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# MARKET TREND SCORE
# ============================================================

print("Building market trend score...")

df["Trend_Score"] = 0.0

df["Trend_Score"] += np.where(
    df["Price_vs_MA20_pct"] > 0,
    1,
    -1
)

df["Trend_Score"] += np.where(
    df["Price_vs_MA50_pct"] > 0,
    1,
    -1
)

df["Trend_Score"] += np.where(
    df["Price_vs_MA200_pct"] > 0,
    1,
    -1
)

df["Trend_Score"] += np.where(
    df["Return_20D"] > 0,
    1,
    -1
)

df["Trend_Score"] += np.where(
    df["Return_60D"] > 0,
    1,
    -1
)


# ============================================================
# MARKET TREND CLASSIFICATION
# ============================================================

print("Classifying market trend...")

def classify_trend(score):

    if pd.isna(score):
        return "Unknown"

    if score >= 4:
        return "Strong Bullish"

    if score >= 2:
        return "Bullish"

    if score <= -4:
        return "Strong Bearish"

    if score <= -2:
        return "Bearish"

    return "Neutral"


df["Market_Trend"] = df["Trend_Score"].apply(
    classify_trend
)


# ============================================================
# VOLATILITY REGIME SCORE
# ============================================================

print("Building volatility regime score...")

df["Volatility_Regime_Score"] = 0.0

# Stress contribution
df["Volatility_Regime_Score"] += np.select(
    [
        df["Volatility_Stress_Score"] >= 75,
        df["Volatility_Stress_Score"] >= 50,
        df["Volatility_Stress_Score"] >= 25
    ],
    [
        3,
        2,
        1
    ],
    default=0
)

# Volatility percentile contribution
df["Volatility_Regime_Score"] += np.select(
    [
        df["Volatility_20D_Percentile"] >= 80,
        df["Volatility_20D_Percentile"] >= 60,
        df["Volatility_20D_Percentile"] >= 40
    ],
    [
        3,
        2,
        1
    ],
    default=0
)

# Z-score contribution
df["Volatility_Regime_Score"] += np.select(
    [
        df["Volatility_20D_ZScore"] >= 2,
        df["Volatility_20D_ZScore"] >= 1
    ],
    [
        2,
        1
    ],
    default=0
)

# Downside-volatility contribution
df["Volatility_Regime_Score"] += np.select(
    [
        df["Downside_Volatility_Ratio"] >= 1.5,
        df["Downside_Volatility_Ratio"] >= 1.2
    ],
    [
        2,
        1
    ],
    default=0
)

# Volatility acceleration contribution
df["Volatility_Regime_Score"] += np.select(
    [
        df["Volatility_Acceleration"] >= 0.30,
        df["Volatility_Acceleration"] >= 0.10
    ],
    [
        2,
        1
    ],
    default=0
)


# ============================================================
# VOLATILITY REGIME CLASSIFICATION
# ============================================================

print("Classifying volatility regime...")

def classify_volatility_regime(score):

    if pd.isna(score):
        return "Unknown"

    if score >= 9:
        return "Extreme"

    if score >= 6:
        return "High"

    if score >= 3:
        return "Elevated"

    return "Low"


df["Detected_Volatility_Regime"] = (
    df["Volatility_Regime_Score"]
    .apply(classify_volatility_regime)
)


# ============================================================
# MARKET REGIME
# ============================================================

print("Combining volatility and trend regimes...")

def classify_market_regime(row):

    volatility = row["Detected_Volatility_Regime"]
    trend = row["Market_Trend"]

    if volatility == "Unknown" or trend == "Unknown":
        return "Unknown"

    if volatility == "Extreme":
        if "Bearish" in trend:
            return "Crisis Bear"
        if "Bullish" in trend:
            return "High Volatility Bull"
        return "Crisis / Unstable"

    if volatility == "High":
        if "Bearish" in trend:
            return "High Volatility Bear"
        if "Bullish" in trend:
            return "High Volatility Bull"
        return "High Volatility Neutral"

    if volatility == "Elevated":
        if "Bearish" in trend:
            return "Elevated Risk Bear"
        if "Bullish" in trend:
            return "Elevated Risk Bull"
        return "Elevated Volatility Neutral"

    # Low volatility
    if trend == "Strong Bullish":
        return "Low Volatility Bull"

    if trend == "Bullish":
        return "Low Volatility Bull"

    if trend == "Strong Bearish":
        return "Low Volatility Bear"

    if trend == "Bearish":
        return "Low Volatility Bear"

    return "Low Volatility Neutral"


df["Market_Regime"] = df.apply(
    classify_market_regime,
    axis=1
)


# ============================================================
# REGIME CONFIDENCE
# ============================================================

print("Calculating regime confidence...")

df["Regime_Confidence_Score"] = 0.0

# Trend agreement
df["Regime_Confidence_Score"] += np.select(
    [
        df["Trend_Score"].abs() >= 4,
        df["Trend_Score"].abs() >= 2
    ],
    [
        30,
        20
    ],
    default=10
)

# Volatility signal strength
df["Regime_Confidence_Score"] += np.select(
    [
        df["Volatility_Stress_Score"] >= 75,
        df["Volatility_Stress_Score"] >= 50,
        df["Volatility_Stress_Score"] >= 25
    ],
    [
        30,
        20,
        10
    ],
    default=5
)

# Volatility percentile confirmation
df["Regime_Confidence_Score"] += np.select(
    [
        df["Volatility_20D_Percentile"] >= 80,
        df["Volatility_20D_Percentile"] >= 60,
        df["Volatility_20D_Percentile"] <= 20
    ],
    [
        20,
        15,
        15
    ],
    default=5
)

# Direction confirmation
df["Regime_Confidence_Score"] += np.where(
    (
        (
            df["Volatility_Direction"].eq("Rising")
            & df["Volatility_Acceleration"].gt(0)
        )
        |
        (
            df["Volatility_Direction"].eq("Falling")
            & df["Volatility_Acceleration"].lt(0)
        )
    ),
    10,
    0
)

df["Regime_Confidence_Score"] = (
    df["Regime_Confidence_Score"]
    .clip(upper=100)
)


# ============================================================
# REGIME CONFIDENCE CLASS
# ============================================================

def classify_confidence(score):

    if pd.isna(score):
        return "Unknown"

    if score >= 75:
        return "High"

    if score >= 50:
        return "Medium"

    return "Low"


df["Regime_Confidence"] = (
    df["Regime_Confidence_Score"]
    .apply(classify_confidence)
)


# ============================================================
# RISK STATE
# ============================================================

print("Determining market risk state...")

def classify_risk_state(row):

    volatility = row["Detected_Volatility_Regime"]
    trend = row["Market_Trend"]

    if volatility == "Extreme":
        return "Critical"

    if volatility == "High":
        return "High Risk"

    if volatility == "Elevated":
        return "Elevated Risk"

    if trend in ["Strong Bearish", "Bearish"]:
        return "Defensive"

    if trend in ["Strong Bullish", "Bullish"]:
        return "Normal"

    return "Neutral"


df["Risk_State"] = df.apply(
    classify_risk_state,
    axis=1
)


# ============================================================
# POSITIONING ENVIRONMENT
# ============================================================

print("Determining positioning environment...")

def classify_positioning(row):

    risk = row["Risk_State"]
    trend = row["Market_Trend"]
    volatility = row["Detected_Volatility_Regime"]

    if risk == "Critical":
        return "Capital Preservation"

    if risk == "High Risk":
        return "Defensive"

    if risk == "Elevated Risk":

        if trend in ["Strong Bullish", "Bullish"]:
            return "Selective Risk"

        return "Defensive"

    if trend in ["Strong Bullish", "Bullish"]:
        return "Risk Seeking"

    if trend in ["Strong Bearish", "Bearish"]:
        return "Capital Preservation"

    if volatility == "Low":
        return "Balanced"

    return "Neutral"


df["Positioning_Environment"] = df.apply(
    classify_positioning,
    axis=1
)


# ============================================================
# REGIME TRANSITION DETECTION
# ============================================================

print("Detecting regime transitions...")

df["Previous_Market_Regime"] = (
    df["Market_Regime"].shift(1)
)

df["Regime_Changed"] = (
    df["Market_Regime"]
    != df["Previous_Market_Regime"]
)

df["Regime_Changed"] = (
    df["Regime_Changed"]
    & df["Previous_Market_Regime"].notna()
)


# ============================================================
# REGIME DURATION
# ============================================================

print("Calculating regime duration...")

regime_group = (
    df["Market_Regime"]
    .ne(df["Market_Regime"].shift())
    .cumsum()
)

df["Regime_Duration_Days"] = (
    df.groupby(regime_group)
    .cumcount()
    + 1
)


# ============================================================
# DRAWDOWN RISK
# ============================================================

print("Classifying drawdown risk...")

def classify_drawdown(drawdown):

    if pd.isna(drawdown):
        return "Unknown"

    if drawdown <= -20:
        return "Severe"

    if drawdown <= -10:
        return "High"

    if drawdown <= -5:
        return "Moderate"

    return "Low"


df["Drawdown_Risk"] = (
    df["Drawdown_252D_pct"]
    .apply(classify_drawdown)
)


# ============================================================
# FINAL REGIME LABEL
# ============================================================

print("Building final regime label...")

df["Final_Regime"] = (
    df["Market_Regime"]
)


# ============================================================
# CLEAN INFINITE VALUES
# ============================================================

print("Cleaning infinite regime values...")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL REGIME DETECTION VALIDATION")
print("=" * 60)
print()

print(
    f"Rows              : {len(df):,}"
)

print(
    f"Columns           : {len(df.columns)}"
)

print(
    f"Start Date        : "
    f"{df['Date'].min().strftime('%Y-%m-%d')}"
)

print(
    f"End Date          : "
    f"{df['Date'].max().strftime('%Y-%m-%d')}"
)

regime_columns = [
    "Trend_Score",
    "Market_Trend",
    "Volatility_Regime_Score",
    "Detected_Volatility_Regime",
    "Market_Regime",
    "Regime_Confidence_Score",
    "Regime_Confidence",
    "Risk_State",
    "Positioning_Environment",
    "Previous_Market_Regime",
    "Regime_Changed",
    "Regime_Duration_Days",
    "Drawdown_Risk",
    "Final_Regime"
]

print(
    f"Regime Features   : {len(regime_columns)}"
)

print()
print("Regime columns:")

for column in regime_columns:
    print(f" - {column}")


# ============================================================
# REGIME DISTRIBUTION
# ============================================================

print()
print("Market regime distribution:")

print(
    df["Market_Regime"]
    .value_counts(dropna=False)
)


# ============================================================
# LATEST REGIME SNAPSHOT
# ============================================================

latest = df.iloc[-1]

print()
print("=" * 60)
print("LATEST MARKET REGIME SNAPSHOT")
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
    f"Regime Confidence          : "
    f"{latest['Regime_Confidence']}"
)

print(
    f"Confidence Score           : "
    f"{latest['Regime_Confidence_Score']:.2f}"
)

print(
    f"Risk State                 : "
    f"{latest['Risk_State']}"
)

print(
    f"Positioning Environment    : "
    f"{latest['Positioning_Environment']}"
)

print(
    f"Regime Duration            : "
    f"{latest['Regime_Duration_Days']} days"
)

print(
    f"Drawdown Risk              : "
    f"{latest['Drawdown_Risk']}"
)

print(
    f"Volatility Environment     : "
    f"{latest['Volatility_Environment']}"
)

print(
    f"Volatility Direction       : "
    f"{latest['Volatility_Direction']}"
)

print(
    f"Volatility Pressure        : "
    f"{latest['Volatility_Pressure']}"
)


# ============================================================
# SAVE OUTPUT
# ============================================================

print()
print("Saving regime detection dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 60)
print("MODULE 05 COMPLETED SUCCESSFULLY")
print("=" * 60)
print()

print(
    f"Output rows    : {len(df):,}"
)

print(
    f"Output columns : {len(df.columns)}"
)

print()
print("Saved to:")
print(OUTPUT_FILE)
print()
print("=" * 60)
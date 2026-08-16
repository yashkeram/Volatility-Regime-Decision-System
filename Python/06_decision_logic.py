import os
import pandas as pd
import numpy as np

# ============================================================
# MODULE 06 - DECISION LOGIC
# ============================================================

print("=" * 60)
print("MODULE 06 - DECISION LOGIC")
print("=" * 60)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Processed"
)

INPUT_FILE = os.path.join(
    PROCESSED_FOLDER,
    "nifty50_regime_detection.csv"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_FOLDER,
    "nifty50_decision_logic.csv"
)

print()
print("Project Root :", PROJECT_ROOT)
print("Input File   :", INPUT_FILE)
print("Output File  :", OUTPUT_FILE)

# ============================================================
# VALIDATE INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Required input file not found:\n{INPUT_FILE}"
    )

# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading regime detection data...")

df = pd.read_csv(INPUT_FILE)

print("Rows loaded    :", len(df))
print("Columns loaded :", len(df))

# ============================================================
# PREPARE DATA
# ============================================================

print()
print("Preparing decision inputs...")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

numeric_columns = [
    "Close",
    "Return_5D",
    "Return_20D",
    "Return_60D",
    "Volatility_20D",
    "Volatility_60D",
    "Volatility_120D",
    "Volatility_252D",
    "Price_vs_MA20_pct",
    "Price_vs_MA50_pct",
    "Price_vs_MA200_pct",
    "Volatility_20D_Percentile",
    "Volatility_20D_ZScore",
    "Volatility_Acceleration",
    "Volatility_Composite",
    "Volatility_Stress_Score",
    "Trend_Score",
    "Volatility_Regime_Score",
    "Regime_Confidence_Score",
    "Regime_Duration_Days",
    "Drawdown_20D_pct",
    "Drawdown_60D_pct",
    "Drawdown_252D_pct",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_value(row, column, default=0.0):
    value = row.get(column, default)

    if pd.isna(value):
        return default

    return float(value)


def classify_signal(score):
    if score >= 70:
        return "Strong Positive"
    elif score >= 55:
        return "Positive"
    elif score >= 45:
        return "Neutral"
    elif score >= 30:
        return "Negative"
    else:
        return "Strong Negative"


# ============================================================
# TREND DECISION SCORE
# ============================================================

print("Building trend decision score...")

def calculate_trend_decision(row):

    score = 50.0

    trend_score = safe_value(
        row,
        "Trend_Score",
        0
    )

    market_trend = str(
        row.get("Market_Trend", "Neutral")
    )

    price_ma20 = safe_value(
        row,
        "Price_vs_MA20_pct",
        0
    )

    price_ma50 = safe_value(
        row,
        "Price_vs_MA50_pct",
        0
    )

    price_ma200 = safe_value(
        row,
        "Price_vs_MA200_pct",
        0
    )

    score += trend_score * 0.25

    if market_trend == "Bullish":
        score += 15

    elif market_trend == "Bearish":
        score -= 15

    else:
        score += 0

    if price_ma20 > 0:
        score += 5
    elif price_ma20 < 0:
        score -= 5

    if price_ma50 > 0:
        score += 5
    elif price_ma50 < 0:
        score -= 5

    if price_ma200 > 0:
        score += 5
    elif price_ma200 < 0:
        score -= 5

    return float(
        np.clip(score, 0, 100)
    )


df["Trend_Decision_Score"] = df.apply(
    calculate_trend_decision,
    axis=1
)

df["Trend_Signal"] = df[
    "Trend_Decision_Score"
].apply(classify_signal)

# ============================================================
# VOLATILITY DECISION SCORE
# ============================================================

print("Building volatility decision score...")

def calculate_volatility_decision(row):

    score = 50.0

    volatility_environment = str(
        row.get(
            "Volatility_Environment",
            "Low"
        )
    )

    volatility_direction = str(
        row.get(
            "Volatility_Direction",
            "Stable"
        )
    )

    volatility_pressure = str(
        row.get(
            "Volatility_Pressure",
            "Stable"
        )
    )

    stress_score = safe_value(
        row,
        "Volatility_Stress_Score",
        50
    )

    percentile = safe_value(
        row,
        "Volatility_20D_Percentile",
        50
    )

    score += (
        50 - stress_score
    ) * 0.35

    score += (
        50 - percentile
    ) * 0.20

    if volatility_environment == "Low":
        score += 15

    elif volatility_environment == "Elevated":
        score += 0

    elif volatility_environment == "High":
        score -= 15

    elif volatility_environment == "Crisis":
        score -= 30

    if volatility_direction == "Falling":
        score += 10

    elif volatility_direction == "Rising":
        score -= 10

    if volatility_pressure == "Decreasing":
        score += 5

    elif volatility_pressure == "Increasing":
        score -= 5

    return float(
        np.clip(score, 0, 100)
    )


df["Volatility_Decision_Score"] = df.apply(
    calculate_volatility_decision,
    axis=1
)

df["Volatility_Signal"] = df[
    "Volatility_Decision_Score"
].apply(classify_signal)

# ============================================================
# RISK DECISION SCORE
# ============================================================

print("Building risk decision score...")

def calculate_risk_score(row):

    score = 50.0

    risk_state = str(
        row.get(
            "Risk_State",
            "Normal"
        )
    )

    drawdown_risk = str(
        row.get(
            "Drawdown_Risk",
            "Unknown"
        )
    )

    positioning = str(
        row.get(
            "Positioning_Environment",
            "Neutral"
        )
    )

    stress_score = safe_value(
        row,
        "Volatility_Stress_Score",
        50
    )

    drawdown_20 = safe_value(
        row,
        "Drawdown_20D_pct",
        0
    )

    drawdown_60 = safe_value(
        row,
        "Drawdown_60D_pct",
        0
    )

    drawdown_252 = safe_value(
        row,
        "Drawdown_252D_pct",
        0
    )

    if risk_state == "Normal":
        score += 15

    elif risk_state == "Caution":
        score -= 5

    elif risk_state == "High Risk":
        score -= 20

    elif risk_state == "Crisis":
        score -= 35

    if drawdown_risk == "Low":
        score += 10

    elif drawdown_risk == "Moderate":
        score -= 5

    elif drawdown_risk == "High":
        score -= 20

    elif drawdown_risk == "Severe":
        score -= 35

    if positioning == "Risk Seeking":
        score += 10

    elif positioning == "Balanced":
        score += 0

    elif positioning == "Defensive":
        score -= 10

    elif positioning == "Capital Preservation":
        score -= 20

    if drawdown_20 < -5:
        score -= 5

    if drawdown_60 < -10:
        score -= 10

    if drawdown_252 < -20:
        score -= 10

    score -= (
        max(stress_score - 50, 0)
        * 0.20
    )

    return float(
        np.clip(score, 0, 100)
    )


df["Risk_Decision_Score"] = df.apply(
    calculate_risk_score,
    axis=1
)

df["Risk_Signal"] = df[
    "Risk_Decision_Score"
].apply(classify_signal)

# ============================================================
# CONFIDENCE ADJUSTMENT
# ============================================================

print("Building decision confidence...")

df["Decision_Confidence_Score"] = (
    df["Regime_Confidence_Score"] * 0.50
    + df["Trend_Decision_Score"] * 0.20
    + df["Volatility_Decision_Score"] * 0.15
    + df["Risk_Decision_Score"] * 0.15
)

df["Decision_Confidence_Score"] = (
    df["Decision_Confidence_Score"]
    .clip(0, 100)
    .round(2)
)

df["Decision_Confidence"] = (
    df["Decision_Confidence_Score"]
    .apply(
        lambda x:
        "High" if x >= 70
        else "Medium" if x >= 50
        else "Low"
    )
)

# ============================================================
# MARKET BIAS SCORE
# ============================================================

print("Building market bias score...")

df["Market_Bias_Score"] = (
    df["Trend_Decision_Score"] * 0.45
    + df["Volatility_Decision_Score"] * 0.30
    + df["Risk_Decision_Score"] * 0.25
)

df["Market_Bias_Score"] = (
    df["Market_Bias_Score"]
    .clip(0, 100)
    .round(2)
)

df["Market_Bias"] = df[
    "Market_Bias_Score"
].apply(
    lambda x:
    "Strong Bullish" if x >= 75
    else "Bullish" if x >= 60
    else "Neutral" if x >= 45
    else "Bearish" if x >= 30
    else "Strong Bearish"
)

# ============================================================
# EXPOSURE GUIDANCE
# ============================================================

print("Determining exposure guidance...")

def exposure_guidance(row):

    market_bias = row["Market_Bias"]
    risk_state = str(
        row.get("Risk_State", "Normal")
    )
    volatility_environment = str(
        row.get(
            "Volatility_Environment",
            "Low"
        )
    )
    confidence = row["Decision_Confidence_Score"]

    if risk_state == "Crisis":
        return "Minimal"

    if volatility_environment == "Crisis":
        return "Minimal"

    if market_bias == "Strong Bearish":
        return "Minimal"

    if market_bias == "Bearish":

        if confidence >= 60:
            return "Reduced"

        return "Defensive"

    if market_bias == "Neutral":
        return "Balanced"

    if market_bias == "Bullish":

        if confidence >= 70:
            return "High"

        return "Moderately High"

    if market_bias == "Strong Bullish":

        if confidence >= 70:
            return "High"

        return "Moderately High"

    return "Balanced"


df["Exposure_Guidance"] = df.apply(
    exposure_guidance,
    axis=1
)

# ============================================================
# PORTFOLIO POSTURE
# ============================================================

print("Determining portfolio posture...")

def portfolio_posture(row):

    market_bias = row["Market_Bias"]
    risk_state = str(
        row.get("Risk_State", "Normal")
    )

    volatility_environment = str(
        row.get(
            "Volatility_Environment",
            "Low"
        )
    )

    positioning = str(
        row.get(
            "Positioning_Environment",
            "Neutral"
        )
    )

    if risk_state == "Crisis":
        return "Capital Preservation"

    if volatility_environment == "Crisis":
        return "Capital Preservation"

    if market_bias in [
        "Strong Bearish",
        "Bearish"
    ]:
        return "Defensive"

    if market_bias in [
        "Strong Bullish",
        "Bullish"
    ]:

        if positioning == "Risk Seeking":
            return "Aggressive"

        return "Growth"

    return "Balanced"


df["Portfolio_Posture"] = df.apply(
    portfolio_posture,
    axis=1
)

# ============================================================
# VOLATILITY RESPONSE
# ============================================================

print("Determining volatility response...")

def volatility_response(row):

    environment = str(
        row.get(
            "Volatility_Environment",
            "Low"
        )
    )

    direction = str(
        row.get(
            "Volatility_Direction",
            "Stable"
        )
    )

    pressure = str(
        row.get(
            "Volatility_Pressure",
            "Stable"
        )
    )

    if environment == "Crisis":
        return "Protect Capital"

    if environment == "High":
        return "Reduce Risk"

    if environment == "Elevated":

        if direction == "Rising":
            return "Reduce Risk"

        return "Monitor Risk"

    if environment == "Low":

        if direction == "Falling":
            return "Risk Can Expand"

        if pressure == "Increasing":
            return "Monitor Risk"

        return "Normal Risk"

    return "Monitor Risk"


df["Volatility_Response"] = df.apply(
    volatility_response,
    axis=1
)

# ============================================================
# TREND RESPONSE
# ============================================================

print("Determining trend response...")

def trend_response(row):

    trend = str(
        row.get(
            "Market_Trend",
            "Neutral"
        )
    )

    score = safe_value(
        row,
        "Trend_Score",
        0
    )

    if trend == "Bullish":

        if score >= 50:
            return "Follow Trend"

        return "Trend Positive"

    if trend == "Bearish":

        if score <= -50:
            return "Avoid Risk"

        return "Trend Negative"

    return "Wait for Confirmation"


df["Trend_Response"] = df.apply(
    trend_response,
    axis=1
)

# ============================================================
# DRAWdown RESPONSE
# ============================================================

print("Determining drawdown response...")

def drawdown_response(row):

    risk = str(
        row.get(
            "Drawdown_Risk",
            "Unknown"
        )
    )

    drawdown_20 = safe_value(
        row,
        "Drawdown_20D_pct",
        0
    )

    if risk == "Severe":
        return "Capital Protection"

    if risk == "High":
        return "Reduce Exposure"

    if risk == "Moderate":

        if drawdown_20 < -8:
            return "Cautious Accumulation"

        return "Monitor Drawdown"

    if risk == "Low":
        return "Normal"

    return "Monitor"


df["Drawdown_Response"] = df.apply(
    drawdown_response,
    axis=1
)

# ============================================================
# FINAL DECISION SCORE
# ============================================================

print("Building final decision score...")

df["Final_Decision_Score"] = (
    df["Market_Bias_Score"] * 0.45
    + df["Decision_Confidence_Score"] * 0.25
    + df["Risk_Decision_Score"] * 0.20
    + df["Volatility_Decision_Score"] * 0.10
)

df["Final_Decision_Score"] = (
    df["Final_Decision_Score"]
    .clip(0, 100)
    .round(2)
)

# ============================================================
# FINAL DECISION
# ============================================================

print("Determining final decision...")

def final_decision(row):

    market_bias = row["Market_Bias"]

    risk_state = str(
        row.get(
            "Risk_State",
            "Normal"
        )
    )

    volatility_environment = str(
        row.get(
            "Volatility_Environment",
            "Low"
        )
    )

    confidence = row[
        "Decision_Confidence_Score"
    ]

    final_score = row[
        "Final_Decision_Score"
    ]

    if risk_state == "Crisis":
        return "Protect Capital"

    if volatility_environment == "Crisis":
        return "Protect Capital"

    if final_score >= 75 and confidence >= 65:
        return "Increase Exposure"

    if final_score >= 60 and confidence >= 50:
        return "Maintain Bullish Exposure"

    if final_score >= 45:
        return "Maintain Balanced Exposure"

    if final_score >= 30:
        return "Reduce Exposure"

    return "Defensive Positioning"


df["Final_Decision"] = df.apply(
    final_decision,
    axis=1
)

# ============================================================
# DECISION REASON
# ============================================================

print("Generating decision rationale...")

def decision_reason(row):

    return (
        f"Trend: {row['Market_Trend']}; "
        f"Volatility: {row['Volatility_Environment']}; "
        f"Risk: {row['Risk_State']}; "
        f"Market bias: {row['Market_Bias']}; "
        f"Confidence: {row['Decision_Confidence']}; "
        f"Positioning: {row['Positioning_Environment']}."
    )


df["Decision_Reason"] = df.apply(
    decision_reason,
    axis=1
)

# ============================================================
# CLEAN INFINITE VALUES
# ============================================================

print()
print("Cleaning infinite decision values...")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# ============================================================
# FINAL COLUMN ORDER
# ============================================================

base_columns = list(
    pd.read_csv(INPUT_FILE, nrows=0).columns
)

decision_columns = [
    "Trend_Decision_Score",
    "Trend_Signal",
    "Volatility_Decision_Score",
    "Volatility_Signal",
    "Risk_Decision_Score",
    "Risk_Signal",
    "Decision_Confidence_Score",
    "Decision_Confidence",
    "Market_Bias_Score",
    "Market_Bias",
    "Exposure_Guidance",
    "Portfolio_Posture",
    "Volatility_Response",
    "Trend_Response",
    "Drawdown_Response",
    "Final_Decision_Score",
    "Final_Decision",
    "Decision_Reason",
]

final_columns = [
    column
    for column in base_columns
    if column in df.columns
]

final_columns += [
    column
    for column in decision_columns
    if column in df.columns
]

df = df[final_columns]

# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL DECISION LOGIC VALIDATION")
print("=" * 60)

print()
print("Rows              :", len(df))
print("Columns           :", len(df.columns))
print(
    "Start Date        :",
    df["Date"].min().date()
)
print(
    "End Date          :",
    df["Date"].max().date()
)

print()
print("Decision Features  :", len(decision_columns))

print()
print("Decision columns:")

for column in decision_columns:

    if column in df.columns:
        print(" -", column)

# ============================================================
# LATEST SNAPSHOT
# ============================================================

latest = df.iloc[-1]

print()
print("=" * 60)
print("LATEST DECISION LOGIC SNAPSHOT")
print("=" * 60)

print()
print(
    "Date                       :",
    latest["Date"].date()
)

print(
    "Close                      :",
    f"{latest['Close']:.2f}"
)

print(
    "Market Trend               :",
    latest["Market_Trend"]
)

print(
    "Volatility Regime          :",
    latest["Detected_Volatility_Regime"]
)

print(
    "Market Regime              :",
    latest["Market_Regime"]
)

print(
    "Trend Decision Score       :",
    f"{latest['Trend_Decision_Score']:.2f}"
)

print(
    "Volatility Decision Score  :",
    f"{latest['Volatility_Decision_Score']:.2f}"
)

print(
    "Risk Decision Score        :",
    f"{latest['Risk_Decision_Score']:.2f}"
)

print(
    "Market Bias                :",
    latest["Market_Bias"]
)

print(
    "Decision Confidence        :",
    f"{latest['Decision_Confidence_Score']:.2f}",
    f"({latest['Decision_Confidence']})"
)

print(
    "Risk State                 :",
    latest["Risk_State"]
)

print(
    "Exposure Guidance          :",
    latest["Exposure_Guidance"]
)

print(
    "Portfolio Posture          :",
    latest["Portfolio_Posture"]
)

print(
    "Volatility Response        :",
    latest["Volatility_Response"]
)

print(
    "Trend Response             :",
    latest["Trend_Response"]
)

print(
    "Drawdown Response          :",
    latest["Drawdown_Response"]
)

print(
    "Final Decision Score       :",
    f"{latest['Final_Decision_Score']:.2f}"
)

print(
    "FINAL DECISION             :",
    latest["Final_Decision"]
)

# ============================================================
# DECISION DISTRIBUTION
# ============================================================

print()
print("Final decision distribution:")

print(
    df["Final_Decision"]
    .value_counts()
)

# ============================================================
# SAVE OUTPUT
# ============================================================

print()
print("Saving decision logic dataset...")

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("MODULE 06 COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print("Output rows    :", len(df))
print("Output columns :", len(df.columns))

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("=" * 60)
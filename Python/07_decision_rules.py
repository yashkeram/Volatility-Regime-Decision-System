# ============================================================
# MODULE 07 - DECISION RULES
# Volatility Regime Decision System - Refurbished
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_FOLDER = PROJECT_ROOT / "Data" / "Processed"

INPUT_FILE = PROCESSED_FOLDER / "nifty50_decision_logic.csv"
OUTPUT_FILE = PROCESSED_FOLDER / "nifty50_decision_rules.csv"


# ============================================================
# DISPLAY SETTINGS
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_numeric(series):
    """Convert a series to numeric and replace infinite values."""
    return pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )


def normalize_text(series):
    """Normalize categorical text for reliable rule matching."""
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("MODULE 07 - DECISION RULES")
print("=" * 60)

print()
print("Project Root :", PROJECT_ROOT)
print("Input File   :", INPUT_FILE)
print("Output File  :", OUTPUT_FILE)


# ============================================================
# INPUT VALIDATION
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n\n"
        "Run Module 06 first."
    )

PROCESSED_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MODULE 06 DATA
# ============================================================

print()
print("Loading decision logic data...")

df = pd.read_csv(INPUT_FILE)

print("Rows loaded    :", len(df))
print("Columns loaded :", len(df.columns))

if df.empty:
    raise ValueError("Module 06 output is empty.")


# ============================================================
# PREPARE DATES
# ============================================================

print()
print("Preparing decision-rule inputs...")

if "Date" not in df.columns:
    raise ValueError(
        "Required column 'Date' is missing from Module 06 output."
    )

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

if df["Date"].isna().all():
    raise ValueError(
        "Date column could not be converted to valid dates."
    )

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# REQUIRED MODULE 06 COLUMNS
# ============================================================

required_columns = [
    "Close",

    # Module 05 regime information
    "Market_Trend",
    "Detected_Volatility_Regime",
    "Market_Regime",
    "Risk_State",
    "Positioning_Environment",

    # Module 06 decision scores
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

    # Module 06 decision guidance
    "Exposure_Guidance",
    "Portfolio_Posture",
    "Volatility_Response",
    "Trend_Response",
    "Drawdown_Response",

    # Module 06 final decision
    "Final_Decision_Score",
    "Final_Decision",
    "Decision_Reason",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nRequired columns missing from Module 06 output:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# NUMERIC INPUTS
# ============================================================

print("Converting rule inputs to numeric...")

numeric_columns = [
    "Close",
    "Trend_Decision_Score",
    "Volatility_Decision_Score",
    "Risk_Decision_Score",
    "Decision_Confidence_Score",
    "Market_Bias_Score",
    "Final_Decision_Score",
]

for column in numeric_columns:
    df[column] = clean_numeric(
        df[column]
    )


# ============================================================
# TEXT INPUTS
# ============================================================

print("Normalizing categorical inputs...")

trend = normalize_text(
    df["Market_Trend"]
)

volatility = normalize_text(
    df["Detected_Volatility_Regime"]
)

market_regime = normalize_text(
    df["Market_Regime"]
)

risk = normalize_text(
    df["Risk_State"]
)

positioning = normalize_text(
    df["Positioning_Environment"]
)

exposure = normalize_text(
    df["Exposure_Guidance"]
)

posture = normalize_text(
    df["Portfolio_Posture"]
)

vol_response = normalize_text(
    df["Volatility_Response"]
)

trend_response = normalize_text(
    df["Trend_Response"]
)

drawdown_response = normalize_text(
    df["Drawdown_Response"]
)

final_decision = normalize_text(
    df["Final_Decision"]
)


# ============================================================
# RULE 01 - PRIMARY MARKET CONDITION
# ============================================================

print("Applying primary market condition rules...")

df["Rule_Market_Condition"] = np.select(
    [
        market_regime.str.contains(
            "crisis",
            na=False
        ),

        market_regime.str.contains(
            "high volatility",
            na=False
        ),

        market_regime.str.contains(
            "elevated",
            na=False
        ),

        market_regime.str.contains(
            "low volatility",
            na=False
        ),
    ],
    [
        "Crisis",
        "High Volatility",
        "Elevated Risk",
        "Low Volatility",
    ],
    default="Unclassified"
)


# ============================================================
# RULE 02 - TREND CONDITION
# ============================================================

print("Applying trend rules...")

df["Rule_Trend_Condition"] = np.select(
    [
        trend.str.contains(
            "bullish",
            na=False
        ),

        trend.str.contains(
            "bearish",
            na=False
        ),

        trend.str.contains(
            "neutral",
            na=False
        ),
    ],
    [
        "Bullish Trend",
        "Bearish Trend",
        "Neutral Trend",
    ],
    default="Unclassified"
)


# ============================================================
# RULE 03 - VOLATILITY CONDITION
# ============================================================

print("Applying volatility rules...")

df["Rule_Volatility_Condition"] = np.select(
    [
        volatility.str.contains(
            "high",
            na=False
        ),

        volatility.str.contains(
            "elevated",
            na=False
        ),

        volatility.str.contains(
            "low",
            na=False
        ),

        volatility.str.contains(
            "crisis",
            na=False
        ),
    ],
    [
        "High Volatility",
        "Elevated Volatility",
        "Low Volatility",
        "Crisis Volatility",
    ],
    default="Unclassified"
)


# ============================================================
# RULE 04 - RISK CONDITION
# ============================================================

print("Applying risk-state rules...")

df["Rule_Risk_Condition"] = np.select(
    [
        risk.str.contains(
            "critical",
            na=False
        ),

        risk.str.contains(
            "high",
            na=False
        ),

        risk.str.contains(
            "elevated",
            na=False
        ),

        risk.str.contains(
            "normal",
            na=False
        ),

        risk.str.contains(
            "low",
            na=False
        ),
    ],
    [
        "Critical Risk",
        "High Risk",
        "Elevated Risk",
        "Normal Risk",
        "Low Risk",
    ],
    default="Unclassified"
)


# ============================================================
# RULE 05 - TREND SCORE BAND
# ============================================================

print("Classifying trend score bands...")

df["Trend_Score_Band"] = np.select(
    [
        df["Trend_Decision_Score"] >= 75,
        df["Trend_Decision_Score"] >= 60,
        df["Trend_Decision_Score"] >= 40,
        df["Trend_Decision_Score"] >= 25,
    ],
    [
        "Strong Bullish",
        "Bullish",
        "Neutral",
        "Bearish",
    ],
    default="Strong Bearish"
)


# ============================================================
# RULE 06 - VOLATILITY SCORE BAND
# ============================================================

print("Classifying volatility score bands...")

df["Volatility_Score_Band"] = np.select(
    [
        df["Volatility_Decision_Score"] >= 80,
        df["Volatility_Decision_Score"] >= 60,
        df["Volatility_Decision_Score"] >= 40,
        df["Volatility_Decision_Score"] >= 20,
    ],
    [
        "Very Favorable",
        "Favorable",
        "Neutral",
        "Unfavorable",
    ],
    default="Highly Unfavorable"
)


# ============================================================
# RULE 07 - RISK SCORE BAND
# ============================================================

print("Classifying risk score bands...")

df["Risk_Score_Band"] = np.select(
    [
        df["Risk_Decision_Score"] >= 80,
        df["Risk_Decision_Score"] >= 60,
        df["Risk_Decision_Score"] >= 40,
        df["Risk_Decision_Score"] >= 20,
    ],
    [
        "Very Low Risk",
        "Low Risk",
        "Moderate Risk",
        "High Risk",
    ],
    default="Very High Risk"
)


# ============================================================
# RULE 08 - CONFIDENCE BAND
# ============================================================

print("Classifying decision confidence...")

df["Rule_Confidence_Band"] = np.select(
    [
        df["Decision_Confidence_Score"] >= 80,
        df["Decision_Confidence_Score"] >= 65,
        df["Decision_Confidence_Score"] >= 50,
        df["Decision_Confidence_Score"] >= 35,
    ],
    [
        "Very High",
        "High",
        "Medium",
        "Low",
    ],
    default="Very Low"
)


# ============================================================
# RULE 09 - MARKET BIAS
# ============================================================

print("Classifying market bias...")

df["Rule_Market_Bias"] = np.select(
    [
        df["Market_Bias_Score"] >= 75,
        df["Market_Bias_Score"] >= 60,
        df["Market_Bias_Score"] >= 40,
        df["Market_Bias_Score"] >= 25,
    ],
    [
        "Strong Bullish",
        "Bullish",
        "Neutral",
        "Bearish",
    ],
    default="Strong Bearish"
)


# ============================================================
# RULE 10 - EXPOSURE CLASSIFICATION
# ============================================================

print("Classifying exposure guidance...")

df["Rule_Exposure_Level"] = np.select(
    [
        exposure.str.contains(
            "very high",
            na=False
        ),

        exposure.str.contains(
            "moderately high",
            na=False
        ),

        exposure.str.contains(
            "high",
            na=False
        ),

        exposure.str.contains(
            "moderate",
            na=False
        ),

        exposure.str.contains(
            "low",
            na=False
        ),

        exposure.str.contains(
            "minimal",
            na=False
        ),
    ],
    [
        "Very High",
        "Moderately High",
        "High",
        "Moderate",
        "Low",
        "Minimal",
    ],
    default=df["Exposure_Guidance"].astype(str)
)


# ============================================================
# RULE 11 - PORTFOLIO POSTURE
# ============================================================

print("Classifying portfolio posture...")

df["Rule_Portfolio_Posture"] = np.select(
    [
        posture.str.contains(
            "aggressive",
            na=False
        ),

        posture.str.contains(
            "moderately aggressive",
            na=False
        ),

        posture.str.contains(
            "balanced",
            na=False
        ),

        posture.str.contains(
            "defensive",
            na=False
        ),

        posture.str.contains(
            "capital preservation",
            na=False
        ),
    ],
    [
        "Aggressive",
        "Moderately Aggressive",
        "Balanced",
        "Defensive",
        "Capital Preservation",
    ],
    default=df["Portfolio_Posture"].astype(str)
)


# ============================================================
# RULE 12 - TREND + VOLATILITY
# ============================================================

print("Building trend-volatility interaction rules...")

df["Trend_Volatility_Combination"] = np.select(
    [
        (
            trend.str.contains("bullish", na=False)
            & volatility.str.contains("low", na=False)
        ),

        (
            trend.str.contains("bullish", na=False)
            & volatility.str.contains("elevated", na=False)
        ),

        (
            trend.str.contains("bullish", na=False)
            & volatility.str.contains("high", na=False)
        ),

        (
            trend.str.contains("bearish", na=False)
            & volatility.str.contains("low", na=False)
        ),

        (
            trend.str.contains("bearish", na=False)
            & volatility.str.contains("elevated", na=False)
        ),

        (
            trend.str.contains("bearish", na=False)
            & volatility.str.contains("high", na=False)
        ),

        (
            trend.str.contains("neutral", na=False)
            & volatility.str.contains("low", na=False)
        ),

        (
            trend.str.contains("neutral", na=False)
            & volatility.str.contains("elevated", na=False)
        ),

        (
            trend.str.contains("neutral", na=False)
            & volatility.str.contains("high", na=False)
        ),
    ],
    [
        "Bullish / Low Volatility",
        "Bullish / Elevated Volatility",
        "Bullish / High Volatility",

        "Bearish / Low Volatility",
        "Bearish / Elevated Volatility",
        "Bearish / High Volatility",

        "Neutral / Low Volatility",
        "Neutral / Elevated Volatility",
        "Neutral / High Volatility",
    ],
    default="Unclassified"
)


# ============================================================
# RULE 13 - RISK + TREND
# ============================================================

print("Building risk-trend interaction rules...")

df["Risk_Trend_Combination"] = np.select(
    [
        (
            trend.str.contains("bullish", na=False)
            & risk.str.contains("normal", na=False)
        ),

        (
            trend.str.contains("bullish", na=False)
            & risk.str.contains("elevated", na=False)
        ),

        (
            trend.str.contains("bullish", na=False)
            & risk.str.contains("high", na=False)
        ),

        (
            trend.str.contains("bullish", na=False)
            & risk.str.contains("critical", na=False)
        ),

        (
            trend.str.contains("bearish", na=False)
            & risk.str.contains("normal", na=False)
        ),

        (
            trend.str.contains("bearish", na=False)
            & risk.str.contains("elevated", na=False)
        ),

        (
            trend.str.contains("bearish", na=False)
            & risk.str.contains("high", na=False)
        ),

        (
            trend.str.contains("bearish", na=False)
            & risk.str.contains("critical", na=False)
        ),
    ],
    [
        "Bullish / Normal Risk",
        "Bullish / Elevated Risk",
        "Bullish / High Risk",
        "Bullish / Critical Risk",

        "Bearish / Normal Risk",
        "Bearish / Elevated Risk",
        "Bearish / High Risk",
        "Bearish / Critical Risk",
    ],
    default="Neutral / Mixed Risk"
)


# ============================================================
# RULE 14 - PRIMARY ACTION
# ============================================================

print("Determining primary rule action...")

df["Primary_Rule_Action"] = np.select(
    [
        final_decision.str.contains(
            "increase",
            na=False
        ),

        final_decision.str.contains(
            "maintain bullish",
            na=False
        ),

        final_decision.str.contains(
            "maintain balanced",
            na=False
        ),

        final_decision.str.contains(
            "reduce",
            na=False
        ),

        final_decision.str.contains(
            "defensive",
            na=False
        ),
    ],
    [
        "Increase Exposure",
        "Maintain Bullish Exposure",
        "Maintain Balanced Exposure",
        "Reduce Exposure",
        "Defensive Positioning",
    ],
    default="Review"
)


# ============================================================
# RULE 15 - POSITIONING
# ============================================================

print("Determining positioning rule...")

df["Positioning_Rule"] = np.select(
    [
        positioning.str.contains(
            "risk seeking",
            na=False
        ),

        positioning.str.contains(
            "risk neutral",
            na=False
        ),

        positioning.str.contains(
            "defensive",
            na=False
        ),

        positioning.str.contains(
            "capital preservation",
            na=False
        ),
    ],
    [
        "Risk Seeking",
        "Risk Neutral",
        "Defensive",
        "Capital Preservation",
    ],
    default=df["Positioning_Environment"].astype(str)
)


# ============================================================
# RULE 16 - VOLATILITY RESPONSE
# ============================================================

print("Determining volatility response rule...")

df["Volatility_Rule_Action"] = np.select(
    [
        vol_response.str.contains(
            "expand",
            na=False
        ),

        vol_response.str.contains(
            "normal",
            na=False
        ),

        vol_response.str.contains(
            "reduce",
            na=False
        ),

        vol_response.str.contains(
            "protect",
            na=False
        ),

        vol_response.str.contains(
            "defensive",
            na=False
        ),
    ],
    [
        "Risk Can Expand",
        "Normal Volatility Response",
        "Reduce Risk",
        "Protect Capital",
        "Defensive Volatility Response",
    ],
    default=df["Volatility_Response"].astype(str)
)


# ============================================================
# RULE 17 - TREND RESPONSE
# ============================================================

print("Determining trend response rule...")

df["Trend_Rule_Action"] = np.select(
    [
        trend_response.str.contains(
            "positive",
            na=False
        ),

        trend_response.str.contains(
            "neutral",
            na=False
        ),

        trend_response.str.contains(
            "negative",
            na=False
        ),
    ],
    [
        "Trend Positive",
        "Trend Neutral",
        "Trend Negative",
    ],
    default=df["Trend_Response"].astype(str)
)


# ============================================================
# RULE 18 - DRAWDOWN RESPONSE
# ============================================================

print("Determining drawdown response rule...")

df["Drawdown_Rule_Action"] = np.select(
    [
        drawdown_response.str.contains(
            "increase",
            na=False
        ),

        drawdown_response.str.contains(
            "monitor",
            na=False
        ),

        drawdown_response.str.contains(
            "reduce",
            na=False
        ),

        drawdown_response.str.contains(
            "protect",
            na=False
        ),

        drawdown_response.str.contains(
            "critical",
            na=False
        ),
    ],
    [
        "Increase Exposure Carefully",
        "Monitor Drawdown",
        "Reduce Exposure",
        "Protect Capital",
        "Critical Drawdown Protection",
    ],
    default=df["Drawdown_Response"].astype(str)
)


# ============================================================
# RULE 19 - FINAL RULE SCORE
# ============================================================

print("Building final rule score...")

df["Rule_Score"] = (
    df["Trend_Decision_Score"] * 0.35
    + df["Volatility_Decision_Score"] * 0.30
    + df["Risk_Decision_Score"] * 0.20
    + df["Decision_Confidence_Score"] * 0.15
)

df["Rule_Score"] = df["Rule_Score"].clip(
    lower=0,
    upper=100
)


# ============================================================
# RULE 20 - RULE STRENGTH
# ============================================================

print("Classifying final rule strength...")

df["Rule_Strength"] = np.select(
    [
        df["Rule_Score"] >= 80,
        df["Rule_Score"] >= 65,
        df["Rule_Score"] >= 50,
        df["Rule_Score"] >= 35,
    ],
    [
        "Very Strong",
        "Strong",
        "Moderate",
        "Weak",
    ],
    default="Very Weak"
)


# ============================================================
# RULE 21 - SYSTEM ACTION
# ============================================================

print("Building system action...")

bullish_environment = (
    trend.str.contains(
        "bullish",
        na=False
    )
    & ~risk.str.contains(
        "critical|high",
        na=False
    )
)

bearish_environment = (
    trend.str.contains(
        "bearish",
        na=False
    )
    | risk.str.contains(
        "critical",
        na=False
    )
)

low_volatility = volatility.str.contains(
    "low",
    na=False
)

high_volatility = volatility.str.contains(
    "high|elevated",
    na=False
)

high_confidence = (
    df["Decision_Confidence_Score"] >= 65
)

low_confidence = (
    df["Decision_Confidence_Score"] < 50
)


df["System_Action"] = np.select(
    [
        (
            bearish_environment
            & high_volatility
            & high_confidence
        ),

        (
            bearish_environment
            & high_confidence
        ),

        (
            bullish_environment
            & low_volatility
            & high_confidence
        ),

        (
            bullish_environment
            & high_confidence
        ),

        (
            bullish_environment
            & low_confidence
        ),

        low_confidence,
    ],
    [
        "Defensive Positioning",
        "Reduce Exposure",
        "Increase Exposure",
        "Maintain Bullish Exposure",
        "Maintain Balanced Exposure",
        "Maintain Balanced Exposure",
    ],
    default="Review"
)


# ============================================================
# RULE 22 - RULE PRIORITY
# ============================================================

print("Determining rule priority...")

df["Rule_Priority"] = np.select(
    [
        df["System_Action"].eq(
            "Defensive Positioning"
        ),

        df["System_Action"].eq(
            "Reduce Exposure"
        ),

        df["System_Action"].eq(
            "Increase Exposure"
        ),

        df["System_Action"].eq(
            "Maintain Bullish Exposure"
        ),

        df["System_Action"].eq(
            "Maintain Balanced Exposure"
        ),
    ],
    [
        "Critical",
        "High",
        "High",
        "Normal",
        "Normal",
    ],
    default="Review"
)


# ============================================================
# RULE 23 - CONSISTENCY CHECK
# ============================================================

print("Checking decision-rule consistency...")

df["Rule_Consistency"] = np.select(
    [
        (
            trend.str.contains(
                "bullish",
                na=False
            )
            & df["Market_Bias_Score"].ge(60)
            & final_decision.str.contains(
                "bullish|increase",
                na=False
            )
        ),

        (
            trend.str.contains(
                "bearish",
                na=False
            )
            & df["Market_Bias_Score"].lt(40)
            & final_decision.str.contains(
                "reduce|defensive",
                na=False
            )
        ),

        (
            risk.str.contains(
                "critical|high",
                na=False
            )
            & final_decision.str.contains(
                "reduce|defensive",
                na=False
            )
        ),

        (
            volatility.str.contains(
                "low",
                na=False
            )
            & trend.str.contains(
                "bullish",
                na=False
            )
        ),
    ],
    [
        "Consistent Bullish",
        "Consistent Defensive",
        "Consistent Risk Reduction",
        "Consistent Low-Vol Bull",
    ],
    default="Mixed"
)


# ============================================================
# RULE 24 - FINAL RULE LABEL
# ============================================================

print("Building final rule label...")

df["Final_Rule_Label"] = np.select(
    [
        df["System_Action"].eq(
            "Defensive Positioning"
        ),

        (
            df["System_Action"].eq(
                "Reduce Exposure"
            )
            & df["Rule_Priority"].eq(
                "High"
            )
        ),

        (
            df["System_Action"].eq(
                "Increase Exposure"
            )
            & df["Rule_Priority"].eq(
                "High"
            )
        ),

        df["System_Action"].eq(
            "Maintain Bullish Exposure"
        ),

        df["System_Action"].eq(
            "Maintain Balanced Exposure"
        ),
    ],
    [
        "DEFENSIVE",
        "REDUCE",
        "INCREASE",
        "BULLISH",
        "BALANCED",
    ],
    default="REVIEW"
)


# ============================================================
# CLEAN INFINITE VALUES
# ============================================================

print()
print("Cleaning infinite rule values...")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL DECISION RULE VALIDATION")
print("=" * 60)

print()
print("Rows              :", len(df))
print("Columns           :", len(df.columns))

valid_dates = df["Date"].dropna()

if not valid_dates.empty:
    print(
        "Start Date        :",
        valid_dates.min().strftime("%Y-%m-%d")
    )

    print(
        "End Date          :",
        valid_dates.max().strftime("%Y-%m-%d")
    )
else:
    print("Start Date        : N/A")
    print("End Date          : N/A")


# ============================================================
# RULE FEATURE LIST
# ============================================================

rule_columns = [
    "Rule_Market_Condition",
    "Rule_Trend_Condition",
    "Rule_Volatility_Condition",
    "Rule_Risk_Condition",
    "Trend_Score_Band",
    "Volatility_Score_Band",
    "Risk_Score_Band",
    "Rule_Confidence_Band",
    "Rule_Market_Bias",
    "Rule_Exposure_Level",
    "Rule_Portfolio_Posture",
    "Trend_Volatility_Combination",
    "Risk_Trend_Combination",
    "Primary_Rule_Action",
    "Positioning_Rule",
    "Volatility_Rule_Action",
    "Trend_Rule_Action",
    "Drawdown_Rule_Action",
    "Rule_Score",
    "Rule_Strength",
    "System_Action",
    "Rule_Priority",
    "Rule_Consistency",
    "Final_Rule_Label",
]


print()
print("Rule Features      :", len(rule_columns))

print()
print("Rule columns:")

for column in rule_columns:
    print(" -", column)


# ============================================================
# LATEST SNAPSHOT
# ============================================================

latest = df.iloc[-1]

print()
print("=" * 60)
print("LATEST DECISION RULE SNAPSHOT")
print("=" * 60)

print()

print(
    "Date                       :",
    latest["Date"].strftime("%Y-%m-%d")
)

print(
    "Close                      :",
    f"{latest['Close']:.2f}"
    if pd.notna(latest["Close"])
    else "N/A"
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
    "Trend Score Band           :",
    latest["Trend_Score_Band"]
)

print(
    "Volatility Score Band      :",
    latest["Volatility_Score_Band"]
)

print(
    "Risk Score Band            :",
    latest["Risk_Score_Band"]
)

print(
    "Decision Confidence        :",
    f"{latest['Decision_Confidence_Score']:.2f}"
    if pd.notna(
        latest["Decision_Confidence_Score"]
    )
    else "N/A"
)

print(
    "Rule Score                 :",
    f"{latest['Rule_Score']:.2f}"
    if pd.notna(
        latest["Rule_Score"]
    )
    else "N/A"
)

print(
    "Rule Strength              :",
    latest["Rule_Strength"]
)

print(
    "System Action              :",
    latest["System_Action"]
)

print(
    "Rule Priority              :",
    latest["Rule_Priority"]
)

print(
    "Final Rule Label           :",
    latest["Final_Rule_Label"]
)

print(
    "Rule Consistency           :",
    latest["Rule_Consistency"]
)


# ============================================================
# SYSTEM ACTION DISTRIBUTION
# ============================================================

print()
print("System action distribution:")

print(
    df["System_Action"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# FINAL RULE LABEL DISTRIBUTION
# ============================================================

print()
print("Final rule label distribution:")

print(
    df["Final_Rule_Label"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# SAVE DATASET
# ============================================================

print()
print("Saving decision rules dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.6f"
)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

if not OUTPUT_FILE.exists():
    raise RuntimeError(
        "Output file was not created."
    )

saved_df = pd.read_csv(
    OUTPUT_FILE,
    nrows=5
)

if saved_df.empty:
    raise RuntimeError(
        "Output file was created but contains no data."
    )


# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 60)
print("MODULE 07 COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print("Output rows    :", len(df))
print("Output columns :", len(df.columns))

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("=" * 60)
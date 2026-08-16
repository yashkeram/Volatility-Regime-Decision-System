import os
import pandas as pd
import numpy as np


# ============================================================
# MODULE 04 - VOLATILITY MODEL
# ============================================================

print()
print("=" * 60)
print("MODULE 04 - VOLATILITY MODEL")
print("=" * 60)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Processed"
)

INPUT_FILE = os.path.join(
    PROCESSED_FOLDER,
    "nifty50_features.csv"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_FOLDER,
    "nifty50_volatility_model.csv"
)


print()
print(f"Project Root : {PROJECT_ROOT}")
print(f"Input File   : {INPUT_FILE}")
print(f"Output File  : {OUTPUT_FILE}")


# ============================================================
# VALIDATE INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nFeature file not found:\n{INPUT_FILE}"
    )


# ============================================================
# LOAD FEATURE DATA
# ============================================================

print()
print("Loading engineered features...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded    : {len(df):,}")
print(f"Columns loaded : {len(df.columns)}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Close",
    "Log_Return",
    "Volatility_20D",
    "Volatility_60D",
    "Volatility_120D",
    "Volatility_252D",
    "Volatility_Ratio_20_60",
    "Volatility_Ratio_20_120",
    "Volatility_Ratio_60_252",
    "ATR_14D_pct",
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
        "\nRequired columns are missing:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# DATE PREPARATION
# ============================================================

print()
print("Preparing dates...")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

if df["Date"].isna().any():
    raise ValueError(
        "Invalid dates found in feature dataset."
    )

df = df.sort_values(
    "Date"
).reset_index(drop=True)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print("Converting model inputs to numeric...")

model_columns = [
    column
    for column in required_columns
    if column != "Date"
]

for column in model_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# MODEL 01 - VOLATILITY TERM STRUCTURE
# ============================================================

print()
print("Calculating volatility term structure...")

df["Volatility_Term_Spread_20_60"] = (
    df["Volatility_20D"]
    - df["Volatility_60D"]
)

df["Volatility_Term_Spread_60_120"] = (
    df["Volatility_60D"]
    - df["Volatility_120D"]
)

df["Volatility_Term_Spread_120_252"] = (
    df["Volatility_120D"]
    - df["Volatility_252D"]
)


# ============================================================
# MODEL 02 - SHORT VS LONG VOLATILITY
# ============================================================

df["Short_Long_Volatility_Ratio"] = (
    df["Volatility_20D"]
    / df["Volatility_252D"]
)


# ============================================================
# MODEL 03 - VOLATILITY PERCENTILE
# ============================================================

print("Calculating historical volatility percentiles...")

# Expanding percentile avoids look-ahead bias.
def expanding_percentile(series):

    values = []
    history = []

    for value in series:

        if pd.isna(value):
            values.append(np.nan)
            continue

        history.append(value)

        if len(history) == 1:
            values.append(50.0)
            continue

        rank = (
            sum(
                historical_value <= value
                for historical_value in history
            )
            / len(history)
        ) * 100

        values.append(rank)

    return pd.Series(
        values,
        index=series.index
    )


df["Volatility_20D_Percentile"] = (
    expanding_percentile(
        df["Volatility_20D"]
    )
)

df["Volatility_60D_Percentile"] = (
    expanding_percentile(
        df["Volatility_60D"]
    )
)

df["Volatility_120D_Percentile"] = (
    expanding_percentile(
        df["Volatility_120D"]
    )
)


# ============================================================
# MODEL 04 - VOLATILITY Z-SCORE
# ============================================================

print("Calculating volatility z-scores...")

volatility_mean_252 = (
    df["Volatility_20D"]
    .rolling(
        window=252,
        min_periods=60
    )
    .mean()
)

volatility_std_252 = (
    df["Volatility_20D"]
    .rolling(
        window=252,
        min_periods=60
    )
    .std()
)

df["Volatility_20D_ZScore"] = (
    (
        df["Volatility_20D"]
        - volatility_mean_252
    )
    /
    volatility_std_252
)


# ============================================================
# MODEL 05 - REALIZED VOLATILITY ACCELERATION
# ============================================================

print("Calculating volatility acceleration...")

df["Volatility_Acceleration"] = (
    df["Volatility_20D"]
    .pct_change(periods=5)
    * 100
)


# ============================================================
# MODEL 06 - VOLATILITY MOMENTUM
# ============================================================

df["Volatility_Momentum_20D"] = (
    df["Volatility_20D"]
    - df["Volatility_20D"].shift(20)
)


# ============================================================
# MODEL 07 - VOLATILITY OF VOLATILITY
# ============================================================

df["Volatility_of_Volatility_20D"] = (
    df["Volatility_20D"]
    .rolling(
        window=20
    )
    .std()
)


# ============================================================
# MODEL 08 - RETURN DISPERSION
# ============================================================

print("Calculating return dispersion...")

df["Return_Dispersion_20D"] = (
    df["Log_Return"]
    .rolling(
        window=20
    )
    .std()
    * np.sqrt(252)
    * 100
)

df["Return_Dispersion_60D"] = (
    df["Log_Return"]
    .rolling(
        window=60
    )
    .std()
    * np.sqrt(252)
    * 100
)


# ============================================================
# MODEL 09 - UPSIDE / DOWNSIDE VOLATILITY
# ============================================================

print("Calculating upside/downside volatility...")

positive_returns = df["Log_Return"].where(
    df["Log_Return"] > 0
)

negative_returns = df["Log_Return"].where(
    df["Log_Return"] < 0
)

df["Upside_Volatility_20D"] = (
    positive_returns
    .rolling(
        window=20
    )
    .std()
    * np.sqrt(252)
    * 100
)

df["Downside_Volatility_20D"] = (
    negative_returns
    .rolling(
        window=20
    )
    .std()
    * np.sqrt(252)
    * 100
)


# ============================================================
# MODEL 10 - DOWNSIDE VOLATILITY RATIO
# ============================================================

df["Downside_Volatility_Ratio"] = (
    df["Downside_Volatility_20D"]
    /
    df["Volatility_20D"]
)


# ============================================================
# MODEL 11 - RANGE-BASED VOLATILITY
# ============================================================

print("Calculating range-based volatility...")

daily_range_pct = (
    (
        df["High"]
        - df["Low"]
    )
    /
    df["Close"]
) * 100

df["Range_Volatility_20D"] = (
    daily_range_pct
    .rolling(
        window=20
    )
    .mean()
)

df["Range_Volatility_60D"] = (
    daily_range_pct
    .rolling(
        window=60
    )
    .mean()
)


# ============================================================
# MODEL 12 - VOLATILITY COMPOSITE
# ============================================================

print("Building volatility composite...")

# Normalize the main volatility dimensions
# using their historical expanding percentile.

volatility_components = [
    "Volatility_20D_Percentile",
    "Volatility_60D_Percentile",
    "Volatility_120D_Percentile"
]

df["Volatility_Composite"] = (
    df[volatility_components]
    .mean(axis=1)
)


# ============================================================
# MODEL 13 - VOLATILITY STRESS SCORE
# ============================================================

print("Building volatility stress score...")

stress_components = []

# High short-term volatility
stress_components.append(
    df["Volatility_20D_Percentile"]
)

# Short-term volatility above long-term volatility
stress_components.append(
    df["Volatility_Ratio_20_120"]
    .clip(
        lower=0,
        upper=3
    )
    * 33.3333
)

# Volatility acceleration
acceleration_score = (
    df["Volatility_Acceleration"]
    .clip(
        lower=-50,
        upper=100
    )
)

acceleration_score = (
    (acceleration_score + 50)
    / 150
) * 100

stress_components.append(
    acceleration_score
)

df["Volatility_Stress_Score"] = (
    pd.concat(
        stress_components,
        axis=1
    )
    .mean(axis=1)
)


# ============================================================
# MODEL 14 - VOLATILITY DIRECTION
# ============================================================

print("Determining volatility direction...")

df["Volatility_Direction"] = np.select(
    [
        df["Volatility_Momentum_20D"] > 1,
        df["Volatility_Momentum_20D"] < -1
    ],
    [
        "Rising",
        "Falling"
    ],
    default="Stable"
)


# ============================================================
# MODEL 15 - VOLATILITY ENVIRONMENT
# ============================================================

print("Classifying volatility environment...")

df["Volatility_Environment"] = np.select(
    [
        df["Volatility_20D_Percentile"] >= 80,
        df["Volatility_20D_Percentile"] >= 60,
        df["Volatility_20D_Percentile"] >= 40
    ],
    [
        "Extreme",
        "High",
        "Normal"
    ],
    default="Low"
)


# ============================================================
# MODEL 16 - VOLATILITY PRESSURE
# ============================================================

df["Volatility_Pressure"] = np.select(
    [
        (
            (df["Volatility_Ratio_20_60"] > 1.10)
            &
            (df["Volatility_Acceleration"] > 0)
        ),

        (
            (df["Volatility_Ratio_20_60"] < 0.90)
            &
            (df["Volatility_Acceleration"] < 0)
        )
    ],
    [
        "Increasing",
        "Decreasing"
    ],
    default="Neutral"
)


# ============================================================
# REMOVE INFINITE VALUES
# ============================================================

print()
print("Cleaning infinite model values...")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# MODEL COLUMN LIST
# ============================================================

model_output_columns = [
    "Volatility_Term_Spread_20_60",
    "Volatility_Term_Spread_60_120",
    "Volatility_Term_Spread_120_252",
    "Short_Long_Volatility_Ratio",
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
    "Volatility_Pressure"
]


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL VOLATILITY MODEL VALIDATION")
print("=" * 60)

print()
print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns)}")
print(f"Start Date        : {df['Date'].min().date()}")
print(f"End Date          : {df['Date'].max().date()}")
print(
    f"Model Features    : "
    f"{len(model_output_columns)}"
)

print()
print("Model columns:")

for column in model_output_columns:
    print(f" - {column}")


# ============================================================
# CURRENT MODEL SNAPSHOT
# ============================================================

valid_model_data = df.dropna(
    subset=[
        "Volatility_20D",
        "Volatility_60D",
        "Volatility_20D_Percentile"
    ]
)

if not valid_model_data.empty:

    latest = valid_model_data.iloc[-1]

    print()
    print("=" * 60)
    print("LATEST VOLATILITY MODEL SNAPSHOT")
    print("=" * 60)

    print()
    print(
        f"Date                       : "
        f"{latest['Date'].date()}"
    )

    print(
        f"Close                      : "
        f"{latest['Close']:.2f}"
    )

    print(
        f"20D Volatility              : "
        f"{latest['Volatility_20D']:.2f}%"
    )

    print(
        f"60D Volatility              : "
        f"{latest['Volatility_60D']:.2f}%"
    )

    print(
        f"120D Volatility             : "
        f"{latest['Volatility_120D']:.2f}%"
    )

    print(
        f"252D Volatility             : "
        f"{latest['Volatility_252D']:.2f}%"
    )

    print(
        f"20D Volatility Percentile   : "
        f"{latest['Volatility_20D_Percentile']:.2f}"
    )

    print(
        f"Volatility Composite        : "
        f"{latest['Volatility_Composite']:.2f}"
    )

    print(
        f"Volatility Stress Score     : "
        f"{latest['Volatility_Stress_Score']:.2f}"
    )

    print(
        f"Volatility Direction        : "
        f"{latest['Volatility_Direction']}"
    )

    print(
        f"Volatility Environment      : "
        f"{latest['Volatility_Environment']}"
    )

    print(
        f"Volatility Pressure         : "
        f"{latest['Volatility_Pressure']}"
    )


# ============================================================
# SAVE MODEL DATA
# ============================================================

print()
print("Saving volatility model dataset...")

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VERIFY OUTPUT
# ============================================================

if not os.path.exists(OUTPUT_FILE):
    raise RuntimeError(
        "Volatility model output was not created."
    )

saved_df = pd.read_csv(
    OUTPUT_FILE
)

print()
print("=" * 60)
print("MODULE 04 COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print(f"Output rows    : {len(saved_df):,}")
print(f"Output columns : {len(saved_df.columns)}")

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("=" * 60)
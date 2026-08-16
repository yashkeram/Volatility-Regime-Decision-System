import os
import sys
import pandas as pd
import numpy as np


# ============================================================
# MODULE 03 - FEATURE ENGINEERING
# ============================================================

print()
print("=" * 60)
print("MODULE 03 - FEATURE ENGINEERING")
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
    "nifty50_clean.csv"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_FOLDER,
    "nifty50_features.csv"
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
        f"\nInput file not found:\n{INPUT_FILE}"
    )


# ============================================================
# LOAD CLEAN DATA
# ============================================================

print()
print("Loading cleaned NIFTY 50 data...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded  : {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Close",
    "Daily_Return",
    "Log_Return"
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
# DATE CONVERSION
# ============================================================

print()
print("Converting Date column...")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

if df["Date"].isna().any():
    raise ValueError(
        "Invalid dates found in the dataset."
    )

df = df.sort_values(
    "Date"
).reset_index(drop=True)


# ============================================================
# ENSURE NUMERIC DATA
# ============================================================

print("Converting numerical columns...")

numeric_columns = [
    "Adj Close",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "Daily_Return",
    "Log_Return"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# FEATURE 01 - 5 DAY RETURN
# ============================================================

print()
print("Creating return features...")

df["Return_5D"] = (
    df["Close"]
    .pct_change(periods=5)
    * 100
)


# ============================================================
# FEATURE 02 - 10 DAY RETURN
# ============================================================

df["Return_10D"] = (
    df["Close"]
    .pct_change(periods=10)
    * 100
)


# ============================================================
# FEATURE 03 - 20 DAY RETURN
# ============================================================

df["Return_20D"] = (
    df["Close"]
    .pct_change(periods=20)
    * 100
)


# ============================================================
# FEATURE 04 - 60 DAY RETURN
# ============================================================

df["Return_60D"] = (
    df["Close"]
    .pct_change(periods=60)
    * 100
)


# ============================================================
# FEATURE 05 - 20 DAY ROLLING VOLATILITY
# ============================================================

print("Creating rolling volatility features...")

df["Volatility_20D"] = (
    df["Log_Return"]
    .rolling(window=20)
    .std()
    * np.sqrt(252)
    * 100
)


# ============================================================
# FEATURE 06 - 60 DAY ROLLING VOLATILITY
# ============================================================

df["Volatility_60D"] = (
    df["Log_Return"]
    .rolling(window=60)
    .std()
    * np.sqrt(252)
    * 100
)


# ============================================================
# FEATURE 07 - 120 DAY ROLLING VOLATILITY
# ============================================================

df["Volatility_120D"] = (
    df["Log_Return"]
    .rolling(window=120)
    .std()
    * np.sqrt(252)
    * 100
)


# ============================================================
# FEATURE 08 - 252 DAY ROLLING VOLATILITY
# ============================================================

df["Volatility_252D"] = (
    df["Log_Return"]
    .rolling(window=252)
    .std()
    * np.sqrt(252)
    * 100
)


# ============================================================
# FEATURE 09 - 20 DAY MOVING AVERAGE
# ============================================================

print("Creating moving-average features...")

df["MA_20D"] = (
    df["Close"]
    .rolling(window=20)
    .mean()
)


# ============================================================
# FEATURE 10 - 50 DAY MOVING AVERAGE
# ============================================================

df["MA_50D"] = (
    df["Close"]
    .rolling(window=50)
    .mean()
)


# ============================================================
# FEATURE 11 - 100 DAY MOVING AVERAGE
# ============================================================

df["MA_100D"] = (
    df["Close"]
    .rolling(window=100)
    .mean()
)


# ============================================================
# FEATURE 12 - 200 DAY MOVING AVERAGE
# ============================================================

df["MA_200D"] = (
    df["Close"]
    .rolling(window=200)
    .mean()
)


# ============================================================
# FEATURE 13 - PRICE VS 20D MA
# ============================================================

df["Price_vs_MA20_pct"] = (
    (df["Close"] / df["MA_20D"]) - 1
) * 100


# ============================================================
# FEATURE 14 - PRICE VS 50D MA
# ============================================================

df["Price_vs_MA50_pct"] = (
    (df["Close"] / df["MA_50D"]) - 1
) * 100


# ============================================================
# FEATURE 15 - PRICE VS 200D MA
# ============================================================

df["Price_vs_MA200_pct"] = (
    (df["Close"] / df["MA_200D"]) - 1
) * 100


# ============================================================
# FEATURE 16 - VOLATILITY REGIME RATIO
# ============================================================

print("Creating volatility regime features...")

df["Volatility_Ratio_20_60"] = (
    df["Volatility_20D"]
    / df["Volatility_60D"]
)


# ============================================================
# FEATURE 17 - VOLATILITY REGIME RATIO
# ============================================================

df["Volatility_Ratio_20_120"] = (
    df["Volatility_20D"]
    / df["Volatility_120D"]
)


# ============================================================
# FEATURE 18 - VOLATILITY REGIME RATIO
# ============================================================

df["Volatility_Ratio_60_252"] = (
    df["Volatility_60D"]
    / df["Volatility_252D"]
)


# ============================================================
# FEATURE 19 - TRUE RANGE
# ============================================================

print("Creating price-range features...")

previous_close = df["Close"].shift(1)

true_range_1 = (
    df["High"] - df["Low"]
)

true_range_2 = (
    (df["High"] - previous_close).abs()
)

true_range_3 = (
    (df["Low"] - previous_close).abs()
)

df["True_Range"] = pd.concat(
    [
        true_range_1,
        true_range_2,
        true_range_3
    ],
    axis=1
).max(axis=1)


# ============================================================
# FEATURE 20 - ATR 14D
# ============================================================

df["ATR_14D"] = (
    df["True_Range"]
    .rolling(window=14)
    .mean()
)


# ============================================================
# FEATURE 21 - ATR AS % OF PRICE
# ============================================================

df["ATR_14D_pct"] = (
    df["ATR_14D"]
    / df["Close"]
) * 100


# ============================================================
# FEATURE 22 - 20D RETURN VOLATILITY
# ============================================================

df["Return_Volatility_20D"] = (
    df["Daily_Return"]
    .rolling(window=20)
    .std()
)


# ============================================================
# FEATURE 23 - 60D RETURN VOLATILITY
# ============================================================

df["Return_Volatility_60D"] = (
    df["Daily_Return"]
    .rolling(window=60)
    .std()
)


# ============================================================
# FEATURE 24 - 20D MAXIMUM DRAWDOWN
# ============================================================

print("Creating drawdown features...")

rolling_peak_20 = (
    df["Close"]
    .rolling(window=20)
    .max()
)

df["Drawdown_20D_pct"] = (
    (df["Close"] / rolling_peak_20) - 1
) * 100


# ============================================================
# FEATURE 25 - 60D MAXIMUM DRAWDOWN
# ============================================================

rolling_peak_60 = (
    df["Close"]
    .rolling(window=60)
    .max()
)

df["Drawdown_60D_pct"] = (
    (df["Close"] / rolling_peak_60) - 1
) * 100


# ============================================================
# FEATURE 26 - 252D MAXIMUM DRAWDOWN
# ============================================================

rolling_peak_252 = (
    df["Close"]
    .rolling(window=252)
    .max()
)

df["Drawdown_252D_pct"] = (
    (df["Close"] / rolling_peak_252) - 1
) * 100


# ============================================================
# FEATURE 27 - 20D HIGH-LOW RANGE
# ============================================================

df["Range_20D_pct"] = (
    (
        df["High"].rolling(20).max()
        -
        df["Low"].rolling(20).min()
    )
    /
    df["Close"]
) * 100


# ============================================================
# FEATURE 28 - VOLUME CHANGE
# ============================================================

if "Volume" in df.columns:

    df["Volume_Change_pct"] = (
        df["Volume"]
        .pct_change()
        * 100
    )


# ============================================================
# REMOVE INFINITE VALUES
# ============================================================

print()
print("Cleaning infinite feature values...")

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

feature_columns = [
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Return_60D",
    "Volatility_20D",
    "Volatility_60D",
    "Volatility_120D",
    "Volatility_252D",
    "MA_20D",
    "MA_50D",
    "MA_100D",
    "MA_200D",
    "Price_vs_MA20_pct",
    "Price_vs_MA50_pct",
    "Price_vs_MA200_pct",
    "Volatility_Ratio_20_60",
    "Volatility_Ratio_20_120",
    "Volatility_Ratio_60_252",
    "True_Range",
    "ATR_14D",
    "ATR_14D_pct",
    "Return_Volatility_20D",
    "Return_Volatility_60D",
    "Drawdown_20D_pct",
    "Drawdown_60D_pct",
    "Drawdown_252D_pct",
    "Range_20D_pct",
    "Volume_Change_pct"
]

existing_feature_columns = [
    column
    for column in feature_columns
    if column in df.columns
]


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL FEATURE VALIDATION")
print("=" * 60)

print()
print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns)}")
print(f"Start Date        : {df['Date'].min().date()}")
print(f"End Date          : {df['Date'].max().date()}")
print(f"Features Created  : {len(existing_feature_columns)}")

print()
print("Feature columns:")
for feature in existing_feature_columns:
    print(f" - {feature}")

print()
print("Missing values in features:")

missing_features = (
    df[existing_feature_columns]
    .isna()
    .sum()
)

print(missing_features)


# ============================================================
# SAVE FEATURE DATA
# ============================================================

print()
print("Saving feature dataset...")

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
        "Feature dataset was not created."
    )

saved_df = pd.read_csv(
    OUTPUT_FILE
)

print()
print("=" * 60)
print("MODULE 03 COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print(f"Output rows    : {len(saved_df):,}")
print(f"Output columns : {len(saved_df.columns)}")

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("=" * 60)
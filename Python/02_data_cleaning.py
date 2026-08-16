import os
import pandas as pd

# ============================================================
# MODULE 02 - DATA CLEANING
# ============================================================

print()
print("=" * 60)
print("MODULE 02 - DATA CLEANING")
print("=" * 60)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Raw"
)

PROCESSED_FOLDER = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Processed"
)

INPUT_FILE = os.path.join(
    RAW_FOLDER,
    "nifty50_raw.csv"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_FOLDER,
    "nifty50_clean.csv"
)

# ============================================================
# DISPLAY PATHS
# ============================================================

print()
print("Project Root :", PROJECT_ROOT)
print("Raw Folder   :", RAW_FOLDER)
print("Processed    :", PROCESSED_FOLDER)
print("Input File   :", INPUT_FILE)
print("Output File  :", OUTPUT_FILE)

# ============================================================
# VALIDATE INPUT
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)

# ============================================================
# LOAD RAW DATA
# ============================================================

print()
print("Loading raw NIFTY 50 data...")

df = pd.read_csv(INPUT_FILE)

print(f"Raw rows loaded : {len(df):,}")

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Adj Close",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}"
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

invalid_dates = df["Date"].isna().sum()

if invalid_dates > 0:
    print(
        f"Removing {invalid_dates} rows with invalid dates..."
    )

    df = df.dropna(
        subset=["Date"]
    )

# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "Adj Close",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume"
]

print("Converting market columns to numeric...")

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ============================================================
# REMOVE DUPLICATE DATES
# ============================================================

duplicate_count = df["Date"].duplicated().sum()

print()
print(
    f"Duplicate dates found : {duplicate_count:,}"
)

if duplicate_count > 0:

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    by="Date"
).reset_index(
    drop=True
)

# ============================================================
# CHECK MISSING MARKET DATA
# ============================================================

print()
print("Checking missing market values...")

missing_before = df[numeric_columns].isna().sum()

print(missing_before)

# ============================================================
# REMOVE ROWS WITH MISSING OHLC DATA
# ============================================================

price_columns = [
    "Adj Close",
    "Close",
    "High",
    "Low",
    "Open"
]

rows_before_missing = len(df)

df = df.dropna(
    subset=price_columns
).reset_index(
    drop=True
)

rows_removed = (
    rows_before_missing - len(df)
)

if rows_removed > 0:

    print(
        f"Rows removed because of missing prices : "
        f"{rows_removed:,}"
    )

# ============================================================
# REMOVE INVALID PRICES
# ============================================================

invalid_price_mask = (
    (df["Close"] <= 0)
    | (df["Open"] <= 0)
    | (df["High"] <= 0)
    | (df["Low"] <= 0)
    | (df["Adj Close"] <= 0)
)

invalid_price_count = invalid_price_mask.sum()

print(
    f"Invalid price rows : {invalid_price_count:,}"
)

if invalid_price_count > 0:

    df = df[
        ~invalid_price_mask
    ].reset_index(
        drop=True
    )

# ============================================================
# PRICE CONSISTENCY CHECK
# ============================================================

invalid_ohlc = (
    (df["High"] < df["Low"])
    | (df["High"] < df["Open"])
    | (df["High"] < df["Close"])
    | (df["Low"] > df["Open"])
    | (df["Low"] > df["Close"])
)

invalid_ohlc_count = invalid_ohlc.sum()

print(
    f"Invalid OHLC rows : {invalid_ohlc_count:,}"
)

if invalid_ohlc_count > 0:

    df = df[
        ~invalid_ohlc
    ].reset_index(
        drop=True
    )

# ============================================================
# VOLUME VALIDATION
# ============================================================

invalid_volume = (
    df["Volume"] < 0
)

invalid_volume_count = invalid_volume.sum()

print(
    f"Invalid volume rows : {invalid_volume_count:,}"
)

if invalid_volume_count > 0:

    df = df[
        ~invalid_volume
    ].reset_index(
        drop=True
    )

# ============================================================
# DAILY RETURNS
# ============================================================

print()
print("Calculating daily returns...")

df["Daily_Return"] = (
    df["Adj Close"]
    .pct_change()
    * 100
)

# ============================================================
# LOG RETURNS
# ============================================================

print("Calculating log returns...")

import numpy as np

df["Log_Return"] = np.log(
    df["Adj Close"]
    / df["Adj Close"].shift(1)
)

# ============================================================
# FINAL COLUMN ORDER
# ============================================================

df = df[
    [
        "Date",
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
        "Daily_Return",
        "Log_Return"
    ]
]

# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("FINAL DATA VALIDATION")
print("=" * 60)

print()
print(f"Rows              : {len(df):,}")
print(f"Columns           : {len(df.columns)}")
print(f"Start Date        : {df['Date'].min().date()}")
print(f"End Date          : {df['Date'].max().date()}")
print(
    f"Missing Values    : {df.isna().sum().sum():,}"
)
print(
    f"Duplicate Dates   : "
    f"{df['Date'].duplicated().sum():,}"
)

# ============================================================
# SAVE CLEAN DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# COMPLETION
# ============================================================

print()
print("=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)

print()
print(f"Input rows       : {len(pd.read_csv(INPUT_FILE)):,}")
print(f"Clean rows       : {len(df):,}")

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("=" * 60)
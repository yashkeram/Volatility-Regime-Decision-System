import os
import sys
import pandas as pd
import yfinance as yf


# ============================================================
# MODULE 01 - DATA INGESTION
# Volatility Regime Decision System
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TICKER = "^NSEI"

START_DATE = "2015-01-01"

END_DATE = None


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    RAW_FOLDER,
    exist_ok=True
)

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)


# ============================================================
# TERMINAL HEADER
# ============================================================

print()
print("=" * 60)
print("MODULE 01 - DATA INGESTION")
print("=" * 60)
print()

print(
    f"Project Root : {PROJECT_ROOT}"
)

print(
    f"Raw Folder   : {RAW_FOLDER}"
)

print(
    f"Processed    : {PROCESSED_FOLDER}"
)

print()


# ============================================================
# TICKER INPUT
# ============================================================

if len(sys.argv) > 1:

    TICKER = sys.argv[1].upper()

else:

    TICKER = DEFAULT_TICKER


# ============================================================
# OUTPUT FILE NAME
# ============================================================

if TICKER == "^NSEI":

    OUTPUT_FILE = "nifty50_raw.csv"

else:

    safe_ticker = (
        TICKER
        .replace("^", "")
        .replace(".", "_")
        .replace("/", "_")
    )

    OUTPUT_FILE = f"{safe_ticker}_raw.csv"


OUTPUT_PATH = os.path.join(
    RAW_FOLDER,
    OUTPUT_FILE
)


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

print("Data Configuration")
print("-" * 60)

print(
    f"Ticker     : {TICKER}"
)

print(
    f"Start Date : {START_DATE}"
)

print(
    f"End Date   : {END_DATE if END_DATE else 'Latest available'}"
)

print(
    f"Output     : {OUTPUT_PATH}"
)

print()


# ============================================================
# DOWNLOAD DATA
# ============================================================

print("Downloading market data...")
print()


try:

    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False
    )

except Exception as error:

    raise RuntimeError(
        f"Failed to download data for {TICKER}: {error}"
    )


# ============================================================
# VALIDATE DOWNLOAD
# ============================================================

if data is None or data.empty:

    raise ValueError(
        f"No market data was downloaded for {TICKER}."
    )


# ============================================================
# HANDLE YFINANCE MULTIINDEX COLUMNS
# ============================================================

if isinstance(
    data.columns,
    pd.MultiIndex
):

    data.columns = data.columns.get_level_values(0)


# ============================================================
# RESET INDEX
# ============================================================

data = data.reset_index()


# ============================================================
# NORMALIZE DATE COLUMN
# ============================================================

if "Date" not in data.columns:

    raise ValueError(
        "Downloaded data does not contain a Date column."
    )


data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce"
)


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]


for column in numeric_columns:

    if column in data.columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

data = data.dropna(
    subset=[
        "Date",
        "Close"
    ]
)


# ============================================================
# SORT DATA
# ============================================================

data = data.sort_values(
    "Date"
)


# ============================================================
# REMOVE DUPLICATE DATES
# ============================================================

data = data.drop_duplicates(
    subset=["Date"],
    keep="last"
)


# ============================================================
# RESET INDEX
# ============================================================

data = data.reset_index(
    drop=True
)


# ============================================================
# VALIDATION
# ============================================================

if data.empty:

    raise ValueError(
        "No valid rows remain after cleaning."
    )


# ============================================================
# SAVE RAW DATA
# ============================================================

data.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("DATA INGESTION COMPLETED")
print("=" * 60)
print()

print(
    f"Ticker        : {TICKER}"
)

print(
    f"Rows          : {len(data):,}"
)

print(
    f"Start         : {data['Date'].min().date()}"
)

print(
    f"End           : {data['Date'].max().date()}"
)

print(
    f"Latest Close  : {data['Close'].iloc[-1]:.2f}"
)

print()

print(
    f"Saved to:"
)

print(
    OUTPUT_PATH
)

print()
print("=" * 60)
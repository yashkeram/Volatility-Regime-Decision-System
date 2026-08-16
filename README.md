# Volatility Regime Decision System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-6f42c1)](https://github.com/ranaroussi/yfinance)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/yashkeram/Volatility-Regime-Decision-System-Refurbished)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A systematic quantitative research framework designed to analyze the **NIFTY 50 Index**. By evaluating market trends, historical volatility, and drawdown conditions, the system detects prevailing market regimes and translates them into structured, actionable capital allocation guidance.

The project is designed as a reproducible research pipeline that transforms raw market data into volatility states, regime classifications, decision rules, and a final portfolio posture.

---

## Research Objective

The objective of this project is to investigate whether classifying market environments (regimes) prior to making allocation decisions can improve risk-adjusted performance and preserve capital during market stress. 

A market can be bullish under very different conditions (e.g., *Low Volatility Bull* vs. *High Volatility Bull*). This research pipeline combines:

* Historical price data
* Volatility term structures
* Regime detection algorithms
* Explicit rule-based decision logic
* Dynamic capital allocation sizing

The goal is not to predict individual market tops or bottoms, but to build and evaluate a **transparent, rules-based investment process** that adapts to changing risk environments.

---

## Research Pipeline

```text
Raw Market Data
       │
       ▼
01. Data Ingestion
       │
       ▼
02. Data Cleaning
       │
       ▼
03. Feature Engineering
       │
       ▼
04. Volatility Model
       │
       ▼
05. Regime Detection
       │
       ▼
06. Decision Logic
       │
       ▼
07. Decision Rules
       │
       ▼
08. Capital Allocation
```

## Project Architecture

```text
Volatility-Regime-Decision-System-Refurbished/
│
├── Data/
│   ├── Raw/
│   │   └── nifty50_raw.csv
│   │
│   └── Processed/
│       ├── nifty50_clean.csv
│       ├── nifty50_features.csv
│       ├── nifty50_volatility_model.csv
│       ├── nifty50_regime_detection.csv
│       ├── nifty50_decision_logic.csv
│       ├── nifty50_decision_rules.csv
│       └── nifty50_capital_allocation.csv
│
├── Notebooks/
│
├── Outputs/
│   ├── Charts/
│   ├── Reports/
│   └── Tables/
│
├── Python/
│   ├── 01_data_ingestion.py
│   ├── 02_data_cleaning.py
│   ├── 03_feature_engineering.py
│   ├── 04_volatility_model.py
│   ├── 05_regime_detection.py
│   ├── 06_decision_logic.py
│   ├── 07_decision_rules.py
│   └── 08_capital_allocation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Technology Stack

| Category        | Technology     |
| --------------- | -------------- |
| Language        | Python         |
| Data Analysis   | Pandas, NumPy  |
| Market Data     | yfinance       |
| Visualization   | Matplotlib     |
| Version Control | Git, GitHub    |
| Research Output | CSV, Markdown  |

## Target Market

This system strictly focuses on the benchmark Indian equities index.

| Market Index | Ticker | Observations | Historical Period |
| :--- | :--- | :--- | :--- |
| NIFTY 50 Index | ^NSEI | 2,860 Trading Days | 2015-01-02 → 2026-08-14 |

## Methodology

The pipeline executes sequentially, with each script transforming the data for the next stage.

| Module | Python Script | Purpose | Primary Output |
| :--- | :--- | :--- | :--- |
| **01. Data Ingestion** | `01_data_ingestion.py` | Downloads raw historical NIFTY 50 market data. | `nifty50_raw.csv` |
| **02. Data Cleaning** | `02_data_cleaning.py` | Validates data, handles missing values, and calculates log returns. | `nifty50_clean.csv` |
| **03. Feature Engineering**| `03_feature_engineering.py` | Creates technical features (Moving Averages, ATR, Drawdowns). | `nifty50_features.csv` |
| **04. Volatility Model** | `04_volatility_model.py` | Builds a multi-dimensional volatility profile (Z-Scores, Momentum). | `nifty50_volatility_model.csv` |
| **05. Regime Detection** | `05_regime_detection.py` | Classifies the current market regime (e.g., *Low Volatility Bull*). | `nifty50_regime_detection.csv` |
| **06. Decision Logic** | `06_decision_logic.py` | Scores trend, volatility, and risk to form a decision framework. | `nifty50_decision_logic.csv` |
| **07. Decision Rules** | `07_decision_rules.py` | Applies explicit bounds to determine final system actions. | `nifty50_decision_rules.csv` |
| **08. Capital Allocation** | `08_capital_allocation.py` | Converts rules into portfolio guidance (Target Equity vs. Cash). | `nifty50_capital_allocation.csv` |

## Outputs

### Processed Data
The pipeline progressively builds these datasets, saving them in the processed data folder:

```text
Data/Processed/
├── nifty50_clean.csv
├── nifty50_features.csv
├── nifty50_volatility_model.csv
├── nifty50_regime_detection.csv
├── nifty50_decision_logic.csv
├── nifty50_decision_rules.csv
└── nifty50_capital_allocation.csv
```
## Reproducibility

To recreate the research outputs from scratch, run the modules sequentially. Each stage depends on the data generated by the step right before it.

```bash
python Python/01_data_ingestion.py
python Python/02_data_cleaning.py
python Python/03_feature_engineering.py
python Python/04_volatility_model.py
python Python/05_regime_detection.py
python Python/06_decision_logic.py
python Python/07_decision_rules.py
python Python/08_capital_allocation.py
```
## Research Design Principles

This framework was built on several core quantitative principles:

* **Layered Architecture:** Strictly separate measurement (*what is happening?*), classification (*what regime are we in?*), and decision-making (*what should we do?*).
* **Independent Risk Assessment:** Treat market risk and volatility as completely independent dimensions from the price trend.
* **Capital Preservation:** Convert market conditions into explicit exposure guidance. The system prioritizes preserving capital when risk conditions deteriorate.
* **Rules Over Discretion:** Prefer a transparent, fully inspectable rule framework over an opaque "black-box" signal. 
* **Confirmation Required:** Only increase portfolio exposure when multiple market conditions consistently support it.

## Limitations

This is a quantitative research project and should **not** be interpreted as investment advice. Important limitations to keep in mind include:

* **Historical Bias:** Past historical performance does not guarantee future results.
* **Unprecedented Events:** Rule-based systems can fail during extreme or unprecedented market conditions.
* **Rapid Shifts:** Volatility regimes can change rapidly and unpredictably.
* **Missing Costs:** The current system does not model real-world transaction costs, slippage, or taxes.
* **Not a Live Strategy:** Extensive backtesting and out-of-sample validation should be rigorously performed before treating this framework as a live trading strategy.

---

## Future Development

I am continuously looking to improve this framework. Planned future extensions include:

* **Alternative Data:** Integration of India VIX and macroeconomic variables.
* **Cross-Asset Validation:** Confirming signals using other asset classes.
* **Granular Analysis:** Sector-level regime analysis.
* **Robust Testing:** Walk-forward validation and rigorous out-of-sample testing.
* **Real-World Modeling:** Adding transaction-cost modeling and risk-adjusted performance metrics.
* **Reporting:** Building interactive dashboards and automated scenario/stress testing.

---

## Disclaimer

**For educational, research, and quantitative-analysis purposes only.**

The outputs generated by this system are research signals and decision-support information. Nothing in this project constitutes financial, investment, tax, or trading advice. Users are strictly responsible for conducting their own research and evaluating their own financial circumstances and risk tolerance before making any investment decisions.

---

## Author

**Yash Keram**


























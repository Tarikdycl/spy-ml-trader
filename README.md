# SPY ML Trader

A machine learning research project exploring whether engineered market features can provide useful signal for short-horizon SPY price direction prediction.

The project focuses on building a leakage-aware time-series machine learning workflow using historical SPY market data, chronological validation, and walk-forward evaluation.

> **Status:** Work in progress. The current version includes market data preparation, feature engineering, baseline modeling, and walk-forward validation. Backtesting and paper-trading components are planned next.

---

## Project Goal

The goal of this project is to study a realistic machine learning workflow for financial time-series data.

Instead of using random train/test splits, the project preserves chronological order and evaluates models on future data only.

**Research question:** Can engineered market features provide useful predictive signal for short-horizon SPY price movements?

This project is intended as an ML research and learning project, not a production trading system or financial advice.

---

## Current Pipeline

```text
Market Data
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Target Construction
    ↓
Baseline Model
    ↓
Chronological Validation
    ↓
Walk-Forward Evaluation
    ↓
Backtesting
    ↓
Paper Trading
```

- [x] Historical SPY market data
- [x] Data preparation
- [x] Feature engineering
- [x] Short-horizon target construction
- [x] Baseline machine learning model
- [x] Chronological train/test validation
- [x] Walk-forward validation
- [ ] Probability-based signal design
- [ ] Backtesting
- [ ] Risk and transaction-cost analysis
- [ ] Paper-trading experiment

---

## Dataset

Historical SPY market data is downloaded using `yfinance`.

The project works with:
- Adjusted Close
- Open
- High
- Low
- Close
- Volume

Generated CSV files are excluded from the repository and can be recreated through the notebooks.

---

## Target

The current classification target is based on the future 3-day SPY return.

```python
future_return_3d = Adj Close.shift(-3) / Adj Close - 1
```

```text
1 -> future 3-day return > 0.5%
0 -> otherwise
```

---

## Feature Engineering

The project explores normalized market features rather than relying on raw price levels.

Feature groups include:
- Momentum
- Trend
- Volatility
- Market behavior

Feature engineering uses only information available at each point in time to reduce look-ahead leakage.

---

## Baseline Modeling

The first baseline model is Logistic Regression.

The goal is to create an interpretable benchmark before testing more complex methods.

The workflow includes:
- feature scaling where appropriate
- chronological train/test splitting
- probability predictions
- classification evaluation

---

## Time-Series Validation

Financial data requires different validation logic from many standard machine learning datasets.

This project uses:
- chronological train/test splits
- walk-forward validation
- strictly forward-looking evaluation

The goal is to simulate how a model would have performed using only information available at that point in history.

---

## Project Structure

```text
spy-ml-trader/
│
├── notebooks/
│   ├── 01_market_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   └── 04_walk_forward_validation.ipynb
│
├── data/
├── models/
├── reports/
├── src/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notebooks

### 01 - Market Data
Downloads and prepares historical SPY market data.

### 02 - Feature Engineering
Creates momentum, trend, volatility, and market-behavior features.

### 03 - Modeling Baseline
Builds the first machine learning baseline using Logistic Regression.

### 04 - Walk-Forward Validation
Evaluates the model using chronological walk-forward validation rather than random cross-validation.

---

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- yfinance
- Matplotlib
- Jupyter

---

## Next Steps

1. Analyze walk-forward model performance
2. Inspect probability calibration and decision thresholds
3. Convert model probabilities into trading signals
4. Build a backtesting framework
5. Include transaction costs and realistic trading assumptions
6. Evaluate risk-adjusted performance
7. Experiment with additional baseline models
8. Eventually test the system through paper trading

---

## Limitations

This project is still under active development.

Current limitations include:
- no completed trading strategy yet
- no transaction-cost modeling yet
- no finalized backtest
- limited model experimentation
- financial markets are noisy and difficult to predict
- historical performance would not guarantee future profitability

---

## Disclaimer

This repository is an educational machine learning research project and is not financial advice.

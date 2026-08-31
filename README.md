# \# SPY ML Trader

# 

# A machine learning research project exploring whether engineered market features can provide useful signal for short-horizon SPY price direction prediction.

# 

# The project focuses on building a leakage-aware time-series machine learning workflow using historical SPY market data, chronological validation, and walk-forward evaluation.

# 

# > Status: Work in progress. The current version includes market data preparation, feature engineering, baseline modeling, and walk-forward validation. Backtesting and paper-trading components are planned next.

# 

# \---

# 

# \## Project Goal

# 

# The goal of this project is to study a realistic machine learning workflow for financial time-series data.

# 

# Instead of using random train/test splits, the project preserves chronological order and evaluates models on future data only.

# 

# The current research question is:

# 

# \*\*Can engineered market features provide useful predictive signal for short-horizon SPY price movements?\*\*

# 

# This project is intended as an ML research and learning project, not a production trading system or financial advice.

# 

# \---

# 

# \## Current Pipeline

# 

# ```text

# Market Data

# &#x20;   ↓

# Data Cleaning

# &#x20;   ↓

# Feature Engineering

# &#x20;   ↓

# Target Construction

# &#x20;   ↓

# Baseline Model

# &#x20;   ↓

# Chronological Validation

# &#x20;   ↓

# Walk-Forward Evaluation

# &#x20;   ↓

# Backtesting

# &#x20;   ↓

# Paper Trading


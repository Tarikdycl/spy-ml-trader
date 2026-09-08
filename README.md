# SPY ML Trader

A machine learning research project exploring whether engineered market features can provide useful predictive signal for short-horizon SPY price movements.

The project focuses on building a leakage-aware financial time-series machine learning workflow using historical SPY data, chronological train/test splits, walk-forward validation, target diagnosis, and ensemble models.

> **Status:** Active development. The current version includes data preparation, feature engineering, multiple model families, target comparison, walk-forward validation, and final model selection. Backtesting and paper-trading components are planned next.

---

## Project Goal

The goal of this project is to study a realistic machine learning workflow for financial time-series data.

Rather than using random train/test splits, the project preserves chronological order and evaluates models only on future observations.

The main research question is:

> Can engineered market features provide stable predictive signal for short-horizon SPY price movements?

The project also investigates a second question:

> Is weak predictive performance caused by limited model complexity, weak features, noisy target definitions, or the inherent difficulty of financial markets?

This repository is intended as a machine learning research and learning project, not a production trading system or financial advice.

---

## Current Pipeline

```text
Historical SPY Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Initial Target Construction
        ↓
Baseline Modeling
        ↓
Decision Trees & Ensembles
        ↓
Walk-Forward Validation
        ↓
Feature Diagnosis
        ↓
Target Diagnosis
        ↓
Alternative Target Comparison
        ↓
Final Model Comparison
        ↓
Selected Random Forest Model
        ↓
Backtesting
        ↓
Paper Trading
```

### Progress

- [x] Historical SPY market data
- [x] Data preparation
- [x] Feature engineering
- [x] Initial short-horizon target construction
- [x] Logistic Regression baseline
- [x] Decision Tree experiments
- [x] Random Forest modeling
- [x] Gradient Boosting modeling
- [x] Chronological train/test evaluation
- [x] Walk-forward validation
- [x] Bias-variance and overfitting analysis
- [x] Feature importance analysis
- [x] Feature separation analysis
- [x] Feature correlation analysis
- [x] Alternative target comparison
- [x] Walk-forward target validation
- [x] Final model comparison
- [ ] Expanded market-regime features
- [ ] Probability-based trading signal design
- [ ] Backtesting
- [ ] Transaction-cost analysis
- [ ] Risk-adjusted evaluation
- [ ] Paper-trading experiment

---

## Dataset

Historical SPY market data is downloaded using `yfinance`.

The raw market dataset includes:

- Adjusted Close
- Open
- High
- Low
- Close
- Volume

Generated CSV files are excluded from the repository and can be recreated through the notebooks.

The final modeling dataset contains more than 8,000 daily observations spanning from 1993 through 2026.

---

## Feature Engineering

The project uses normalized market features rather than raw price levels.

The current feature set includes:

### Momentum Features

- 1-day return
- 3-day return
- 5-day return
- 10-day return
- 20-day return

### Trend Features

- price relative to 5-day moving average
- price relative to 20-day moving average
- 5-day moving average relative to 20-day moving average

### Volatility Features

- 5-day realized volatility
- 20-day realized volatility
- volatility ratio
- daily high-low range

### Market Behavior Features

- overnight gap
- intraday return
- volume ratio

All features are constructed using information available at the current point in time in order to reduce look-ahead leakage.

---

## Initial Target

The project initially used a 3-day forward return target.

```python
future_return_3d = AdjClose.shift(-3) / AdjClose - 1
```

The original classification target was:

```text
1 -> future 3-day return > 0.5%
0 -> otherwise
```

This target produced only modest predictive performance, which motivated a deeper investigation into target design.

---

## Target Diagnosis

Several alternative prediction targets were evaluated using the same feature set and Random Forest model.

The tested targets included:

```text
3-day return > 0%
3-day return > 0.5%
5-day return > 0.5%
5-day return > 1%
```

Initial test-period ROC-AUC results suggested that larger short-horizon moves were more learnable than simple direction prediction.

The strongest candidate was:

```text
5-day future return > 1%
```

Walk-forward validation confirmed this result.

### Walk-Forward Target Comparison

```text
Target        Mean Validation ROC-AUC    Std
5D > 1%              0.5973            0.0339
3D > 0.5%            0.5687            0.0244
5D > 0.5%            0.5378            0.0250
3D > 0%              0.5121            0.0203
```

The results suggest that predicting very small directional movements is close to random using the current feature set, while larger 5-day moves contain somewhat more learnable structure.

---

## Feature Diagnosis

The project also investigates whether the existing feature set provides meaningful class separation.

Feature means were compared between positive and negative target classes and standardized by each feature's standard deviation.

The strongest separation appeared in features related to:

- daily trading range
- medium-term trend position
- realized volatility
- recent returns

The analysis suggested a weak relationship between higher volatility, weaker recent momentum, and increased probability of future positive moves.

However, standardized differences remained relatively small, indicating that the classes are not cleanly separable.

---

## Feature Correlation

A correlation matrix was used to inspect redundancy among the engineered features.

Several natural feature groups appeared:

```text
Momentum / Trend
    return_3d
    return_5d
    return_10d
    return_20d
    price_vs_sma5
    price_vs_sma20
    sma5_vs_sma20

Volatility / Range
    volatility_5d
    volatility_20d
    volatility_ratio
    high_low_range

Market Behavior
    overnight_gap
    intraday_return
    volume_ratio
```

The feature set does not suffer from extreme redundancy, but much of the information is concentrated in price, momentum, and volatility.

Future work should therefore focus on adding new sources of market information rather than simply adding more variations of existing return features.

---

## Baseline Modeling

Logistic Regression was used as the initial interpretable baseline.

The workflow includes:

- chronological train/test splitting
- feature scaling through a Scikit-Learn pipeline
- probability predictions
- ROC-AUC evaluation
- precision and recall analysis

For the selected 5-day > 1% target, Logistic Regression achieved approximately:

```text
Test ROC-AUC: 0.545
```

---

## Decision Tree Experiments

An unrestricted Decision Tree was intentionally tested to study overfitting behavior.

The tree grew to approximately:

```text
Depth: 32
Leaves: 1360
Nodes: 2719
```

It achieved perfect training performance but nearly random test performance.

```text
Train ROC-AUC: 1.000
Test ROC-AUC:  0.507
```

This provided a direct example of high variance and overfitting.

A regularized tree with reduced depth and larger minimum leaf size substantially improved generalization.

---

## Random Forest

Random Forest was introduced to reduce the instability of individual decision trees through:

- bootstrap sampling
- multiple decision trees
- random feature subsets
- probability aggregation

The selected Random Forest configuration uses:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)
```

For the final selected target:

```text
Test ROC-AUC: approximately 0.578
```

---

## Gradient Boosting

Gradient Boosting was also evaluated as a sequential ensemble method.

The model used small decision trees that progressively corrected the errors of the existing ensemble.

Its final test performance was very close to Random Forest.

```text
Test ROC-AUC: approximately 0.576
```

---

## Walk-Forward Model Comparison

The final model families were compared using five chronological validation folds.

```text
Model                 Mean Validation ROC-AUC    Std
Random Forest                 0.5974             0.0342
Logistic Regression           0.5940             0.0315
Gradient Boosting             0.5900             0.0289
```

Random Forest achieved the highest average validation ROC-AUC.

However, all three models performed similarly.

This suggests that model complexity is no longer the main bottleneck.

Future improvements are more likely to come from:

- stronger features
- additional market context
- regime information
- improved signal design

---

## Time-Series Validation

Financial time-series data requires different validation logic from many standard machine learning datasets.

This project uses:

- chronological train/test splits
- expanding-window validation
- Scikit-Learn `TimeSeriesSplit`
- walk-forward evaluation
- future-only validation periods

The goal is to simulate how a model would have performed using only information available at that point in history.

Random shuffling is avoided throughout the primary modeling workflow.

---

## Project Structure

```text
spy-ml-trader/
│
├── notebooks/
│   ├── 01_market_data.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_walk_forward_validation.ipynb
│   ├── 05_theory_notes.ipynb
│   ├── 06_logreg_from_scratch.ipynb
│   ├── 07_tree_models.ipynb
│   ├── 08_feature_target_diagnosis.ipynb
│   └── 09_final_modeling.ipynb
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

### 01 — Market Data

Downloads and prepares historical SPY market data.

### 02 — Feature Engineering

Creates momentum, trend, volatility, volume, and market-behavior features.

### 03 — Modeling Baseline

Builds the initial Logistic Regression baseline.

### 04 — Walk-Forward Validation

Introduces chronological and walk-forward evaluation.

### 05 — Theory Notes

Contains supporting machine learning theory and project notes.

### 06 — Logistic Regression From Scratch

Implements Logistic Regression manually to better understand the optimization process behind the model.

### 07 — Tree Models

Compares:

- unrestricted Decision Tree
- regularized Decision Tree
- Random Forest
- Gradient Boosting

This notebook also explores bias, variance, overfitting, ensemble learning, feature importance, and Random Forest walk-forward tuning.

### 08 — Feature & Target Diagnosis

Investigates:

- target balance
- feature separation
- standardized feature differences
- feature correlation
- alternative prediction horizons
- alternative thresholds
- walk-forward target comparison

This analysis leads to the selection of the 5-day > 1% target.

### 09 — Final Modeling

Rebuilds the modeling pipeline using the selected target and compares:

- Logistic Regression
- Random Forest
- Gradient Boosting

Models are evaluated using both chronological test data and walk-forward validation.

Random Forest is selected as the current final model.

---

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- yfinance
- Matplotlib
- Jupyter
- Git / GitHub

---

## Current Findings

The project currently suggests that:

1. Simple short-term SPY direction prediction is extremely noisy.
2. Larger 5-day movements appear somewhat more predictable with the current feature set.
3. Unrestricted decision trees severely overfit financial time-series data.
4. Regularization and ensemble methods improve generalization.
5. Random Forest currently provides the strongest average walk-forward performance.
6. Logistic Regression remains surprisingly competitive with more complex models.
7. Model complexity alone is unlikely to provide major additional performance gains.
8. Future improvements should focus primarily on feature quality and new sources of market information.

---

## Next Steps

### Feature Expansion

Potential new features include:

- RSI
- rolling drawdown
- distance from recent highs and lows
- volatility regime indicators
- volatility acceleration
- trend acceleration
- volume surprise
- VIX
- broader market context
- market breadth
- calendar features

### Trading Layer

The next stage of the project will investigate:

1. probability calibration
2. decision thresholds
3. probability-based trading signals
4. trade entry and exit logic
5. backtesting
6. transaction costs
7. slippage assumptions
8. risk-adjusted returns
9. benchmark comparison
10. paper trading

---

## Limitations

The current project remains an experimental machine learning research system.

Important limitations include:

- predictive performance remains modest
- current features are primarily derived from SPY price and volume data
- market regimes change over time
- financial data contains substantial noise
- ROC-AUC does not directly imply trading profitability
- transaction costs and slippage have not yet been modeled
- no finalized trading strategy exists yet
- no completed backtest exists yet
- historical predictive performance does not guarantee future profitability

---

## Disclaimer

This repository is an educational machine learning research project.

It is not financial advice, an investment recommendation, or a production trading system.

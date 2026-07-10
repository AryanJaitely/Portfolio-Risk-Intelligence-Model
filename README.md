# PRISM – Portfolio Risk Intelligence and Selection Model

PRISM is an end-to-end machine learning and portfolio optimization framework designed to construct diversified mutual fund portfolios using historical market data.

The project combines financial time-series analysis, machine learning, portfolio theory, and optimization techniques to recommend portfolio allocations that balance expected return, risk, diversification, holdings overlap, and sector concentration.

Unlike traditional portfolio optimization approaches that primarily consider return and variance, PRISM evaluates portfolios across multiple dimensions to produce more robust investment allocations.


## Features

- Historical NAV data analysis
- Financial feature engineering
- Random Forest-based return and volatility prediction
- Walk-forward time-series validation
- Portfolio risk analysis
- Diversification measurement
- Holdings overlap analysis
- Sector concentration penalty
- Portfolio consistency scoring
- Simulated Annealing optimization
- Historical backtesting
- Benchmark comparison
- Explainable portfolio recommendations

---

## System Workflow

```
Historical NAV Data
        │
        ▼
Data Processing
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Prediction
        │
        ▼
Portfolio Risk Analysis
        │
        ▼
Portfolio Scoring
        │
        ▼
Simulated Annealing Optimization
        │
        ▼
Backtesting & Benchmark Evaluation
        │
        ▼
Portfolio Recommendation
```

---

## Machine Learning

The prediction engine uses a **Random Forest Regressor** trained on engineered financial features extracted from historical NAV data.

Features include:

- Historical returns
- Rolling volatility
- Momentum
- Rolling Sharpe Ratio
- Fund age
- Historical performance statistics

To avoid look-ahead bias, the model is evaluated using **walk-forward validation**, which better reflects real-world financial forecasting.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Directional Accuracy

---

## Portfolio Optimization

Each candidate portfolio is evaluated using multiple criteria:

- Expected Return
- Predicted Volatility
- Diversification Ratio
- Holdings Overlap
- Sector Concentration
- Consistency Score

All metrics are normalized before being combined into the **.P.R.I.S.M. Score**, ensuring no individual factor dominates the optimization process.

Portfolio weights are optimized using **Simulated Annealing**, enabling efficient exploration of the search space.

---

## Backtesting

The optimized portfolio is evaluated on unseen historical data using a formation-period/test-period split.

Performance is compared against benchmark allocation strategies using metrics such as:

- Portfolio Return
- Portfolio Volatility
- Maximum Drawdown



## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Random Forest
- Simulated Annealing



## Dataset

This repository does not include the complete historical NAV dataset due to GitHub file size limitations.

To run the project, download the historical mutual fund NAV data separately and place it in the appropriate data directory along with:

- "fund_meta.csv"
- "holdings.csv"
- "sectors.csv"


## Future Improvements

- Deep learning-based forecasting
- Alternative optimization algorithms
- Interactive portfolio dashboard
- Automated data ingestion
- Hyperparameter optimization


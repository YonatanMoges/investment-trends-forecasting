
# Stock Market Analysis and Forecasting

This project provides a framework for analyzing and forecasting stock market trends with a focus on optimizing a portfolio containing Tesla (TSLA), S&P 500 ETF (SPY), and Vanguard Bond ETF (BND). The notebook demonstrates each step of the workflow: data preprocessing, model training, forecasting, and portfolio optimization.

## Project Overview

### Key Tasks

1. **Data Preprocessing and Exploration**: 
   - Load, clean, and analyze historical data for the assets.
   - Calculate daily returns, volatility, and seasonal patterns.
   - Functions can be found in `scripts/data_preprocessing.py`
   
2. **Model Training**:
   - Train ARIMA, SARIMA, and LSTM models to forecast stock prices, using historical data.
   - Functions can be found in `scripts/time_series_models.py`

3. **Market Trend Forecasting**:
   - Use trained LSTM models to forecast future stock prices and assess trends.
   - Functions can be found `scripts/forecasting.py`

4. **Portfolio Optimization**:
   - Optimize the portfolio based on forecasted returns, aiming to maximize the Sharpe Ratio and manage risk.
   - Functions can be found in `portfolio_optimization.py`

---

## Project Structure

```plaintext
.
├── data/                           # Contains historical data for TSLA, SPY, BND (if available)
├── models/                         # Stores trained model files (e.g., LSTM models)
├── notebook/
│   └── Stock_Market_Analysis_and_Forecasting_Demo.ipynb  # Main notebook for analysis and forecasting
├── scripts/
│   ├── data_preprocessing.py       # Class for data loading, cleaning, and exploratory analysis
│   ├── time_series_models.py       # Class for training ARIMA, SARIMA, and LSTM models
│   ├── forecasting.py              # Class for market forecasting using trained LSTM models
│   └── portfolio_optimization.py   # Class for optimizing portfolio based on forecasted returns
├── src/
├── tests/
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## How to Run

1. **Clone the Repository**

   Begin by cloning this repository to your local machine:
   ```bash
   git clone https://github.com/YonatanMoges/investment-trends-forecasting.git
   cd investment-trends-forecasting
   ```

2. **Set Up the Environment**

   It’s recommended to use a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
   pip install -r requirements.txt
   ```
---

## Usage

1. **Data Preprocessing and Exploration**
   - Load and clean historical stock data.
   - Perform exploratory data analysis, including volatility analysis and Sharpe Ratio calculation.
   - Example code:
     ```python
     from data_preprocessing import StockDataProcessor
     
     tickers = ['TSLA', 'SPY', 'BND']
     processor = StockDataProcessor(tickers, '2015-01-01', '2023-01-01')
     processor.download_data()
     processor.clean_data()
     processor.basic_stats()
     ```

2. **Model Training**
   - Train ARIMA, SARIMA, and LSTM models on historical data to predict future prices.
   - Example code:
     ```python
     from time_series_models import ModelTrainer
     
     trainer = ModelTrainer(processor.data['TSLA'])
     arima_model = trainer.train_arima(order=(5, 1, 0))
     lstm_model = trainer.train_lstm(sequence_length=60, units=50, epochs=10, batch_size=32)
     ```

3. **Market Forecasting**
   - Use the trained LSTM model to forecast future prices for each asset.
   - Example code:
     ```python
     from forecasting import MarketForecaster
     
     forecaster = MarketForecaster(lstm_model, processor.data['TSLA'])
     predictions = forecaster.forecast(steps=180)
     ```

4. **Portfolio Optimization**
   - Optimize the asset allocation in a portfolio to maximize returns and minimize risk.
   - Example code:
     ```python
     from portfolio_optimization import PortfolioOptimizer
     
     optimizer = PortfolioOptimizer(processor.daily_returns[['TSLA', 'SPY', 'BND']])
     optimal_weights = optimizer.optimize_portfolio()
     ```

---

## Notebook Walkthrough

The `Stock_Market_Analysis_and_Forecasting_Demo.ipynb` notebook demonstrates the end-to-end workflow, covering all tasks with modular, reusable classes for each stage of analysis.

---

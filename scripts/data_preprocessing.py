# task_1_preprocessing.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

class StockDataProcessor:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.daily_returns = None

    def download_data(self):
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        self.data.dropna(inplace=True)
    
    def clean_data(self):
        self.data = self.data.interpolate(method='linear').dropna()

    def basic_stats(self):
        print(self.data.describe())

    def calculate_daily_returns(self):
        self.daily_returns = self.data.pct_change().dropna()

    def plot_closing_prices(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.data['TSLA'], label='TSLA Closing Price')
        plt.title('TSLA Closing Prices')
        plt.show()

    def analyze_volatility(self):
        rolling_mean = self.data['TSLA'].rolling(window=30).mean()
        rolling_std = self.data['TSLA'].rolling(window=30).std()
        return rolling_mean, rolling_std

    def decompose_seasonality(self):
        decomposition = seasonal_decompose(self.data['TSLA'], model='additive', period=365)
        decomposition.plot()
        plt.show()

    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        mean_return = self.daily_returns['TSLA'].mean()
        std_dev = self.daily_returns['TSLA'].std()
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev
        return sharpe_ratio

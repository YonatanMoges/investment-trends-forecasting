import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

class DataPreprocessing:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}

    def fetch_data(self):
        """
        Fetch historical data using YFinance.
        """
        for ticker in self.tickers:
            print(f"Fetching data for {ticker}...")
            self.data[ticker] = yf.download(ticker, start=self.start_date, end=self.end_date)
        return self.data

    def clean_data(self):
        """
        Clean the data: handle missing values and check data types.
        """
        for ticker, df in self.data.items():
            # Fill missing values using forward fill
            df.fillna(method='ffill', inplace=True)
            df.dropna(inplace=True)
            # Ensure correct data types
            df['Volume'] = df['Volume'].astype(float)
            self.data[ticker] = df
        return self.data

    def normalize_data(self):
        """
        Normalize the 'Adj Close' prices using MinMaxScaler.
        """
        scaler = MinMaxScaler()
        for ticker, df in self.data.items():
            df['Normalized Close'] = scaler.fit_transform(df[['Adj Close']])
            self.data[ticker] = df
        return self.data

    def visualize_data(self):
        """
        Visualize the closing prices and normalized prices.
        """
        plt.figure(figsize=(14, 7))
        for ticker, df in self.data.items():
            plt.plot(df.index, df['Adj Close'], label=f'{ticker} Adj Close')
        plt.title("Adjusted Close Prices of Assets Over Time")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.show()

    def analyze_volatility(self, window=30):
        """
        Calculate and plot rolling mean and standard deviation for each ticker.
        """
        plt.figure(figsize=(14, 7))
        
        for ticker, df in self.data.items():
            # Ensure the DataFrame has a DateTime index
            df = df.copy()
            df.index = pd.to_datetime(df.index)

            # Calculate rolling mean and standard deviation
            rolling_mean = df['Adj Close'].rolling(window=window).mean()
            rolling_std = df['Adj Close'].rolling(window=window).std()

            # Drop NaN values
            rolling_mean = rolling_mean.dropna()
            rolling_std = rolling_std.dropna()

            # Align the indexes of rolling mean and std
            rolling_mean, rolling_std = rolling_mean.align(rolling_std, join='inner')

            # Calculate the lower and upper bounds
            lower_bound = (rolling_mean - rolling_std).dropna()
            upper_bound = (rolling_mean + rolling_std).dropna()

            # Plot rolling mean
            plt.plot(rolling_mean.index, rolling_mean, label=f'{ticker} Rolling Mean ({window} days)')

            plt.fill_between(
            rolling_mean.index,
            lower_bound.loc[rolling_mean.index].squeeze(),
            upper_bound.loc[rolling_mean.index].squeeze(),
            alpha=0.2
            )

        plt.title(f"Rolling Mean and Volatility (Window = {window} days)")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.show()



    def decompose_trend_seasonality(self, ticker, model='additive'):
        """
        Decompose the time series into trend, seasonal, and residual components.
        """
        df = self.data[ticker]
        decomposition = seasonal_decompose(df['Adj Close'], model=model, period=365)
        fig = decomposition.plot()
        fig.set_size_inches(14, 10)
        plt.show()

# Example usage
if __name__ == "__main__":
    tickers = ['TSLA', 'BND', 'SPY']
    start_date = '2015-01-01'
    end_date = '2024-10-31'

    dp = DataPreprocessing(tickers, start_date, end_date)
    dp.fetch_data()
    dp.clean_data()
    dp.normalize_data()
    dp.visualize_data()
    dp.analyze_volatility()
    dp.decompose_trend_seasonality(ticker='TSLA')

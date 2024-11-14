# task_3_forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

class MarketForecaster:
    def __init__(self, model, data, scaler, sequence_length=60):
        self.model = model
        self.data = data
        self.scaler = scaler
        self.sequence_length = sequence_length

    def forecast(self, steps=30):
        # Reshape last_data to have the same dimensions as when scaler was fitted
        last_data = self.data[-self.sequence_length:].values.reshape(-1, 1)
        scaled_last_data = self.scaler.transform(last_data)
        
        # Prepare the input sequence for LSTM
        X_test = np.array([scaled_last_data[i:i + self.sequence_length] for i in range(len(scaled_last_data) - self.sequence_length + 1)])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Forecast future prices
        predictions = []
        for _ in range(steps):
            predicted = self.model.predict(X_test[-1].reshape(1, self.sequence_length, 1))
            predictions.append(predicted[0, 0])
            
            # Update X_test with the predicted value
            new_scaled_data = np.append(X_test[-1][1:], predicted[0, 0])
            X_test = np.vstack([X_test, new_scaled_data.reshape(1, self.sequence_length, 1)])
        
        # Inverse transform to get the actual prices
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions.flatten()

    def plot_forecast(self, predictions):
        # Plot forecasted prices along with historical data
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data, label='Historical Data')
        
        # Generate forecast dates starting from the next day after the last historical date
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=len(predictions))
        plt.plot(forecast_dates, predictions, label='Forecast', color='orange')
        
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Stock Price Forecast')
        plt.legend()
        plt.show()

    def analyze_forecast(self, forecast_prices):
        """
        Analyze the forecast to identify trends and risks.

        Parameters:
        - forecast_prices (list): Predicted stock prices.
        """
        # Calculate basic statistics
        mean_price = np.mean(forecast_prices)
        volatility = np.std(forecast_prices)

        # Trend analysis
        if forecast_prices[-1] > forecast_prices[0]:
            trend = "Upward Trend"
        elif forecast_prices[-1] < forecast_prices[0]:
            trend = "Downward Trend"
        else:
            trend = "Stable Trend"

        # Display analysis
        print("Forecast Analysis:")
        print(f"Mean Forecast Price: {mean_price:.2f}")
        print(f"Volatility (Standard Deviation): {volatility:.2f}")
        print(f"Identified Trend: {trend}")

        # Highlight market opportunities and risks
        if volatility > 0.1 * mean_price:
            print("High volatility detected. Caution advised for investors.")
        if trend == "Upward Trend":
            print("Potential market opportunity: Expected price increase.")
        elif trend == "Downward Trend":
            print("Potential market risk: Expected price decline.")
